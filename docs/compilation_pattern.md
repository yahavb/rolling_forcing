# Compilation Pattern: Stateless Compile + NKI Eager

A pattern for compiling transformer models on Neuron that achieves high throughput without fighting Dynamo graph breaks.

## The Problem

Compiling entire transformer blocks with `torch.compile(block, backend='neuron')` fragments into many small NEFFs due to graph breaks from:
- Python dict access (`kv_cache["k"]`)
- Data-dependent branches (`if num_new_tokens > kv_cache_size`)
- Dynamic slicing with computed indices (`cache[0, start:end]`)
- In-place mutation (`dst.copy_(src)`)
- Dynamic tensor creation (`torch.tensor(python_int, device=...)`)

Result: hundreds of tiny NEFFs with high dispatch overhead between them.

## The Pattern

**Compile stateless compute. Run stateful logic + attention as NKI kernels in eager.**

```python
_compile = lambda m: torch.compile(m, backend='neuron', dynamic=False, fullgraph=True)
```

### What to compile

| Component | Why |
|-----------|-----|
| Linear projections (Q, K, V, O) | Pure matmul, no state, static shapes |
| FFN (gate, up, down) | Pure matmul + activation, no state |
| Norms (LayerNorm, RMSNorm) | Elementwise, no state |
| Embeddings (patch, text, time) | Lookup + linear, no state |
| Head (final projection) | Pure matmul |

### What to leave in eager (as NKI kernels)

| Component | Why |
|-----------|-----|
| Self-attention (QKV → scores → output) | Fused NKI flash attention via `wrap_nki` HOP |
| Cross-attention | Same — fused NKI kernel |
| RoPE | NKI kernel (needs grid_sizes, frame positions) |
| KV cache read/write | `nki_op` with `mutates_args` for in-place DMA |

### What stays in Python eager (no compile, no NKI)

| Component | Why |
|-----------|-----|
| KV cache index arithmetic | Data-dependent integers, branches |
| Cache eviction logic | Conditional control flow |
| Sequence position tracking | Mutable Python state |
| Collective communication setup | `@torch.compiler.disable` |

## Implementation

### DiT / Video Diffusion (rolling_forcing, wan2-i2v-14b)

```python
# Compile stateless sub-modules
for block in model.blocks:
    block.q = _compile(block.q)       # or fused block.qkv
    block.k = _compile(block.k)
    block.v = _compile(block.v)
    block.o = _compile(block.o)
    block.ffn = _compile(block.ffn)

model.patch_embedding = _compile(model.patch_embedding)
model.text_embedding = _compile(model.text_embedding)
model.time_embedding = _compile(model.time_embedding)
model.head = _compile(model.head)

# NKI kernels run in eager via HOP
@wrap_nki
@nki.jit
def flash_self_attention(q, k, v, ...):
    # Fused attention on NeuronCore SBUF
    ...

# KV cache writes via mutation-aware custom op
@nki_op("dit::kv_cache_copy", mutates_args={"k_dst", "v_dst"})
@nki.jit
def kv_cache_copy(k_dst, k_src, v_dst, v_src):
    # DMA copy directly to pre-allocated cache
    ...
```

### LLM Decode (qwen3-vl)

```python
for layer in model.layers:
    layer.self_attn.q_proj = _compile(layer.self_attn.q_proj)
    layer.self_attn.k_proj = _compile(layer.self_attn.k_proj)
    layer.self_attn.v_proj = _compile(layer.self_attn.v_proj)
    layer.self_attn.o_proj = _compile(layer.self_attn.o_proj)
    layer.mlp.gate_proj = _compile(layer.mlp.gate_proj)
    layer.mlp.up_proj = _compile(layer.mlp.up_proj)
    layer.mlp.down_proj = _compile(layer.mlp.down_proj)

model.lm_head = _compile(model.lm_head)

# Attention: NKI prefill_attention / decode_attention kernels
# KV cache: managed in eager Python with nki_op for writes
```

## Why This Works

1. **No graph breaks** — each compiled unit is a pure function (tensor in → tensor out), no state, no branches, no dicts. `fullgraph=True` verifies this at compile time.

2. **NKI kernels are faster than traced attention** — 2-3× over what the compiler generates for attention patterns. Running them in eager via HOP avoids the graph break problem entirely.

3. **Low dispatch overhead** — fewer but larger NEFFs for the compute-heavy ops (matmul). The NKI kernel launches are lightweight (single HOP dispatch each).

4. **Collectives compose cleanly** — `all_reduce` after compiled O-projection works because the compiled graph outputs a contiguous tensor. TP/SP communication stays outside compiled regions.

## Anti-patterns

| Don't | Why |
|-------|-----|
| `torch.compile(entire_block)` | Graph breaks from cache logic → many tiny NEFFs |
| NKI kernels inside compiled graphs | Works only if `.contiguous()` is preserved (SDK-version-dependent) |
| `dynamic=True` | Neuron backend doesn't support dynamic shapes |
| Compiling code with `if` on tensor values | Data-dependent guards → recompilation every call |
| Compiling code that mutates inputs | Dynamo treats inputs as immutable; use `nki_op(mutates_args=...)` instead |

## Results

| Approach | NEFFs/rank | FPS | Issue |
|----------|-----------|-----|-------|
| Whole-block compile | 434 | 0.43 | Graph break fragmentation |
| Sub-module compile + NKI eager | ~370 | 7.8-8.6 | None — production config |

The sub-module approach produces more total NEFFs than the theoretical minimum (~60-70 for whole-block), but each NEFF executes without dispatch stalls between fragments. The 18× FPS improvement comes from eliminating the inter-NEFF dispatch overhead that dominated the whole-block approach.

## Validated On

- **rolling_forcing**: Wan2.1-T2V-1.3B DiT, 32 layers, TP=4 SP=2, 8 NeuronCores
- **qwen3-vl**: Qwen3-VL LLM, TP=8, prefill + decode
- **wan2-i2v-14b**: Wan2.1-I2V-14B DiT, TP across multiple NDs
