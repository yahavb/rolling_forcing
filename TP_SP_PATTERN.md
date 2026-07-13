# TP×SP Parallelism Pattern on Neuron

## The Constraint

Tensor Parallelism (TP) shards attention heads across ranks. The hard requirement:

```
num_heads % tp_degree == 0
```

If the model's head count doesn't evenly divide by the number of available NeuronCores, pure TP won't work. The solution: combine TP with Sequence Parallelism (SP) so that `tp_degree × sp_degree = world_size`, using all cores while respecting the head divisibility constraint.

## How the Groups Are Laid Out

With 8 ranks, TP=4, SP=2:

```
TP groups (heads split within each group):
  [0, 1, 2, 3]   — SP partition 0
  [4, 5, 6, 7]   — SP partition 1

SP groups (sequence split within each group):
  [0, 4]   — tp_rank=0
  [1, 5]   — tp_rank=1
  [2, 6]   — tp_rank=2
  [3, 7]   — tp_rank=3
```

- **TP** splits attention heads and FFN hidden dims → all-reduce on output projection
- **SP** splits the sequence (token) dimension → all-gather/reduce-scatter around attention

## Model Comparison

| Model | Heads | TP=8 valid? | Config Used | Rationale |
|-------|-------|-------------|-------------|-----------|
| [Rolling Forcing 1.3B](#rolling-forcing-13b) | 12 | No (12÷8=1.5) | TP=4, SP=2 | Only factoring that divides heads evenly |
| [Wan2.2-I2V-A14B](#wan22-i2v-a14b) | 40 | Yes (40÷8=5) | TP=4, SP=2 | SP halves per-rank activation memory for long sequences |
| [Wan2.2-TI2V-5B](#wan22-ti2v-5b) | 24 | Yes (24÷8=3) | TP=8, SP=1 | Heads divide cleanly — no SP needed |

---

## Rolling Forcing 1.3B

**Path:** `~/rolling_forcing/`

- **Architecture:** dim=2048, 12 heads, 32 layers (Wan2.1-T2V-1.3B with DMD distillation)
- **Problem:** 12 heads ÷ 8 cores = 1.5 — not an integer. Pure TP=8 is impossible.
- **Solution:** TP=4 (3 heads/rank) × SP=2 (half the sequence per rank) = 8 total ranks.
- **Group init:** `models/dit_pipeline.py:init_parallel_groups()` creates `"attn-tp"` (size 4) and `"attn-sp"` (size 2) groups.
- **Commit removing TP=8 attempt:** `2400474` — "Remove TP=8 job — model has 12 heads, not divisible by 8"

## Wan2.2-I2V-A14B

**Path:** `~/wan2-i2v-14b/`

- **Architecture:** dim=5120, 40 heads, 40 layers
- **Head divisibility:** 40÷8=5 heads/rank — TP=8 is technically valid.
- **Why TP=4×SP=2 anyway:** The 14B model at 81 frames produces large activation tensors. SP halves the sequence length each rank must store in attention, reducing peak HBM usage. Pure TP=8 would give 5 heads/rank but require each rank to hold the full sequence in self-attention.
- **Group init:** `models/parallel_state.py:init_sp_group(tp_degree=4)` creates SP groups orthogonal to TP groups.

## Wan2.2-TI2V-5B

**Path:** `~/wan2-ti2v-5b/`

- **Architecture:** dim=3072, 24 heads, 30 layers
- **Head divisibility:** 24÷8=3 heads/rank — divides cleanly.
- **Config:** TP=8, SP=1 (pure tensor parallelism, no sequence parallelism).
- **Why no SP:** The 5B model's activations fit comfortably in per-rank HBM with full-sequence attention. No need to add the communication overhead of SP.
- **This model does NOT use the TP×SP combination pattern** — it demonstrates the simpler case where heads divide evenly by the core count.

## When to Use TP×SP vs Pure TP

Use **pure TP** when:
- `num_heads % num_cores == 0`
- Per-rank activation memory fits in HBM with full sequence

Use **TP×SP** when:
- `num_heads % num_cores != 0` (forced — find largest TP that divides heads, use remaining cores for SP)
- Activation memory is too large for full sequence per rank (chosen for memory — even if pure TP is valid)
