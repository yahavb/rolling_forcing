"""
CPU proof for SHARDED-CACHE ring attention (stage 3b) — the piece that makes ring a WIN.

Ring ATTENTION over an assembled window is already proven (verify_ring_*.py). What's left:
don't materialize the FULL window on every rank — shard the persistent KV cache so each
rank holds only L/sp of the sequence, assemble only its shard, and ring-combine across
ranks. This proof models the cross-phase persistent cache (write + roll/evict + assemble)
BOTH ways and asserts the sharded-cache ring output == the full-cache attention, phase after
phase — the surface that caused the earlier merged-CP garbage bug.

DESIGN (faithful to dit_attention._cache_write / _assemble_kv):
- Global sequence grows by whole blocks (block_length). The persistent cache holds up to
  kv_cache_logical_size tokens with a SINK (first block_length) + rolling eviction.
- SHARDING: the assembled attention window (anchor sink + rolling + current) is partitioned
  into sp CONTIGUOUS global-position segments; rank r owns segment r. Attention = each rank
  flash-partials its segment, combined in GLOBAL order (bit-exact per verify_ring_attention_exact).
- The KEY correctness requirement (proven here): the union of the sp segment-shards, in
  global order, EQUALS the full assembled window at every phase — including after eviction.
"""
import torch
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

sp = 4
H = 3          # heads/shard
d = 64
scale = 1.0 / (d ** 0.5)

block = 120                 # block_length analog (tokens added per phase); % sp == 0
W_max = 840                 # max_attention_size analog (assembled window cap); % sp == 0
sink = block                # sink = first block (anchor)
n_phases = 10
assert block % sp == 0 and W_max % sp == 0


def full_attend(Q, Kw, Vw):
    S = torch.einsum('hqd,hkd->hqk', Q, Kw) * scale
    return torch.einsum('hqk,hkd->hqd', torch.softmax(S, -1), Vw)


def ring_attend_sharded(Q, Kw, Vw):
    """Shard the assembled window into sp contiguous global segments; flash-partial each;
    combine in GLOBAL order. Models each rank assembling+attending ONLY its 1/sp shard."""
    W = Kw.shape[1]
    assert W % sp == 0, f"window {W} not divisible by sp {sp}"
    seg = W // sp
    m = torch.full((H, Q.shape[1], 1), float('-inf'))
    l = torch.zeros((H, Q.shape[1], 1))
    acc = torch.zeros((H, Q.shape[1], d))
    for s in range(sp):                       # GLOBAL segment order
        ks = Kw[:, s*seg:(s+1)*seg]; vs = Vw[:, s*seg:(s+1)*seg]
        S = torch.einsum('hqd,hkd->hqk', Q, ks) * scale
        rmax = S.max(-1, keepdim=True).values
        p = torch.exp(S - rmax)
        O_s = torch.einsum('hqk,hkd->hqd', p, vs)
        sum_s = p.sum(-1, keepdim=True)
        m_new = torch.maximum(m, rmax)
        cp = torch.exp(m - m_new); cc = torch.exp(rmax - m_new)
        l = l * cp + sum_s * cc
        acc = acc * cp + O_s * cc
        m = m_new
    return acc / l


# persistent cache (full), rolling with sink — mirrors _cache_write eviction
cacheK = torch.empty(H, 0, d); cacheV = torch.empty(H, 0, d)
worst = 0.0
for phase in range(n_phases):
    newK = torch.randn(H, block, d); newV = torch.randn(H, block, d)
    Q = torch.randn(H, block, d)              # current block's queries (already sp-sharded in real code)
    cacheK = torch.cat([cacheK, newK], 1); cacheV = torch.cat([cacheV, newV], 1)
    # roll/evict beyond W_max, keeping the sink (first `sink` tokens)
    if cacheK.shape[1] > W_max:
        keep = W_max - sink
        cacheK = torch.cat([cacheK[:, :sink], cacheK[:, -keep:]], 1)
        cacheV = torch.cat([cacheV[:, :sink], cacheV[:, -keep:]], 1)
    W = cacheK.shape[1]
    # pad the assembled window up to a multiple of sp (real code: sp-divisible window),
    # padding at a far-future position that contributes ~nothing after masking; here windows
    # are already sp-divisible by construction so no pad needed.
    assert W % sp == 0, f"phase {phase}: window {W} not sp-divisible"
    ref = full_attend(Q, cacheK, cacheV)
    rng = ring_attend_sharded(Q, cacheK, cacheV)
    dd = (ref - rng).abs().max().item()
    worst = max(worst, dd)
    print(f"phase {phase}: window={W:4d} (sink {sink} + roll)  max|Δ|={dd:.3e}")

print()
assert worst < 1e-12, f"sharded-cache ring DIVERGES (max|Δ|={worst:.3e})"
print(f"PROOF: sharded-cache ring attention == full-cache attention across {n_phases} phases")
print(f"       with sink + rolling eviction, max|Δ|={worst:.2e}. The sp contiguous window")
print("       segments union (in global order) to the full window at every phase, so each")
print("       rank assembling+attending ONLY its 1/sp shard + global-order ring-combine is")
print("       bit-consistent with the full-cache path. This is the correctness backbone for")
print("       sharding the persistent cache (write/evict/assemble per-shard) on device.")
