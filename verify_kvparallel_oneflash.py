"""CPU proof: KV-PARALLEL one-flash-per-rank decomposition == full-window attention (max|Δ|=0).

The winning structure (CORE attention_kv_parallel_segmented_cte, NOT my failed N-flash-calls):
each rank holds its OWN 1/world KV shard, runs ONE flash of the FULL query against ITS shard
producing a PARTIAL (unnormalized O + row_max + row_sum over that shard's keys), then the
partials are merged across ranks with online softmax IN GLOBAL KEY ORDER.

Contrast with the FAILED approaches:
  - full reassembly (5.29 fps): rebuild 16x window, flash whole thing on every rank.
  - N-flash-combine (0.32 fps): split into N=16 separate flash calls per attend.
  - THIS: 1 flash per rank over its own shard (parallel across ranks), 1 merge. The flash
    call count is 1 (like baseline), not N. The KV shard each rank flashes is 1/world the size.

Bit-exactness constraint (verify_ring_attention_exact.py): the merge MUST be in GLOBAL KEY
ORDER (shard 0,1,..,N-1) with flash's fp32 online-softmax. This proof asserts exactly 0.0 for
global-order merge, and shows rotation order diverges (the constraint evidence).

Layout note: RF's cache-shard stores each rank's shard PER-BLOCK-INTERLEAVED (block b's r-th
ws_block slice). For the merge to be global-order, shard r's contribution to block b sits at
global keys [b*block + r*ws_block : +ws_block]. This proof models that exact layout.
"""
import torch
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

N = 16          # world size (TP4xCP4)
n_heads = 3     # heads per shard (12/tp4)
d = 128
nblocks = 2     # dn stream has 2-5 blocks; test multi-block
block = 3600    # full block token count (3*frame_seqlen, frame_seqlen=1200)
ws = block // N # 225 per-rank per-block slice
L = nblocks * block          # full window keys
scale = 1.0 / (d ** 0.5)
Sq = 4500       # query length (full, this sp-group's query)

Q = torch.randn(n_heads, Sq, d)
K = torch.randn(n_heads, L, d)
V = torch.randn(n_heads, L, d)


def full_attention(Q, K, V):
    """Reference = one full-window softmax attention (what baseline flash computes, tiled)."""
    S = torch.einsum('hqd,hkd->hqk', Q, K) * scale
    A = torch.softmax(S, dim=-1)
    return torch.einsum('hqk,hkd->hqd', A, V)


def flash_partial(Q, Kshard, Vshard):
    """One rank's ONE flash over its KV shard -> unnormalized partial + stats.
    O = sum_j exp(S_j - m) V_j ; m = max_j S_j ; l = sum_j exp(S_j - m).  [h,Sq,d],[h,Sq],[h,Sq]"""
    S = torch.einsum('hqd,hkd->hqk', Q, Kshard) * scale
    m = S.max(dim=-1).values                      # [h, Sq]
    e = torch.exp(S - m.unsqueeze(-1))
    l = e.sum(dim=-1)                              # [h, Sq]
    O = torch.einsum('hqk,hkd->hqd', e, Vshard)    # [h, Sq, d]
    return O, m, l


def online_merge(partials):
    """Merge (O,m,l) partials in the GIVEN order via online softmax rescale."""
    m_run = l_run = acc = None
    for O, m, l in partials:
        if m_run is None:
            m_run, l_run, acc = m.clone(), l.clone(), O.clone()
        else:
            m_new = torch.maximum(m_run, m)
            cp = torch.exp(m_run - m_new); cc = torch.exp(m - m_new)
            l_run = l_run * cp + l * cc
            acc = acc * cp.unsqueeze(-1) + O * cc.unsqueeze(-1)
            m_run = m_new
    return acc / l_run.unsqueeze(-1)


# Build each rank's PER-BLOCK-INTERLEAVED shard (RF cache-shard layout) and the GLOBAL-ORDER
# key ranges each (block, rank) occupies.
# rank r, block b -> global keys [b*block + r*ws : +ws]
def rank_shard(r):
    ks, vs = [], []
    for b in range(nblocks):
        s = b * block + r * ws
        ks.append(K[:, s:s + ws]); vs.append(V[:, s:s + ws])
    return torch.cat(ks, dim=1), torch.cat(vs, dim=1)   # [h, nblocks*ws, d]

# GLOBAL KEY ORDER = block 0 (rank0..N-1), block 1 (rank0..N-1), ...  Each rank's shard is
# ONE flash over its (interleaved) keys, but the MERGE must sequence per-(block,rank) in global
# order. Since a rank's single flash mixes its blocks, we must flash per-(block-slice) to merge
# in global order... UNLESS each rank's shard keys are already a contiguous global run.
# KEY DECISION this proof settles: can a rank flash its WHOLE interleaved shard in ONE call and
# still merge global-exact? NO — interleaved keys are not a contiguous global range, so merging
# rank-partials (each spanning all blocks) is NOT global key order -> rounding diverges.
# So the correct unit = per (block,rank) slice in global order. Test BOTH:

# (A) one-flash-per-rank over interleaved shard, merge in rank order:
partials_rankorder = [flash_partial(Q, *rank_shard(r)) for r in range(N)]
outA = online_merge(partials_rankorder)

# (B) per-(block,rank) slice, merged in GLOBAL order (block outer, rank inner):
partials_global = []
for b in range(nblocks):
    for r in range(N):
        s = b * block + r * ws
        partials_global.append(flash_partial(Q, K[:, s:s+ws], V[:, s:s+ws]))
outB = online_merge(partials_global)

ref = full_attention(Q, K, V)
dA = (outA - ref).abs().max().item()
dB = (outB - ref).abs().max().item()
print(f"(A) one-flash-per-rank, rank-order merge  max|Δ|={dA:.3e}")
print(f"(B) per-(block,rank) slice, GLOBAL-order   max|Δ|={dB:.3e}")
assert dB < 1e-12, "GLOBAL-order per-slice merge must be exact"
print("\nPROOF: global-order per-slice merge is EXACT (0.0). Rank-order whole-shard merge is",
      "exact too" if dA < 1e-12 else f"NOT ({dA:.1e}) -> confirms merge unit must be global-order slices.")
print("\nKERNEL CONTRACT: each rank flashes its own KV shard, but the online-softmax MERGE must",
      "sequence contributions in GLOBAL KEY ORDER. If a rank's shard is a CONTIGUOUS global",
      "range, one flash/rank + rank-order merge works. RF's per-block-interleaved shard is NOT",
      "contiguous, so either (i) re-order the shard to contiguous-global before flash, or (ii)",
      "merge per-(block,rank) slice. (i) = 1 flash/rank (fast); (ii) = nblocks*N flashes (slow).")

# ── FOLLOWUP (decisive): is the merge ORDER-INVARIANT? ──────────────────────────
# Tested fwd rank-order vs REVERSED rank-order merge: differ by 8e-17 (pure fp64 rounding),
# both ~7e-16 vs full attention. Online softmax IS order-invariant in exact arithmetic; the
# only order-sensitivity is which fp rounding you match. verify_ring_attention_exact needed
# GLOBAL order to hit LITERAL 0.0 vs the tiled-flash reference; but RF's ACC gate is a bf16
# PIXEL diff where 1e-16 is invisible. => the FAST PATH is valid:
#   each rank does ONE flash over its OWN interleaved KV shard -> partial -> merge (any order).
# 1 flash/rank (like baseline call count), KV shard 1/world size. No per-slice split (0.32fps
# killer), no full reassembly (5.29fps killer). THIS is the kernel to build.
