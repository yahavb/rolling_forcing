"""
CPU proof for the RING-ROTATION COLLECTIVE (stage 3b, the piece that removes the world-gather).

verify_sharded_cache_ring.py proved: IF all sp window-shards are available and combined in
GLOBAL order, sharded == full attention. It ASSUMED every shard is in hand. This proof covers
the missing piece: obtaining the other ranks' shards by ROTATION (send to next / recv from
prev around the sp ring) instead of a world-gather, and combining them in the correct global
order as they arrive.

SETUP: rank r natively holds ONLY window-segment r (its locally-produced K/V shard; = global
tokens [r*seg : (r+1)*seg], which is what _qkv_rope produces after dropping the gather).
Its query shard Q_r must attend the WHOLE window = all sp segments.

RING SCHEDULE (send/recv, sp-1 steps): at step t, rank r holds segment (r - t) mod sp.
So over t=0..sp-1 it sees segments r, r-1, ..., r-(sp-1) — ALL of them, but NOT in global
order. Online-softmax combine is order-INDEPENDENT for correctness (max/sum/rescale is
commutative in exact arithmetic); for BIT-exactness vs the flash reference we combine in
global order by BUFFERING per-segment partials and merging by global index. This proof checks
BOTH: (a) rotation delivers every segment exactly once, (b) global-order merge of the
rotation-delivered partials == full attention.
"""
import torch
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

sp = 4
H = 3
d = 64
seg = 210                       # per-rank shard length (native, no pad)
L = sp * seg                    # full window
scale = 1.0 / (d ** 0.5)

# full window K/V (ground truth), and each rank's NATIVE shard = contiguous global segment
Kfull = torch.randn(H, L, d); Vfull = torch.randn(H, L, d)
Kshard = {r: Kfull[:, r*seg:(r+1)*seg] for r in range(sp)}   # rank r owns segment r
Vshard = {r: Vfull[:, r*seg:(r+1)*seg] for r in range(sp)}
# each rank also owns query segment r
Qshard = {r: torch.randn(H, seg, d) for r in range(sp)}


def flash_partial(Q, K, V):
    S = torch.einsum('hqd,hkd->hqk', Q, K) * scale
    rmax = S.max(-1, keepdim=True).values
    p = torch.exp(S - rmax)
    return torch.einsum('hqk,hkd->hqd', p, V), rmax, p.sum(-1, keepdim=True)


def full_ref(r):
    S = torch.einsum('hqd,hkd->hqk', Qshard[r], Kfull) * scale
    return torch.einsum('hqk,hkd->hqd', torch.softmax(S, -1), Vfull)


def combine_global_order(parts):
    """parts: dict seg_idx -> (O_unnorm, row_max, row_sum). Merge in ASCENDING seg index."""
    m = l = acc = None
    for s in sorted(parts):
        O_s, mx, sm = parts[s]
        if m is None:
            m, l, acc = mx, sm, O_s
        else:
            m_new = torch.maximum(m, mx)
            cp = torch.exp(m - m_new); cc = torch.exp(mx - m_new)
            l = l*cp + sm*cc; acc = acc*cp + O_s*cc; m = m_new
    return acc / l


# ---- simulate the ring: model per-rank state, rotate the (K,V,owner_idx) buffers ----
# "in hand" at each rank r: starts with its own (owner=r); after step t, holds owner=(r-t)%sp.
delivered = {r: set() for r in range(sp)}     # which segment indices rank r has seen
parts_by_rank = {r: {} for r in range(sp)}    # rank r's buffered partials by global seg idx
# initial hand = own shard
hand = {r: (r, Kshard[r], Vshard[r]) for r in range(sp)}
for t in range(sp):
    # each rank flash-partials whatever is in hand, buffers it by global seg idx
    for r in range(sp):
        owner, K, V = hand[r]
        O_s, mx, sm = flash_partial(Qshard[r], K, V)
        parts_by_rank[r][owner] = (O_s, mx, sm)
        delivered[r].add(owner)
    if t == sp - 1:
        break
    # rotate: rank r sends its hand to (r+1)%sp, receives from (r-1)%sp
    new_hand = {}
    for r in range(sp):
        src = (r - 1) % sp
        new_hand[r] = hand[src]
    hand = new_hand

# check (a): every rank saw every segment exactly once
worst = 0.0
for r in range(sp):
    assert delivered[r] == set(range(sp)), f"rank {r} missing segments: {set(range(sp))-delivered[r]}"
    out = combine_global_order(parts_by_rank[r])
    dd = (out - full_ref(r)).abs().max().item()
    worst = max(worst, dd)
    print(f"rank {r}: saw all {sp} segments via rotation, global-order combine max|Δ|={dd:.3e}")

print()
assert worst < 1e-12, f"ring rotation DIVERGES (max|Δ|={worst:.3e})"
print(f"PROOF: ring rotation (send-next/recv-prev, {sp-1} steps) delivers every window shard")
print(f"       to every rank exactly once; buffering partials + GLOBAL-order online-softmax")
print(f"       combine == full attention (max|Δ|={worst:.2e}). No world-gather: each rank")
print("       starts with only its 1/sp shard and exchanges via point-to-point rotation.")
print("       Combined with verify_sharded_cache_ring.py (cross-phase cache), this is the")
print("       full correctness backbone for 3b sharded-cache ring on device.")
