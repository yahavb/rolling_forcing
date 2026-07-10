"""
CPU accuracy proof for merged-path CP K/V deinterleave ELIMINATION.

PROFILE FINDING (run 100726225029 vs 100726214517): after 69e0fdc dropped the
per-layer k_full=cat copy (11.87 -> 12.14 fps), the remaining cap is TWO stable
gpsimd-94%/tensor-8% pure-layout kernels:
    1c20c6ca395b0a (994KB, ~8633us)  +  c326d995d5f134 (830KB, ~7224us)
= ~15.8k us of zero-matmul work every merged step. These are the cu/dn K/V
DEINTERLEAVE in forward_merged:

    _, k_full, v_full = self._gather_qkv(..., gather_q=False)  # ONE interleaved world-gather
    k_di = k_full.view(N, L_cu_N + L_dn_N, dim)                # gathered = [r0cu,r0dn,r1cu,r1dn,...]
    k_cu_di = k_di[:, :L_cu_N].reshape(L_cu, dim)              # <- strided slice -> gpsimd COPY (1c20..)
    k_dn_di = k_di[:, L_cu_N:].reshape(L_dn, dim)              # <- strided slice -> gpsimd COPY (c326..)
    # (same for v)

The strided [:, :L_cu_N] slice is non-contiguous (each rank's cu is separated by
that rank's dn), so .reshape() materializes a full-window copy of K and V.

CANDIDATE FIX (this proof): split k_local/v_local into their cu and dn parts
BEFORE the gather (cheap local contiguous slices, L_full_N each), then world-gather
each part separately. all_gather concatenates in RANK order, so gathering cu_local
over world lands exactly [r0cu, r1cu, ..., r(N-1)cu] = contiguous [L_cu, dim] with
NO deinterleave copy. Same for dn, and for v.

    k_cu_local = k_local[:L_cu_N]                # local contiguous slice (cheap)
    k_dn_local = k_local[L_cu_N:]                # local contiguous slice (cheap)
    k_cu_di = world_gather(k_cu_local, L_cu)     # contiguous, no strided copy
    k_dn_di = world_gather(k_dn_local, L_dn)     # contiguous, no strided copy

Trade vs current: 2 world-gathers (k,v) -> 4 (k_cu,k_dn,v_cu,v_dn); same total
bytes, +2 collective launches/layer. Correctness is EXACT (proven here); the
launch-vs-copy perf question is a device call (RF is collective-launch-sensitive).

This proof models the world all-gather as rank-order concatenation (which is exactly
what all_gather_into_tensor does) and asserts the fix produces byte-identical
k_cu_di / k_dn_di / v_cu_di / v_dn_di to the current deinterleave. Since every
downstream consumer (RoPE, _slice_heads, _cache_write, _assemble_kv, _attend) reads
ONLY these four tensors, byte-identity of them is the sufficient condition for the
entire merged forward — including the cross-phase KV cache — to be unchanged.
"""
import torch
torch.manual_seed(0)

tp, sp = 4, 4
N = tp * sp
fs = 1200            # frame_seqlen
nfpb_cu = 3
f_full = 15
dim = 1536
L_cu = nfpb_cu * fs
L_dn = (f_full - nfpb_cu) * fs
L_full = L_cu + L_dn
L_cu_N, L_dn_N = L_cu // N, L_dn // N
L_full_N = L_full // N
assert L_cu % N == 0 and L_dn % N == 0

# Global K and V, distinct per token so any layout mistake shows up.
# Tag cu with a huge offset so cu/dn can never be confused.
BIG = 10**7
k_cu_global = torch.arange(L_cu, dtype=torch.float64).reshape(L_cu, 1).repeat(1, dim)
k_dn_global = (torch.arange(L_dn, dtype=torch.float64) + BIG).reshape(L_dn, 1).repeat(1, dim)
v_cu_global = k_cu_global + 0.5          # distinct from k, same layout
v_dn_global = k_dn_global + 0.5

# ---- per-rank LOCAL shard: rank r holds [cu[r*L_cu_N:], dn[r*L_dn_N:]] concatenated ----
# This is the CP-merged input layout (dit_model.py x-shard) that forward_merged sees
# reflected in k_local/v_local after the qkv projection.
def local_shard(cu_global, dn_global):
    return {
        r: torch.cat([cu_global[r * L_cu_N:(r + 1) * L_cu_N],
                      dn_global[r * L_dn_N:(r + 1) * L_dn_N]], dim=0)  # [L_full_N, dim]
        for r in range(N)
    }

k_local = local_shard(k_cu_global, k_dn_global)
v_local = local_shard(v_cu_global, v_dn_global)


def world_gather(local_by_rank):
    """Model all_gather_into_tensor over 'world': concat rank contributions in rank order."""
    return torch.cat([local_by_rank[r] for r in range(N)], dim=0)


# ================= CURRENT PATH (single interleaved gather + strided deinterleave) =================
def current_deinterleave():
    k_full = world_gather(k_local)                     # [L_full, dim], interleaved [r0cu,r0dn,r1cu,...]
    v_full = world_gather(v_local)
    k_di = k_full.view(N, L_cu_N + L_dn_N, dim)
    k_cu_di = k_di[:, :L_cu_N].reshape(L_cu, dim)      # gpsimd copy 1c20..
    k_dn_di = k_di[:, L_cu_N:].reshape(L_dn, dim)      # gpsimd copy c326..
    v_di = v_full.view(N, L_cu_N + L_dn_N, dim)
    v_cu_di = v_di[:, :L_cu_N].reshape(L_cu, dim)
    v_dn_di = v_di[:, L_cu_N:].reshape(L_dn, dim)
    return k_cu_di, k_dn_di, v_cu_di, v_dn_di


# ================= FIX (split-before-gather: two contiguous gathers per tensor) =================
def split_before_gather():
    # local contiguous slices (cheap; L_full_N-sized, no strided copy)
    k_cu_local = {r: k_local[r][:L_cu_N] for r in range(N)}
    k_dn_local = {r: k_local[r][L_cu_N:] for r in range(N)}
    v_cu_local = {r: v_local[r][:L_cu_N] for r in range(N)}
    v_dn_local = {r: v_local[r][L_cu_N:] for r in range(N)}
    # two contiguous world-gathers per tensor -> already-contiguous [cu]/[dn]
    k_cu_di = world_gather(k_cu_local)                 # [L_cu, dim] contiguous, NO strided copy
    k_dn_di = world_gather(k_dn_local)                 # [L_dn, dim]
    v_cu_di = world_gather(v_cu_local)
    v_dn_di = world_gather(v_dn_local)
    return k_cu_di, k_dn_di, v_cu_di, v_dn_di


cur = current_deinterleave()
fix = split_before_gather()
names = ["k_cu_di", "k_dn_di", "v_cu_di", "v_dn_di"]
expected_shapes = [(L_cu, dim), (L_dn, dim), (L_cu, dim), (L_dn, dim)]

maxd = 0.0
for name, a, b, shp in zip(names, cur, fix, expected_shapes):
    assert a.shape == shp == b.shape, f"{name} shape {a.shape}/{b.shape} != {shp}"
    d = (a - b).abs().max().item()
    maxd = max(maxd, d)
    print(f"{name:9s}: shape {tuple(a.shape)}  max|Δ| = {d:.3e}")

# Sanity: prove cu tokens really are the low-tag cu values and dn the high-tag, i.e.
# the fix didn't just make two identical-but-wrong tensors.
assert fix[0].max().item() < BIG, "k_cu_di leaked dn tokens"
assert fix[1].min().item() >= BIG, "k_dn_di leaked cu tokens"
# Prove global token order preserved: k_cu_di row i must equal global cu token i.
assert (fix[0] - k_cu_global).abs().max().item() == 0.0, "k_cu_di not in global order"
assert (fix[1] - k_dn_global).abs().max().item() == 0.0, "k_dn_di not in global order"

print(f"\nK/V DEINTERLEAVE ELIMINATION end-to-end max|Δ| = {maxd:.3e}")
assert maxd < 1e-9, "DIVERGES — split-before-gather is NOT bit-identical, do not ship"
print("PROOF PASSED: split-before-gather (two contiguous world-gathers per k/v) produces")
print("              byte-identical k_cu_di/k_dn_di/v_cu_di/v_dn_di to the strided")
print("              deinterleave. Every downstream consumer reads only these four, so")
print("              RoPE / cache-write / assemble / attend — and the cross-phase KV")
print("              cache — are unchanged. The gpsimd copies (1c20.., c326..) are removed;")
print("              net perf (2 extra gather launches vs a full-window copy) is a DEVICE")
print("              call and must be confirmed by the on-device ACC GATE + fps run.")
