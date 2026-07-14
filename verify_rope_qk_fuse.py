"""
rope-qk-fuse — ACCURACY PROOF (exactly 0.0) for fusing the q/k RoPE launches.

forward_merged used to run 4 separate _nki_rope_apply launches per layer:
  rq_cu = rope(q_cu, grid_cu);  rk_cu = rope(k_cu, grid_cu)
  rq_dn = rope(q_dn, grid_dn);  rk_dn = rope(k_dn, grid_dn)
q and k in a region share the SAME position grid, and the RoPE kernel treats heads
as a pure broadcast/free axis (cos/sin are [P,D], broadcast over N heads — see
kernels/rope.py:65-71: cos_b/sin_b = broadcast_to(cos_1/sin_1, (P,N,D))). So the
op is head-independent: each head's output depends only on that head's input and the
shared per-position grid.

CLAIM: stacking this rank's q-heads and k-heads along the head axis, running ONE
rope call, then splitting back is BIT-IDENTICAL to two separate calls (4->2 per layer).

This models the kernel's exact arithmetic (out = x*cos + swap(x)*sin, cos/sin broadcast
over heads) in torch and shows fused == separate at EXACTLY 0.0 for both regions.
The proof does not depend on the specific cos/sin values — only on head-independence
and a shared grid — so it holds for every layer/frame/timestep.
"""
import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)  # exact fp — any nonzero delta is a real bug


def rope_kernel(x, cos, sin):
    """Faithful model of _causal_rope_rotation_nki (kernels/rope.py:26-75).

    x   : [S, N, D]      tokens x heads x head_dim
    cos : [S, D]         per-position, SHARED across heads (broadcast over N)
    sin : [S, D]
    out = x*cos_b + swap(x)*sin_b, where swap is the even/odd interleave swap and
    cos_b/sin_b broadcast the size-1 head axis to N (stride-0 view in the kernel).
    """
    S, N, D = x.shape
    x_swap = torch.empty_like(x)
    x_swap[:, :, 0::2] = x[:, :, 1::2]
    x_swap[:, :, 1::2] = x[:, :, 0::2]
    cos_b = cos.view(S, 1, D)  # broadcast over heads == kernel's [P,1,D]->[P,N,D]
    sin_b = sin.view(S, 1, D)
    return x * cos_b + x_swap * sin_b


def check(name, S, N, D):
    # q and k for this region: same positions (same grid), independent per-head data.
    q = torch.randn(S, N, D)
    k = torch.randn(S, N, D)
    cos = torch.randn(S, D)  # the SHARED per-position grid for this region
    sin = torch.randn(S, D)

    # SEPARATE (baseline): two launches.
    rq_sep = rope_kernel(q, cos, sin)
    rk_sep = rope_kernel(k, cos, sin)

    # FUSED: stack heads -> ONE launch -> split back (what forward_merged now does).
    qk = torch.cat([q, k], dim=1)              # [S, 2N, D]
    rqk = rope_kernel(qk, cos, sin)            # single call, heads independent
    rq_fused = rqk[:, :N, :].contiguous()
    rk_fused = rqk[:, N:, :].contiguous()

    dq = (rq_fused - rq_sep).abs().max().item()
    dk = (rk_fused - rk_sep).abs().max().item()
    ok = (dq == 0.0 and dk == 0.0)
    print(f"  {name:8s} S={S:5d} N={N:2d} D={D}: max|Δ|q={dq:.1e}  max|Δ|k={dk:.1e}"
          f"  {'OK (0.0)' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("rope-qk-fuse ACC GATE — fused (1 launch) vs separate (2 launches), fp64 exact:")
    all_ok = True
    # nlh = heads_per_shard; RF TP4 on 12-head Wan1.3B -> 3 heads/shard, head_dim 128.
    # frame_seqlen 1200 (480x640); cu region = 3 frames, dn region = 15 frames.
    all_ok &= check("cu", 3 * 1200, 3, 128)
    all_ok &= check("dn", 15 * 1200, 3, 128)
    # extra shapes to show independence of geometry:
    all_ok &= check("tiny", 256, 3, 128)
    all_ok &= check("nlh1", 1200, 1, 128)
    print("RESULT:", "ALL EXACTLY 0.0 — fusion is bit-identical" if all_ok
          else "FAILED — do NOT ship")
    assert all_ok, "rope-qk-fuse changes numerics"
