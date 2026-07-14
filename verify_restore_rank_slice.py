"""CPU proof: restore_layout_rank_slice builds ONLY this rank's output slice directly from
`gathered`, bit-identical to the old build-full-then-slice, at ~N x less DMA (max|Δ|=0).

Old path (dit_attention forward_merged tail): restore_layout DMA-copies ALL N ranks' data
into the deinterleaved `full` [L_full,dim], then the caller keeps full[rank*L_full_N:...] and
DISCARDS the other N-1/N. That NEFF is 93% dma (pure data movement) — copying N x what's used.
New: emit only this rank's L_full_N rows (2-6 contiguous dma sub-ranges). Same bytes kept,
N x fewer bytes moved."""
import torch

for N, nfpb, max_frames, fsl, dim in [(16,3,15,1200,8),(16,3,12,1200,8),(8,3,15,1200,8),(4,3,15,1200,8)]:
    L_cu = nfpb * fsl; L_dn = max_frames * fsl; L_full = L_cu + L_dn
    L_full_N = L_full // N; L_cu_N = L_cu // N; L_dn_N = L_dn // N
    gathered = torch.arange(N * L_full_N * dim).reshape(N * L_full_N, dim).float()

    # reference: old restore_layout builds full, caller slices
    full = torch.empty(L_full, dim)
    for w in range(N):
        full[w*L_cu_N:(w+1)*L_cu_N] = gathered[w*L_full_N : w*L_full_N+L_cu_N]
        full[L_cu+w*L_dn_N:L_cu+(w+1)*L_dn_N] = gathered[w*L_full_N+L_cu_N : w*L_full_N+L_cu_N+L_dn_N]

    def rank_slice(rank):
        s = rank * L_full_N; e = s + L_full_N
        out = torch.empty(L_full_N, dim); r = s
        while r < e:
            if r < L_cu:
                w = r // L_cu_N; off = r % L_cu_N
                n = min(L_cu_N - off, e - r, L_cu - r); src0 = w*L_full_N + off
            else:
                rp = r - L_cu; w = rp // L_dn_N; off = rp % L_dn_N
                n = min(L_dn_N - off, e - r); src0 = w*L_full_N + L_cu_N + off
            out[r-s:r-s+n] = gathered[src0:src0+n]; r += n
        return out

    worst = max((rank_slice(rk) - full[rk*L_full_N:(rk+1)*L_full_N]).abs().max().item() for rk in range(N))
    assert worst == 0, f"N={N} mf={max_frames} DIVERGES {worst}"
    print(f"N={N} nfpb={nfpb} max_frames={max_frames}: rank-slice == full-then-slice max|Δ|={worst:.1e}")

print("\nPROOF PASSED: per-rank slice is bit-identical to build-full-then-slice, all configs.")
print("Old NEFF moved N x L_full_N rows/rank; new moves L_full_N. ~N x less DMA on a 93%-dma op.")
