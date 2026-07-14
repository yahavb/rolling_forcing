"""CPU proof: all_to_all neighbor-shift halo == all_gather halo (max|Δ|=0).

VAE _halo_exchange_w OLD: all_gather every rank's 2 edges to ALL world ranks, keep only the 2
neighbors (discard world-2). Probe showed halo = 76% of VAE conv-path (~18ms, 33 collectives/
decode) = launch-bound. NEW: all_to_all where rank r sends its RIGHT edge to r+1 (r+1's
halo_left) and LEFT edge to r-1 (r-1's halo_right), zeros elsewhere; receives halo_left from
r-1, halo_right from r+1. Same collective count, but each moves only neighbor edges not a
16-way broadcast."""
import torch
torch.manual_seed(0)
for world, feat in [(16, 5), (8, 12), (4, 3)]:
    E = torch.randn(world, 2, feat)  # [rank, {left,right}, feat]
    def ref(r):
        return (E[r-1, 1] if r > 0 else None, E[r+1, 0] if r < world-1 else None)
    send = [[torch.zeros(feat) for _ in range(world)] for _ in range(world)]
    for r in range(world):
        if r+1 < world: send[r][r+1] = E[r, 1]
        if r-1 >= 0:    send[r][r-1] = E[r, 0]
    worst = 0.0
    for d in range(world):
        recv = [send[s][d] for s in range(world)]       # all_to_all: recv[d][s]=send[s][d]
        hl = recv[d-1] if d > 0 else None
        hr = recv[d+1] if d < world-1 else None
        rl, rr = ref(d)
        if rl is not None: worst = max(worst, (hl-rl).abs().max().item())
        if rr is not None: worst = max(worst, (hr-rr).abs().max().item())
    print(f"world={world} feat={feat}: a2a-halo vs all_gather-halo max|Δ|={worst:.2e}")
    assert worst == 0.0
print("\nPROOF PASSED: all_to_all neighbor shift == all_gather halo, bit-identical.")
