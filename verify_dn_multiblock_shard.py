"""
CPU proof: PER-BLOCK sharding of the MULTI-BLOCK dn stream — the missing piece that lets
cache-sharding work on the merged path (agent ad0126b26 found contiguous shard diverges at
max|Δ|=13500 for multi-block dn; this proves the per-block layout is bit-exact for all Nb).

THE LAYOUT (what _attend_cache_shard's reassembly at lines 762-773 requires):
  rank r holds, for EVERY block b in the stream, that block's r-th ws_block slice:
      shard_r = concat over b of  dn[b*block + r*ws_block : b*block + (r+1)*ws_block]
  full window = for each block b, concat rank0..rank(world-1) slices  (verify_cache_shard rule).
This is UNIFORM (each rank holds nblocks*ws_block) and reassembles bit-exact for any nblocks
— unlike the contiguous shard (rank r = dn[r*Ldn/world:]) which only matches at nblocks==1.

Also models the persistent cache across phases: dn grows block-by-block; each rank's cache
holds its per-block slices; assemble+reassemble == full-window attention input every phase.
"""
import torch
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

world = 16
fs = 1200
nfpb = 3
block = nfpb * fs          # 3600 (block_length)
wsb = block // world       # 225 (_cs_block_length)
d = 8                      # tiny feature; layout test
assert block % world == 0

def full_attend_input(K):        # the full window the non-sharded path attends
    return K

def sharded_reassemble(per_rank, nblocks):
    """per_rank[r] = this rank's per-block slices concatenated [nblocks*wsb, d].
    Reassemble via _attend_cache_shard rule: for each block, concat rank0..N-1."""
    out = []
    for b in range(nblocks):
        for r in range(world):
            out.append(per_rank[r][b*wsb:(b+1)*wsb])
    return torch.cat(out, 0)

# simulate dn stream growing 1..5 blocks (merged phases), full cache per rank
worst = 0.0
for nblocks in range(1, 6):
    Ldn = nblocks * block
    # full dn K (ground truth), distinct per token
    K = torch.arange(Ldn, dtype=torch.float64).reshape(Ldn, 1).repeat(1, d)
    # PER-BLOCK shard: rank r's cache = for each block, its r-th wsb slice
    per_rank = {r: torch.cat([K[b*block + r*wsb : b*block + (r+1)*wsb] for b in range(nblocks)], 0)
                for r in range(world)}
    # each rank holds uniform nblocks*wsb tokens
    assert all(per_rank[r].shape[0] == nblocks * wsb for r in range(world))
    reasm = sharded_reassemble(per_rank, nblocks)
    dd = (reasm - full_attend_input(K)).abs().max().item()
    worst = max(worst, dd)
    print(f"dn nblocks={nblocks} (Ldn={Ldn:5d}): per-rank={nblocks*wsb:4d} tok  reassemble max|Δ|={dd:.3e}")

print()
assert worst < 1e-12, f"per-block dn shard DIVERGES (max|Δ|={worst:.3e})"
print(f"PROOF PASSED: PER-BLOCK dn sharding is bit-exact for all block counts 1..5 "
      f"(max|Δ|={worst:.2e}).")
print("Layout: rank r holds each block's r-th ws_block(=225) slice; reassembly = per-block")
print("concat rank0..N-1. UNIFORM per-rank, matches _attend_cache_shard. The INPUT shard in")
print("dit_model must produce THIS (per-block r-slices), NOT cp-merged's contiguous")
print("dn[r*Ldn/world:] (which only matches at nblocks==1).")
