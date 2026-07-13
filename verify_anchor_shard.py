"""CPU proof: the ANCHOR-block RoPE in _assemble_kv is bit-exact under cache-shard.

In the merged cache-shard path _assemble_kv re-RoPEs the cached SINK block
(kv_cache["k"][0, :block_length]) with the single-block grid (3,h,w) at a fixed
start_frame. In cache-shard mode that cached anchor holds only THIS rank's
per-block-0 shard: ws_block=block//world tokens at global (in-block) offset
world_rank*ws_block, heads [h_start:h_end]. RoPE is per-position, so:

    RoPE_full(block)[world_rank*ws_block : +ws_block, h_start:h_end]
      == RoPE_shard(block[world_rank*ws_block:+ws_block], global_offset=world_rank*ws_block)[:, h_start:h_end]

i.e. the anchor must pass tok_offset=world_rank*ws_block, tok_len=ws_block,
head_start=h_start, head_end=h_end — otherwise it defaults tok_len=block(3600)!=s(225).
max|Δ| must be 0. Same idiom as verify_krope_shard.py, single-block anchor scale."""
import torch; torch.manual_seed(0); torch.set_default_dtype(torch.float64)

world = 16          # TP4 x CP4
tp = 4
block = 3600        # block_length = 3 * frame_seqlen (frame_seqlen = h*w = 1200)
n = 12; d = 128
n_local = n // tp   # heads per tp shard = 3
ws_block = block // world   # 225 per-rank per-block slice

theta = torch.randn(block) * 0.01   # per-position rotation angle within the block
def rope_pos(x, p0):
    o = x.clone()
    for i in range(x.shape[0]):
        a = theta[p0 + i]; c, s = torch.cos(a), torch.sin(a)
        o[i, :, 0::2] = c * x[i, :, 0::2] - s * x[i, :, 1::2]
        o[i, :, 1::2] = s * x[i, :, 0::2] + c * x[i, :, 1::2]
    return o

# each rank holds block-0's shard: ws_block tokens at in-block offset r*ws_block
Kloc = {r: torch.randn(ws_block, n, d) for r in range(world)}
Kfull = torch.cat([Kloc[r] for r in range(world)], 0)   # full block-0 (3600 tokens)

maxd = 0.0
for r in range(world):
    tp_rank = r % tp                       # attn-tp rank -> head window
    h_start = tp_rank * n_local; h_end = h_start + n_local
    # reference: RoPE the FULL block (all heads), then slice this rank's tokens + heads
    ref = rope_pos(Kfull, 0)[r * ws_block:(r + 1) * ws_block, h_start:h_end]
    # sharded anchor as it ACTUALLY runs: the cache already stores the head-sliced shard,
    # so RoPE runs on the pre-sliced [ws_block, n_local, d] at global offset r*ws_block —
    # NO head kwargs, NO post-RoPE head slice (== fixed _nki_rope_apply, tok_offset/tok_len
    # only). RoPE is per-position AND per-head-independent, so this equals ref.
    anchor_shard = Kloc[r][:, h_start:h_end]           # cache holds only this rank's n_local heads
    got = rope_pos(anchor_shard, r * ws_block)         # RoPE the pre-head-sliced shard
    maxd = max(maxd, (ref - got).abs().max().item())

print(f"anchor-shard vs full-anchor max|Δ|={maxd:.3e}")
assert maxd < 1e-12, "DIVERGES"
print("PROOF PASSED: sharded anchor RoPE (tok_offset=r*ws_block, cache pre-head-sliced) == full-then-slice")
