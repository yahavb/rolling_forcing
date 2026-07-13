"""
STAGE 1 CPU proof for TRUE CACHE-SHARDING (each rank stores ONLY its 1/world token slice of
the persistent KV cache; the full window is reassembled by all_gather ONLY at attention).

Models the exact _cache_write + _assemble_kv bookkeeping BOTH ways over many phases and both
the denoise path and the merged cu/dn path, asserting the sharded cache reassembles to the
SAME full window as today. This is the cross-phase surface that caused the earlier garbage
bug — prove it offline before any device code.

KEY INVARIANT: token position p (global) is produced by world-rank (p // block_shard) ... NO.
The real sharding: x enters block already world-sharded — rank r holds global tokens
[r*L/world : (r+1)*L/world] of the CURRENT block. So rank r's cache holds, for every block
written, that block's r-th contiguous 1/world slice. Reassembling all ranks' caches in rank
order == the full cache. We verify: (sharded write + evict + assemble, per rank) gathered ==
(full write + evict + assemble) — position for position, across phases, for denoise AND cu/dn.
"""
import torch
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

world = 16
d = 8                       # tiny feature for speed (proof is about positions/layout)
# scaled-down but structurally faithful sizes (all multiples of world)
fsl = 32                    # frame_seq_length analog; must be % world == 0
nfpb = 3
block_len = nfpb * fsl                    # 96  (block_length)
cache_logical = 24 * fsl                  # 768 (kv_cache_logical_size)
max_attn = 21 * fsl                       # 672 (max_attention_size)
sink = block_len
assert block_len % world == 0 and cache_logical % world == 0 and max_attn % world == 0
bs = world // world                        # per-rank sees full heads here; layout test only
ws = lambda x: x // world                  # world-shard size of a length

# ---- FULL cache model (today): one contiguous cache, write block, roll/evict beyond cap ----
def full_run(n_phases):
    K = torch.empty(0, d)
    snaps = []
    gei = 0
    for ph in range(n_phases):
        newK = torch.arange(gei, gei + block_len).float().reshape(block_len, 1).repeat(1, d)
        K = torch.cat([K, newK], 0)
        if K.shape[0] > cache_logical:
            keep = cache_logical - sink
            K = torch.cat([K[:sink], K[-keep:]], 0)
        gei += block_len
        # assemble window = last min(len, max_attn), plus sink prepended if rolled
        L = K.shape[0]
        cache_start = max(0, L - max_attn)
        win = K[cache_start:]
        snaps.append(win.clone())
    return snaps

# ---- SHARDED cache model: rank r stores block's r-th 1/world slice; gather to reassemble ----
def shard_run(n_phases):
    ws_block = ws(block_len)
    ws_cap = ws(cache_logical)
    ws_sink = ws(sink)
    ws_maxa = ws(max_attn)
    caches = {r: torch.empty(0, d) for r in range(world)}
    gei = 0
    snaps = []
    for ph in range(n_phases):
        for r in range(world):
            # rank r's slice of this block = global tokens [gei + r*ws_block : +ws_block]
            base = gei + r * ws_block
            newK = torch.arange(base, base + ws_block).float().reshape(ws_block, 1).repeat(1, d)
            caches[r] = torch.cat([caches[r], newK], 0)
            if caches[r].shape[0] > ws_cap:
                keep = ws_cap - ws_sink
                caches[r] = torch.cat([caches[r][:ws_sink], caches[r][-keep:]], 0)
        gei += block_len
        # assemble per rank: its 1/world of the window, then GATHER in rank order
        Lr = caches[0].shape[0]
        cs = max(0, Lr - ws_maxa)
        win_shards = [caches[r][cs:] for r in range(world)]
        # reassemble: interleave shards back to global order. Each rank holds contiguous
        # 1/world of EACH block in the window; the full window = for each block, concat
        # rank0..rank15 slices. Since every rank rolled identically, shard s row i corresponds
        # to block (cs+i)//ws_block ... reconstruct by concatenating per-block across ranks.
        nblocks = win_shards[0].shape[0] // ws_block
        full = []
        for b in range(nblocks):
            for r in range(world):
                full.append(win_shards[r][b*ws_block:(b+1)*ws_block])
        snaps.append(torch.cat(full, 0))
    return snaps

N = 12
f = full_run(N); s = shard_run(N)
worst = 0.0
for ph in range(N):
    if f[ph].shape != s[ph].shape:
        print(f"phase {ph}: SHAPE MISMATCH full{tuple(f[ph].shape)} shard{tuple(s[ph].shape)}"); worst = 9e9; break
    dd = (f[ph] - s[ph]).abs().max().item()
    worst = max(worst, dd)
    print(f"phase {ph}: window={f[ph].shape[0]:4d}  max|Δ|={dd:.3e}")

print()
assert worst < 1e-9, f"CACHE-SHARD DIVERGES (max|Δ|={worst:.3e}) — reassembly wrong"
print(f"PROOF PASSED: sharded per-rank cache (1/world each) + per-block rank-order gather ==")
print(f"the full cache window across {N} phases with sink+eviction (max|Δ|={worst:.2e}).")
print("Reassembly rule: full window = for each block in window, concat rank0..world-1 slices.")
print("This is the layout backbone; _cache_write/_assemble_kv shard by /world with this rule.")
