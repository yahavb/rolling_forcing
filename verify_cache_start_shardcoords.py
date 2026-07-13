"""CPU proof: under cache-shard, cache_start MUST be converted to sharded coords
(cache_start // world) so that _cache_write's index bookkeeping stays self-consistent.

_cache_write derives cache_end = cache_start + block_length, num_new_tokens =
cache_end - global_end_index, num_evicted = num_new_tokens + local_end_index -
kv_cache_size, and the evict slice src_start = sink + num_evicted. In cache-shard
mode block_length / kv_cache_size / sink are ALL sharded (÷world), but the caller
still passes cache_start in FULL sequence coords (advances by full block=3600/phase).
Mixing full cache_start with sharded sizes explodes num_evicted -> the observed
_cache_copy_inplace src(0,...) != dst(1350,...) crash.

FIX: divide cache_start by world in shard mode. This proof runs the EXACT bookkeeping
loop (multi-phase, with eviction) both ways and asserts every sharded index ==
full index // world, position for position. max|Δ|=0."""

world = 16
frame_seqlen = 1200
block_full = 3 * frame_seqlen            # 3600
cache_full = 24 * frame_seqlen           # 28800  (kv_cache_logical_size)
assert block_full % world == 0 and cache_full % world == 0

def run(block_length, kv_cache_size, cache_start_seq, n_phases):
    """Exact _cache_write index bookkeeping. cache_start_seq(ph) returns the cache_start
    passed for phase ph. Returns per-phase (num_new, num_evicted, local_start, local_end,
    evict_src_start, evict_rolled) — evict_* are None when no eviction."""
    sink = block_length
    gei = 0            # global_end_index
    lei = 0            # local_end_index
    out = []
    for ph in range(n_phases):
        cache_start = cache_start_seq(ph)
        cache_end = cache_start + block_length
        num_new = cache_end - gei
        num_evicted = 0
        ev_src = ev_rolled = None
        if num_new > 0 and num_new + lei > kv_cache_size:
            num_evicted = num_new + lei - kv_cache_size
            ev_rolled = kv_cache_size - 2 * sink
            ev_src = sink + num_evicted
        lei = lei + num_new - num_evicted
        lsi = lei - block_length
        if num_new > 0:
            gei = cache_end
        out.append((num_new, num_evicted, lsi, lei, ev_src, ev_rolled))
    return out

N = 12
# FULL: cache_start advances by full block each phase
full = run(block_full, cache_full, lambda ph: ph * block_full, N)
# SHARD (BUGGY): sharded sizes but FULL cache_start -> reproduce the crash
buggy = run(block_full // world, cache_full // world, lambda ph: ph * block_full, N)
# SHARD (FIXED): sharded sizes AND sharded cache_start (// world)
fixed = run(block_full // world, cache_full // world, lambda ph: (ph * block_full) // world, N)

print(f"{'ph':>2} | {'full(nnew,nevict,lsi,lei,src,roll)':>42} | {'fixed':>34}")
worst = 0
for ph in range(N):
    f = full[ph]; x = fixed[ph]
    # every fixed index must equal the full index // world (None stays None)
    exp = tuple(None if v is None else v // world for v in f)
    d = 0 if x == exp else 1
    worst = max(worst, d)
    print(f"{ph:>2} | {str(f):>42} | {str(x):>34}  {'OK' if d==0 else 'DIVERGE'}")

# show the buggy path blowing up (num_evicted huge / src beyond cache) at first eviction
first_evict = next((ph for ph in range(N) if buggy[ph][1] > 0), None)
if first_evict is not None:
    b = buggy[first_evict]
    print(f"\nBUGGY first eviction @ph{first_evict}: num_evicted={b[1]} src_start={b[4]} "
          f"(cache_shard alloc={cache_full//world}) -> src slice empty == the crash")

assert worst == 0, "FIXED sharded bookkeeping != full // world"
print("\nPROOF PASSED: cache_start // world makes sharded _cache_write bookkeeping ==")
print("full bookkeeping // world at every phase (num_new, evict, local idx, evict slice).")
