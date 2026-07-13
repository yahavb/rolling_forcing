#!/bin/bash
# Representative per-cluster NEFF profiler — run INSIDE the rf pod.
#
# Instead of capturing all ~752 executed NEFFs (hours; drowns hotspots in noise),
# cluster them by exact byte size (identical compiled subgraph -> identical size),
# capture ONE representative per cluster (~44), then weight each row by its cluster
# POPULATION so a tiny-but-numerous class (e.g. the 347x 11264B cluster — the SD
# kv-cache concat/pad risk) ranks by its class-TOTAL cost, not its per-NEFF cost.
#
# Usage (from a host with kubectl):
#   kubectl cp scripts/sample_neffs_by_cluster.sh <pod>:/tmp/sample.sh
#   kubectl exec -it <pod> -- bash /tmp/sample.sh
# Reads NEFFs from $PROFILE_DIR (default /tmp/neuron_profile), writes:
#   /tmp/sample_out/per_neff_sampled.txt   <- weighted engine table + priority
#   /tmp/sample_out/*.json                 <- raw summary-json per rep
set +e +o pipefail

PROFILE_DIR="${PROFILE_DIR:-/tmp/neuron_profile}"
OUT="${OUT:-/tmp/sample_out}"
POUT="$OUT/json"; rm -rf "$OUT"; mkdir -p "$POUT"

# ── 1) cluster executed NEFFs by byte size; pick one representative path per size ──
# emit "repr_path<TAB>size<TAB>population"
declare -A REP CNT
while IFS= read -r nf; do
  sz=$(stat -c %s "$nf" 2>/dev/null || stat -f %z "$nf" 2>/dev/null)
  [ -z "$sz" ] && continue
  CNT[$sz]=$(( ${CNT[$sz]:-0} + 1 ))
  [ -z "${REP[$sz]:-}" ] && REP[$sz]="$nf"
done < <(find "$PROFILE_DIR" -name '*.neff' 2>/dev/null)

NCLUST=${#REP[@]}
NTOTAL=$(find "$PROFILE_DIR" -name '*.neff' 2>/dev/null | wc -l)
echo "clusters (distinct sizes): $NCLUST   total NEFFs: $NTOTAL"
[ "$NCLUST" -eq 0 ] && { echo "no NEFFs under $PROFILE_DIR"; exit 1; }

# ── 2) capture+view ONE rep per cluster; stamp population into the json filename ──
OK=0; i=0
for sz in "${!REP[@]}"; do
  i=$((i+1))
  neff="${REP[$sz]}"; pop="${CNT[$sz]}"
  h=$(basename "$neff" .neff)
  ntff="$POUT/${h}.ntff"
  # population encoded in name as __popN so the aggregator can weight without a sidecar
  json="$POUT/${h}__pop${pop}__sz${sz}.json"
  echo "[$i/$NCLUST] size=$sz pop=$pop  $h"
  NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1 timeout 300 \
    neuron-profile capture --single-io -n "$neff" -s "$ntff" >/dev/null 2>&1 || continue
  [ -f "$ntff" ] || continue
  timeout 180 neuron-profile view -n "$neff" -s "$ntff" \
    --output-format summary-json > "$json" 2>/dev/null || true
  [ -s "$json" ] && OK=$((OK+1))
done
echo "captured+viewed OK: $OK / $NCLUST clusters"

# ── 3) aggregator: raw per-rep AND population-weighted class view ──
cat > "$OUT/agg.py" <<'PYEOF'
import sys, os, json, glob, re
jdir = sys.argv[1]
def pick(d):
    nodes = [v for v in d.values() if isinstance(v, dict)] if isinstance(d, dict) else d
    return max(nodes or [d], key=lambda x: (x.get("total_time", 0) or 0))
rows = []
for f in sorted(glob.glob(os.path.join(jdir, "*.json"))):
    base = os.path.basename(f)
    m_pop = re.search(r"__pop(\d+)__", base); m_sz = re.search(r"__sz(\d+)", base)
    pop = int(m_pop.group(1)) if m_pop else 1
    sz  = int(m_sz.group(1))  if m_sz  else 0
    try: n = pick(json.load(open(f)))
    except Exception: continue
    g = lambda k: float(n.get(k, 0) or 0)
    pct = lambda k: g(k + "_percent") * 100.0
    t = g("total_time") * 1e6
    rows.append({"hash": base.split("__")[0][:14], "pop": pop, "sz": sz,
                 "tot_us": t, "class_us": t * pop,
                 "tensor": pct("tensor_engine_active_time"),
                 "gpsimd": pct("gpsimd_engine_active_time"),
                 "vector": pct("vector_engine_active_time"),
                 "dma":    pct("dma_active_time")})
if not rows:
    print("NO valid per-rep json — capture failed."); sys.exit(0)

def classify(r):
    if r["tensor"] < 15 and r["dma"] >= 40: return "OVERHEAD-BOUND (dma/stall) <- OPTIMIZE"
    if r["tensor"] >= 40: return "COMPUTE-BOUND (matmul-hot)"
    return "MIXED"

# ---- view 1: per-representative (one row per structural class) ----
rows.sort(key=lambda r: r["tot_us"], reverse=True)
print(f"=== PER-CLUSTER REP ENGINE UTIL ({len(rows)} clusters) — per-NEFF time ===")
print(f"{'neff':14} {'pop':>4} {'size_B':>9} {'tot_us':>9} {'tensor%':>8} {'gpsimd%':>8} {'vector%':>8} {'dma%':>7}  class")
for r in rows:
    print(f"{r['hash']:14} {r['pop']:>4} {r['sz']:>9} {r['tot_us']:>9.1f} "
          f"{r['tensor']:>8.1f} {r['gpsimd']:>8.1f} {r['vector']:>8.1f} {r['dma']:>7.1f}  {classify(r)}")

# ---- view 2: POPULATION-WEIGHTED (class total = per-NEFF time x population) ----
# This is the one that surfaces "many tiny NEFFs" — the SD kv-cache pattern.
rows.sort(key=lambda r: r["class_us"], reverse=True)
tot = sum(r["class_us"] for r in rows) or 1.0
print("\n=== CLASS-TOTAL WEIGHTED (tot_us x population) — WHERE THE WALL TIME LIVES ===")
print("  NOTE: population = # distinct graphs of this class, NOT runtime invocations.")
print("        Cross with ntrace.pb (harmony) invocation counts for true weight.")
print(f"{'neff':14} {'pop':>4} {'class_us':>11} {'%oftot':>7} {'tensor%':>8} {'dma%':>7}  class")
cum = 0.0
for r in rows[:15]:
    share = 100*r["class_us"]/tot; cum += share
    print(f"{r['hash']:14} {r['pop']:>4} {r['class_us']:>11.1f} {share:>6.1f}% "
          f"{r['tensor']:>8.1f} {r['dma']:>7.1f}  ({'cum %.1f%%'%cum})  {classify(r)}")
PYEOF
python3 "$OUT/agg.py" "$POUT" 2>&1 | tee "$OUT/per_neff_sampled.txt"
echo "=== done. table: $OUT/per_neff_sampled.txt ; raw json: $POUT ==="
