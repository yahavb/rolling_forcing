---
name: rolling-forcing-trn2-benchmark
description: |
  Benchmark and profile Rolling Forcing (Wan2.1-T2V-1.3B text-to-video, TP4×SP2,
  8 NeuronCores) on AWS Trn2 via a Kubernetes Job, and emit a grounded report
  from `kubectl logs`. The job runs the 3-run methodology (warmup → clean fps →
  profiled), assembles a neuron-explorer.harmony.a2z.com upload bundle, profiles
  EVERY executed NEFF for an engine-utilization + MFU table with a priority-ranked
  optimization list, and persists everything (incl. generated mp4 frames) to one
  timestamped run dir on the S3-backed PVC: /var/mdl/rolling-forcing/runs/<ddmmyyHHMMSS>/.

  Use when the user says "benchmark rolling forcing on trn2", "profile RF",
  "generate the RF neuron report", "get RF fps / MFU", "build the harmony trace
  bundle for RF", or "compare rolling forcing vs streamdiffusion fps". The
  companion skill `neuron-3run-benchmark` is the general version; this one is
  wired to RF's actual entry point (e2e_pipeline.py), topology, and outputs.
---

# Rolling Forcing — Trn2 benchmark + profile (k8s Job → grounded report)

Builds a Kubernetes Job that benchmarks Rolling Forcing on a Trn2 host and writes
a report you read from `kubectl logs`, plus durable artifacts on the PVC. The
working, validated job is `references/rf-job.yaml` — **start from that file**.

## RF is NOT StreamDiffusion — what's different (and why a 1:1 copy fails)

| Aspect | StreamDiffusionV2 (`sd-job.yaml`) | Rolling Forcing (this skill) |
|---|---|---|
| Entry point | `inference_neuron.py --benchmark --warmup --iters` | `e2e_pipeline.py` (NO warmup/iters flags) |
| Latency signal | `median=…ms throughput=… frame/s` | per-block `block N: DiT … VAE … <fps> fps` + `T5: … ms` |
| Topology | TP-4, 4 ranks | TP=4 × SP=2 = **8 ranks** (`torchrun --nproc_per_node 8`) |
| Per-stage timing gate | always on | only under `PROFILE_E2E_PIPELINE=1` |
| Unique NEFFs executed | ~318 | ~2 at TP4×SP2, but **752 at TP4×SP4** — cluster-sample, see below |
| Reference fps | ~0.52 fps (broken/slow) | **~8.1 fps** steady-state median |

## THE FLOW (this is the whole skill — `references/rf-job.yaml` does exactly this)

```
Job (ONE pod, 8 ranks on m-trn2):
[1/3] WARMUP   compile-if-cold / load NEFFs (not timed)
[2/3] CLEAN    PROFILE_E2E_PIPELINE=1 MEASURE_TFLOPS=1, NO device profiler
               -> per-block fps + per-stage ms + tflops_report.json  (REAL latency)
               -> steady-state MEDIAN fps (excludes block 0 of each prompt; compile-contaminated)
[3a] HARMONY PASS  NEURON_RT_INSPECT_* only (system trace) -> harmony artifacts in
                   /tmp/neuron_profile: trace_info.pb, ntrace.pb, cpu_util.pb, trace.json
     build_harmony_bundle -> neuron-explorer.harmony.a2z.com dir; PERSIST TO /var/mdl NOW
                   (don't wait for [3b]) so the visual timeline can be uploaded immediately.
[3b] NATIVE PASS   NEURON_PROFILE_ENABLE only -> writes neff_*.neff that [5] re-captures.
     >>> MUST be a SEPARATE pass from [3a]: RT_INSPECT + NEURON_PROFILE_ENABLE in ONE run
         collide on the device notification channel ("Only one subscriber allowed! type=0 /
         system trace already started") -> profiler fails, no trace. Two passes, one job.
[4] (harmony bundle assembled in [3a])
                  trace_info.pb (REQUIRED) + *.pb + *.ntff + *.neff (ALL ranks, deduped)
                  NOTE: trace.json is NOT in the spec and is ~700MB -> kept OUT of the upload dir
[5] per_neff_mfu          -> for each executed NEFF:
                  neuron-profile capture --single-io -> <hash>.ntff
                  neuron-profile view --output-format summary-json -> <hash>.json
                  embedded python aggregator -> per-NEFF engine table (tensor/gpsimd/vector/dma %)
                                              + model MFU (from tflops_report.json)
                                              + PRIORITY list: OVERHEAD-BOUND vs COMPUTE-BOUND
PERSIST -> /var/mdl/rolling-forcing/runs/<ddmmyyHHMMSS>/
             frames/  harmony_bundle/  per_neff_mfu.txt  tflops_report.json
             clean.txt  profiled.txt  trace.json.gz  rf_results.tar.gz
```

## Launch

```bash
kubectl delete job rolling-forcing --ignore-not-found
kubectl apply -f rf-job.yaml          # = references/rf-job.yaml
kubectl logs -f job/rolling-forcing
kubectl logs job/rolling-forcing > /tmp/rf.log   # for analysis
```

## Reading the report from the log

- **fps (headline):** `steady-state MEDIAN = <N> fps`. Use the MEDIAN, not mean —
  the timed clean run's first prompt and every prompt's block 0 carry NEFF
  compilation (block times 50–150s vs ~1.5s steady); median rejects them.
- **MFU:** `MODEL-LEVEL MFU (from tflops_report.json)` — achieved TFLOPS vs 1520
  (8 cores × 190 TFLOPS bf16 peak).
- **What to optimize:** the `OPTIMIZATION PRIORITY` list. `OVERHEAD-BOUND
  (dma/stall)` NEFFs (tensor%<15, dma%≥40) are the safe, high-leverage targets —
  a layout/kernel fix with no quality risk. `COMPUTE-BOUND (matmul-hot)` NEFFs are
  already efficient (lower ceiling, needs an algo/precision change).

## Analyzing HUNDREDS of NEFFs — cluster-sample, do NOT size-filter

The "~2 per rank" assumption (see HARD-WON FIX 7) is WRONG at higher topologies:
the TP4×SP4 run (`080726205353`) compiled **752 distinct NEFFs**. Capturing all
752 takes hours and drowns the signal; but you must NOT shortcut by profiling only
the largest NEFFs — that hides the real hotspot. The method that works:

1. **Cluster by exact byte size.** Identical compiled subgraph → identical byte
   count. 752 NEFFs collapse to ~**44 structural classes**. This is the sampling
   frame. (`per_neff_mfu()` in `references/rf-job.yaml` now does this automatically:
   `stat -c %s` → one representative per size → `__pop<N>__sz<B>` stamped into the
   json filename so the aggregator can weight.)
2. **Capture ONE rep per class** (~44 captures, minutes) — engine breakdown per
   class.
3. **Weight by class-total = per-NEFF `tot_us` × population**, NOT per-NEFF time.
   A class of 347 tiny graphs at 26µs each = 9193µs total, which OUTRANKS a single
   2.1MB matmul graph. Sorting by per-NEFF time or by byte size would bury it.
4. **Cross-check invocation counts against `ntrace.pb`** (harmony timeline).
   `population` = # distinct compiled graphs, NOT runtime invocation count — it's a
   proxy. The device trace has the true fire-count; confirm there before investing.

**Why size-filtering is the trap (proven on run 080726205353):** the biggest
class by population was 347× 11264-byte graphs — the smallest non-stub size. Its
class-total (4.9%) made it the **3rd-heaviest class in the whole run**. Filtering
"tiny NEFFs" as noise would have made the single largest overhead source invisible.
This is the SD kv-cache concat/pad pattern: matmul-free, individually tiny, lethal
in aggregate.

**The headline shape of RF's profile: there is NO single hotspot.** The
distribution is FLAT — top class ~6.8%, and it takes ~15 classes to reach 61%.
A 13→16 fps push (~19% wall-time cut) will NOT come from one kernel; it comes from
the OVERHEAD-BOUND pile (~18–20% of wall time doing zero matmul). Baseline profile:

| Class    | pop | class% | tensor% | dominant     | mapped to |
|----------|-----|--------|---------|--------------|-----------|
| 017737cf | 347 | 4.9%   | 8.0     | dma 44.6     | kv-cache copy glue (`kernels/kv_cache_copy.py`) |
| b949927c | 1   | 4.7%   | 5.9     | gpsimd 89.5  | **RoPE** strided even/odd swap (`kernels/rope.py:49-50`) |
| b6041713 | 1   | 3.9%   | 5.8     | gpsimd 89.5  | RoPE, sibling shape (cu vs dn grid) |
| 0ed08cdb | 19  | 3.6%   | 0.6     | vector 94.6  | modulation/norm (`dit_attention.py:48-57`, `dit_layers.py:43`) |
| 0003fa28 | 72  | 3.3%   | 2.5     | dma 79.1     | more kv/cache dma glue |
| 051f9b4f | 59  | —      | 3.3     | dma 58.4     | tiny DMA glue |

**Interpreting engine dominance → kernel:** GPSIMD-dominant (≥80%, ~0 matmul) =
a strided gather/scatter NKI kernel (RoPE interleave, layout restore). VECTOR-
dominant = elementwise/norm broadcasts (RMSNorm, modulation). DMA-dominant with
high population = many small `dma_copy` launches (kv-cache write/assemble, conv3d
cache, halo exchange). None of these are matmul — they are the safe layout/kernel
wins.

**Profiling a COMPLETED run's persisted NEFFs (no re-run):** the harmony bundle on
the PVC holds all `.neff` files. `neuron-profile capture --single-io` needs only
the `.neff` + a Neuron device, so a throwaway job can mount the PVC and re-profile
in place. `scripts/sample_neffs_by_cluster.sh` does exactly this; the run dir name
is `ddmmyyHHMMSS` (day-first, NOT lexically sortable — always pin the run, never
trust newest-dir auto-detect). `/var/mdl` = `s3://621547421844-ap-southeast-4/`,
so `aws s3 cp` reaches the same files without kubectl.

## Attributing a code change to the traces (NEFF-diff, text-only — do this, don't guess)

**The core insight that makes attribution possible.** NEFF hashes are content-addressed —
the hash IS the hash of the compiled subgraph. So when you change code, the affected
subgraphs recompile into **new hashes**, and the old ones disappear. That is the attribution
lever:

> **Diff the per-NEFF table of run-before vs run-after a change.** Classes that appear /
> vanish / change population or wall-time are directly attributable to the change. This is
> actually **more precise for attribution than the visual harmony tool** — the tool shows you
> *what takes time*; a two-run diff shows you *what your change did to what takes time*. And
> it's 100% text: no upload, no neuron-explorer, no visual step.

Never assert "this DMA-bound NEFF is my new copy" from a single run — hashes are opaque and
overhead-bound glue exists in the baseline too. Prove it with a diff, or say you haven't.

**Three tiers of what's achievable — honest boundaries.**

- **Tier 1 — NEFF-class-set diff (fully doable, text-only, the default).** Two runs (e.g.
  the flag off = baseline vs on = change) each emit the per-NEFF table (the `[3b]` native
  pass / `per_neff_mfu.txt`). Match classes by hash + byte size and report: **new classes
  (µs), removed classes, classes that grew/shrank in wall-time or population.** If a
  DMA-bound class disappears or shrinks after removing a copy, THAT is the attribution —
  proven, not guessed. This is the "confirmed vs hallucinated" discipline: an experiment
  self-attributes by diffing its per-NEFF table against the named baseline run.

- **Tier 2 — per-NEFF timeseries from the trace (via CLI, on a trainium host).**
  `neuron-profile view --output-format summary-text` / `summary-json` on the `.neff` +
  `.ntff` extracts the **same per-NEFF timeseries the harmony tool renders** — invocation
  count, wall-time, per-engine breakdown — as text. That is the "what took the time" data
  without the web tool. The job already does `capture` + `view`; dump `summary-text` for the
  top classes when needed. Limitation: it runs where the NEFFs live (the pod / a trainium
  box), not an arbitrary sandbox — but the output is text.

- **Tier 3 — hash → exact source line (NOT reliably doable — stated boundary).** Mapping a
  NEFF back to the exact Python/kernel line needs the NEFF compiled with debug/source
  metadata, and even then the mapping is coarse (a fused subgraph spans many ops). Do NOT
  promise "this NEFF == that `torch.cat` on line N." What IS promisable is Tier 1's "this
  change made this DMA class shrink by X µs" — which is the attribution that actually matters.

**Practical rule:** every change-vs-baseline experiment should carry a Tier-1 diff of its
per-NEFF table against the named baseline run dir, so the effect is measured in text, exactly
replicating what you'd eyeball in neuron-explorer — without the visual round-trip.

## HARD-WON FIXES baked into references/rf-job.yaml (do not regress these)

1. **`set +e +o pipefail` before the benchmark section.** The first version died
   after 8h with `Error` on a SIGPIPE: `find … | sort -rn | head -1` under
   `pipefail` killed the pod before archiving. Setup (clone/deps/weights) stays
   strict; everything after the banner is best-effort with an ERR trap.
2. **fps field is `$(NF-1)`, not `$NF`.** Block lines end `… 8.09 fps`; `$NF` is
   the literal word "fps". Averaging `$NF` gave `0.00 fps`. Use `$(NF-1)`.
3. **CLEAN run must set `PROFILE_E2E_PIPELINE=1`** or RF prints no per-block/fps
   lines at all — you'd have nothing to compare. This flag is cheap perf_counter
   timing, NOT the device profiler.
4. **Device trace is `ntrace.pb`, NOT `*.ntff`.** Keying explorer/bundle on
   `*.ntff` finds nothing. Harmony reads the `.pb`.
5. **NEFFs live under `/tmp/neuron_profile/.../neff_*.neff`**, in a DIFFERENT
   sibling subdir than `trace_info.pb` (per rank: one `<ts>/neff_*.neff`, one
   `<ts>/trace_info.pb …`). NOT in `/tmp/neff_cache`. Walk up to the pid dir.
6. **Keep `trace.json` OUT of the harmony upload dir** — ~700MB, not in the spec,
   bloated the bundle to 784MB. Save it gzipped separately.
7. **Profile EXECUTED NEFFs only, and cluster-sample when there are hundreds.**
   Never re-capture the whole compile cache (SD's 581/318-NEFF loop took hours).
   At TP4×SP4 there are 752 executed NEFFs → cluster by byte size to ~44 classes,
   capture one rep each, weight by population. See "Analyzing HUNDREDS of NEFFs".
8. **One timestamped run dir + the FRAMES.** Output is mp4s
   (`prompt_NNN.mp4` from `gather_and_save`) written to the run's `--output_folder`;
   they must be copied to `frames/`. Everything lands in
   `/var/mdl/rolling-forcing/runs/<ddmmyyHHMMSS>/` — nothing scattered elsewhere.
9. **PROFILE IN TWO SEPARATE PASSES (same job), PERSIST EACH IMMEDIATELY.**
   NEURON_RT_INSPECT (harmony/system trace) and NEURON_PROFILE_ENABLE (native) CANNOT
   run in the same pass — they collide on the device notification channel ("Only one
   subscriber allowed! type=0 / system trace already started", run xjwzg) and the
   profiler fails to start (no trace). Run [3a] harmony-only then [3b] native-only.
   Create the /var/mdl run dir UP FRONT and drop each pass's artifacts the moment it
   finishes (harmony_bundle + trace.json.gz after [3a]; per_neff_mfu.txt after [3b]) —
   so the harmony traces are uploadable WITHOUT waiting for the whole job. Do NOT spawn
   separate jobs for this; it's two passes inside the one benchmark job.

## Constraints (RF project rules — never violate)

- Never set `NEURON_LOGICAL_NC_CONFIG`.
- No T5/CPU offload — all models stay on Neuron.
- Resource claim / cpu / memory must match `rolling-forcing-job.yaml` (m-trn2, 24, 256Gi).
- `neuron-profile` (CLI) for per-NEFF capture+view; harmony bundle is the upload
  artifact. Do not introduce a `neuron-explorer` server / SQL / parquet path here.
