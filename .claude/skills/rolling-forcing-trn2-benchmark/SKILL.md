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
| Unique NEFFs executed | ~318 | **~2 per rank** (small stable set) |
| Reference fps | ~0.52 fps (broken/slow) | **~8.1 fps** steady-state median |

## THE FLOW (this is the whole skill — `references/rf-job.yaml` does exactly this)

```
Job (ONE pod, 8 ranks on m-trn2):
[1/3] WARMUP   compile-if-cold / load NEFFs (not timed)
[2/3] CLEAN    PROFILE_E2E_PIPELINE=1 MEASURE_TFLOPS=1, NO device profiler
               -> per-block fps + per-stage ms + tflops_report.json  (REAL latency)
               -> steady-state MEDIAN fps (excludes block 0 of each prompt; compile-contaminated)
[3/3] PROFILED NEURON_RT_INSPECT_* (system) + NEURON_PROFILE_ENABLE (native) together
               -> per-rank harmony artifacts in /tmp/neuron_profile:
                  trace_info.pb, ntrace.pb, cpu_util.pb, host_mem.pb, trace.json, neff_*.neff
[4] build_harmony_bundle  -> a dir matching neuron-explorer.harmony.a2z.com spec:
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
7. **Profile only the ~2 EXECUTED NEFFs**, not the whole compile cache. SD's
   581/318-NEFF re-capture loop is what made runs take hours.
8. **One timestamped run dir + the FRAMES.** Output is mp4s
   (`prompt_NNN.mp4` from `gather_and_save`) written to the run's `--output_folder`;
   they must be copied to `frames/`. Everything lands in
   `/var/mdl/rolling-forcing/runs/<ddmmyyHHMMSS>/` — nothing scattered elsewhere.

## Constraints (RF project rules — never violate)

- Never set `NEURON_LOGICAL_NC_CONFIG`.
- No T5/CPU offload — all models stay on Neuron.
- Resource claim / cpu / memory must match `rolling-forcing-job.yaml` (m-trn2, 24, 256Gi).
- `neuron-profile` (CLI) for per-NEFF capture+view; harmony bundle is the upload
  artifact. Do not introduce a `neuron-explorer` server / SQL / parquet path here.
