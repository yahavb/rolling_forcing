# Parallelism Strategies for Real-Time Diffusion Video Inference on AWS Trainium (Neuron)

**Paper-source notes — inference side.** To be converged with the training-side project.
Target venue: *Advanced Computing* (gold OA). Topic fit: Systems/Networks/Communication
(distributed & parallel systems, high-performance computing, performance analysis);
AI (foundation models, large-scale); open-source reproducibility.

---

## 1. Abstract (draft)

Autoregressive video diffusion models generate frames in a rolling window, imposing a
long-context self-attention over a sliding KV cache that must be produced at interactive
frame rates. We study distributed inference of a 1.3B-parameter text-to-video diffusion
transformer (Wan2.1-T2V, DMD-distilled for rolling-forcing denoising) on a single AWS
Trainium2 (Trn2) instance across 8–32 NeuronCores. We characterize the achievable
throughput/efficiency envelope of tensor parallelism (TP), sequence/context parallelism
(SP/CP), and their interaction with the Neuron collective-communication runtime. Using a
system-level tracer (harmony/neuron-explorer) and a per-NEFF (compiled subgraph)
engine-utilization method, we show the workload is **collective-barrier- and
memory-bandwidth-bound at ~4% model FLOPs utilization (MFU)**, not compute-bound, and that
naive optimizations (kernel fusion, launch reordering) fail because the Neuron compiler
already schedules the separated ops efficiently. We identify that the baseline
sequence-parallel implementation performs a redundant full-sequence query all-gather each
layer, and replace it with **true context parallelism** — keeping the query sequence-sharded
and gathering only within the tensor-parallel subgroup — yielding a **+14% end-to-end
throughput** improvement at identical output (bit-exact) and identical core count. We further
map the topology surface (TP×{CP2,CP4,CP8}) and establish that throughput peaks at 16
NeuronCores, declining beyond as collective-barrier cost dominates.

---

## 2. System under test

### 2.1 Model
- **Base:** Wan-AI/Wan2.1-T2V-1.3B (T5 text encoder → DiT transformer → VAE decoder).
- **Checkpoint:** TencentARC/RollingForcing DMD-distilled DiT ("rolling forcing" = causal,
  chunked autoregressive denoising with a sliding KV window).
- **DiT dims:** dim=1536, 12 heads, head_dim=128, 32 transformer layers (self-attn +
  text cross-attn + FFN per layer). bf16.
- **Denoising structure:** video generated in chunks of `nfpb` (num_frame_per_block=3)
  latent frames; 5 DMD denoising steps; a "rolling window" over blocks. Per generation the
  scheduler runs `window_num = num_blocks + num_denoising_steps − 1` phases (11 for 21
  output frames): **phase 0 = "denoise" mode (1 call), phases 1..N = "merged" mode (10
  calls)** where merged fuses a cache-update sub-sequence (cu) and a denoise sub-sequence
  (dn) into one forward.

### 2.2 Hardware — AWS Trainium2 (Trn2)
- Per NeuronCore (LNC1): ~190 TFLOP/s bf16 peak; 12 GB HBM.
- **Logical NeuronCore Config (LNC):** LNC1 = 1 physical core/logical (12 GB/core);
  LNC2 = 2 physical cores fused/logical (24 GB/core, ~2× peak/logical core).
- **Hard collective-runtime constraint (measured):** total Logical NeuronCore group size
  must be **1, 2, 4, 8, 16, or a multiple of 32** — e.g. 24 is rejected
  ("Unsupported topology"). This bounds the legal (TP×SP) products.
- Resource claims (k8s device plugin): `m-lnc1-trn2` (16 cores @ LNC1), `l-lnc1-trn2`
  (32 cores @ LNC1), `l-trn2` (16 cores @ LNC2, 24 GB/core).

### 2.3 Parallelism axes (process groups)
- **`attn-tp` (TP):** tensor parallelism over attention heads. Must divide num_heads(12) →
  legal TP ∈ {1,2,3,4,6,12}; and TP must be a legal collective group size. TP is pinned at
  **4** (12/4 = 3 heads/rank, balanced; larger TP narrows the per-rank matmul and grows the
  TP all-reduce).
- **`attn-sp` (SP) / context parallelism (CP):** shards the token sequence. SP degree =
  world_size / TP. This is the same axis the literature calls context parallelism (CP).
- **`world`:** all ranks; used for the query/key/value all-gather in the baseline.

---

## 3. Methodology

### 3.1 Three-run benchmark protocol
For every configuration, one k8s Job runs: **[1] warmup** (compile/load NEFFs, untimed);
**[2] clean** (per-block fps + per-stage ms + analytic TFLOPS/MFU, no device profiler —
the real latency); **[3] profiled** (native torch_neuronx profiler + system tracer →
harmony/neuron-explorer upload bundle). Steady-state fps reported as the **median** over
per-block measurements excluding block 0 of each prompt (which carries NEFF (re)compilation,
50–150 s vs ~1 s steady). All artifacts persisted to a timestamped run dir on S3-backed
storage.

### 3.2 Per-NEFF engine-utilization via cluster sampling
A single generation compiles **hundreds** of distinct NEFFs (compiled subgraphs) — 752 at
TP4×SP4, 1151 at TP4×SP8. Profiling all of them is intractable and misleading. We introduce
**cluster-by-byte-size sampling**: identical compiled subgraphs have identical byte size, so
752 NEFFs collapse to ~44 structural classes; we capture one representative per class
(`neuron-profile capture --single-io` + `view --output-format summary-json`) and **weight
each class by its population** (`per-NEFF wall time × #graphs in class`). This surfaces the
"many-tiny-NEFFs" pattern that size- or top-N filtering hides — e.g. a class of 347 tiny
11 KB graphs was the 3rd-heaviest by aggregate wall time.
**Caveat established:** `--single-io` per-NEFF times measure a kernel *in isolation*, not on
the *critical path*; they mislead about end-to-end cost. The system-level `ntrace.pb`
timeline (event names + durations) is what reveals true critical-path structure.

### 3.3 On-device accuracy gating
Correctness of a parallelism change cannot be validated on CPU (the RoPE/attention kernels
are Neuron-only NKI). We use two layers: **(a) a CPU numerical proof** that models the exact
gather/shard/slice index math of old vs new paths on synthetic tensors and asserts
bit-identical output (max|Δ|=0); **(b) an on-device accuracy gate** that, before timing,
runs the pipeline in reference vs new mode with identical seeds/prompts and diffs the output
video frames pixel-by-pixel, aborting the benchmark if they diverge. Both gates guard every
result reported here.

---

## 4. Characterization findings

### 4.1 The workload is not compute-bound; it is collective- and bandwidth-bound
At TP4×SP4 (16 cores, 480×640), steady state is **~12.4 fps, DiT ≈785 ms/block, MFU ≈4.06%**
(123 of 3040 TFLOPS). Two stacked causes, both from the profiler:
1. **Between graph launches:** a recurring ~150 µs host/device gap per launch dominated by
   collective sync — `nrt_model_submit` (~71 µs), host barrier `enc_barrier` (~43 µs),
   hardware barrier `cc_exec_barrier` (~31 µs) — plus ~70 µs host-launch latency before
   submit. The collective *work* (`cc_running` ~683 µs) is already overlapped with compute;
   only the barrier rendezvous is exposed on the critical path.
2. **Inside compute NEFFs:** the dominant matmul NEFFs run at ~65% tensor-engine active /
   ~80% DMA active — i.e. **memory-bandwidth-bound**, stalling on data movement. TP4 shards
   12 heads → 3/rank, making per-rank matmuls small (low arithmetic intensity).
   The two largest single NEFFs are the pointwise RoPE kernels (~90% GPSIMD, ~6% tensor).

### 4.2 What did NOT work (negative results — valuable for the paper)
Every micro-optimization aimed at the above failed, and the reasons are instructive:
| Change | Result | Root cause |
|---|---|---|
| RoPE kernel rewrite (loop hoist, rotate_half layout) | flat (12.4) | RoPE overlapped with compute; not on critical path |
| Renoise-noise prefetch (host→device pipelining) | flat | once-per-phase; bubble is once-per-launch (wrong frequency) |
| `NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS` (implicit async) | crash | conflicts with the framework's explicit async API (mutually exclusive) |
| Increasing chunk size (num_frame_per_block 3→7) | OOM / kernel asserts | nfpb=3 baked into NKI kernel shape validation; HBM blowup |
| Fuse 3 QKV all-gathers → 1 collective | **−6%** | compiler already pipelines the 3; one big collective serializes 3× payload |
| Fuse 3 QKV Linears → one [d,3d] GEMM | **−43%** | compiler's 3 small GEMMs tile better than one fused GEMM |
| Bigger matmul via higher resolution (latent_w 80→112) | MFU 4.06→5.62%, fps 12.4→10.4 | confirms bandwidth-bound cause, but adds pixels → lower fps (diagnostic, not a fix) |

**Cross-cutting lesson:** on Neuron, hand-fusion and launch-level reordering repeatedly lose
to the compiler's existing scheduling. The `--single-io` profile overstated the headroom;
critical-path analysis was required to avoid chasing overlapped work.

### 4.3 Topology surface (TP fixed at 4)
| Config | Cores | Resolution | fps | Notes |
|---|---|---|---|---|
| TP4×SP2 | 8 | 480×512 | 10.80 | biggest per-rank matmul, fewest barriers |
| **TP4×SP4** | 16 | 480×640 | **12.4** | baseline |
| TP4×SP8 | 32 | 480×512 | 8.0 | MFU collapses to 1.27%; a collective-glue NEFF class balloons to 28% of wall time |
| TP6×SP4 | 24 | — | illegal | 24 not a legal collective group size |
Throughput does **not** scale with cores past 16; beyond that, per-rank matmuls shrink and
collective-barrier count grows, net-negative. Also: SP degree constrains resolution — every
sharded token-run (`frame_seq_length`) must be divisible by world size (e.g. SP8=32 requires
fs%32=0, forcing 480×512 not 480×640).

---

## 5. Core contribution: true context parallelism vs. the baseline "SP"

### 5.1 The redundancy
The baseline sequence-parallel attention (`_qkv_rope`) **all-gathers the query over the full
`world` group to the complete sequence length L, applies RoPE to all L tokens, then discards
(SP−1)/SP of it**, keeping only this rank's shard. Keys/values are legitimately gathered full
(attention reads the whole KV window), but the query never needs the full sequence
materialized. This is a per-layer, on-critical-path waste of (a) an all-gather over 16 ranks
vs. the 4-rank TP subgroup, and (b) RoPE compute on 4× the tokens actually used.

### 5.2 The fix (context parallelism on the query path)
Keep the query **sequence-sharded end-to-end**. Gather it only over the `attn-tp` subgroup —
which, by the rank layout `rank = sp_rank·TP + tp_rank`, delivers exactly this SP-shard's
tokens — and apply RoPE only to that shard using the position-matched grid slice. Formally we
prove (and verify at max|Δ|=0):
- `gather_{attn-tp}(q_local)` == `slice_{sp}( gather_{world}(q_local) )` (rank-layout identity), and
- `RoPE(q)[shard]` == `RoPE(grid[shard], q[shard])` (RoPE is per-position → slicing commutes).
K/V and the output path are unchanged. Result is bit-exact; the per-layer query all-gather
shrinks from L over 16 ranks to L/SP over 4, and query RoPE from L to L/SP tokens.

### 5.3 Result
| Config (480×640, 16 cores) | fps | Δ | MFU | Output |
|---|---|---|---|---|
| Baseline SP4 (full-query gather) | 12.4 | — | 4.06% | ref |
| **CP4 (query sequence-sharded)** | **14.13** | **+14%** | higher | bit-identical (max|Δ|=0), quality-confirmed |
This is a pure communication/compute-redundancy elimination: same cores, same resolution,
same numerics. Validated across CP2 (10.58 @640 / 10.80 @512), CP4 (14.13), CP8 (8.19 @512),
all bit-exact vs. reference and visually confirmed.

### 5.4 Remaining headroom (in progress)
The above optimizes only the **denoise-mode** call (phase 0 = 1 of 11 generator calls). The
**merged-mode** path (phases 1..10 = 10 of 11 calls, the bulk of runtime) still performs the
full-query gather because its input is a single world-contiguous shard of an interleaved
`[cu | dn]` (cache-update + denoise) sequence — so a TP-subgroup gather yields the wrong
cu/dn mixture per rank. The proper fix shards **cu and dn independently** across `world` so
each TP-subgroup gather (after a deinterleave) reconstructs the correct per-shard cu and dn;
this ripples through the time-embedding shard, KV-cache indexing, and the output
`restore_layout` reassembly. A CPU proof of the full 5-site change is bit-identical
(max|Δ|=0); on-device validation and end-to-end throughput are pending. Because merged mode
is 10/11 of the calls, the projected headroom is large (target: interactive 16 fps).

---

## 6. Reproducibility
- Single `trn2.48xlarge`; k8s Job clones code + config, restores model weights from an
  S3-backed cache, runs the 3-run protocol, persists all artifacts (frames, harmony bundle,
  per-NEFF MFU table, logs) to one timestamped run dir.
- Entry point `e2e_pipeline.py` (T5→DiT→VAE fused), `torchrun --nproc_per_node {8,16,32}
  --tp_degree 4`. Topology selected by NPROC + resource claim; no `NEURON_LOGICAL_NC_CONFIG`
  set in code (LNC chosen by the claim). Profiling via `neuron-profile` CLI (not a server);
  harmony/neuron-explorer for system traces.
- All parallelism changes gated by CPU numerical proof + on-device frame-diff before any
  reported number.

---

## 6a. Why context parallelism fits this workload (intuition)

*A first-principles framing of why CP — not DP/TP/PP — is the parallelism that matches
rolling-forcing video diffusion on Trainium. Useful as the paper's motivating narrative.*

### Context parallelism in theory
CP shards the **sequence/context dimension**: rank *r* owns tokens `[r·L/P : (r+1)·L/P]` and
computes their layers locally. Contrast the other axes — **DP** shards the batch, **TP**
shards *within* a layer (heads/hidden), **PP** shards layers across stages. CP shards the
*tokens themselves*: it is the axis to reach for when a single sequence is too large to sit
efficiently on one device, or when one is **latency-bound on a single sample** (batch = 1).

The subtlety: nearly every op is embarrassingly parallel over tokens (norm, MLP,
projections act per-token) — **except attention**, which mixes tokens. So CP is fundamentally
a statement about *attention*: how to compute attention over the full sequence when a rank
owns only a slice of the queries, and — in *true* CP — only a slice of K/V. Two flavors:
- **Partial CP** (shipped on `main`): shard Q, but all-gather full K/V to every rank. Simple,
  but every rank materializes the whole K/V window.
- **Full CP / ring attention**: shard Q, K, and V; rotate K/V shards around the ring, merge
  with online-softmax. No rank ever holds the full window.

### Why RF is an unusually good CP target
1. **Batch = 1, latency-bound.** RF generates one video interactively toward 16 fps; there is
   no batch to hide latency behind, so DP is inapplicable. The only way to speed up a single
   sample is to split *that sample's* work — and the sample's cost *is* its token sequence.
   CP is the axis that matches the problem.
2. **The sequence is genuinely large.** 480×640 over the rolling window is thousands of latent
   tokens per denoise step, across 32 layers × multiple denoise steps. The token axis is
   where the FLOPs and the memory live.
3. **The bottleneck is data movement, not compute.** At ~4% MFU (Section 4.1) the workload is
   collective/DMA-bound. The baseline "SP" all-gathered the full Q every layer and discarded
   (P−1)/P of it — pure wasted movement on the critical path. CP's value here is *removing
   movement*, which is exactly RF's bottleneck (hence denoise-CP's +14% came from shrinking a
   collective, not the math — Section 5).
4. **Attention is the only cross-token op, and it is already an online-softmax flash kernel.**
   Full CP (ring) is a natural *extension* of machinery RF already has (per-tile running
   max/sum + correction factor), not a new algorithm — the cross-rank merge is the existing
   cross-tile merge with tiles living on different ranks.

### Why Trainium's architecture makes CP the right lever (and bounds it)
- **Small, fixed memory per core** (LNC1 ≈ 12 GB, LNC2 ≈ 24 GB). Partial CP forces every rank
  to hold the *full* K/V window inside that budget; full CP shards K/V so each rank holds
  1/P of the window — **CP directly relieves the per-core memory ceiling** (the same ceiling
  that decides whether the profiler even fits, LNC2 vs LNC1). This is a memory-architecture
  argument, not only a speed one.
- **Collectives are the tax, with fixed legal group sizes** (1,2,4,8,16, or mult of 32; §2.2).
  Double-edged: CP *helps* by replacing a `world` all-gather (16 ranks) with a smaller-group
  gather (attn-tp, 4) or ring point-to-point — less data moved, smaller barrier. But CP also
  *adds* per-layer communication, and past 16 ranks the fabric tax **dominates** — precisely
  why SP8/CP8 regress (Section 4.3). CP is therefore "shard to remove wasted movement, up to
  where the fabric tax overtakes it"; on this hardware that optimum is CP4.
- **TP is pinned at 4** (12 heads divide only by 4), so the sequence axis (`world/TP`) is
  structurally *the* remaining scaling axis — CP is the parallelism RF has left to exploit.

### The honest tension (full CP / ring)
Ring attention adds point-to-point K/V rotation *every layer*. On a collective-bound machine
this is a genuine risk: the win materializes only if the transfers overlap with compute and
the removed world-gather outweighs the added ring steps. This is why the methodology strictly
separates **correctness-proven** from **throughput-proven** — bit-exactness of the ring
kernel does not by itself guarantee an fps gain; the fabric tax must be measured, not assumed.

---

## 7. Figures/tables to produce for the paper
1. **Fig. Harmony timeline** annotated with the per-launch barrier gap
   (nrt_model_submit / enc_barrier / cc_exec_barrier / kbl_exec_wait).
2. **Fig. Per-NEFF class table** (cluster-sampled, population-weighted): tensor/gpsimd/vector/
   dma % per class, showing bandwidth-bound compute + tiny-NEFF aggregate cost.
3. **Table. Negative results** (Section 4.2) — the "compiler beats hand-fusion" evidence.
4. **Table. Topology surface** (Section 4.3) — fps/MFU vs cores; the 16-core peak; legal
   group-size constraint.
5. **Fig. CP vs SP** schematic (full-query-gather-then-discard vs sequence-sharded query) +
   the +14% result and bit-exactness proof.
6. **Fig. Roofline / arithmetic-intensity** — matmul size vs MFU (latent_w sweep) showing the
   bandwidth-bound regime.

## 8. Key claims (for the contributions list)
1. A reproducible **characterization** of a rolling-forcing video-diffusion transformer on
   Trn2 showing it is collective-barrier- and bandwidth-bound (~4% MFU), with a
   critical-path-grounded method (cluster-sampled per-NEFF + system tracer) that corrects the
   misleading single-op profile.
2. A catalog of **negative results** demonstrating that hand kernel/collective fusion and
   launch reordering underperform the Neuron compiler's native scheduling.
3. A **context-parallelism** reformulation of the query path that removes a redundant
   full-sequence all-gather, giving **+14%** throughput at bit-exact output and fixed cores,
   plus a topology-surface study establishing a 16-core throughput optimum under the
   hardware's legal collective-group-size constraint.
4. An **accuracy-gating methodology** (CPU numerical proof + on-device frame-diff) for
   safely validating parallelism transforms on accelerators whose kernels are not
   CPU-executable.

---
*Source data: session benchmark log at
`~/.claude/projects/.../memory/reference_rf_benchmark_log.md` and
`project_rf_collective_bound.md`. Branches: `main` (shipped CP4), `cp-merged-path` (in-progress
merged CP). Converge with the training-side parallelism project (NOT the serving project).*
