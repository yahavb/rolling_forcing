# VERIFIED 14 fps BASELINE — Rolling Forcing inference (read this instead of trusting session memory)

This branch (`verified-14fps-baseline`, off `main` @ 4e7e1d6) pins the EXACT config that
produces ~14 fps, measured from real k8s runs. If a future session claims a different
baseline, distrust the session and trust THIS file + the commit message + the logs cited below.

## THE NUMBER (measured, not remembered)
- **~14 fps** = block-1 steady-state at TP4×CP4, 480×640, frame_seq_length 1200, 5 denoise steps.
- Two independent runs, SAME code (main @ 4e7e1d6), SAME config, both RING-enabled (RF_RING=1):
  - run `rf-iter1000-mhjhb` (iter1000-job.yaml, CP4-only): block-1 median **13.92**, max **14.16**.
  - run `rolling-forcing-9jhs7` (rf-job.yaml, CP4 pass of it): block-1 median **13.89**, max **14.03**.
- Full-rolling-window median (blocks 1-6, fps tapers as KV window fills) = **13.2**. The "14.1"
  people quote is the **block-1 peak** (fresh KV window). Both are the same run — different summary.

## THE RECIPE (all fields verified against main's committed files + the run logs)
| field | value | source |
|-------|-------|--------|
| branch | **main** (@ 4e7e1d6) | log provenance banner |
| topology | **TP4 × CP4 = 16 ranks** (`--tp_degree 4`, NPROC=16) | log |
| resolution | **480×640** (`--latent_w 80`) → **frame_seq_length 1200** | configs/rolling_forcing_dmd.yaml:23 |
| denoise steps | **5** — `denoising_step_list [1000,800,600,400,200]` | config |
| checkpoint | **checkpoints/rolling_forcing_dmd.pt** (shipped T=5 DMD, 16G) | log |
| ring | `RF_RING=1` (optional — see below) | job env |
| frames | `--num_output_frames 21 --chunk-size 3 --fps 16 --use_ema` | job cmd |
| claim | m-lnc1-trn2 (16 cores, LNC1) | job |

## RING IS OPTIONAL — the 14 fps is the CP-QUERY path, not ring
- CP query path (gather Q over attn-tp=4 ranks, RoPE only this rank's shard, never materialize
  full Q) ships on main UNCONDITIONALLY for world_size>1 (commit 6a480c4). That alone = 14.13
  (run ssdrv, memory reference_rf_benchmark_log.md:40).
- `RF_RING=1 RF_RING_NSEG=1` = ring fast path = 14.18 (run wznpt) — SAME as CP-query within noise
  ("ring plumbing is FREE when not sharding"). So ring on/off both give ~14 at 1200.
- SP (the OLD path, all-gather Q over all 16 → RoPE full → discard 3/4) is what the pre-CP
  commits used. Do NOT benchmark on a pre-6a480c4 commit.

## WHY main's rf-job.yaml LOOKED like it only did 9 fps (the half-day trap)
- `main:rf-job.yaml` hardcodes `BRANCH: tp4-sp4-16core` (env, ~line 516) — a commit (d51799b,
  Jul 3) that PREDATES the CP query path (6a480c4) and ring (6fa6403). So its 16-rank pass ran
  OLD SP code (~13.2) and its 8-rank pass = 9 fps.
- `rf-job.yaml` ALSO runs a TP4xSP2 8-rank BASELINE pass FIRST (the 9 fps you kept seeing) before
  the 16-rank CP4 pass. The 9 fps is the 8-rank denominator, NEVER the CP4 number.
- To reproduce 14 with rf-job.yaml you MUST: set BRANCH=main, and read the 16-rank CP4 pass
  (block-1), NOT the SP2 pass.

## HOW TO RUN THIS (CP4-only, clean 14 fps, no SP2 confusion)
Use `iter1000-job.yaml` on THIS branch (or main): CP4-only, no SP2 pass, RF_RING=1, BRANCH=main.
```
kubectl delete job rf-iter1000 --ignore-not-found
kubectl apply -f iter1000-job.yaml
kubectl logs rf-iter1000-<pod> | grep -E "branch :|block +1:|MEDIAN"
```
Expect: `branch : main`, block-1 lines ~14 fps, no `TP4xSP2` banner.

## SEQ-LEN FACTS (so 1480/1500 confusion never repeats)
- Legal frame_seq_length at world=16 must be %16==0. Steps of 240 at 480p (latent_h=60):
  1200 (latent_w 80, 480×640) → 1440 (w96, 480×768) → 1680 (w112, 480×896).
- 1480 %16=8, 1500 %16=12, 1560 %16=8 → ALL ILLEGAL at 16 ranks (fail the CP shard assert).
  The user ORIGINALLY asked for 1500; a prior session mis-logged it as "1480" — both illegal.
- ring NSEG=1 seq-len sweep (run gsd24): 1200=14.18 → 1440=13.77 (+20% tokens = −2.9% fps,
  near-flat → collective/launch-bound, not attention-compute-bound). 1440 IS servable ~13.8.

## RUN-ID LEDGER (pin the run, logs are ground truth)
- rf-iter1000-mhjhb : main 4e7e1d6, iter1000-job.yaml, CP4 1200 RING → 13.92 block-1
- rolling-forcing-9jhs7 : main 4e7e1d6, rf-job.yaml (my edit), CP4 1200 RING=1 NSEG=1 → 13.89 block-1
- rolling-forcing-cwd8x/knjcp/... : pre-CP branch tp4-sp4-16core d51799b → 8-9 (SP2) / ~13.2 (SP 16r)

## KV-OVERLAP EXPERIMENT RESULT (branch kv-overlap-exchange, run xlskl, 2026-07-13)
In-kernel ncc.all_gather cache-shard (RF_RING_CACHESHARD=1 RF_CACHESHARD_INKERNEL=1):
- COMPILES + RUNS end-to-end (after 3 fixes: in-func import, strided-view ap(), dma_copy reassembly).
- fps = **5.29 steady @ 1200** (DiT ~2200ms) vs 14.18 baseline. **2.7x SLOWER — regressed.**
- Cause: gather did NOT overlap compute. dma_copy->shared_hbm->ncc.all_gather->dma_copy-reassembly
  ->flash is a SERIAL chain; flash waits for the full window. Worse than torch-shard (12.25).
- To actually win: interleave the gather with the flash K-tile loop (gather block N+1 while
  computing block N) like attention_kv_parallel_segmented_cte — a full kernel rewrite, not a wrapper.
- Correctness: ran to completion producing video; no max|Δ|=0 numeric gate in this path (RING GATE only).

## KV-OVERLAP: WHY THE NAIVE KERNEL REGRESSED + THE CORRECT DESIGN (2026-07-13)
ROOT CAUSE of 5.29fps (MFU 3.16%, DiT 2200ms vs baseline 680ms): the kernel REASSEMBLED the
FULL window (16x KV in HBM, twice, via strided dma_copy) then flashed the WHOLE window on
EVERY rank. That is baseline attention compute PLUS a 16x gather PLUS 2 fat HBM round-trips —
strictly worse. It inverted the point of sharding.

THE POINT OF SHARDING (what CORE attention_kv_parallel_segmented_cte does): each rank flashes
ONLY its 1/world KV shard (16x LESS attention compute) -> emits a PARTIAL (unnorm O, row_max,
row_sum) -> combine partials across ranks. Exchanged data = partials [Sq,bs,d]+[Sq,bs], NOT the
16x window. Overlap = compute local partial WHILE the collective moves other ranks' partials.
RF ALREADY has the partial machinery: wan_flash_self_attn(return_partials=True) + _attend_ring
online-softmax combine.

KEY: in the cache-shard path each rank ALREADY HOLDS its native 1/world shard (k_own/v_own,
_attend_cache_shard:782 — k_len_int is the SHARDED length). So local flash needs ZERO pre-gather;
only the PARTIALS get combined after.

HARD CONSTRAINT (verify_ring_attention_exact.py): ACC-gate max|Δ|=0 REQUIRES the online-softmax
merge in GLOBAL POSITION ORDER (shard 0,1,..,15) with the same fp32 as flash. Rotation/arrival
order -> ~1e-16 -> fails the pixel gate. AND each rank's shard is PER-BLOCK-INTERLEAVED (block b's
r-th slice), not contiguous global positions — so the combine must map interleaved->global order.

NEXT (Option A, the right small step): sharded local flash (return_partials) on k_own/v_own +
all_gather the small PARTIALS + ordered global combine. Reuses proven combine math; tests whether
sharded-partial (NOT full-reassembly) recovers fps. Option B = full in-kernel all_to_all ring
(harder, ordered-merge baked into schedule). Do A first, ACC-gate, then profile overlap.

## KV-OVERLAP DIAGNOSTIC RESULT (RF_CACHESHARD_COMBINE, run dfdbr) — DECISIVE
gather + ordered partial-combine, NO reassembly = **0.32 fps, DiT 37,600ms/block** (55x SLOWER).
Not a crash — runs, but catastrophic. CAUSE: the combine splits ONE flash call into N=16
per-shard flash calls (nblocks=1 here), each on the FULL Sq query (4500) with its own padded
buffer + partial. 16x the flash LAUNCHES. This is the "padded-segment scaffold" failure
(benchmark log: RF_RING_SHARD nseg=sp = ~3-5fps) x16.

CONCLUSION (rules out a whole family): on this hardware, splitting the single full-window flash
into per-shard flash calls is ruinous — per-call launch+padding overhead dominates, regardless
of whether you reassemble (5.29) or combine partials (0.32). The 14fps baseline's ONE flash call
over the full (world-gathered) window is HARD TO BEAT at the torch/kernel-call granularity.
The ONLY way sharding wins = a SINGLE fused kernel that does gather+attend internally with the
KV transfer overlapping the matmul pipeline (true CORE-style), NOT N separate wan_flash calls.
That is a large kernel rewrite. Torch-level and wrapper-level sharding are both DEAD (measured).

## FUSED KV-PARALLEL KERNEL — DESIGN LOCKED, BUILD PLAN (2026-07-13)
CPU proof (verify_kvparallel_oneflash.py) settles CORRECTNESS: 1-flash-per-rank over own KV
shard -> partial (O,max,sum) -> online merge (ORDER-INVARIANT, ~7e-16 vs full). This is the
ONLY structure that keeps flash-call-count = 1 (not N -> 0.32fps, not reassembly -> 5.29fps).

RF PER-RANK GEOMETRY (confirmed, differs from CORE which gathers full Q):
- query per rank = L/sp (sp=4)  [sp-sharded over attn-tp, gather_q_tp]
- KV per rank    = 1/world (=1/16) [world-sharded, per-block-interleaved]
- nblocks VARIABLE 1..7 (dn window fills to _cs_max_attention_size=1575=21*1200/16)
- from run d8q77: q[3,128,4500] k_shard[3,128,225] v_shard[3,225,128]

CORE (attention_kv_parallel_segmented_cte, 473 lines) is PAGED (block_tables/APC/sliding/
round-robin) — do NOT graft whole. REUSE only: (1) dma->named shared_hbm->ncc idiom (proven,
compiles), (2) the _merge_partial_attention_outputs online-softmax merge structure.

WHERE THE WIN MUST COME FROM (measure, don't assume): baseline gathers full KV (barrier) then
1 flash. This design: each rank flashes its OWN shard (no pre-gather) -> exchange PARTIALS
(~same volume as KV) -> merge. Partials are NOT smaller, so the ONLY win = the partial
all_to_all/all_gather OVERLAPPING the flash matmul. If it doesn't overlap, this ties or loses.

BUILD SEQUENCE (each stage device-validated before next; ACC gate max|Δ|=0 bf16 pixel):
1. Kernel does local flash(q_sp_shard, own_kv_shard) -> return partial. NO collective yet;
   torch does the partial all_gather + merge. Confirms local-flash-partial correctness + fps
   floor (should ~= current 5-12 range, non-overlapped).
2. Move the merge into the kernel (recv partials as input). Still torch collective.
3. Move the collective INTO the kernel (ncc.all_gather of partials) — the overlap play.
   Profile: does the collective hide behind the flash matmul? THIS is the fps test.
Stage 1 first. Do NOT write all 3 at once (that produced 3 broken deploys).

## STAGE 1 RESULT (RF_CACHESHARD_STAGE1, run 72gj5) — 3.49 fps, floor is DAMNING
KV-parallel non-overlapped floor = 3.49 fps (DiT ~3200ms), WORSE than reassembly (5.29) and
torch-shard (12.25). Ran clean (correct). Cause: gathers 3 fp32 PARTIALS (O[4500,3,128] fp32
> the bf16 KV shard) as barriers, AND each rank still flashes the full Sq=4500 query (query is
L/sp, not shrunk further) so flash cost didn't drop. Full-query flash + 3 fat partial barriers.

ALL SHARDING VARIANTS NOW MEASURED, ALL BELOW 14.18 BASELINE:
  torch-shard+reassembly 12.25 | in-kernel reassembly 5.29 | KV-parallel Stage1 3.49 | N-flash 0.32
Baseline = 1 full-window flash after 1 KV all_gather = 14.18. Every decomposition adds collective
+ per-call overhead that does not overlap enough to pay for itself on this hardware (collective/
launch-bound, MFU ~3-4%). Stage 3 overlap would have to hide the ENTIRE partial-exchange+merge
behind the flash JUST to reach 5-12, let alone beat 14 — implausible given the 3.49 floor.
VERDICT: KV-window sharding for FPS is exhausted. 14.18 (main, shipped) stands.
