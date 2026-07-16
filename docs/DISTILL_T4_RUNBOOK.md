# Faster Rolling Forcing via T=4 Re-Distillation — Runbook

Goal: push RF inference past 16 fps (from ~14 fps at T=5) by distilling the
Wan2.1-T2V-1.3B causal student down to **4 denoising steps** and deploying it on
the existing Trn2 inference stack.

## Why this works (the fps lever)

RF does one DiT forward per rolling window, over `num_blocks + T − 1` windows,
where `T = len(denoising_step_list)`. Each forward processes a merged sequence of
`(T+1) × num_frame_per_block` frames at mixed noise levels
(`models/dit_pipeline.py`: `full_frames = (T+1)*nfpb`). So DiT cost per block
scales ~`(T+1)`:

| T | window frames | DiT cost vs T=5 | projected fps (from ~14) |
|---|---------------|-----------------|--------------------------|
| 5 (today) | 18 | 1.00× | 14 |
| **4** | 15 | ~0.83× | **~16–17** |
| 3 | 12 | ~0.67× | ~18–20 |

You **cannot** just shorten `denoising_step_list` on the shipped T=5 checkpoint:
the DMD student only predicts accurate x0 at the noise levels it was distilled
on. Fewer steps ⇒ re-distill. The inference pipeline itself is fully
parameterized on `len(denoising_step_list)`, so a T=4 checkpoint needs only a
config change + a Neuron recompile — no inference code changes.

## Pieces in this repo

| File | Role |
|------|------|
| `configs/rolling_forcing_dmd_t4.yaml` | T=4 training/inference config (4-entry `denoising_step_list`) |
| `training/` | vendored + Neuron-adapted upstream DMD training stack (see `training/README.md`) |
| `rf-distill-job.yaml` | Trn2 k8s Job: stage weights → precompute embeds → torchrun DMD training → mirror ckpts to S3 |

## Step 1 — Train (produces a T=4 EMA checkpoint)

```bash
kubectl delete job rf-distill --ignore-not-found
kubectl apply -f rf-distill-job.yaml
kubectl logs -f job/rf-distill
```

The job:
1. Stages Wan2.1-T2V-1.3B (student/critic base + T5 + VAE), Wan2.1-T2V-14B
   (frozen teacher), `ode_init.pt` (student init), and the prompt file — all
   S3-tar-cached under `/var/mdl/rolling_forcing/`.
2. Precomputes T5 prompt embeds once on CPU (frees ~9.6GB HBM in the 16-rank
   training job).
3. Runs upstream DMD score-distillation via `training/train.py` with the T=4
   config. Checkpoints land in `$RUN/checkpoint_model_XXXXXX/model.pt` and are
   mirrored live to `/var/mdl/rolling_forcing/distill/<TS>/`.

### What to watch for convergence

The DMD "loss" is a stop-grad surrogate (`0.5·MSE(x0,(x0−grad).detach())`) whose
gradient equals the DMD gradient — its **scalar value is flat by construction and
is NOT a convergence signal.** Do not watch `generator_loss`. Instead (tensorboard
in `$RUN/tensorboard`):

| Signal | Meaning | Healthy |
|--------|---------|---------|
| **`[fp32-master check]` at step 50** | binary go/no-go: did the tracked generator param actually move? | passes (max\|Δ\|>0). If it asserts, training is silently frozen — stop. |
| **`dmdtrain_gradient_norm`** | **primary convergence metric**: normalized `\|fake−real\|` = distance from teacher distribution | starts high, **trends down and stabilizes** |
| **`critic_loss`** | is the critic tracking the student's output? (gradient is only valid if so) | drops early, then **low + stable** |
| `generator_grad_norm` / `critic_grad_norm` | optimization health (clip=10) | bounded, non-zero, no NaN |
| `generator_loss` | surrogate — **ignore absolute value** | (flat) |

**Ground truth is visual.** DMD converges by eyeballing rendered samples (losses are
adversarial/flat). Render the EMA checkpoint every ~200 steps (mirrored to the PVC
live) and stop when quality plateaus. Upstream runs ~3k steps.

### Proof-of-concept: single prompt (configured)

The T=4 config's `data_path` is `prompts/distill_poc_single.txt` — example prompt
000 (the western/horse shot), repeated so `DistributedSampler(drop_last=True)`
yields ≥1 sample per rank (a 1-line file would give 0 samples/rank across 16 ranks
and hang at step 0). `precompute_embeds.py` keys embeds by prompt string, so the
duplicates collapse to one embed; every rank overfits the same prompt.

**Built-in A/B:** compare the T=4 EMA render of this prompt against the existing
**T=5** baseline `rf_sp4_videos/prompt_000.mp4` (same prompt, your current 14fps
demo). That side-by-side is the proof.

To broaden later: set `data_path` to `prompts/example_prompts.txt` (15) or the full
vidprom corpus (and set `PROMPTS_SRC` if staging from the PVC).

## Step 2 — Wire the checkpoint into inference

1. Copy the EMA checkpoint into the inference checkpoints dir:
   ```bash
   cp /var/mdl/rolling_forcing/distill/<TS>/checkpoint_model_XXXXXX/model.pt \
      checkpoints/rolling_forcing_dmd_t4.pt
   ```
   (The inference loader reads the `generator_ema` key with `--use_ema`.)

2. Point the deploy/serve config at the T=4 config + checkpoint. In
   `rf-deploy.yaml`:
   ```yaml
   - name: CONFIG_PATH
     value: "configs/rolling_forcing_dmd_t4.yaml"
   - name: CHECKPOINT_PATH
     value: "checkpoints/rolling_forcing_dmd_t4.pt"
   ```
   The `denoising_step_list` lives in the config; the pipeline picks up T=4
   automatically. `e2e_pipeline.py` now derives the step count from
   `len(pipe.denoising_step_list)` (no hardcoded 5), so MFU accounting is
   correct.

3. First serve run recompiles NEFFs for the new merged-sequence shapes
   (`full_frames` shrinks 18→15 frames). Expect a one-time compile.

## Step 3 — Benchmark

Use the `rolling-forcing-trn2-benchmark` skill (or `rf-job.yaml`) to measure
clean fps + MFU on the T=4 checkpoint and compare against the T=5 baseline.
Confirm the ~16 fps target and check quality on your prompt set.

## Risks (see training/README.md for the full list)

- FSDP-on-Neuron (HYBRID_SHARD, `summon_full_params`, `clip_grad_norm_`) is
  unverified on device — the biggest unknown.
- fp32-master + autocast precision on Neuron (the weight-change assert guards the
  silent-no-train failure mode).
- flex_attention→SDPA dense-mask equivalence + O(L²) mask memory.
- HBM footprint: frozen 14B teacher + 1.3B student + 1.3B critic + fp32 masters +
  AdamW moments across 16 cores.
- T=4 quality is unvalidated by the upstream authors (they use T=5); it trades
  quality for throughput. Validate on your target prompts before shipping.

---

## Render / quality results (2026-07-13)

Serving the distilled checkpoints end-to-end (train → ckpt on PVC → e2e_pipeline
render → eyeball vs the shipped T=5 baseline `rf_sp4_videos/prompt_000.mp4`).

### Quality (the verdict)
- iter200 EMA (T=4) — BLURRY
- iter400 EMA (T=4) — BLURRY
- iter1000 EMA (T=4) — STILL BLURRY (EMA has ~800 G-steps avg by here; not "too early")
- shipped `rolling_forcing_dmd.pt` (T=5) — SHARP (ground truth)
- => our T=4 distillation is UNDER-CONVERGED / not usable yet. Training is the blocker,
  not serving. prompt_000 (western/horse) is the ONLY trained prompt; it is blurry, so
  untrained prompts being blurry tells us nothing extra.

### Open confound (resolve before more training)
Render the SHIPPED ckpt at T=4 (not T=5): isolates "4 steps too few" from "our weights bad".
- shipped@T=4 blurry -> T=4 inherently degrades quality; distillation must specifically fix it.
- shipped@T=4 sharp  -> our distilled weights are undertrained; train more / fix critic balance.

### EMA (added 2026-07-13, commit 31dd612)
- DMD raw generator loss oscillates by design (avg50 cycled 0.35->0.68->0.49 over ~1000 iters
  — a full lobe, NOT divergence). Ship the EMA weights, not raw snapshots.
- Impl: sharded fp32 EMA, NO FSDP.summon_full_params (avoids the 5.3GB/rank spike that got
  EMA disabled before); full_tensor()->CPU only at save. Saves `generator_ema` in the ckpt.
  Config: ema_weight=0.999, ema_start_step=200. Render with `--use_ema`.
- loss_fake drifts up late (spikes 1.5-2.2) = critic destabilizing. Lever: raise
  dfake_gen_update_ratio 5->10 and/or lower generator lr (standard DMD2 generator/critic balance).

### fps note (SEPARATE thread — do not block training)
- T=4 renders measured ~2 fps vs ~14 fps T=5 baseline on the SAME 16 cores (TP4xCP4).
  Pathological (T=4 is LESS work). profiler per_neff_mfu.txt (run 120726204813): 2 hottest
  NEFFs OVERHEAD-BOUND (tensor ~6%, gpsimd ~89%, dma ~48%) under RF_RING=1 @ 1200 seq len.
- CAUSE DISPUTED: user validated main+ring = 14fps at 1480 seq len. Control queued
  (iter1000-job.yaml edited: original ckpt @ T=5 @ ring-on @ 1200) — NOT yet run.
- Render jobs: iter{200,400,1000}-job.yaml. BRANCH=main (+ committed restore_layout T=4 fix
  4e7e1d6 so T=4 geometry doesn't crash), RF_RING=1, TP4xCP4, latent_w 80 (1200), --use_ema.
- Frames land in-pod at /tmp/results/clean_out/prompt_NNN.mp4 (kubectl cp) and on the PVC at
  /var/mdl/rolling-forcing/runs/<TS>/frames/cp4_16rank/ (aws s3 cp) after the job's persist step.

---

## Denoise-steps vs fps study (2026-07-15) — MEASURED

Swept denoise steps on the GOLDEN main rf-job.yaml (shipped ckpt, TP4xCP4, 16 ranks,
fs1200/480x640, RF_RING=1, --use_ema). ONLY `denoising_step_list` changed per run;
everything else byte-identical (diffed vs main:rf-job.yaml = job name + 1 patch line).
NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=0 in ALL runs (constant, not a variable).

| T (steps) | denoising_step_list             | max_frames (nds*3) | full_frames (nfpb+max) | DiT/block | fps  | regime |
|-----------|---------------------------------|--------------------|------------------------|-----------|------|--------|
| 3         | [1000,667,333]                  | 9                  | 12                     | ~1700 ms  | 6.5  | SLOW   |
| 4         | [1000,750,500,250]              | 12                 | 15                     | ~2150 ms  | 5.2  | SLOW (worst) |
| 5         | [1000,800,600,400,200] (shipped)| 15                 | 18                     | ~700 ms   | 14.3 | FAST   |
| 6         | [1000,833,667,500,333,167]      | 18                 | 21                     | ~700 ms   | 14.3 | FAST   |
| 7         | [1000,857,714,571,429,286,143]  | 21                 | 24                     | ~750 ms   | 13.5 | FAST   |

**Finding: a hard fps CLIFF between T=4 and T=5.** T<=4 slow (5-6.5 fps), T>=5 fast
(13.5-14.3 fps). ~3x jump at the max_frames 12->15 (full_frames 15->18) boundary.
NOT a smooth "more steps = faster" gradient (T5/T6/T7 are a flat plateau; T3 is slow
but less slow than T4). T=5 is the shipped design point (num_training_frames=21).

**Ruled OUT as the cause (with data):**
- More work at T=4: NO. Code path proven (upstream + our port): T=4 dispatches a SMALLER
  full_frames tensor (15 vs 18), T-scaled max_frames, fixed 21-frame attention cap
  (max_attention_size=21*frame_length, identical upstream & ours). Every shape <= T=5.
- Compilation / recompile: NO. NEFFs are per-shape, recompiled every run regardless of T;
  fps numbers are warm/steady blocks (post-compile).
- async setting: NO. NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=0 in BOTH the 14fps (T5)
  and 5fps (T4) runs — constant, so not the differentiator.
- checkpoint / ring / /var/mdl I/O: NO. ckpt read once into HBM; ring & FS identical across T.
- per-NEFF single-io cost: IDENTICAL T4 vs T5 (same hashes, same us/NEFF in per_neff_mfu.txt).
- invocation count: T4 executes FEWER compiled-region launches than T5 (146,752 vs 158,560).

**What the trace shows (host CPU trace, runs 150726020531=T5, 160726000211=T4):**
- T4 total device compute is LESS than T5 (busy 142.6s vs 151.0s) yet wall-clock is MORE
  (574s vs 399s). Device utilization: T4=24.9% vs T5=37.9%. The extra ~175s is IDLE, not work.
- i.e. same-or-less work, far more idle => the device sits waiting between ops at T<=4.

**Leading (UNPROVEN) hypothesis: device/compiler execution artifact of the small shape.**
full_frames=18/21 (T5/6/7) likely hit an efficient tiling; full_frames=15 (T4) / 12 (T3)
fall to a slower kernel variant / worse overlap. This is a neuronx-cc shape->tiling effect,
NOT a Python-source bug (both codebases dispatch LESS for T=4). Could also be a collective/DMA
that pipelines only above a size threshold. NOT distinguishable from the CPU trace.

**To CONFIRM + localize (next step, NOT done):** device profile — `neuron-profile` per-NEFF
DEVICE timing on one T=4 vs one T=5 NEFF (names the slow NEFF + whether tile factor differs).
Cheap diagnostic+fix to try FIRST: pad T=4's full_frames 15->16 or ->18 (shape hint). If speed
recovers, it confirms shape/tiling AND fixes it without a hand-written kernel. Do NOT author an
NKI kernel blind — the slow NEFF is likely compiler-generated (no kernel to edit); the lever is
a compiler flag / shape pad, decided AFTER the device profile.

**Implication for distillation goal (fewer steps + quality + fps):** fewer denoise steps does
NOT automatically get faster on this stack — T<=4 lands in the slow regime by SHAPE, below the
T=5 (21-frame) design point. A T=4 distilled model needs the shape/tiling fix above to serve at
full fps; otherwise T=4 serves at ~5 fps despite doing less compute than the 14 fps T=5.
Separately: our distilled T=4 checkpoints render BLURRY (undertrained) vs shipped-ckpt@T=4 which
renders SHARP — so quality is a TRAINING problem, independent of this fps/shape issue.
