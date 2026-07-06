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
