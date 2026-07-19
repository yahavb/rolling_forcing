# Distillation Divergence — Root Cause (config drift from the known-good recipe)

## The observed failure (measured, not speculated)

Quality sweep `quality_sweep_190726155355` (prompt_000, wnlfl run 202607152026), by eye:

| iter | quality |
|------|---------|
| 200  | best (but still WORSE than the project's shipped rolling_forcing_dmd.pt) |
| 1000 | ≈ 200 |
| 2000 | worse |
| 4000 | almost complete noise |

**Quality DEGRADES with training → the run DIVERGES (adversarial/DMD collapse), it does not
converge.** And even the best (iter200) is worse than the shipped checkpoint. So OUR RECIPE is
producing bad checkpoints — this is a recipe problem, not a train-longer problem.

`dmdnorm_avg50` was flat ~0.44–0.60 the whole time and NEVER revealed this — the loss proxy
hid the collapse; only the rendered video showed it. Do not trust dmdnorm as a quality signal.

## Root cause: we drifted from the upstream recipe that made the GOOD checkpoint

`training/configs/rolling_forcing_dmd.yaml` is the upstream recipe that produced the good
shipped `rolling_forcing_dmd.pt`. Diffing it against our `configs/rolling_forcing_dmd_t4.yaml`:

**lr, lr_critic, dfake_gen_update_ratio, guidance_scale are IDENTICAL** (1.5e-6 / 4e-7 / 5 /
3.0) — so the raw lr/critic knobs are NOT the cause. What we CHANGED away from the good recipe:

| knob | upstream (good) | ours (diverged) | why it matters |
|------|-----------------|-----------------|----------------|
| **grad_accum** | 4 (default) | **1** | **PRIME SUSPECT.** We cut effective batch 4x->1x to dodge a 14B G-step OOM. DMD's critic/generator balance is tuned for the larger effective batch; at batch=1 the gradient estimates are noisy, the critic can't track the generator, the pair diverges. We traded convergence for memory. |
| **ema_weight** | 0.99 | **0.999** | 10x LONGER averaging window (not "slightly longer" as our comment claimed). By iter4000 the EMA has averaged in thousands of degrading raw weights -> the noise. |
| num_training_frames | 21 | 3 (14B) / 6 | far shorter rollout -> less signal per step |
| data_path | vidprom (many) | single prompt | single-prompt overfit (intended for the demo) |
| sharding_strategy | hybrid_full | full | memory, not dynamics |
| fsdp wrap | size | transformer | memory, not dynamics |

## Fix priority (highest-confidence, cheapest first)

1. **grad_accum 1 -> 4** (match upstream). Restores effective batch = the most likely
   divergence cause. NOTE: grad_accum=1 was set to fix a 14B G-step OOM — but the run we
   judged is the 1.3B (wnlfl), which CAN afford grad_accum=4. The 14B needs a different memory
   fix (not batch=1) if it's to converge.
2. **ema_weight 0.999 -> 0.99** (match upstream). Stop the EMA dragging in bad late weights.
3. Re-run the 1.3B distill with these, re-sweep renders (200/1000/2000/4000) to confirm the
   curve now RISES or at least holds instead of collapsing.

## Consequence for the other runs

- **p003 (1.3B) and 14B runs use the SAME drifted recipe** (grad_accum=1, ema=0.999) -> almost
  certainly the SAME divergence. Their LATE checkpoints are likely worse, not better.
- The 14B specifically CANNOT just take grad_accum=4 (that's the OOM we dodged). It needs a
  real memory fix (more student ranks / smaller rollout / activation offload) so it can run a
  proper effective batch WITHOUT batch=1. Until then, 14B convergence is not expected.

## The honest miss

Prior analysis in this thread treated flat dmdnorm as "healthy equilibrium" and pushed
"train longer / render iter3000+." That was backwards — the run was diverging and the good
checkpoint (if any) is early. The lesson: gate distillation on periodic RENDERS, not on the
loss proxy.
