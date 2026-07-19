# Distillation Quality vs Iterations — Empirical Sweep

## The question (settle it with frames, not loss)

Renders of the 1.3B DMD-distilled student have looked **blurry** at every checkpoint we've
eyeballed (prompt_000 iter1000/1200; prompt_003 iter200). The loss (`dmdnorm_avg50`) is a
flat ~0.44–0.60 equilibrium across iters 200→4000 — it does **NOT** decrease with training,
so it cannot answer "will more training sharpen the output." DMD is adversarial-like: the
generator loss orbits an equilibrium, it doesn't descend. **Only the rendered video answers
the quality question.**

## The experiment

`rf-render-sweep-job.yaml`: render the SAME prompt from the SAME distill run at a series of
checkpoints, in one pod, and compare sharpness by eye.

- Default: prompt_000 (the trained prompt for the wnlfl run `202607152026`) at iters
  `200 1000 2000 4000` — spans the trajectory (200/1000 = known-blurry anchors from prior
  eyeballing; 2000 = untested middle; 4000 = latest/most-trained endpoint).
- Knobs: `ITERS` (space-separated checkpoints, must exist on the PVC), `RUN_DIR` (which
  distill run), `PROMPT_IDX` (which prompt line to render).
- Output: `/var/mdl/rolling-forcing/runs/quality_sweep_<TS>/iter<IT>_prompt<IDX>.mp4`, one
  mp4 per checkpoint.

```
kubectl apply -f rf-render-sweep-job.yaml
kubectl logs -f job/rf-render-sweep
aws s3 cp --recursive s3://621547421844-ap-southeast-4/rolling-forcing/runs/quality_sweep_<TS>/ ./sweep/
```

## How to read the result (pre-committed, so we don't rationalize after)

- **Later checkpoints visibly sharper** → training length IS the lever. Keep training
  (iter6000/8000/10000), re-sweep.
- **All equally blurry (1200 ≈ 4000)** → training length is NOT the lever. The blur is a
  ceiling elsewhere. Candidates to test next, in order:
  1. **Student capacity** — 1.3B may not be able to match the 14B teacher's detail regardless
     of iters. Test: this is exactly what the 14B student distill answers.
  2. **EMA smoothing** — ema_weight=0.999 averages ~1000 steps; may soften high-freq detail.
     Test: render the RAW (non-EMA) weights at the same iter (`--use_ema` off).
  3. **Recipe** — lr / DMD timestep buckets / guidance_scale hitting equilibrium early.

## Facts on the table (measured, not speculated)

- `dmdnorm_avg50` is flat across iters (proxy, useless for the quality question).
- Renders eyeballed so far: prompt_000 iter1000/1200 blurry; specialization IS real
  (prompt_000-ckpt renders prompt_000 >> prompt_003 — the trained prompt wins).
- This sweep is the first thing that measures quality-vs-iter directly.

## Method note (why a sweep, not one render)

One render at one iter can't distinguish "undertrained" from "at ceiling" — you need the
*trajectory*. Rendering 200→4000 in one pass shows whether the curve is still moving. If it's
flat, no single later checkpoint will surprise you; if it's rising, you know to keep going.
This is the cheap decisive test before committing more training compute.

## Results — MEASURED (run quality_sweep_190726155355, prompt_000, by eye)

| iter | render quality (by eye) |
|------|-------------------------|
| 200  | **BEST** |
| 1000 | ≈ 200 (same/similar) |
| 2000 | WORSE than 200/1000 |
| 4000 | almost complete NOISE |

**VERDICT: quality DEGRADES with training — the distillation DIVERGES, it does not converge.**
Best checkpoint is the EARLIEST (iter200); more training actively destroys the output.

### What this means
- **"Train longer = better" is FALSE here.** The opposite. This is adversarial/DMD collapse:
  the generator and critic lose their balance and the student degenerates toward noise.
- **dmdnorm_avg50 was misleading.** Its flat-then-oscillating ~0.44–0.60 was NOT a healthy
  equilibrium — it was the run drifting/diverging. The loss proxy did not reveal the collapse;
  only the rendered video did. Do not trust dmdnorm magnitude as a quality signal, ever.
- **Every "wait for iter3000+" instinct was backwards.** The usable model is early.

### For the "distill a prompt QUICK" demo
This is actually the RIGHT answer for the goal: quick works. The demo checkpoint is iter200
(or earlier). Do NOT run to iter10000 — it makes it worse.

### Root-cause candidates to investigate (the divergence, not the blur)
1. **Learning rate / critic-generator balance** — lr 1.5e-6, lr_critic 4e-7,
   dfake_gen_update_ratio 5. Collapse suggests the balance is wrong for this stack; the
   generator over-updates relative to the critic and walks off.
2. **EMA window** — ema_weight 0.999 averages ~1000 steps; by iter4000 the EMA has averaged
   in thousands of increasingly-bad raw weights. Early EMA (iter200, ~few hundred steps of
   history) is closer to the good ode_init. Test: does raw (non-EMA) diverge slower/faster?
3. **No early stopping** — the run has no quality gate; it trains to iter10000 blindly while
   the output has already collapsed by iter2000. Needs a periodic-render quality check to
   stop at the best iter.

### Immediate consequence
The other long runs (p003 at iter1718, 14B) are likely on the SAME divergence trajectory —
their late checkpoints are probably WORSE, not better. Render THEIR early checkpoints
(iter200-ish), not their latest, before drawing any conclusion from them.
