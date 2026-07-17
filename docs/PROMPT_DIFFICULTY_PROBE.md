# Prompt Difficulty & Cost-Effective Distillation Demos on Neuron

**Goal of this work:** we are *not* training a production model. We want the cheapest,
fastest way to **demonstrate that knowledge-distillation training runs on AWS Neuron
(Trn2)**. A single-prompt DMD distillation is the vehicle. The open question: **not all
prompts converge equally fast — which prompt gives the most convincing result for the
least compute?**

## What we observed (the trigger)

Distilling the 1.3B causal student (T=5 DMD, single prompt, ode_init) on two different
prompts, at the SAME early iterations:

| prompt | scene | dmdnorm_avg50 @ it10–45 |
|---|---|---|
| prompt_000 | cinematic western, galloping horse, sweeping desert | ~0.50–0.60 |
| prompt_003 | boxing kangaroo, modern gym, contained motion | ~0.30 |

prompt_003 starts with a **smaller DMD distribution gap** → less for distillation to close
→ converges faster → cheaper demo.

## What dmdnorm means (and its limits)

`dmdnorm` = magnitude of the DMD update signal = how far the student's score is from the
teacher−critic target. Lower = student already closer to the teacher's distribution on
that prompt.

- **Lower dmdnorm ⇒ faster convergence ⇒ less compute.** This is the lever we want.
- **BUT dmdnorm magnitude is NOT a cross-prompt quality score.** It measures *distance to
  close*, which depends on how far the base model started for that scene. The render A/B
  (does the distilled prompt beat an undistilled one) remains the ground truth.

## What does NOT change with prompt (a correction worth stating)

Prompt choice does **not** change checkpoint size, matmul shapes, or serving fps. Every
1.3B distilled checkpoint is 825 tensors / 11 GB with dim=2048/16-head/32-layer regardless
of prompt — only the weight *values* differ. So "easy prompt → smaller model → more fps"
is false. The only real payoff of an easy prompt is **fewer training iterations = lower
GPU-hours to a convincing result.** fps is fixed by architecture (and by T, which has a
hard cliff at T<5 on this stack — see DISTILL_T4_RUNBOOK.md).

## Hypothesis: what makes a prompt "easy" to distill

- Closer to the teacher's pretraining distribution (common scenes/objects).
- Simpler, more contained motion (the causal student matches the bidirectional teacher
  more easily when dynamics are simple).
- prompt_003 (single subject, gym, contained motion) < prompt_000 (galloping horse,
  sweeping camera, dramatic lighting) on all three counts.

## The probe (cheap measurement, not 16 full distillations)

`rf-prompt-probe-job.yaml` (branch `rf-prompt-probe`): loops all 16 `example_prompts.txt`
entries, runs a short `PROBE_ITERS`-iter (default 60) DMD distillation on each from the
same ode_init, saves NO checkpoints, and emits a ranking by final `dmdnorm_avg50`.

```
kubectl apply -f rf-prompt-probe-job.yaml
kubectl logs -f job/rf-prompt-probe
```

Output: a ranked TSV (`idx  final_avg50  min_avg50  prompt`) at
`/var/mdl/rolling_forcing/probe/<TS>/results.tsv`, sorted lowest-first. Lowest =
converges fastest = the prompt to use for a demo.

Cost: 16 × (~40s model load + 60 iters) ≈ under an hour on one 16-core node, zero
checkpoints written.

## How to use the result

1. Pick the lowest-`final_avg50` prompt.
2. Full-distill 1.3B on just that prompt (existing `rf-distill-*-job.yaml` recipe,
   `data_path` = that single prompt).
3. It should reach a sharp render in fewer iters than prompt_000 did → the cost-effective
   "distillation runs on Neuron" demo.

## Related experiments (branch map)

- `rf-distill-t4` — original 1.3B prompt_000 distill (the running baseline).
- `rf-distill-prompt003` — control: distill on prompt_003, expect 003-render > 000-render
  (proves specialization is prompt-specific, not "000 is easy").
- `rf-distill-14b` — 14B causal student scale-up (whole-node, xxl-trn2).
- `rf-prompt-probe` — THIS: rank all prompts by convergence speed.

---

## Blog post ideas (seeds)

1. **"Not all prompts cost the same to distill"** — the dmdnorm-gap observation, why
   prompt_003 converges faster than prompt_000, and a cheap probe to rank prompts. Angle:
   picking your demo prompt is a real cost lever in knowledge distillation.
2. **"Running video-diffusion distillation on AWS Trainium"** — the end-to-end story:
   14B teacher → 1.3B causal student, DMD on Neuron, the three-group placement + FSDP2
   memory fights, single-node Trn2. Angle: distillation training (not just inference) on
   Neuron.
3. **"What 'easy' means for a distilled model"** — dmdnorm as a distribution-gap proxy,
   the correction that prompt choice ≠ model size ≠ fps, and where the real cost lever is
   (iterations, not architecture).
4. **"Scaling a distillation student 1.3B → 14B on one Trn2 instance"** — the 64-core
   claim, the 5 hardcoded-1.3B-assumptions we peeled out, the first-G-step backward OOM
   and the frames 6→3 fix. Angle: what it actually takes to go bigger on fixed hardware.
