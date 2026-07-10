# RF T=4 Distillation — RESUME STATE (read this first on return)

Last session captured this so we don't restart from scratch. Everything is in git on
branch **`rf-distill-t4`** (pushed). The training loop **runs stably** (memory war won);
the open question is **convergence**.

---

## TL;DR — where we are

- **Memory/OOM: SOLVED.** The 3-model DMD loop trains stably on 16 Trn2 cores, memory flat
  across G-steps, past all historical OOM walls (ran to iter ~250).
- **Convergence: OPEN.** Last observed `dmdnorm_avg50` **rose** 0.51→0.60 over 250 iters
  (should DECLINE toward 0). Root cause found + fixed (paper-verified): the
  `DISTILL_FUNCTIONAL_ATTN` flag was blinding ALL rollout blocks (deleting the KV temporal +
  global context that IS Rolling Forcing). Fix committed `74c996d` — **not yet re-run.**
- **NEXT ACTION:** re-run the job (below), watch the new `[grad] grad_norm=` (proves weights
  move) and `dmdnorm_avg50` (should stop rising / decline). If still rising with grad_norm>0,
  the next audited divergence is the **scoring-timestep coupling** (see "Open divergences").

---

## The workflow (how we make + test changes — MUST follow)

1. **All training code lives in `training/`** (vendored + Neuron-adapted from upstream
   RollingForcing). The job **git-clones** the branch, so **changes must be committed AND
   pushed** before they take effect in a run. Local edits alone do nothing.
2. **Branch:** always work on **`rf-distill-t4`**. The job's `BRANCH` env = `rf-distill-t4`.
3. **Commit + push, then launch:**
   ```bash
   cd ~/rolling_forcing
   git add <files> && git commit -m "..."       # descriptive, cite the failure it fixes
   git push origin rf-distill-t4
   kubectl delete job rf-distill --ignore-not-found && kubectl apply -f rf-distill-job.yaml
   ```
4. **Read a run:**
   ```bash
   kubectl get po | grep rf-distill                     # find the pod
   kubectl logs <pod-name> > /tmp/rf-distill-<id>.log    # dump, then grep
   ```
   The log opens with a PROVENANCE banner (JOB + branch + commit) — **always confirm the
   commit matches what you pushed**, or you're reading a stale/old-code run.
5. **Diagnostics convention:** temporary debug prints are tagged **`[dbg ...]`** (grep-strip
   with `grep -v '\[dbg'` once training is confirmed). Real output = `it N/... dmdnorm`,
   `loss_fake`, `[grad] grad_norm`, `[ckpt] wrote`.
6. **Reference for CORRECTNESS = the paper + upstream code, NOT StreamDiffusionV2.**
   - Paper: `2509.25161v1.pdf` (RollingForcing, Liu et al. NTU+Tencent).
   - Upstream code: `~/RollingForcing/` and the vendored `training/model/dmd.py`,
     `training/model/base.py`, `training/pipeline/rolling_forcing_training.py`.
   - SD (`~/StreamDiffusionV2`) was ONLY the parallelism/OOM reference — do NOT use it for
     training-accuracy decisions.

---

## Current topology / config (final, working)

- **16 NeuronCores** (l-trn2 = 2 chips × 8, LNC=1). Job: `rf-distill-job.yaml`.
- **Three-group asymmetric placement:** teacher ranks 0-7 (14B, frozen, FSDP1, bf16),
  student 8-11 (1.3B, **FSDP2 fully_shard**), critic 12-15 (1.3B, FSDP1). Cross-group via
  `dist.broadcast` (Neuron has no P2P).
- **Key env (in `rf-distill-job.yaml`):** `DISTILL_THREE_GROUP=1`, `DISTILL_FUNCTIONAL_ATTN=1`,
  `NPROC=16`, `NEURON_FALLBACK_ENABLED=1`, `NEURON_LAUNCH_BLOCKING=1`.
- **Config `configs/rolling_forcing_dmd_t4.yaml`:** `num_training_frames: 6` (2 blocks),
  `grad_accum: 1`, `gradient_checkpointing: true`, `dfake_gen_update_ratio: 5`, `warmup: 10`,
  `denoising_step_list: [1000,750,500,250]` (T=4), `guidance_scale: 3.0`, `lr: 1.5e-6`,
  `lr_critic: 4e-7`, `tp_degree: 4` (+ implicit teacher_tp=8/student_tp=4/fake_tp=4 in trainer).

---

## What each metric means (how to read a run)

- `it N [G-step] dmdnorm=X dmdnorm_avg50=Y` — a **generator** update ran (every 5th iter
  after warmup). `dmdnorm` = mean|DMD grad| this step; `dmdnorm_avg50` = rolling mean of last
  50 real dmdnorm values. **CONVERGENCE = dmdnorm_avg50 trending DOWN toward 0.**
- `it N [critic-only] dmdnorm=nan` — only the critic trained this iter (normal; nan = no
  G-step, not an error).
- `it N loss_fake=X` — critic health (every iter). Should stay low/stable; **rising = critic
  losing track** (bad).
- `[grad] it N: grad_norm=X` — (NEW, `74c996d`) logged each G-step. **>0 = student weights are
  actually updating.** ≈0 = frozen model (SD's silent-failure mode). This is the guardrail.
- `[ckpt] wrote model.iterN.pt` — checkpoint saved (every `save_every`=200) to the PVC
  `/var/mdl/rolling_forcing/distill/<TS>/`.

---

## Open divergences from upstream dmd.py (audited; check IN THIS ORDER if still not converging)

My flat trainer (`training/trainer/distillation_3group.py`) **reimplements** the DMD update
(it must — the 3-group split means real_score is on other ranks, so upstream's
`compute_distribution_matching_loss` can't be called directly). Audited vs upstream:

1. **[FIXED, `74c996d`] functional attn on ALL blocks** → blinded the rollout. Now gated on
   `env AND torch.is_grad_enabled()`: no_grad blocks use the full KV cache, only the with-grad
   exit block is functional. Paper-confirmed (Fig.3: non-gradient windows preserve KV context).
2. **[SUSPECT #1 if still rising] scoring-timestep coupling.** Upstream samples a FRESH random
   scoring timestep inside `compute_distribution_matching_loss`; mine reuses the rollout's
   bucketed `tt` for `x_t`. Upstream decouples them. If dmdnorm still rises, decouple: sample
   a fresh (bucketed) timestep + build a fresh noisy `x_t` for teacher/critic scoring,
   independent of the rollout's exit timestep.
3. **[SUSPECT #2, likely benign] critic flow target.** Mine: `flow_pred(raw model out)` vs
   `flow_tgt = noise - x0`. Upstream: `flow_pred = _convert_x0_to_flow_pred(pred_x0, xt, t) =
   (xt-x0)/sigma_t` vs `(noise-x0)`. Verify the raw model output == the sigma-normalized
   conversion; if not, use upstream's `_convert_x0_to_flow_pred` exactly.
4. **[known approximation, gradient-path only] the with-grad exit block attends only its own
   tokens** (functional), not the KV cache — vs the paper's ideal (gradient windows attend the
   cache bidirectionally). Bounded; revisit last.
5. **gradient_mask** — upstream masks the first block from the DMD loss ONLY when
   `num_generated_frames > 21`. At `num_training_frames=6` this is never triggered → not a bug
   at current config. (Would matter if frames raised back toward 21.)

---

## The RESUME command sequence

```bash
cd ~/rolling_forcing
git checkout rf-distill-t4 && git pull            # confirm HEAD has 74c996d (functional-attn fix)
git log -1 --format="%h %s"

# launch the fixed run:
kubectl delete job rf-distill --ignore-not-found && kubectl apply -f rf-distill-job.yaml

# after ~2-3 min (first-call NEFF compiles are slow, NOT a hang), read:
kubectl get po | grep rf-distill
kubectl logs <pod> > /tmp/rf-distill-<id>.log
grep -E "commit |^it [0-9]+/|grad_norm|dmdnorm_avg50|loss_fake|ckpt wrote|OOM|Traceback" /tmp/rf-distill-<id>.log | tail -40
```

Decision after the run:
- **grad_norm > 0 AND dmdnorm_avg50 declining** → converging; let it run to a checkpoint,
  then RENDER the checkpoint at T=4 vs the T=5 baseline (ground truth — the paper insists loss
  alone is not enough).
- **grad_norm ≈ 0** → weights frozen; graph severed somewhere (debug the DMD grad path).
- **grad_norm > 0 but dmdnorm_avg50 still rising** → apply Open-divergence #2 (decouple the
  scoring timestep), commit, push, rerun.

---

## Full history / provenance
- Complete fix ledger + memory notes: `~/.claude/.../memory/reference_dmd_rollout_graph_oom_fix.md`
- Paper-source material (parallelism, training side): `docs/PAPER_parallelism_neuron_training.md`
- Commit index: see the paper doc §10, and `git log --oneline 3d9230d..HEAD`.
- Runbook (pre-existing): `docs/DISTILL_T4_RUNBOOK.md`.
