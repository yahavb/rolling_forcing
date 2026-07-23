# RF T=4 Distillation — RESUME STATE (read this first on return)

Last session captured this so we don't restart from scratch. Everything is in git on
branch **`rf-distill-t4`** (pushed). The training loop **runs stably** (memory war won);
the open question is **convergence**.

---

## LATEST (2026-07-22) — HARDWARE RULED OUT: blur is the RECIPE, not Neuron compute

CPU↔Neuron fp32 numerical-parity check on the 1.3B student (ckpt
`distill/202607220649/model.iter200.pt`). Ran ONE student rollout (noise→x0), fixed
seed 0, TRUE fp32 on BOTH paths, and diffed the x0 tensors:

| metric | value |
|---|---|
| cosine sim | **1.000005** |
| mean rel err | **0.000136** |
| mean \|A−B\| | 0.000018 |
| max \|A−B\| | 0.0012 (on a [-2.79, 3.06] range) |
| A vs B mean/std | 0.12178/1.16651 vs 0.12179/1.16651 |

**VERDICT: the on-device compute is CORRECT** — matmuls, RoPE, norms, scheduler math
all match CPU to op-order noise. So the blur + non-convergence is the **DMD recipe /
training dynamics**, NOT a hardware-accuracy bug. (Consistent with the training log:
`dmdnorm_avg50` flat ~0.46, `grad_norm` oscillating — recipe-shaped, not corrupted.)

- SCOPE: this validated the Neuron EAGER backend in fp32. It did NOT test the bf16 NKI
  attention kernels (fp32 bypasses them). To also rule those out: a separate bf16 run
  (`DTYPE=bf16 USE_NKI_KERNELS=1 ATTN_DTYPE=bf16`). Given the eager path is clean and the
  failure is recipe-shaped, chase the recipe first.
- HOW (reproduce): `training/validate_student_cpu.py` (single-process, no FSDP/dist;
  forces true fp32 via `ATTN_DTYPE=fp32` + `USE_NKI_KERNELS=0`). Two k8s jobs run it
  off the training pod so the pod stays free:
  - `rf-validate-cpu-job.yaml` — m5 CPU node, no Neuron SDK; precomputes embeds to the
    PVC + writes `parity.cpu.pt`.
  - `rf-validate-neuron-job.yaml` — own `m-lnc1-trn3` claim (NOT the training pod, which
    holds all its cores); reads the SAME PVC embeds, writes `parity.neuron.pt`, diffs.
  - Both read/write ONE tar object on `/var/mdl` (S3-FUSE) — never `cp -r` a dir there
    (a 2275-file tree wedged a pod 36 min). See the S3-PVC memory / skill HARD RULE.
- NEXT on the recipe (documented drift, check in this order): grad_accum, ema decay/start,
  the 16-bucket timestep sampling vs upstream's continuous, generator lr / critic balance.

## LATEST (2026-07-13) — render/quality findings, read FIRST

- **Quality verdict: our T=4 distilled model is BLURRY, undertrained. NOT usable yet.**
  Rendered the distilled checkpoints end-to-end (train→ckpt→serve→eyeball):
  - iter200 EMA (T=4) → blurry. iter400 EMA (T=4) → blurry. **iter1000 EMA (T=4) → still blurry.**
  - Ground truth: shipped `rf_sp4_videos/prompt_000.mp4` (T=5) is **sharp**. Ours is much worse
    on the SAME prompt it was trained on (prompt_000 = the western/horse POC prompt).
  - iter1000's EMA has ~800 G-steps of averaging (w=0.999, start 200) — so "EMA too early"
    is NOT the excuse anymore. The student is genuinely **under-converged**.
- **EMA was added** (commit `31dd612`, sharded/no-summon, saves `generator_ema`; config
  `ema_weight=0.999 ema_start_step=200`). DMD raw loss OSCILLATES by design (avg50 cycled
  0.35→0.68→0.49 over ~1000 iters — a full lobe, NOT divergence); you ship the EMA weights.
  EMA render loads via `--use_ema`. But even EMA is blurry → training, not serving, is the blocker.
- **OPEN CONFOUND before more training:** iter1000 differs from the sharp baseline by TWO
  things — our weights AND T=4. To tell if blur is from **T=4 (4 steps too few)** vs **our
  distillation being bad**, render the SHIPPED ckpt at T=4:
  - shipped@T=4 blurry → 4 steps inherently degrades; distillation must fix that (hard).
  - shipped@T=4 sharp → our weights are undertrained; more/better training is the fix.
- **fps side-note (unresolved, low priority):** T=4 renders measured ~2 fps vs the ~14 fps
  T=5 baseline on the SAME 16 cores — pathological since T=4 is LESS work. Profiler
  (`per_neff_mfu.txt`, run 120726204813) showed the 2 hottest NEFFs are OVERHEAD-BOUND
  (tensor ~6%, gpsimd ~89%, dma ~48%) with RF_RING=1 at 1200 seq len. Whether ring is the
  cause is DISPUTED by the user (validated main+ring=14fps at 1480 seq len). Control queued:
  original ckpt @ T=5 @ ring-on @ 1200 (iter1000-job.yaml, edited) to isolate. NOT yet run.
  Serving-fps is a SEPARATE thread from convergence; do not let it block training work.

## TL;DR — where we are (training convergence)

- **Memory/OOM: SOLVED.** The 3-model DMD loop trains stably on 16 Trn2 cores, memory flat
  across G-steps, past all historical OOM walls (ran to iter ~1050+, EMA adds ~1.3GB/rank sharded).
- **Convergence: OPEN — and now known INSUFFICIENT for quality.** `dmdnorm_avg50` oscillates
  ~0.35↔0.68 (expected for DMD; not divergence). `loss_fake` drifts up with spikes to 1.5–2.2
  late (critic destabilizing — the generator/critic imbalance lever: raise
  `dfake_gen_update_ratio` 5→10 and/or lower generator lr). The render proves the current
  recipe under-converges regardless of EMA.
- **NEXT ACTION:** (1) render shipped ckpt @ T=4 to resolve the T=4-vs-undertrained confound;
  (2) if undertrained → train far longer and/or fix critic balance, re-render a late EMA ckpt.

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
