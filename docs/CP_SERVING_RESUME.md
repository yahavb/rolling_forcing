# RF Serving Optimization — Context-Parallelism (CP) — SESSION RESUME

**This is the INFERENCE/SERVING optimization thread.** There is a SEPARATE training thread
(RF T=4 distillation — see `docs/DISTILL_T4_RESUME.md` / memory `project_rf_t4_distill_resume`).
Do NOT mix them.

## HOW TO RELOAD THIS SESSION (say this after a reboot)
> "Read docs/CP_SERVING_RESUME.md and resume the RF CP serving work."

That's it. This doc + the memory files below carry all goals/methods/state — do not
re-explain the project.

---

## GOAL
Increase Rolling Forcing (RF) text-to-video **inference fps** on a single AWS Trn2, at
**fixed 480×640**, toward **16 fps** (interactive). Baseline was TP4×SP4 = 12.4 fps.

## THE MODEL / HARDWARE (one-liner)
Wan2.1-T2V-1.3B + DMD-distilled rolling-forcing DiT (dim=1536, 12 heads, 32 layers, bf16).
Trn2: LNC1=12GB/core, LNC2=24GB/core. Legal collective group sizes: 1,2,4,8,16, or mult of 32.
TP pinned at 4 (12 heads only divide by 4). SP = world/TP is RF's context-parallel axis.

## THE KEY FINDING & CURRENT LEVER (context-parallelism)
RF's original "SP" wastefully **all-gathers the full query over `world` every layer, RoPEs
all L tokens, then discards (SP-1)/SP of it.** True CP keeps the query sequence-sharded:
gather Q only over the `attn-tp` subgroup → this rank's shard directly; RoPE only that shard.
K/V still world-gathered (attention needs the whole KV window).
- **Denoise path (phase 0)** CP: DONE, shipped to main (`6a480c4`). CP4 = **14.13 fps (+14%)**,
  bit-identical (max|Δ|=0), quality-confirmed.
- **Merged path (phases 1-10, = 10/11 of generator calls = bulk of runtime):** IN PROGRESS on
  branch `cp-merged-path`. Requires cu/dn-separate input sharding (world-contiguous [cu|dn]
  breaks attn-tp gather). All 5 sites implemented + CPU-proven + on-device gate PASSED
  (max|Δ|=0, quality good) BUT slower: 11.87 fps < 14.13 — added K/V deinterleave copies.
  Currently removing those copies toward true full CP (also shard K/V so world-gather shrinks).

## CURRENT STATE (as of last session)
- **Branch under test:** `cp-merged-path` (latest `69e0fdc` = removed redundant k_full=cat copy).
- **main:** `6a480c4` (denoise CP shipped). DO NOT merge to main without explicit confirmation.
- **Last merged-CP result:** 11.87 fps, gate PASS, run dir `/var/mdl/rolling-forcing/runs/100726214517`
  (that's the baseline to Tier-1-diff the next run against).
- **Next action:** deploy `69e0fdc`, check gate PASS + fps vs 11.87, Tier-1 NEFF-diff vs 100726214517.

## THE FISH LADDER OF NEGATIVE RESULTS (do NOT retry — all failed at fixed 480×640)
RoPE kernel rewrites (flat), noise prefetch (flat), NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT (crash,
forbidden — RF uses explicit async), nfpb 3→7 (nfpb=3 welded in kernels; OOM/asserts),
qkv-gather-fuse (−6%), qkv-matmul-fuse (−43%). LESSON: **hand-fusion loses to the Neuron
compiler.** latent_w↑ raises MFU only by making a bigger video (fps drops). SP8/CP8 = slower
(collective tax). The ONLY thing that raised fps = eliminating the redundant query all-gather
(=CP). Full table: memory `reference_rf_benchmark_log.md`.

---

## THE WORKFLOW (every experiment — this is the loop)
1. **THINK / DESIGN.** State the hypothesis + what it changes. For any parallelism/layout
   change, **CPU-proof it bit-identical FIRST** (numpy/torch sim of the index math, assert
   max|Δ|=0) — CP kernels are NKI/Neuron-only, can't run on CPU, so prove the plumbing offline.
   See `verify_merged_cp.py` for the pattern.
2. **CHANGE CODE** in a git worktree off the right base (usually `origin/main` for a fresh
   experiment, or the current experiment branch to extend it).
   `git worktree add -b <exp-branch> /tmp/rf-<x> origin/main`
3. **COMMIT + PUSH to the dedicated branch** (NEVER main):
   `git add <files> && git commit -m "..." && git push -u origin <exp-branch>`
   (The k8s job clones this branch by name — the branch MUST be pushed before deploy.)
4. **POINT THE MANIFEST at the branch.** Edit `rf-job.yaml`:
   - `BRANCH` env value → `<exp-branch>`
   - topology: `NPROC` (8/16/32 = CP2/CP4/CP8, tp_degree stays 4), `LATENT_W`
     (80=480×640 fs1200; 64=480×512 fs960 — SP8/CP8 needs fs%32=0 so must use 64), and the
     resource claim (m-lnc1-trn2=16c/12GB, l-lnc1-trn2=32c/12GB, l-trn2=16c/24GB-LNC2 for profiler)
   - any feature flag env (e.g. `RF_CP_MERGED=1`) in the accuracy gate [0/3] + clean [2/3] +
     profiled [3a]/[3b] as needed.
5. **DEPLOY:**
   ```
   cd /Users/yahavb/rolling_forcing
   kubectl delete job rolling-forcing --ignore-not-found
   kubectl apply -f rf-job.yaml
   ```
6. **READ (user dumps the log):** `kubectl logs rolling-forcing-<pod> > /tmp/<pod>.log`
   Read order — **QUALITY FIRST, THEN FPS** (user's rule):
   - `ACC GATE PASS max|Δ|=0` (+ user eyeballs frames) — correctness gate; must pass or job aborts.
   - `steady-state MEDIAN = N fps` — the number.
   - **Tier-1 NEFF-diff** the run's `per_neff_mfu.txt` vs the named baseline run dir to
     ATTRIBUTE the change (which DMA class shrank/grew) — prove, don't guess.

## THE rf-job.yaml STRUCTURE (what runs)
`[0/3]` accuracy gate (ref vs CP frame-diff, aborts on divergence) → `[1/3]` warmup →
`[2/3]` CLEAN (fps, no profiler) → `[3a]` HARMONY trace pass (NEURON_RT_INSPECT only) persists
harmony_bundle to /var/mdl immediately → `[3b]` NATIVE profile pass (NEURON_PROFILE_ENABLE only)
→ `[5]` per-NEFF table. **3a and 3b MUST be separate passes** (running both profilers in one
pass = "Only one subscriber allowed" collision). Profiler needs LNC2/24GB (l-trn2).
Artifacts → `/var/mdl/rolling-forcing/runs/<ddmmyyHHMMSS>/` (= `s3://621547421844-ap-southeast-4/...`).

## HARD RULES (memory-backed — never violate)
- NEVER merge/push to `main` until user explicitly confirms after extensive testing
  (`feedback_no_merge_to_main_unconfirmed`).
- NEVER suggest giving up / stopping / banking on CP — always propose the next fix
  (`feedback_never_give_up_cp`).
- NEVER set `NEURON_LOGICAL_NC_CONFIG` (LNC chosen by resource claim only).
- No T5/CPU offload. Accuracy: CPU proof + on-device frame-diff before any reported number.
- Creds: user runs kubectl/aws (I'm usually unauthed). Log dumps come from the user.
- `/var/mdl` = `s3://621547421844-ap-southeast-4/`; run dirs are `ddmmyyHHMMSS` (day-first,
  NOT lexically sortable — always pin the run).

## MEMORY FILES (auto-loaded each session — the durable state)
- `reference_rf_benchmark_log.md` — EVERY run: config/branch/fps/MFU/result + the CP sweep + merged-CP status
- `project_rf_collective_bound.md` — full diagnosis (4% MFU, collective/DMA-bound) + CP breakthrough
- `feedback_never_give_up_cp.md`, `feedback_no_merge_to_main_unconfirmed.md` — the hard rules
- Skill `rolling-forcing-trn2-benchmark` (RF-wired) / `neuron-3run-benchmark` (general, in
  github.com/yahavb/claude-skills) — the benchmark+profile methodology incl. NEFF-diff attribution.

## PAPER
`docs/PAPER_parallelism_neuron.md` — paper-source (Advanced Computing, systems/parallelism).
Converge with the TRAINING thread's parallelism work (NOT this serving thread's serving specifics).
