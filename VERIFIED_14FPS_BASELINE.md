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
