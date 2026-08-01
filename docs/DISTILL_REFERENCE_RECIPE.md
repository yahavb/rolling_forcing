# Distillation Reference Recipe — what the PUBLISHED numbers actually are

Written 2026-08-01. **Read this before changing any DMD hyperparameter in this repo.**

This doc exists because we spent a long time optimizing against a reference that was never
validated — our own earlier config comments — and two of those comments were factually
wrong. Everything below is from primary sources (papers, upstream repo configs read
verbatim, upstream code greps) with the source named. Where no primary evidence exists,
that is stated instead of guessed.

---

## 0. TL;DR — the corrections that matter most

1. **`grad_accum=4` does NOT "match upstream effective batch."** That comment (in
   `configs/rolling_forcing_dmd_distill.yaml` and `_14b.yaml`) is **FALSE**. Upstream has
   **no gradient accumulation at all** and reaches its batch purely by data parallelism.
2. **`total_batch_size: 64` in our configs is a silent no-op.** Verified by grep of
   upstream: `total_batch_size` is referenced in exactly ONE place,
   `trainer/ode.py:96-98`, and only as an *assertion that accumulation is absent*
   (`"Gradient accumulation is not supported for ODE training"`).
   `trainer/distillation.py` — the DMD trainer — **never reads it**.
3. **Every shipped video-DMD checkpoint trained at effective batch 64.** Our 1.3B ran at
   effective batch **4** (batch_size 1 x grad_accum 4, data-parallel width 1).
4. **Our rollout is 3-7x shorter than every published recipe** (6 or 3 frames vs 21-27).
5. **There is NO published quality metric gate.** Practitioners eyeball rendered samples.
   Our latent-sharpness curve is our own instrument, not an industry standard.
6. **No published 14B-STUDENT recipe exists anywhere.** All five shipped checkpoints use a
   1.3B student with a 14B *teacher*. At 14B student we are past the published frontier.

---

## 1. The published recipes (all produced SHIPPED, sharp checkpoints)

| knob | RollingForcing | Self-Forcing | CausVid | LongLive |
|---|---|---|---|---|
| **effective batch** | **64** (yaml `total_batch_size`) | **64** (Table 3) | 64 (8x8 GPUs) | **64** |
| **steps to converge** | **3,000** | **600** | 6,000 | **3,000** |
| **rollout frames** | **27**, N~U[21,27], DMD loss on **last 21** | **21** | 21 | — |
| lr | 1.5e-6 | 2e-6 | 2e-6 | 1e-5 (actor) |
| lr_critic | 4e-7 | 4e-7 | — | 2e-6 |
| `dfake_gen_update_ratio` | 5 | 5 | 5 | — |
| EMA | 0.99 from step 200 | 0.99 | none in configs | 0.99 from step 200 |
| guidance_scale | 3.0 | 3.0 | 3.5 | — |
| denoising steps | T=5 [1000,800,600,400,200] | T=4 [1000,750,500,250] | 4-step | — |
| student size | **1.3B** | **1.3B** | **1.3B** | **1.3B** |

Sources: RollingForcing arXiv **2509.25161** §4.1 + Appendix A and
`configs/rolling_forcing_dmd.yaml` @ github.com/TencentARC/RollingForcing;
Self-Forcing arXiv **2506.08009** Appendix A Table 3 + README;
CausVid arXiv **2412.07772**; LongLive arXiv **2509.22622** (NVlabs).

### Direct quotes worth keeping
- Self-Forcing: *"Most of our training runs use 64 NVIDIA GPUs (80GB memory each) with a
  per-GPU batch size of 1."*
- Self-Forcing README: *"By implementing gradient accumulation, it should be possible to
  reproduce the results in less than 16 hours using 8 H100 GPUs."*
  **← this is the sanction for buying batch with TIME instead of width.**
- LongLive: *"Training is conducted on 64 GPUs with one sample per GPU (global batch
  size = 64)"*; *"For the 60 s setting, we train for 3,000 iterations."*
- RollingForcing issue #6, a successful reproducer: *"I use 64 A800 with bs=64."*
- Causal Forcing (thu-ml, arXiv **2602.02214**, reproduced RF successfully): *"For bs64, we
  recommend training for no more than 1K steps; otherwise the motion dynamics may degrade.
  If bs is smaller (e.g., 8), more training steps is preferable."* And: with a good init
  *"DMD can be trained very few steps (e.g., ~100)."*

### Known ambiguity (do not resolve it by picking the convenient one)
RollingForcing's **paper says batch size 8**; its **config says `total_batch_size: 64`**.
Both searches flagged this. The reproducer evidence (64 A800) and the cross-paper
convergence on 64 favour 64.

---

## 2. What our configs actually run

| | published | `rolling_forcing_dmd_distill.yaml` (1.3B) | `rolling_forcing_dmd_14b.yaml` |
|---|---|---|---|
| effective batch | 64 | **4** (1 x accum 4, DP width 1) | **4** |
| rollout frames | 21-27 | **6** | **3** |
| lr / lr_critic / ratio / EMA / CFG | see table | **identical, correct** | **identical, correct** |
| student | 1.3B | 1.3B | **14B (no precedent)** |

**Why our effective batch is 4, not 64:** three-group placement gives the student its own
rank group (`student_tp=4` of world 16 for the 1.3B; 32 of 64 for the 14B). Those ranks are
*tensor/FSDP-sharded replicas of one sample*, not data-parallel replicas — so data-parallel
width is **1** and effective batch = `batch_size x grad_accum` = 4.

**Upstream reaches 64 by width, we can only reach it by accumulation.** For the 14B, 64
data-parallel replicas of a 14B student do not fit one instance at any resolution.

---

## 3. Documented failure modes at small effective batch

These match what we observed, and are the reason to suspect batch before suspecting
training length:

- **Self-Forcing issue #70** (4 GPUs): *"after around 1000 iterations, the generated videos
  become completely noisy"* — not fixed by cutting LR to 1/16. First reply:
  *"Did you open the gradient accumulation to make the total batch-size up to 64?"*
- **Self-Forcing issue #73** (8xH200, EBS=64 via hand-rolled accum): DMD loss plateaus,
  critic can't keep up even at lr_critic 4e-7 -> 1e-6.
- **Our own `DISTILL_QUALITY_SWEEP.md`**: iter200 best -> iter4000 near-noise, at
  effective batch 4. Consistent with the above.

## 3b. The upstream `ode_init.pt` is BROKEN — this may matter more than batch

**thu-ml (Causal Forcing, arXiv 2602.02214) identified a mathematical fallacy in
RollingForcing's released `ode_init.pt`.** Multiple independent reproducers on
RollingForcing issue #6 report the same symptoms at 3k steps: **slow motion, camera drift,
subject disappearance**; at 15k steps *"strange camera movements and repetitive content."*
Swapping in Causal Forcing's corrected **causal** ODE init fixed the drift.

`configs/rolling_forcing_dmd_distill.yaml` loads `checkpoints/ode_init.pt` — i.e. the
suspect artifact. **So even upstream never cleanly converged from its own released init.**
Fixing our batch may therefore not be sufficient on the 1.3B path.

(They also changed `denoising_step_list` to `[1000,750,500,250]` to match their
pretraining — worth noting if adopting their init.)

---

## 4. On quality gates — there is no published metric

- CausVid author Tianwei Yin, asked how to tell a good run from a bad one (CausVid issue
  #8): *"I had similar observation and I don't have a good way to tell too... I just looked
  at sampled videos for some intuition."*
- Self-Forcing issue #73: *"video quality seems to improve even when loss stagnates."*
- Published gates are VBench / VBench-Long and eyeballed samples. DMD2's practical
  heuristic is to watch general image statistics (they used pixel brightness).

**Our latent-sharpness metric (`SHARP_EVERY`, variance-normalized 3x3 Laplacian on x0) is
our own instrument.** It was validated against human ranking of three prompt_000 renders
(reference 321.9 > iter1000 40.0 > iter3400 28.4 pixel-space; 0.40434 / 0.07993 / 0.03015
at latent res) and two rival metrics were REJECTED for inverting that ranking (FFT
high-freq ratio; inter-frame abs diff, which tracks motion). It is better instrumentation
than the published work has — but it is not externally validated. **Ground truth remains a
render.**

---

## 5. DMD2 (arXiv 2405.14867) — what it does and does NOT say

**Does say** — the generator/critic balance:
- TTUR = **5** critic updates per generator update. Appendix C: ratio **1** *"suffers from
  training instability"*; ratio **10** gives *"excellent stability"* but *"significantly
  slows down the training process"*; **5** is *"the best balance between stability and
  convergence speed."*
- Failure signature without TTUR: *"the average brightness, along with other statistics, of
  generated samples fluctuates significantly."* FID 2.62 -> 3.48, restored to 2.61 by TTUR.
- Root cause: the critic *"does not track the fake score accurately, since it is
  dynamically optimized on the non-stationary output distribution of the generator"* ->
  *"approximation errors and biased generator gradients."*
- Its own batches are large: ImageNet **280** / 200K iters; SD1.5 **2048** / 40K;
  SDXL **128** / 20K.

**Does NOT say:** DMD2 makes **no claim that small batch size causes failure**. All its
stability analysis is about *critic update frequency*. The batch-64 requirement for video is
**empirical convention across shipped checkpoints, not a stated theorem.** Do not cite DMD2
as proof that batch=1 breaks.

---

## 6. Odyssey o2 and Lightricks LTX — NO published recipe (verified negative)

- **Odyssey o2**: HF org `odyssey-systems` contains exactly three public repos, all CUDA
  kernel libraries. `o2-14b` / `o2-1.3b` return **401/404** — private or nonexistent. No
  paper, no model card, no blog with hyperparameters. The "finalized FP8 model" claim is
  **not verifiable from any public source.** Treat any o2 number as folklore.
- **Lightricks LTX-2**: checkpoints definitively ship (`Lightricks/LTX-2.3`, 2.19M
  downloads; `-fp8`, `-nvfp4` repos exist). But arXiv **2601.03233** has **no distillation
  section at all**, and a grep of github.com/Lightricks/LTX-2 for
  `dmd|score_distill|teacher|adversarial` returns **zero hits** — the public repo has only
  LoRA fine-tune configs. The **only** published statement of method is the nvfp4 card's
  phrase *"trained by Quantization Aware Distillation for improved accuracy."*
  Inference-side only: distilled = **8 steps, CFG=1**.
- So: **quantization-aware distillation is real and shipped, but nobody publishes the
  recipe.** Our FP8 plan cannot copy one.

---

## 7. Memory cost of moving toward the reference (MEASURED, this repo)

- **`grad_accum` 4 -> 64: ~ZERO memory cost.** Gradients accumulate into ONE buffer
  (`loss.backward()` adds into `p.grad` in place); 64 accumulations use the same bytes as 4.
  Cost is wall-clock only: with `dfake_gen_update_ratio=5`, one optimizer step becomes
  **5 x 64 = 320 iterations**.
- **`num_training_frames` 6 -> 21: EXPENSIVE.** This repo already measured the rollout's
  last-block attention working cache (`3600 x (N x 1200) x 12 heads x 2B`):

  | N | cache | verdict (measured) |
  |---|---|---|
  | 21 | **2.18 GB** | **overflowed** |
  | 18 | 1.87 GB | *"still ~1.7GB over"* |
  | 15 | ~1.24 GB | *"clears it with headroom"* |
  | 6 (current) | ~0.5 GB | fits |

  So **21 frames does NOT fit** on world=16/student=4. Levers if you want it: raise
  `student_tp` (the 1.3B has room the 14B doesn't) or run the 1.3B on world 32/64.

**Cadence trap:** at `grad_accum=64` a single optimizer step costs 320 iters, so any
iter-denominated cadence must be rescaled or it fires before a weight update happens.
`save_every=200` would write 1.6 checkpoints per optimizer step; `SHARP_EVERY=20` would log
16 sharpness points per weight update, all measuring the SAME weights.

---

## 8. The 1.3B sharpness run — the most promising checkpoint we have

Pod `rf-distill-sharp-nxkqf`, branch `rf-sharpness-metric` @ `044a473`, run
`sharp_202607300331`, 1.3B student, prompt_000, T=5, **grad_accum=4** (effective batch 4),
world=16 teacher8/student4/critic4.

At it 2605/10000, 130 sharpness samples:

| window | raw mean | median |
|---|---|---|
| it 0-400 | 0.445 | 0.397 |
| it 800-1200 | 0.557 | 0.489 |
| it 1600-2000 | 0.728 | 0.668 |
| **it 2000-2400** | **2.496** | 2.244 |
| **it 2400-2800** | **3.391** | 3.311 |

- **EMA 0.593 -> 3.090 (5.2x).** Slope +1.02e-3/iter, monotone across every window, with a
  distinct **inflection near it2000** (~3.4x jump).
- `dmdnorm` **declining** 0.825 (it400-800 peak) -> 0.482. Correct direction.
- `loss_fake` **falling** 0.204 -> 0.122. Critic tracking, not diverging.
- `grad_norm` 0.08-0.75, calm.

**This is the opposite of the earlier diverging run** (where quality fell while dmdnorm sat
flat). **IMPORTANT INFERENCE:** this is improving 5x *at effective batch 4*, so
under-batching is **not preventing learning** on the 1.3B — it may cap the ceiling, but the
"batch 4 explains the blur" story is too simple. Do not change this config's `grad_accum`
until iter2600 EMA has been rendered; it is currently the best checkpoint in flight.

**NEXT ACTION: render `sharp_202607300331/model.iter2600.pt` with `--use_ema`** and compare
against iter1000. That render is what converts a 5x metric rise into a verdict.

**Also note the earlier EMA-seeding artifact:** the `stale=N` counter is seeded from the
FIRST sample, so `best` anchors to sample #1 and `stale` inflates even while the raw curve
climbs (it read `stale=39` at it800 while genuinely improving). Fix the seeding before
wiring `SHARP_PATIENCE` to auto-stop, or a good run gets killed.

---

## 9. Decision guide

**If the goal is a SHARP checkpoint:** the 1.3B is the only path with a published recipe
(every shipped checkpoint is a 1.3B student). Watch the run in §8 first — it may already be
converging. If it plateaus, the levers in order are (a) effective batch toward 64 via
`grad_accum` (free, memory-wise), (b) rollout length toward 21 (needs a topology change),
(c) Causal Forcing's corrected causal ODE init (§3b).

**If the goal is "KD training runs on Neuron":** already demonstrated. The mechanics
(three-group placement, FSDP2, functional attention, memory discipline) are the
contribution; do not overclaim quality.

**If the goal is a 14B student:** accept that no reference recipe exists. See
`~/odyssey-trainium` branch `exp/o2-distill-14b` for the o2-14b port, which reached a
running G-step loop at effective batch 64 via `grad_accum=64` on one trn3 instance.

---

## 10. Cross-refs
- `DISTILL_QUALITY_SWEEP.md` — the measured iter200-best/iter4000-noise collapse.
- `DISTILL_T4_RESUME.md`, `DISTILL_T4_RUNBOOK.md` — the T=4 thread and the fps cliff.
- `PAPER_parallelism_neuron_training.md` §8 — the two-forward same-noise correctness rule.
- Note: `DISTILL_DIVERGENCE_ROOTCAUSE.md` exists only on branch `rf-distill-14b`. Its
  grad_accum/EMA diagnosis is superseded on the batch point by §0-2 here.
