# Parallelism for Multi-Model Diffusion **Training** on AWS Trainium (Neuron) — Source Material

**Paper-source notes — training side.** Companion to `docs/PAPER_parallelism_neuron.md`
(inference side); to be converged into one paper. Target venue: *Advanced Computing*
(gold OA). Topic fit: Systems (distributed & parallel systems, HPC, performance analysis);
AI (foundation models, large-scale training); open-source reproducibility.

> Scope: **model TRAINING** (DMD distillation), not serving. Collects every parallelism /
> memory / distributed technique established while making a **3-model** DMD distillation
> loop (Rolling Forcing, Wan2.1-T2V) run on a 16-core Trn2 node. Where the inference paper
> studies the *throughput/efficiency envelope* of TP×CP for one model, this side studies
> *fitting and correctly training three co-resident models* under Neuron's memory and
> collective constraints.
>
> Every claim is grounded in a measured run on branch `rf-distill-t4` and/or the
> proven-stable StreamDiffusionV2 (`SD`) reference (`exp/fps-1.3b`). Commit hashes cited.

---

## 0. One-paragraph thesis

Training a multi-model diffusion pipeline (student + frozen teacher + critic) on a single
Trainium node is bounded not by compute nor by any one model's size, but by **(a) three-way
model co-residency** exceeding per-core HBM and **(b) a distillation rollout whose autograd
graph is retained across optimizer steps under FSDP1**. The fix is a **role-partitioned,
asymmetric model-parallel placement** (each model on its own rank subgroup, sized to its
parameter count) combined with **FSDP2 `reshard_after_forward`** to free the rollout graph,
and a **functional, run-constant-gated attention path** that survives activation-checkpoint
recompute. We also contribute a **measurement methodology** (per-rank, per-substep
device-tensor probes) that distinguishes compile-latency from deadlock and memory-peak from
graph-retention — the distinctions that make these failures diagnosable at all.

---

## 1. The workload (what makes it a hard parallelism problem)

**Distribution Matching Distillation (DMD)** of a video diffusion model on Trainium. Three
transformer models are co-trained adversarially in one loop:

| Role | Model | Params | Trainable | Attention |
|---|---|---|---|---|
| generator (student) | Wan2.1-T2V-1.3B causal | 1.3B | yes | causal, streaming KV-cache |
| real_score (teacher) | Wan2.1-T2V-**14B** | 14B | **frozen** | non-causal, full-clip |
| fake_score (critic) | Wan2.1-T2V-1.3B | 1.3B | yes | non-causal |

- Hardware: one **trn2 node = 2 chips × 8 NeuronCores = 16 cores** (LNC=1), claim `l-trn2`.
- Per-core HBM budget ≈ **24 GB**.
- Data-free: the student **rolls out its own samples** (a multi-block causal video rollout),
  the teacher and critic **score** them, and the DMD gradient `(fake_score − real_score)`
  trains the student. This rollout is the source of the hard memory problem.
- The naive footprint: 14B + 1.3B + 1.3B co-resident **on every core** + the rollout's
  autograd graph = far over 24 GB/core. Making this fit is the paper's core contribution.

**Why this is a parallelism paper and not just a memory paper:** the fix is *placement +
sharding strategy*, and the interesting failures are all distributed-execution failures
(collective deadlocks, FSDP reshard semantics, checkpoint recompute determinism under
sharding, Neuron-specific collective/compile constraints).

---

## 2. Neuron-specific constraints that shape the design

These are the hardware/runtime facts a GPU practitioner would not anticipate; they drive
every design choice below.

1. **Collectives yes, point-to-point no.** Neuron supports `dist.broadcast` / all-reduce /
   all-gather, but **not P2P send/recv**. → cross-model-group data transfer must be done
   with **global broadcasts** (every rank calls in lockstep; only the source group supplies
   real data, others supply zeros).
2. **Per-op collective lockstep is fatal if unmatched.** Any collective that a subset of
   ranks reaches while others don't → **deadlock, no error, no traceback** (silent hang).
   This is the dominant failure mode of group-parallel training and is invisible without
   per-rank instrumentation.
3. **Eager (`torch_neuronx`) defers frees.** `del`/`gc.collect()` do not immediately reclaim
   HBM; a `torch.neuron.synchronize()` is needed to flush pending frees before the next
   allocation. (But note §6: this did **not** free a retained autograd graph — a key result.)
4. **First-call compilation dominates wall-clock.** Every distinct NEFF compiles on first
   execution (minutes each). A DMD iteration has ≥4 compile points (student rollout, teacher
   score, critic score, grad recompute), each compiling once then cached. **A 14-min "hang"
   was actually first-call compile, not a deadlock** — only per-rank timestamped logging
   distinguished them.
5. **NEFF accumulation from dynamic shapes/values.** A *continuous* random timestep each
   iter makes `torch.compile` trace a **new graph → a new `module.neff` that loads and never
   unloads → scratchpad creeps → OOM ~iter 12.** Fix: sample from a **fixed discrete bucket
   set** so the loaded NEFF set stays bounded. (SD `86cb4d3`; RF `5590d72`.)
6. **No native complex dtype.** `torch.view_as_complex` / complex `aten::cat` (used by
   upstream RoPE) fail to compile. → real-valued cos/sin RoPE required (RF `8d417c3`,
   numerically identical, max|Δ|=4.8e-7 vs complex).
7. **No CUDA flash-attn; stock SDPA has a large-sequence compile ceiling.** A full-clip
   `bf16[1, 18000, 40, 128]` attention fails to compile under stock
   `F.scaled_dot_product_attention`; a **tiled NKI flash-attention kernel** compiles it
   (RF `1cf63bd`, ported from a working 14B-on-Neuron reference). Same math, tiled.
8. **fp32 master vs fp32 compute must be separated.** FSDP `MixedPrecision(param_dtype=fp32)`
   forces fp32 *compute* → `aten.mm` dtype mismatch vs bf16 activations. Use
   `param_dtype=bfloat16` (bf16 compute) while FSDP keeps the fp32-loaded flat param as the
   optimizer master (RF `ed1e28f`). One knob conflated two concepts.

---

## 3. The parallelism architecture (the core contribution)

### 3.1 Why tensor-parallel alone fails
TP shards **weights** (matmul width), not **activations** or the **autograd graph**. Each
core still materializes the full activation tensors for its shard. → for a 30-block video
rollout the retained graph is ~20–22 GB/core regardless of TP degree. **Models are not
inherently core-memory-bound; TP-only is.** Real large-model training needs TP + FSDP/ZeRO
+ activation checkpointing (+ sometimes sequence/pipeline parallel).

### 3.2 Three-group (model-parallel) placement
Instead of sharding all three models across all 16 cores (co-residency ≈ 11–20 GB/core of
weights before any activation), give **each model its own rank group** so a core holds
**one** model:

```
teacher (14B, frozen)  -> ranks 0..7    (8 ranks)
student (1.3B, train)  -> ranks 8..11   (4 ranks)
critic  (1.3B, train)  -> ranks 12..15  (4 ranks)
```

- **Asymmetric on purpose** (RF `0bde225`): the 14B teacher is ~10× the 1.3B nets, so it
  gets 8 ranks; student/critic get 4 each. Uses all 16 cores, no idle ranks.
  `make_distill_groups(teacher_tp=8, student_tp=4, fake_tp=4)`.
- **Cross-group transfer = global broadcast** (Neuron has no P2P). One DMD iteration:
  1. student rolls out `x0` (its group), broadcasts `x_t`, `t`, `x0` to all;
  2. teacher scores `x_t` (its group), broadcasts `real_pred` back;
  3. critic scores `x_t` (its group), broadcasts `fake_pred` back;
  4. student computes DMD gradient and updates.
  Every rank calls every broadcast; idle ranks send zeros.
- **Result:** per-core resident weights dropped from ~11.2 GB (co-resident) to one model
  (student core `Tensors` 11.2 → 6.6 GB). This is the "3 nets co-located → OOM" wall SD
  documented; SD noted even a 2-group split still OOM'd, so **3 groups is required**.

### 3.3 The teacher-in-fp16 + more-ranks lever
Frozen models need **no fp32 master** (no optimizer). Load the 14B teacher **directly in
bf16** (`torch_dtype=bfloat16`), sharded ÷8: 14B×2B/8 ≈ **3.5 GB/core**, vs 14B×4B/4 =
14 GB/core if loaded fp32 on 4 ranks (RF `62f525f` + `0bde225`).
- **Caution learned:** `low_cpu_mem_usage=True` (accelerate meta-device init) **deadlocked**
  8 teacher ranks for 25 min on the Neuron eager backend (RF `71a5181` reverted it). With
  ample host RAM it's unnecessary; plain `from_pretrained(torch_dtype=bf16)` loads in ~30 s.

### 3.4 Per-block sharding granularity (FSDP wrap policy)
FSDP `size`-based auto-wrap with `min_num_params=5e7` wrapped the whole 1.3B model as **one
shard unit** → a single 5.287 GB all-gather to run the forward → OOM. **Per-transformer-block
wrap** (`transformer_auto_wrap_policy` on the block class) all-gathers **one block at a time**
→ peak = one block, not the whole model (RF `3d58c28`). This is the single most important
FSDP knob for large-model training and mirrors SD's per-block `fully_shard`.

---

## 4. FSDP1 vs FSDP2 — the decisive result for training

**This is the headline parallelism finding.** The DMD generator-update backward retained a
full rollout autograd graph **per optimizer step**, accumulating ~10 GB/G-step until OOM,
and *no Python-level intervention freed it* (see §6). Root cause and fix:

- **FSDP1** (`FullyShardedDataParallel`): under the manual DMD update pattern (recompute-
  with-grad rollout → `loss.backward()`), FSDP1's backward **unshard buffers were not
  resharded/freed** — the memprobe showed the graph persisting across steps.
- **FSDP2** (`torch.distributed._composable.fsdp.fully_shard(reshard_after_forward=True)`):
  reshards params after **both forward and backward**, so the graph frees each step.
  **Measured:** post-G-step memory went from monotonically climbing (27→32→36→40 GB → OOM)
  to **flat** (26,996 MB identical at iter 10 and iter 15). (RF `022054e`.)

**FSDP2 integration specifics for a subgroup (paper-worthy detail):**
- FSDP2 wants a `DeviceMesh`, RF uses explicit process groups. **`init_device_mesh` over the
  world is collective** — calling it only on student ranks deadlocks. Instead build
  `DeviceMesh("neuron", torch.tensor(student_ranks))` from the **explicit student rank list**
  (reuses the existing subgroup, no world collective).
- Only the **student** needs FSDP2 (it has the multi-step backward). Frozen teacher (no
  backward) and critic (single forward+backward) stay FSDP1 — never OOM'd. **Mixed FSDP1 +
  FSDP2 in one job, by role.**
- Downstream API differences: FSDP2 modules have **no `.clip_grad_norm_`** (use
  `torch.nn.utils.clip_grad_norm_` on params); state-dict via
  `get_model_state_dict(full_state_dict=True, cpu_offload=True)` with a DTensor
  `.full_tensor()` fallback, not FSDP1 `FSDP.state_dict_type`.

---

## 5. Activation checkpointing under sharding — determinism traps

FSDP2 applies **per-block `NO_REENTRANT` activation checkpointing**
(`apply_activation_checkpointing` + `checkpoint_wrapper`), which recomputes the block in
backward. This exposes **three determinism requirements** that stock diffusion code violates,
each a distinct failure:

1. **No in-place mutation of views.** The streaming KV-cache does
   `kv_cache["k"][:, a:b] = roped_key` (in-place slice write). Under recompute this raises
   `RuntimeError: Output 0 of SliceBackward0 is a view and is being modified inplace`. →
   **functional (out-of-place) attention path for training**: the exit block attends only its
   own `roped_key/v` via out-of-place SDPA, no cache assembly, no eviction (RF, SD `42cd403`).
2. **Forward and recompute must take the identical code path.** Gating the functional path on
   `self.training` breaks: **NO_REENTRANT recompute / FSDP can flip `module.training`** between
   the original forward and the recompute → forward takes the functional branch, recompute
   takes the cache branch → *different number of saved tensors* →
   `CheckpointError: 72 vs 62 tensors`. → **gate on a constant env flag**
   (`DISTILL_FUNCTIONAL_ATTN`), never on mutable module state (RF `8603b83`, `d7f9b30`; SD
   `1d5f752`).
3. **Stateful caches anywhere in the block must be bypassed in training.** The *cross*-
   attention `crossattn_cache["is_init"]` flag is the same trap as (1)/(2): forward sees
   `is_init=False` (compute+store), recompute sees `is_init=True` (read) → tensor-count
   mismatch. → functional cross-attn too (RF `d7f9b30`).
4. **Do not double-checkpoint.** If FSDP applies block checkpointing, the model's own
   internal `torch.utils.checkpoint` must be off, or the recompute tensor counts diverge
   (SD `42b59b0`). And **disabling checkpointing entirely is worse**: the with-grad rollout
   then stores all activations up front (17,754 → 44,142 MB, OOM before backward) — RF
   `3a378f9` (tested and reverted). Checkpointing is load-bearing.

**General principle for the paper:** *activation-checkpoint recompute demands a pure,
stateful-side-effect-free forward.* Streaming-inference code (in-place KV caches, `is_init`
flags, `self.training` branches) is fundamentally incompatible; a training-mode functional
path, gated on a run-constant, is required.

---

## 6. Measurement methodology (a transferable contribution)

The debugging discipline itself is paper-worthy: **guessing failed repeatedly; measurement
resolved it.** Techniques:

- **Per-rank timestamped logging** to distinguish *slow first-call compile* from *deadlock*
  (both present as "no output"). E.g. proved a 94-min "hang" was ~14-min compiles, not a hang.
- **Per-iteration device-tensor probe** (`gc.get_objects()` → count + MB of `device=='neuron'`
  tensors) to separate a **memory-peak** (fits after one step) from a **creep/leak** (climbs
  every step). This is what proved the NEFF-accumulation creep and, separately, the FSDP1
  graph-retention leak.
- **Per-sub-step probe inside the optimizer step** (before/after rollout, before/after
  backward, after opt.step, after del) to localize the leak to `backward` + FSDP unshard.
- **Top-retained-tensor-shape dump** (`Counter` over shapes) to *name* the leak: the retained
  set was `(1, 3600, 1536)` block activations going **0 → 300 → 600 (+300 per G-step)** — i.e.
  a full 30-block rollout graph retained per step. This turned "some leak" into "the autograd
  graph is retained below Python," which ruled out all Python-level frees and pointed at FSDP2.
- **Key negative result:** `del` + `gc.collect()` + `torch.neuron.synchronize()` +
  nulling the pipeline caches freed **99 MB of a 10 GB leak** → *proof* the retention was in
  FSDP1's C++/backward machinery, not a Python reference. This is what justified the FSDP2
  rewrite over more frees.

---

## 7. Additional memory levers (secondary, but real)

- **Latent-space only:** DMD scores latents; the VAE is not needed on-device. Removing it +
  pinning frame count (so the VAE re-encode branch is dead code) freed its HBM (RF `d53f7c3`).
- **Rollout length = activation size.** `num_training_frames` sets `max_attention_size =
  frames × frame_seq`, which sizes the rollout attention working cache. Cutting 27→21→15→6
  frames is the direct activation lever (SD `02fa0bf` "fewer frames = 55% less activation").
  Note this is a *recipe* tradeoff (train-vs-deploy sequence length), not free.
- **No EMA on device.** EMA via `FSDP.summon_full_params` all-gathers the full fp32 params
  onto every rank (5.287 GB spike). SD trains without EMA; the invariant is **the full
  unsharded fp32 model must never co-reside on-device** — any full-param gather (checkpoint
  save) must use `cpu_offload=True` (RF `8a62157`).
- **`grad_accum` freeing discipline:** with FSDP1, deferring `zero_grad` to the next G-step
  retained grads across the intervening critic-only steps; `zero_grad(set_to_none=True)`
  immediately after `opt.step()` releases them (RF `464b2a3`). (Made moot but correct under
  FSDP2.)

---

## 8. Correctness under all this parallelism (must not be lost)

Memory fixes must not silently break learning — a real risk when restructuring the graph:

- **Two-forward same-noise:** the scoring rollout (no_grad) and the grad recompute must use
  the **same fixed noise**, or the DMD gradient (computed on the scored `x0`) is applied to a
  *different* `x0` — silently wrong training. Reuse `_rollout_noise` for both (RF `0a1ac10`;
  SD `5d90c6b`).
- **The DMD gradient is injected via** `loss = 0.5·MSE(x0, (x0 − grad).detach())`, whose
  `backward()` delivers exactly `grad = (fake − real)/normalizer` into the student weights.
  The loss *value* is meaningless by construction; the **gradient** is the signal.
- **Convergence metric = `dmdnorm`** = `mean(|(fake − real)/normalizer|)`, a 50-step rolling
  mean; as the student matches the teacher, `(fake − real) → 0` so it should trend to 0.
- **Open guardrail (recommended for the paper's rigor):** log `grad_norm` every generator
  step and assert a tracked param actually changes after N steps — the SD lesson that a
  severed graph (grad_norm ≈ 0) trains loss-looks-fine but never converges. `dmdnorm` alone
  cannot distinguish "converging slowly" from "frozen."

---

## 9. Result / status (as of writing)

After the full stack (three-group placement + asymmetric ranks + bf16 frozen teacher +
per-block FSDP2 `reshard_after_forward` on the student + functional env-gated attention +
NKI flash attention + real RoPE + discrete-timestep NEFF bounding), the 3-model DMD loop
**trains stably**: post-G-step device memory is **flat across steps** (no leak), and the loop
runs past the historical OOM walls (iter 12 creep, iter 15 graph-retention) into steady state
(observed to iter ~110+, checkpointing at `save_every`). Final topology: 16 cores, teacher
8 / student 4 / critic 4, student FSDP2 + rest FSDP1.

---

## 10. Reproducibility index (commits — branch `rf-distill-t4`)

| Theme | Commit | What it establishes |
|---|---|---|
| Three-group placement | `74d463c` | model-parallel groups + broadcast dance |
| Asymmetric ranks | `0bde225` | teacher 8 / student 4 / critic 4 |
| bf16 frozen teacher | `62f525f`, `ec2245c` | frozen → no fp32 master → ÷8 = 3.5 GB/core |
| Per-block FSDP wrap | `3d58c28` | one-block all-gather vs whole-model |
| **FSDP2 fully_shard (student)** | `022054e` | **reshard_after_forward frees the backward graph** |
| Functional self-attn | (7d9cfff) | no in-place KV write under recompute |
| Env-gated functional attn | `8603b83`, `d7f9b30` | fwd/recompute path determinism |
| Real cos/sin RoPE | `8d417c3` | no complex dtype on Neuron |
| bf16 compute params | `ed1e28f` | fp32 master ≠ fp32 compute |
| NKI flash attention | `1cf63bd` | full-clip attn compiles (tiled) |
| Discrete-timestep NEFF bound | `5590d72` | stops NEFF-accumulation creep |
| EMA off (no on-device full gather) | `8a62157` | full fp32 model never co-resident |
| Two-forward same-noise | `0a1ac10` | correctness of the graph-free DMD |
| Measurement probes | `f91e5a3`, `1f6a4a2` | creep-vs-peak, shape-named leak |

Reference implementation that pre-solved much of this: **StreamDiffusionV2** branch
`exp/fps-1.3b` (`distill/distill_sdv2.py`), commits `5d90c6b` (structural two-forward),
`86cb4d3` (discrete timesteps), `42cd403`/`1d5f752` (functional attn), FSDP2 build
(`f0708f9`/`b669b07`). And the 14B-on-Neuron attention reference `~/wan2-i2v-14b`.

---

## 11. Related work & what differs from GPU FSDP training

Positioning for reviewers — this is deliberately *not* "we ran FSDP on a new accelerator."

**Standard large-model training stack (GPU).** ZeRO/FSDP shards params+grads+optimizer state;
activation checkpointing trades compute for memory; TP shards matmul width; PP pipelines
layers. Reference multi-model RLHF/distillation stacks (e.g. actor+critic+reward) typically
**offload or time-slice** the frozen/secondary models, or place them on separate hosts. Our
setting is one node, three models, all resident, adversarial per-iteration coupling.

**What is different on Trainium / with this workload:**

1. **No P2P → model-parallel groups communicate by broadcast, not send/recv.** GPU multi-model
   training routes actor↔critic tensors with P2P or NCCL P2P; on Neuron every cross-group
   transfer is a **world broadcast in lockstep**, which reshapes both the algorithm (zeros on
   non-source ranks) and the failure surface (any unmatched collective = silent deadlock).
2. **Asymmetric role-partitioned placement is first-class, not incidental.** Because the three
   models differ ~10× in size, the *right* parallel layout is **unequal rank counts per model**
   (14B→8 cores, 1.3B→4 each) on one node — a scheduling problem GPU stacks usually sidestep by
   using more hosts. We show the asymmetric single-node layout and why symmetric groups waste
   cores / OOM the big model.
3. **FSDP1 vs FSDP2 is decision-critical here, not a version bump.** On GPU the two are largely
   interchangeable for throughput. Here, FSDP1 **retained the DMD rollout's backward graph
   across optimizer steps** (measured +10 GB/step → OOM) while FSDP2 `reshard_after_forward`
   freed it (flat). The manual recompute-with-grad DMD update is exactly the pattern that
   exposes FSDP1's reshard gap.
4. **Activation-checkpoint recompute vs streaming-inference state.** GPU diffusion training
   rarely reuses the *streaming-inference* forward (in-place KV cache, `is_init`, causal
   windows). Reusing it under NO_REENTRANT recompute is autograd-fatal (in-place-on-view) and
   non-deterministic (`self.training` flips, cache `is_init` flips → tensor-count mismatch).
   We formalize the requirement: **recompute demands a pure forward gated on a run-constant.**
5. **Compiler-driven memory creep from dynamic values.** GPU eager execution has no analogue of
   a *new compiled NEFF per distinct timestep value* accumulating in a scratchpad. The
   discrete-bucket fix is Neuron/graph-compiler-specific.
6. **Kernel availability changes the parallel plan.** Absence of CUDA flash-attn means a
   full-clip attention that "just works" on GPU must be routed through a tiled **NKI** kernel to
   compile at all — a constraint that couples the *kernel* layer to the *parallelism* layer.

**Closest prior art to cite:** ZeRO/FSDP (Rajbhandari et al.; PyTorch FSDP), activation
checkpointing (Chen et al.), DMD/DMD2 distillation (Yin et al.), Self-Forcing / Rolling
Forcing causal video diffusion, and AWS Neuron SDK docs (NKI, `torch_neuronx`, collective
runtime). The gap we fill: **single-node, multi-model, adversarial-coupled diffusion training
on a graph-compiled non-CUDA accelerator**, where placement + FSDP-generation + recompute
purity + kernel routing must be co-designed.

---

## 12. Figure & table list (for the merged paper)

Artifacts to produce; data already exists in the cited run logs (`/tmp/rf-distill-*.log`)
and memory notes.

**Figures**
- **F1 — Placement diagram.** 16 cores → teacher[0-7] / student[8-11] / critic[12-15];
  arrows = the 4 broadcasts of one DMD iteration (x_t/t/x0 out; real_pred, fake_pred back).
  Contrast panel: naive "all 3 models on all 16 cores" (co-resident, OOM).
- **F2 — Device memory vs iteration, FSDP1 vs FSDP2.** The headline curve. FSDP1: post-G-step
  MB climbs 27→32→36→40 → OOM at iter 15. FSDP2: flat at ~27 GB across iters 10 and 15+.
  (Data: per-iter memprobe from runs `qlrk2`/`zh67d` (FSDP1) vs `hkbcr` (FSDP2).)
- **F3 — Per-substep memory within one G-step.** Bars at e0 before-rollout / e1 after-rollout
  / e3 after-backward / e4 after-opt.step / e6 after-cleanup, for FSDP1 (e6 does not drop) vs
  FSDP2 (e6 returns to baseline). Localizes the leak to backward-unshard.
- **F4 — "Named leak" bar.** Retained tensor count by shape across it9/it11/it16:
  `(1,3600,1536)` 0→300→600, `(1,7200,12,128)` 60→120→180. Shows a full rollout graph
  retained per step.
- **F5 — Per-core resident weights by placement.** co-resident 11.2 GB → three-group student
  6.6 GB → +bf16 teacher ÷8 = 3.5 GB/core. Stacked bar.
- **F6 — Convergence.** `dmdnorm_avg50` (and `loss_fake`) vs generator-step, once a long run
  exists. (Pending — see §8 guardrail before claiming convergence.)

**Tables**
- **T1 — Neuron constraints → design response** (from §2): 8 rows.
- **T2 — Failure → root cause → fix → commit** (from §10), the full sequence, as the
  "engineering narrative" table.
- **T3 — Ablation: which single change is load-bearing.** grad-ckpt on/off (44 GB vs fits),
  FSDP1/FSDP2 (leak/flat), per-block vs whole-model wrap (5.287 GB spike), functional attn
  on/off (CheckpointError). Each row = remove-one-and-it-breaks.

**Reproducibility appendix:** the config (`configs/rolling_forcing_dmd_t4.yaml`), the job
(`rf-distill-job.yaml`: `DISTILL_THREE_GROUP=1`, `DISTILL_FUNCTIONAL_ATTN=1`, `NPROC=16`,
`l-trn2`), and the commit index (§10).
