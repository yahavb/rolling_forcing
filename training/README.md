# RollingForcing DMD Distillation — Training on AWS Trn2 / Neuron

This is a **self-contained vendored copy** of the upstream RollingForcing DMD
score-distillation training stack (`/Users/yahavb/RollingForcing`), adapted to
run on AWS Trn2 / Neuron. It produces a DiT checkpoint compatible with this
repo's inference (`e2e_pipeline.py`, `generate_latents.py`, `serve.py`).

It is **completely independent** of this repo's top-level `utils/` and `models/`
(which are the INFERENCE stack) — nothing here imports them and nothing here
overwrites them.

## Layout

```
training/
  train.py                 # entry point (upstream, unchanged logic)
  configs/
    default_config.yaml    # base config (upstream) — loaded by RELATIVE path
    rolling_forcing_dmd.yaml
  trainer/                 # distillation.py is the DMD trainer used here
  model/                   # DMD / CausVid / SiD / GAN / base
  pipeline/                # rolling_forcing_training.py (+ inference pipelines)
  utils/                   # dataset, distributed, loss, misc, scheduler, wan_wrapper, lmdb
  wan/                     # vendored Wan2.1 package (modules/, configs/, distributed/, utils/)
```

## Import strategy (fewest edits)

**I kept all intra-package imports UNCHANGED** (`from utils.wan_wrapper import ...`,
`from model import DMD`, `from pipeline import ...`, `from wan.modules... import`).

This works because the k8s manifest launches the job as:

```
cd training && torchrun ... train.py -- --config_path ../configs/rolling_forcing_dmd_t4.yaml --logdir $RUN --disable-wandb
```

With `cwd = training/`, Python puts `training/` on `sys.path`, so the top-level
package names `utils`, `model`, `pipeline`, `wan`, `trainer` resolve to the
vendored copies here — **not** to this repo's inference `utils/`/`models/`.
(Note this repo's inference package is `models/`, plural; the training package is
`model/`, singular — they never collide.)

**Consequence:** you MUST launch with `cd training` (as the manifest does).
Running `torchrun training/train.py` from the repo root would put the repo root
on `sys.path` and shadow `utils` with the inference `utils/`. Zero import edits
were needed by honoring the `cd training` launch contract.

`train.py` also does `OmegaConf.load("configs/default_config.yaml")` with a
**relative** path, which is why `training/configs/default_config.yaml` MUST exist
(it does — copied from upstream). The T=4 override config is passed explicitly.

## Launch

```bash
cd training
torchrun --nnodes=1 --nproc_per_node=<N_NEURONCORES> train.py \
    --config_path ../configs/rolling_forcing_dmd_t4.yaml \
    --logdir "$RUN" \
    --disable-wandb
```

- `--logdir $RUN` is where checkpoints are written:
  `$RUN/checkpoint_model_XXXXXX/model.pt`. This matches the manifest's `$RUN`.
- If env var `CKPT_MIRROR_DIR` is set, each saved checkpoint dir is also copied
  there (rank 0 only) so checkpoints land live on the S3-backed PVC and the
  local logdir (often `/tmp`) stays bounded.

Required Neuron env: `NEURON_FALLBACK_ENABLED=0`. **Do NOT** set
`NEURON_LOGICAL_NC_CONFIG` (forbidden in this project).

## Precompute T5 embeddings (frees ~9.6GB HBM)

`precompute_embeds.py` removes the UMT5 text encoder from the training graph.
Run it ONCE on CPU (single process), then point training at the `.pt`:

```bash
cd training
# 1) precompute (CPU, 1 process) — reads data_path + negative_prompt from the config
python3 precompute_embeds.py \
    --config_path ../configs/rolling_forcing_dmd_t4.yaml \
    --out /tmp/embeds.pt

# 2) train with T5 removed from device
PRECOMPUTED_EMBEDS=/tmp/embeds.pt torchrun ... train.py \
    --config_path ../configs/rolling_forcing_dmd_t4.yaml --logdir "$RUN" --disable-wandb
```

- **Flag names (match the manifest):** `precompute_embeds.py` takes
  `--config_path <yaml>` and `--out <path.pt>`. It reads `data_path` and
  `negative_prompt` from the config (OmegaConf merge over
  `configs/default_config.yaml`, same as `train.py`) — no `--captions` arg. Run
  from `cwd=training/` so relative config paths resolve.
- **Env var (matches the manifest):** the trainer reads `PRECOMPUTED_EMBEDS`
  (path to the `.pt`). Set → skip T5 entirely and look up embeds by prompt
  string. Unset → upstream in-loop T5 (no behavior change).
- **Output:** `torch.save({"prompt_embeds": {prompt_str: [1,512,4096]},
  "negative_prompt_embeds": [1,512,4096]})`.
- **Fail-loud:** a training prompt absent from `prompt_embeds` raises an
  `AssertionError` in `fwdbwd_one_step` (never silently skipped).

Trainer changes (`trainer/distillation.py`, all gated on `PRECOMPUTED_EMBEDS`):
1. In `__init__`, after building `self.model`: load the `.pt`, stash
   `precomputed_prompt_embeds` / `precomputed_negative_embeds`, and set
   `self.model.text_encoder = None` (frees the CPU encoder before any move).
2. The `text_encoder` `fsdp_wrap(...)` is skipped when using precomputed embeds
   — this is where device HBM is freed.
3. In `fwdbwd_one_step` Step 2, the precomputed branch stacks per-prompt
   `[1,512,4096]` into `[B,512,4096]` and repeats the negative embed to `[B,...]`;
   the else-branch is the verbatim upstream T5 path.

## Neuron adaptations (exhaustive, with file:line references)

All changes are surgical and commented in-code with a `# Neuron:` prefix.

### A. Distributed backend / device — `utils/distributed.py`
- `launch_distributed_job(backend="nccl")` → `backend="neuron"` (matches this
  repo's inference `dist.init_process_group(backend="neuron")`). Removed
  `torch.cuda.set_device(local_rank)` (not applicable on Neuron; the runtime
  binds one NeuronCore per rank via `LOCAL_RANK`).
- `fsdp_wrap(...)` `device_id=torch.cuda.current_device()` →
  `device_id=int(os.environ.get("LOCAL_RANK", 0))`.

### A'. Device in the trainer — `trainer/distillation.py`
- `self.device = torch.cuda.current_device()` →
  `torch.device("neuron")` when CUDA is absent.
- TF32 matmul flags guarded behind `torch.cuda.is_available()`.
- All `torch.cuda.empty_cache()` → `_maybe_empty_cache()` (guarded no-op on Neuron).
- `generate_video()` hard-coded `device="cuda"` → `device=self.device`.

### A''. Device — `pipeline/rolling_forcing_training.py` (~line 420-433)
- Four `.cuda()` moves in the `denoised_timestep_*` argmin lookups →
  `.to(self.scheduler.timesteps.device)` (device-agnostic; same numerics).

### A'''. Device — `utils/wan_wrapper.py:33` and `wan/modules/t5.py:~478`
- `WanTextEncoder.device` property `return torch.cuda.current_device()` →
  `return next(self.text_encoder.parameters()).device`.
- `T5EncoderModel.__init__(device=torch.cuda.current_device())` (an
  import-time-evaluated default that crashes on a CUDA-less host) →
  `device=None`, resolved lazily. (Training path uses `umt5_xxl` directly and
  never constructs `T5EncoderModel`, but the import must not crash.)

### B. Attention backend (flash-attn) — `wan/modules/attention.py`
No edit needed. Upstream already guards `import flash_attn*` with
`ModuleNotFoundError` and falls through to a `scaled_dot_product_attention`
path in `attention()` when neither FA2/FA3 is available. On Neuron flash-attn is
absent, so the SDPA fallback runs and `flash_attention()` (with its
`assert q.device.type == 'cuda'`) is never called.

### C. flex_attention → SDPA (THE delicate change) — `wan/modules/causal_model.py`
torch `flex_attention` has no Neuron backend. Two helpers were added at module
top:
- `_dense_mask_from_fn(mask_fn, q_len, kv_len, device)` — evaluates the SAME
  `attention_mask(b, h, q_idx, kv_idx)` closure the upstream code passed to
  `create_block_mask`, on broadcast index grids, producing a dense boolean
  `[Lq, Lk]` mask (True = attend). Every upstream predicate OR-s in
  `q_idx == kv_idx`, so no row is ever all-False → SDPA softmax never NaNs
  (including in the 128-padding region).
- `_neuron_flex_attention(query, key, value, block_mask)` — calls
  `F.scaled_dot_product_attention(q, k, v, attn_mask=block_mask, is_causal=False)`.
  `is_causal=False` because the dense mask already encodes causality. Flex's
  default softmax scale is `1/sqrt(head_dim)`, identical to SDPA's default, so
  numerics match.

Call sites changed:
- The two `flex_attention(...)` calls in `CausalWanSelfAttention.forward`
  (teacher-forcing and non-TF branches) → `_neuron_flex_attention(...)`. The
  `[:, :, :-padded_length]` un-pad slice and the 128-padding are **preserved**.
- The three `create_block_mask(attention_mask, ...)` calls in
  `_prepare_blockwise_causal_attn_mask`, `_prepare_teacher_forcing_mask`, and
  `_prepare_blockwise_causal_attn_mask_i2v` → `_dense_mask_from_fn(...)`.

### D. Other CUDA deps
torchao / nvidia-tensorrt / pycuda / onnx are **not imported** by the training
path (they were only in the upstream inference/export tooling) — nothing to
guard. `utils/misc.py`'s `torch.cuda.manual_seed_all(seed)` safely no-ops
without CUDA and was left unchanged (minimal footprint).

## fp32 master weights (THE critical distillation fix) — `utils/distributed.py` + `trainer/distillation.py`

**Where:** `fsdp_wrap()` gained an `fp32_master: bool` parameter. When
`mixed_precision=True` **and** `fp32_master=True`, the `MixedPrecision` policy is
built with `param_dtype=torch.float32` (instead of `torch.bfloat16`),
keeping the sharded master params in fp32. `reduce_dtype`/`buffer_dtype` stay
fp32. bf16 compute is expected from autocast if enabled.

In `trainer/distillation.py`, the two TRAINABLE modules are wrapped with
`fp32_master=True`:
- `self.model.generator` (the student being distilled)
- `self.model.fake_score` (the critic)

The two FROZEN modules keep bf16 params (`fp32_master=False`, the default) to
save HBM:
- `self.model.real_score` (the frozen multi-step bidirectional Wan2.1-T2V-14B teacher)
- `self.model.text_encoder`

**Why:** with `lr=1.5e-6`, `Δp/p ~ 1.5e-4`. bf16 resolution near 1.0 is ~3.9e-3,
so a bf16-only master rounds every update to zero and the model silently never
trains.

**Verification assertion:** `trainer/distillation.py` snapshots one trainable
generator parameter at init and, after `config.weight_change_check_step`
(default 50) optimizer steps, asserts `max|Δ| > 0`. If the param is
bit-identical it raises with a message pointing at this exact bf16-master bug.
Set `weight_change_check_step: 0` in config to disable.

## Teacher checkpoint

The teacher (`real_score`) is the BASE multi-step bidirectional
**Wan2.1-T2V-14B** (`real_name: Wan2.1-T2V-14B`, `is_causal=False`) — the
upstream config already sets `real_name: Wan2.1-T2V-14B`, and this was kept
unchanged. Do NOT point the teacher at a distilled few-step checkpoint.

## Version tension (diffusers / transformers)

This repo's INFERENCE pins `diffusers==0.37.1` / `transformers==5.8.1`. Upstream
TRAINING used `diffusers==0.31.0` / `transformers>=4.49.0`. **The training job
should install the upstream training pins** (the k8s manifest installs them into
the training container). Do NOT change this repo's inference pins. This
divergence is expected and intentional; the two stacks run in different
containers.

## RISKS / UNVERIFIED ON DEVICE

None of this has been executed on Trn2 (parse-checked only — no torch on the dev
host). The following need validation on real hardware:

1. **FSDP on Neuron.** `torch.distributed.fsdp.FullyShardedDataParallel` with
   `backend="neuron"`, `ShardingStrategy.HYBRID_SHARD`, `use_orig_params=True`,
   and `device_id=LOCAL_RANK` is unverified on Neuron. Sharded all-gather /
   reduce-scatter collectives, `summon_full_params` (used by `EMA_FSDP` and
   `fsdp_state_dict`), and `clip_grad_norm_` on an FSDP module may behave
   differently or be unsupported. This is the single biggest unknown.

2. **fp32-master + autocast interplay.** `param_dtype=torch.float32` in the
   MixedPrecision policy keeps params fp32, but there is no explicit
   `torch.autocast(device_type="...")` context in the trainer — bf16 compute
   was expected from GPU autocast upstream. On Neuron, confirm the matmuls
   actually run in the intended precision and that the weight-change assertion
   passes (it will catch the silent no-train failure if not).

3. **SDPA dense-mask equivalence.** `_dense_mask_from_fn` must reproduce the
   flex `create_block_mask` semantics exactly. Verify: (a) the dense mask
   matches a reference flex `create_mask` on a small case, (b) SDPA on Neuron
   accepts a boolean `attn_mask` broadcasting `[Lq,Lk]` over `[B,H,Lq,Lk]`,
   (c) no NaNs from the padded rows, and (d) the softmax scale (`1/sqrt(d)`)
   matches. The dense `[Lq,Lk]` mask is O(L^2) memory — for `21*1560≈32760`
   tokens (×128-padding) this is ~1 GiB as bool; if that is too large, tile the
   mask or push it into an NKI attention kernel.

4. **HBM footprint.** Frozen 14B teacher (`real_score`) + trainable 1.3B student
   (`generator`) + trainable 1.3B critic (`fake_score`) + text encoder + fp32
   master copies of the two trainable nets + AdamW moments (fp32) must fit
   across the NeuronCores after FSDP sharding. The fp32-master requirement
   roughly doubles the trainable-param memory vs a bf16-only run. Validate the
   sharding degree / `sharding_strategy` (`hybrid_full`) actually fits, and
   consider `text_encoder_cpu_offload` / gradient checkpointing (already on via
   `gradient_checkpointing: true`).

5. **`../configs/rolling_forcing_dmd_t4.yaml` is MISSING** from this repo's
   `configs/` at the time of this port. The manifest passes it by relative path.
   It must be created (in this repo's inference `configs/` dir — outside this
   `training/` package) before the job will run. It should mirror
   `configs/rolling_forcing_dmd.yaml` with T=4 frame settings and MUST keep
   `denoising_step_list` identical to the inference config used at serve time.

6. **Dataset path / `TextDataset`.** `config.data_path`
   (`prompts/vidprom_filtered_extended.txt` in the upstream config) must be
   present in the training container, and `num_workers=8` DataLoader behavior on
   the Neuron host is unverified.
