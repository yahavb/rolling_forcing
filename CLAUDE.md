# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Rolling Forcing text-to-video inference on AWS Trn2 (Neuron). Generates video from text prompts using a three-stage pipeline (T5 → DiT → VAE) running distributed across 8 NeuronCores at LNC=1 on a single Trn2 chip.

Based on Wan2.1-T2V-1.3B with DMD-distilled DiT weights for rolling-forcing denoising.

## Running

All scripts use `torchrun` with 8 ranks. Must run on a Trn2 host with Neuron SDK installed.

```bash
# Full end-to-end pipeline (preferred)
bash scripts/run_e2e_pipeline_distributed.sh

# Individual stages for debugging/profiling
bash scripts/run_encode_prompt_distributed.sh    # T5 → text_embeds/
bash scripts/run_generate_latents_distributed.sh # DiT → output_latent.pt
bash scripts/run_decode_latents_distributed.sh   # VAE → output.mp4
```

Environment variables for profiling: `PROFILE_E2E_PIPELINE=1`, `PROFILE_PIPELINE=1`, `PROFILE_T5=1`, `PROFILE_VAE=1`.

Required Neuron env vars (set by scripts): `NEURON_FALLBACK_ENABLED=0`, `NEURON_LOGICAL_NC_CONFIG=1`, `NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=0`.

## Architecture

### Parallelism Strategy (8 ranks)

| Stage | Parallelism | Entry Point |
|-------|-------------|-------------|
| T5 encoder | TP=8 | `encode_prompt.py` |
| DiT denoising | TP=4 × SP=2 | `generate_latents.py` |
| VAE decoder | W-shard × 8 | `decode_latents.py` |
| Fused pipeline | All three | `e2e_pipeline.py` |

### Process Groups (`utils/parallel_state.py`)

Named groups registered via `ps.register_group()`:
- `"world"` — all 8 ranks
- `"attn-tp"` — tensor-parallel group for DiT self-attention (4 ranks)
- `"attn-sp"` — sequence-parallel group for DiT (2 ranks)
- `"vae-sp"` — VAE width-sharding group (all 8 ranks)

### Compilation

All modules/functions intended for Neuron are wrapped with `_compile()` from `utils/__init__.py`, which calls `torch.compile(backend="neuron", dynamic=False, fullgraph=True)`. The dynamo cache limit is set to 128.

### Key Abstractions

- **`CausalInferencePipeline`** (`models/dit_pipeline.py`) — orchestrates the rolling-forcing denoising loop with KV cache management, timestep patterns, re-noising, and streaming output. Supports both full-video (`inference_rolling_forcing`) and streaming (`inference_rolling_forcing_stream`) modes.
- **`CausalWanModel`** (`models/dit_model.py`) — the DiT transformer loaded via `diffusers` `from_pretrained`. Self-attention is TP-sharded; sequence is SP-sharded across ranks.
- **`WanVAEWrapper`** (`models/vae.py`) — VAE decoder with width-sharding via halo exchanges (`_halo_exchange_w`). Uses causal 3D convolutions with temporal caching for streaming decode.
- **`NoiseProducer`** (`utils/noise_producer.py`) — async CPU noise generation for re-noising steps.

### NKI Kernels (`kernels/`)

Custom Neuron Kernel Interface kernels: flash self-attention, cross-attention, RoPE, KV-cache copy, causal conv3d cache updates, width-edge extraction for halo exchange, and layout restoration.

### Config Merging

`configs/default_config.yaml` provides base parameters; `configs/rolling_forcing_dmd.yaml` overrides with DMD-specific settings. Merged with OmegaConf.

## Dependencies

- `torch-neuronx`, `neuronx-cc`, `nki` (Neuron SDK alpha wheels)
- `diffusers==0.37.1`, `transformers==5.8.1`, `huggingface-hub==1.16.4`
- `omegaconf`, `einops`, `click`, `ftfy`, `regex`, `av`

## Model Weights

- `wan_models/Wan2.1-T2V-1.3B/` — T5 encoder + VAE decoder (from HuggingFace)
- `checkpoints/rolling_forcing_dmd.pt` — DMD-distilled DiT (from HuggingFace TencentARC/RollingForcing)
- `cpu_rng_states/` — deterministic RNG states for reproducible generation (contact authors)
