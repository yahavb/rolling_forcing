# Roofline / Parallelism Analysis — Rolling Forcing (Wan2.1-T2V-1.3B) on trn2

Same structure as the Odyssey O2 (14B) parallelism-scheme analysis, but for the **RF 1.3B**
model as actually deployed in this repo. All model shapes are read from the code
(`models/dit_model.py`, `utils/flops.py`, `configs/rolling_forcing_dmd.yaml`), and all fps /
TFLOPS / MFU numbers are **measured on trn2** via the `rf-*-job.yaml` benchmark jobs
(branch `rope-qk-fuse`, 21 frames, no profiler unless noted). Where a value is derived rather
than measured it is marked (derived).

---

## 0. Executive summary

RF 1.3B runs **real-time today** at 480×640: **14.3 fps** steady-state (DiT+VAE), TP4×CP4,
16 NeuronCores. The open question this analysis answers is **how fps scales with sequence
length and with the LNC (compute-per-rank) configuration** — the RF analogue of Odyssey's
"can we hit 16 fps" budget question.

Key measured result: **RF is per-rank-bandwidth-bound.** Each (core-type, seq-len) point runs
at ~full speed until it crosses a per-rank bandwidth knee, then fps rolls off ∝ sequence
length. The knee is one seq-len rung higher on LNC2 (l-trn2) than on LNC1 (m-lnc1-trn2).

---

## 1. Model: shapes and FLOPs (RF 1.3B, from code)

| Symbol | Value | Source |
|---|---|---|
| D (hidden dim) | 2048 | `dit_model.py` dim |
| F (FFN dim) | 8192 | `dit_model.py` ffn_dim |
| L (layers) | 32 | `dit_model.py` num_layers |
| num_heads | 16 | `dit_model.py` |
| head_dim | 128 | D/num_heads |
| patch_size | (1,2,2) | `dit_model.py` |
| text_len (Lt) | 512 | `dit_model.py` |
| nfpb (latent frames/block) | 3 | `configs/rolling_forcing_dmd.yaml` num_frame_per_block |
| denoising steps | 5 | denoising_step_list [1000,800,600,400,200] |
| frame_seq_length | 1200 / 1440 / 1680 | latent_w 80/96/112 (swept) |

**Token counts** (per denoise forward), `seq_len = num_frames × frame_seq_length`:
the merged rolling-forcing forward processes nfpb (cache-update) + up to `max_frames = 5×3 = 15`
denoise frames. At fs1200 the steady merged call is ~18 frames × 1200 = 21,600 tokens; the
per-frame token count = frame_seq_length itself.

**FLOPs per denoise step** (from `utils/flops.py`, BF16 = 2·M·N·K), per layer:

| Component | Formula | Note |
|---|---|---|
| self_attn qkv proj | 3·2·M·D·D | |
| self_attn QKᵀ + AV | 2·(2·num_heads·M·M·head_dim) | ∝ M² — dominates as fs grows |
| self_attn o proj | 2·M·D·D | |
| cross_attn q/k/v proj | 3·2·M·D·D over M and text_len | |
| cross_attn QKᵀ + AV | 2·(2·num_heads·M·Lt·head_dim) | small (Lt=512) |
| ffn (up+down) | 2·M·D·F + 2·M·F·D | |

Unlike Odyssey (14B, w=12 window → S=43,200), RF's self-attention window is the rolling
KV window (≤ 5 blocks), so RF's absolute FLOP is ~13× smaller — which is exactly why RF is
launch/DMA-bound rather than FLOP-bound at these sizes (see §5).

---

## 2. Hardware: trn2, the two claim types benchmarked

| | l-trn2 | m-lnc1-trn2 |
|---|---|---|
| LNC | 2 (2 physical cores fused → 1 logical) | 1 |
| HBM / rank | ~24 GB | ~12 GB |
| Per-rank compute / BW | ~2× | 1× |
| Ranks (TP4×CP4) | 16 | 16 |

Peak used by the harness MFU column: 3040 TFLOPS @ 16 cores (an LNC2 figure). **Caveat:** the
harness reports m-lnc1 MFU against this SAME LNC2 peak, so m-lnc1 "MFU %" below understates
true per-rank utilization by ~2×. Compare fps and mean-DiT-ms across claims, not the raw MFU %.

---

## 3. MEASURED DATA POINTS (the roofline dataset)

Topology fixed at **TP4 × CP4 = 16 ranks**, 21 frames, branch `rope-qk-fuse`, profiler OFF.
fps = DiT+VAE steady-state per-block median (warm blocks 1–6, all 16 prompts). mean DiT ms =
mean warm per-block DiT time. TFLOPS/MFU = harness `MEASURE_TFLOPS` (whole DiT+VAE, vs 3040 peak).

| latent_w | pixel (480×W) | frame_seq_length | LNC (claim) | fps (DiT+VAE) | mean DiT ms/block | achieved TFLOPS | MFU (vs 3040) | OOM? | log |
|---|---|---|---|---|---|---|---|---|---|
| 80  | 640 | 1200 | LNC2 (l-trn2)      | **14.3**  | 695 | ~137.7 | ~4.53% | no  | brp8j / main baseline |
| 96  | 768 | 1440 | LNC2 (l-trn2)      | **14.60** | 672 | ~170.4 | ~5.61% | no  | 1440-kcdvk |
| 112 | 896 | 1680 | LNC2 (l-trn2)      | **11.35** | 883 | ~180.3 | ~5.93% | no  | seqlen-ztg8b |
| 80  | 640 | 1200 | LNC1 (m-lnc1-trn2) | **13.95** | 700 | ~137.7 | ~4.53%* | no  | 1200-mlnc1-hlg4z |
| 96  | 768 | 1440 | LNC1 (m-lnc1-trn2) | **11.3**  | ~820 | — | — | **only under profiler** | (earlier w4gnr OOM w/ profiler; fit no-profiler) |
| 112 | 896 | 1680 | LNC1 (m-lnc1-trn2) | **9.84**  | 965 | ~169.5 | ~5.58%* | no  | 1680-mlnc1-hgcjr |

\* m-lnc1 MFU reported against the LNC2 3040-peak → true per-rank MFU ≈ 2× shown.

### 2×3 fps grid

| | fs1200 | fs1440 | fs1680 |
|---|---|---|---|
| **LNC2** | 14.3 | 14.60 | 11.35 |
| **LNC1** | 13.95 | 11.3  | 9.84  |

### mean DiT ms/block grid

| | fs1200 | fs1440 | fs1680 |
|---|---|---|---|
| **LNC2** | 695 | 672 | 883 |
| **LNC1** | 700 | ~820 | 965 |

---

## 4. Interpretation — the bandwidth-knee model

1. **Below the knee, fps is flat / seq-len is free.** LNC2 fs1200→fs1440: 14.3→14.60 (mean
   DiT 695→672 ms — no cost, within noise). The longer sequence hides under the per-rank
   DMA/launch ceiling.
2. **At the knee, fps rolls off ∝ sequence.** LNC2 fs1440→fs1680: 14.60→11.35 (DiT 672→883).
   LNC1 fs1200→fs1440→fs1680: 13.95→11.3→9.84 (monotonic — not a floor).
3. **LNC only matters near/above the knee.** At fs1200 LNC1 ≈ LNC2 (13.95 vs 14.3, mean DiT
   700 vs 695 — identical); LNC2's extra bandwidth is idle headroom. The LNC gap is
   0.35 → 3.3 → 1.5 fps across fs1200/1440/1680: it **peaks where the cores straddle their
   knees** (LNC1 rolled off, LNC2 not yet) and narrows once both are past.
4. **Knee locations:** LNC1 ≈ fs1200–1440; LNC2 ≈ fs1440–1680.

This is the direct RF analogue of Odyssey's roofline: RF operates far below the FLOP roofline
(MFU 4–6%), so the binding resource is per-rank memory bandwidth + kernel-launch overhead, not
compute. Doubling per-rank BW (LNC1→LNC2) shifts the knee up one seq-len rung; it does not
give a flat 2× speedup.

---

## 5. Why RF sits at 4–6% MFU (vs Odyssey targeting 45–66%)

RF 1.3B's per-layer FLOP is small (window-limited self-attention, D=2048), so at TP4×CP4 the
matmuls are tiny per rank and the model is **launch/DMA-bound** — every executed NEFF profiles
as OVERHEAD-BOUND / MIXED with 0% matmul MFU (see `project_rf_collective_bound` in memory).
This is why the only optimizations that ever moved RF fps were **launch-count reductions**
(RoPE gpsimd batching: Win1/Win5), not FLOP or precision changes. Odyssey 14B has ~13× the
FLOP and a 43k-token window, so it is genuinely compute-adjacent and its 45–66% MFU targets
are meaningful; RF's are not the right lens — for RF, **launches and bandwidth are the axes.**

---

## 6. Memory (derived)

RF 1.3B: weights ~1.3B×2 = ~2.6 GB; KV cache = 2 × (kv_cache_logical_size = 24×frame_length) ×
D × L × 2B, scales with frame_length. At fs1200 this fits comfortably in both 12 GB (LNC1) and
24 GB (LNC2). fs1440 fit LNC1 only WITHOUT the profiler (the profiler's device-session buffer
was the ~102 MB straw that OOM'd — confirmed: fs1680 fit LNC1 no-profiler). So for RF the HBM
ceiling is not the binding constraint at these resolutions; bandwidth is.

---

## 7. Open items / next data points

- **fs1920 (latent_w=128) / fs2160 (latent_w=144) on LNC2** — map the post-knee slope; watch
  for the HBM ceiling appearing at the top. Job: `rf-rope-qk-fuse-seqlen-job.yaml`, set LATENT_W.
- **A plain-main (no qk-fuse) fs1440/1680 point** — isolate the qk-fuse kernel's effect at
  higher seq len (current higher-fs points are all on branch `rope-qk-fuse`).
- **Valid seq-len rungs** (TP4×CP4, latent_h=60): frame_seq_length = 15×latent_w, must be
  %16==0 ⇒ latent_w %16 ⇒ {80/1200, 96/1440, 112/1680, 128/1920, 144/2160}. 104/1560 FAILS.

---

## 8. Reproduce

All jobs: branch under test in `BRANCH` env, `LATENT_W` sets the rung, claim (`l-trn2` vs
`m-lnc1-trn2`) sets LNC, profiler on/off via the `NEURON_PROFILE_ENABLE`/`NEURON_RT_INSPECT_*`
block. fps computed from warm blocks:

```
grep -oE "block  [1-6]: DiT +[0-9.]+ ms +VAE +[0-9.]+ ms +([0-9]+) frames" LOG \
 | sed -E 's/.*DiT +([0-9.]+) ms +VAE +([0-9.]+) ms +([0-9]+) frames/\1 \2 \3/' \
 | awk '$1<3500{dv=$3*1000/($1+$2);a[++n]=dv;sd+=$1} END{...median...; print sd/n}'
```
