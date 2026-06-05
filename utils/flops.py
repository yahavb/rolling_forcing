# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TFLOPS and MFU (Model FLOPs Utilization) measurement for Rolling Forcing inference.

Hardware reference:
  - Trn2 NeuronCore: 190 TFLOPS BF16 per core (TensorE)
  - 8 cores at LNC=1 → 1520 TFLOPS aggregate peak BF16
"""

import os
import json
import time

import torch

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Trn2 NeuronCore peak BF16 TFLOPS (per core)
NEURONCORE_PEAK_TFLOPS_BF16 = 190.0


def compute_dit_flops_per_step(
    dim: int = 2048,
    ffn_dim: int = 8192,
    num_heads: int = 16,
    num_layers: int = 32,
    text_len: int = 512,
    num_frames: int = 15,
    frame_seq_length: int = 1560,
    batch_size: int = 1,
) -> float:
    """
    Estimate FLOPs for one DiT forward pass (one denoising step).

    The DiT processes a sequence of (num_frames * frame_seq_length / world_size)
    tokens through self-attention + cross-attention + FFN per layer.

    For BF16 matmul: FLOPs = 2 * M * N * K per operation.
    """
    head_dim = dim // num_heads
    seq_len = num_frames * frame_seq_length

    flops = 0
    for _ in range(num_layers):
        # Self-attention: Q, K, V projections (3 linear layers)
        flops += 3 * (2 * batch_size * seq_len * dim * dim)
        # Self-attention: QK^T
        flops += 2 * batch_size * num_heads * seq_len * seq_len * head_dim
        # Self-attention: attn @ V
        flops += 2 * batch_size * num_heads * seq_len * seq_len * head_dim
        # Self-attention: output projection
        flops += 2 * batch_size * seq_len * dim * dim

        # Cross-attention: Q projection
        flops += 2 * batch_size * seq_len * dim * dim
        # Cross-attention: K, V projections (over text_len)
        flops += 2 * (2 * batch_size * text_len * dim * dim)
        # Cross-attention: QK^T
        flops += 2 * batch_size * num_heads * seq_len * text_len * head_dim
        # Cross-attention: attn @ V
        flops += 2 * batch_size * num_heads * seq_len * text_len * head_dim
        # Cross-attention: output projection
        flops += 2 * batch_size * seq_len * dim * dim

        # FFN: two linear layers (gate + fc1 → dim*ffn_dim, fc2 → ffn_dim*dim)
        # Gate: dim → ffn_dim
        flops += 2 * batch_size * seq_len * dim * ffn_dim
        # FC1: dim → ffn_dim
        flops += 2 * batch_size * seq_len * dim * ffn_dim
        # FC2: ffn_dim → dim
        flops += 2 * batch_size * seq_len * ffn_dim * dim

    return float(flops)


def compute_t5_flops(
    dim: int = 4096,
    dim_ffn: int = 10240,
    num_heads: int = 64,
    num_layers: int = 24,
    seq_len: int = 512,
    batch_size: int = 1,
) -> float:
    """Estimate FLOPs for one T5 encoder forward pass."""
    head_dim = dim // num_heads
    flops = 0
    for _ in range(num_layers):
        # Self-attention QKV
        flops += 3 * (2 * batch_size * seq_len * dim * dim)
        # QK^T
        flops += 2 * batch_size * num_heads * seq_len * seq_len * head_dim
        # attn @ V
        flops += 2 * batch_size * num_heads * seq_len * seq_len * head_dim
        # Output projection
        flops += 2 * batch_size * seq_len * dim * dim
        # FFN: gate + fc1 + fc2
        flops += 2 * batch_size * seq_len * dim * dim_ffn  # gate
        flops += 2 * batch_size * seq_len * dim * dim_ffn  # fc1
        flops += 2 * batch_size * seq_len * dim_ffn * dim  # fc2
    return float(flops)


def compute_vae_flops_approx(
    latent_channels: int = 16,
    height: int = 60,
    width: int = 104,
    num_frames: int = 3,
    base_dim: int = 96,
) -> float:
    """
    Rough VAE decoder FLOPs estimate (conv3d dominant).
    This is approximate — the VAE is not the bottleneck.
    """
    # Approximate: count conv3d ops at each resolution stage
    # Stage dims: 384→384→384, 384→192, 192→96, 96→3
    # Spatial resolutions double at upsample stages
    h, w = height, width
    t = num_frames
    flops = 0

    dim_mult = [4, 4, 2, 1]
    dims = [base_dim * m for m in dim_mult]

    # Initial conv: 16 → 384, kernel 3x3x3
    flops += 2 * t * h * w * latent_channels * dims[0] * 27

    # Middle blocks (2 residual + 1 attn at 384)
    flops += 2 * (2 * t * h * w * dims[0] * dims[0] * 27)
    # Attention at lowest res
    spatial = h * w
    flops += 2 * t * spatial * spatial * dims[0]

    # Upsample stages (rough)
    for i, (in_d, out_d) in enumerate(zip(dims, dims[1:] + [3])):
        for _ in range(3):  # num_res_blocks + 1
            flops += 2 * t * h * w * in_d * out_d * 27
            in_d = out_d
        h *= 2
        w *= 2
        if i < 2:
            t *= 2

    return float(flops)


class TFLOPSMeter:
    """Tracks elapsed time and computes TFLOPS/MFU for inference runs."""

    def __init__(self, num_cores: int = 8):
        self.num_cores = num_cores
        self.peak_tflops = num_cores * NEURONCORE_PEAK_TFLOPS_BF16
        self.records = []

    def record(self, stage: str, elapsed_s: float, flops: float):
        tflops = flops / elapsed_s / 1e12
        mfu = tflops / self.peak_tflops
        self.records.append({
            "stage": stage,
            "elapsed_s": elapsed_s,
            "flops": flops,
            "tflops_achieved": tflops,
            "mfu": mfu,
            "peak_tflops": self.peak_tflops,
            "num_cores": self.num_cores,
        })
        logger.info(
            "  [%s] %.2f ms | %.2f TFLOPS | MFU %.2f%% (peak %.0f TFLOPS @ %d cores)",
            stage, elapsed_s * 1000, tflops, mfu * 100,
            self.peak_tflops, self.num_cores,
        )

    def summary(self):
        if not self.records:
            return
        total_flops = sum(r["flops"] for r in self.records)
        total_time = sum(r["elapsed_s"] for r in self.records)
        avg_tflops = total_flops / total_time / 1e12 if total_time > 0 else 0
        avg_mfu = avg_tflops / self.peak_tflops

        logger.info("=" * 60)
        logger.info("TFLOPS / MFU SUMMARY")
        logger.info("=" * 60)
        logger.info("  Peak hardware: %.0f TFLOPS BF16 (%d NeuronCores × %.0f)",
                    self.peak_tflops, self.num_cores, NEURONCORE_PEAK_TFLOPS_BF16)
        logger.info("  Total compute: %.2e FLOPs in %.2f s",
                    total_flops, total_time)
        logger.info("  Average: %.2f TFLOPS | MFU %.2f%%",
                    avg_tflops, avg_mfu * 100)
        logger.info("-" * 60)
        for r in self.records:
            logger.info("  %-20s %7.1f ms  %6.2f TFLOPS  MFU %5.2f%%",
                        r["stage"], r["elapsed_s"] * 1000,
                        r["tflops_achieved"], r["mfu"] * 100)
        logger.info("=" * 60)
        return {
            "peak_tflops": self.peak_tflops,
            "num_cores": self.num_cores,
            "total_flops": total_flops,
            "total_time_s": total_time,
            "avg_tflops": avg_tflops,
            "avg_mfu": avg_mfu,
            "per_stage": self.records,
        }

    def save(self, path: str):
        data = self.summary()
        if data:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("TFLOPS report saved to %s", path)
