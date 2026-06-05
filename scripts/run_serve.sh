#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch the serving endpoint locally on a Trn2 host.
#
# Usage:
#   bash scripts/run_serve.sh

set -e

WORLD_SIZE=8

export NEURON_FALLBACK_ENABLED=0
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=0
export TP_DEGREE="${TP_DEGREE:-4}"
export DEFAULT_NUM_FRAMES="${DEFAULT_NUM_FRAMES:-126}"
export DEFAULT_FPS="${DEFAULT_FPS:-16}"
export CONFIG_PATH="${CONFIG_PATH:-configs/rolling_forcing_dmd.yaml}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/rolling_forcing_dmd.pt}"
export WARMUP_FRAMES="${WARMUP_FRAMES:-21}"

echo "============================================"
echo "  Rolling Forcing Serving Endpoint"
echo "  TP=$TP_DEGREE, SP=$((WORLD_SIZE / TP_DEGREE))"
echo "  Warmup: $WARMUP_FRAMES frames"
echo "  Default output: $DEFAULT_NUM_FRAMES frames"
echo "============================================"

torchrun --nproc_per_node "$WORLD_SIZE" serve.py
