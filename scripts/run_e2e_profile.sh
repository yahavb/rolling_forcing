#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Profiling script: runs the e2e pipeline with Neuron Profiler enabled
# and post-processes traces with neuron-explorer.
#
# Usage:
#   bash scripts/run_e2e_profile.sh
#
# Outputs:
#   /tmp/neuron_profile/  — raw NTFF trace files
#   profiles/             — neuron-explorer summary + JSON + TFLOPS/MFU report

set -e

WORLD_SIZE=8
TP_DEGREE=4
CHUNK_SIZE=3
NUM_OUTPUT_FRAMES=21   # short run for profiling (fewer frames = faster)
FPS=16

CONFIG_PATH="configs/rolling_forcing_dmd.yaml"
CHECKPOINT_PATH="checkpoints/rolling_forcing_dmd.pt"
PROMPT_FILE="prompts/example_prompts.txt"
OUTPUT_FOLDER="videos_profile"

# ============================================
# Neuron env
# ============================================
export NEURON_FALLBACK_ENABLED=0
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=0

# ============================================
# Neuron Profiler: capture device-level traces
# ============================================
export NEURON_RT_INSPECT_ENABLE=1
export NEURON_RT_INSPECT_OUTPUT_DIR=/tmp/neuron_profile
export NEURON_RT_INSPECT_DEVICE_PROFILE=session
mkdir -p /tmp/neuron_profile

echo "============================================"
echo "  Neuron Profiler ENABLED"
echo "  Trace output: $NEURON_RT_INSPECT_OUTPUT_DIR"
echo "============================================"

# ============================================
# Run inference with profiling + TFLOPS measurement
# ============================================
PROFILE_E2E_PIPELINE=1 MEASURE_TFLOPS=1 \
    torchrun --nproc_per_node "$WORLD_SIZE" e2e_pipeline.py \
    --prompt_file "$PROMPT_FILE" \
    --output_folder "$OUTPUT_FOLDER" \
    --config_path "$CONFIG_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --tp_degree "$TP_DEGREE" \
    --num_output_frames "$NUM_OUTPUT_FRAMES" \
    --chunk-size "$CHUNK_SIZE" \
    --fps "$FPS" \
    --use_ema \
    --rng_state_path cpu_rng_states

# ============================================
# Post-process with neuron-explorer
# ============================================
PROFILE_OUT="profiles"
mkdir -p "$PROFILE_OUT"

NTFF_DIR=$(find /tmp/neuron_profile -mindepth 2 -maxdepth 2 -type d 2>/dev/null | head -1)

if [[ -n "$NTFF_DIR" ]]; then
    echo ""
    echo "============================================"
    echo "  NEURON EXPLORER ANALYSIS"
    echo "  Trace dir: $NTFF_DIR"
    echo "============================================"

    # Summary text
    echo "=== neuron-explorer view (summary-text) ==="
    neuron-explorer view -d "$NTFF_DIR" \
      --output-format summary-text \
      --ignore-dma-trace 2>&1 | tee "$PROFILE_OUT/neuron_explorer_summary.txt" || true

    # JSON (for programmatic analysis)
    echo ""
    echo "=== neuron-explorer view (JSON) ==="
    neuron-explorer view -d "$NTFF_DIR" \
      --output-format json \
      --output-file "$PROFILE_OUT/neuron_explorer_profile.json" \
      --ignore-dma-trace 2>&1 | tee "$PROFILE_OUT/neuron_explorer_view.log" || true

    # NEFF distribution
    echo ""
    echo "=== NEFF COUNT AND SIZE DISTRIBUTION ==="
    NEFF_COUNT=$(find "$NTFF_DIR" -name '*.neff' | wc -l)
    echo "Total NEFFs: $NEFF_COUNT"
    find "$NTFF_DIR" -name '*.neff' -exec ls -l '{}' ';' | awk '{print $5}' | sort -n | awk '
      BEGIN { count=0; sum=0 }
      { sizes[count++]=$1; sum+=$1 }
      END {
        if (count == 0) { print "No NEFFs found"; exit }
        printf "Total size: %.2f MB\n", sum/1024/1024
        printf "Min: %d bytes\n", sizes[0]
        printf "Max: %d bytes (%.2f MB)\n", sizes[count-1], sizes[count-1]/1024/1024
        printf "Median: %d bytes\n", sizes[int(count/2)]
        printf "Mean: %.0f bytes\n", sum/count
        printf "\nSize buckets:\n"
        small=0; med=0; large=0; xlarge=0
        for(i=0;i<count;i++) {
          if(sizes[i]<10000) small++
          else if(sizes[i]<100000) med++
          else if(sizes[i]<1000000) large++
          else xlarge++
        }
        printf "  <10KB  (tiny):   %d\n", small
        printf "  10-100KB (small): %d\n", med
        printf "  100KB-1MB (med):  %d\n", large
        printf "  >1MB  (large):    %d\n", xlarge
      }'

    # Copy raw traces
    cp -r /tmp/neuron_profile/* "$PROFILE_OUT/" 2>/dev/null || true
else
    echo "WARNING: No NTFF directory found under /tmp/neuron_profile"
    echo "         (profiling may not have captured traces)"
fi

echo ""
echo "============================================"
echo "  DONE — results in: $PROFILE_OUT/"
echo "============================================"
