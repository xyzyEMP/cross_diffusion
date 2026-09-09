#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/yzy/anaconda3/envs/diffusion_planner/bin/python}"
DATA_ROOT="${TARTANGROUND_ROOT:-/media/yzy/1E585F67585F3CA9/data_yzy/TartanGround}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/tartan/outputs/final}"
SAMPLES_PER_TRAJECTORY="${SAMPLES_PER_TRAJECTORY:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DEVICE="${DEVICE:-auto}"
TERRAIN_RESOLUTION="${TERRAIN_RESOLUTION:-0.5}"
DIFFUSION_REPEATS="${DIFFUSION_REPEATS:-3}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/diffusion_planner_matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/diffusion_planner_cache}"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" -m tartan.evaluation.evaluate \
  --data-root "$DATA_ROOT" \
  --args-file checkpoints/args.json \
  --checkpoint checkpoints/model.pth \
  --output-dir "$OUTPUT_DIR" \
  --samples-per-trajectory "$SAMPLES_PER_TRAJECTORY" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --diffusion-repeats "$DIFFUSION_REPEATS" \
  --terrain-resolution "$TERRAIN_RESOLUTION"
