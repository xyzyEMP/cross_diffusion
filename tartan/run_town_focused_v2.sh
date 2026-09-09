#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/yzy/anaconda3/envs/diffusion_planner/bin/python}"
DATA_ROOT="${TARTANGROUND_ROOT:-/media/yzy/1E585F67585F3CA9/data_yzy/TartanGround}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/tartan/outputs/town_focused_v2}"
DEVICE="${DEVICE:-cuda}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/diffusion_planner_matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/diffusion_planner_cache}"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" -m tartan.evaluation.evaluate \
  --data-root "$DATA_ROOT" \
  --args-file checkpoints/args.json \
  --checkpoint checkpoints/model.pth \
  --output-dir "$OUTPUT_ROOT/open_loop" \
  --samples-per-trajectory "${OPEN_LOOP_SAMPLES_PER_TRAJECTORY:-12}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --device "$DEVICE" \
  --diffusion-repeats "${DIFFUSION_REPEATS:-3}" \
  --terrain-resolution 0.5 \
  --no-visualizations

"$PYTHON_BIN" -m tartan.evaluation.closed_loop \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_ROOT/closed_loop" \
  --args-file checkpoints/args.json \
  --checkpoint checkpoints/model.pth \
  --device "$DEVICE" \
  --episodes-per-trajectory "${CLOSED_LOOP_EPISODES_PER_TRAJECTORY:-2}" \
  --diffusion-repeats "${CLOSED_LOOP_DIFFUSION_REPEATS:-2}" \
  --replan-steps "${REPLAN_STEPS:-10}" \
  --min-route-coverage "${MIN_ROUTE_COVERAGE:-0.95}"

"$PYTHON_BIN" -m tartan.evaluation.summarize_v2 \
  --output-root "$OUTPUT_ROOT" \
  --data-root "$DATA_ROOT"

for batch_index in 0 1 2 3 4; do
  "$PYTHON_BIN" -m tartan.viz.animate_results \
    --input-dir "$OUTPUT_ROOT/open_loop/predictions" \
    --output-dir "$OUTPUT_ROOT/visualizations/open_loop" \
    --data-root "$DATA_ROOT" \
    --terrain-cache tartan/cache/elevation \
    --environments ModernCityDowntown OldTownFall \
    --count 6 --seed 20260822 --batch-index "$batch_index" --fps 10

  "$PYTHON_BIN" -m tartan.viz.animate_closed_loop \
    --result-root "$OUTPUT_ROOT/closed_loop" \
    --output-dir "$OUTPUT_ROOT/visualizations/closed_loop" \
    --data-root "$DATA_ROOT" \
    --terrain-cache tartan/cache/elevation \
    --batch-index "$batch_index" --fps 10
done

"$PYTHON_BIN" -m tartan.evaluation.final_report \
  --result-root "$OUTPUT_ROOT"

echo "Town-focused evaluation complete: $OUTPUT_ROOT"
