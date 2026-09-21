#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/zeron-vepfs/tjqc/cross-diffusion}
BASE=${OUTPUT_ROOT:-/tj-share/cross_diffusion_workdir/research_score_v1_2}
DATA="$BASE/01_data_audit/stage03_native_v123_remediation_final_20260911T170000Z_nogit/transfer_primary"
OUT="$BASE/05_transfer/seed11_first_round"
mkdir -p "$OUT/logs"
jobs=(joint_train:1 joint_train:10 joint_train:100 emb_cond_diffusion:1 emb_cond_diffusion:10 emb_cond_diffusion:100)
run_one() {
  local spec=$1 gpu=$2 method budget run
  method=${spec%%:*}; budget=${spec##*:}; run="$OUT/${method}_${budget}pct_seed11"
  CUDA_VISIBLE_DEVICES=$gpu /root/miniconda3/envs/diffusion-planner/bin/python -u -m tartan.research_score.scripts.train_method_smoke \
    --method "$method" --manifest "$DATA/target_train_windows.jsonl" --val-manifest "$DATA/target_val_windows.jsonl" \
    --source-manifest "$DATA/source_length_audit.jsonl" --args "$PROJECT_ROOT/checkpoints/args.json" \
    --checkpoint "$PROJECT_ROOT/checkpoints/model.pth" --output "$run" --budget "$budget" \
    --steps 300 --val-every 100 --batch-size 4 --seed 11 --amp >"$OUT/logs/${method}_${budget}pct_seed11.log" 2>&1
}
pids=()
for i in "${!jobs[@]}"; do run_one "${jobs[$i]}" "$i" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ $failed -eq 0 ]] || exit 1
count=$(find "$OUT" -mindepth 2 -maxdepth 2 -name metrics.json | wc -l)
[[ $count -eq 15 ]] || { echo "Expected 15 results, found $count" >&2; exit 2; }
echo COMPLETE > "$OUT/status.txt"
