#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/zeron-vepfs/tjqc/cross-diffusion}
BASE=${OUTPUT_ROOT:-/tj-share/cross_diffusion_workdir/research_score_v1_2}
DATA="$BASE/01_data_audit/stage03_native_v123_remediation_final_20260911T170000Z_nogit/transfer_primary"
OUT="$BASE/05_transfer/seed11_first_round"
mkdir -p "$OUT/logs"
methods=(pretrain_finetune pretrain_adapter joint_train emb_cond_diffusion)
budgets=(1 10 100)
jobs=()
for method in "${methods[@]}"; do
  for budget in "${budgets[@]}"; do jobs+=("$method:$budget"); done
done
run_one() {
  local spec=$1 gpu=$2 method budget run
  method=${spec%%:*};budget=${spec##*:};run="$OUT/${method}_${budget}pct_seed11"
  if [[ -s "$run/metrics.json" ]]; then return 0; fi
  if [[ -e "$run" ]]; then echo "Incomplete run exists: $run" >&2; return 2; fi
  CUDA_VISIBLE_DEVICES=$gpu /root/miniconda3/envs/diffusion-planner/bin/python -u -m tartan.research_score.scripts.train_method_smoke \
    --method "$method" --manifest "$DATA/target_train_windows.jsonl" --val-manifest "$DATA/target_val_windows.jsonl" \
    --source-manifest "$DATA/source_length_audit.jsonl" --args "$PROJECT_ROOT/checkpoints/args.json" \
    --checkpoint "$PROJECT_ROOT/checkpoints/model.pth" --output "$run" --budget "$budget" \
    --steps 300 --val-every 100 --batch-size 4 --seed 11 --amp >"$OUT/logs/${method}_${budget}pct_seed11.log" 2>&1
}
export -f run_one
export PROJECT_ROOT BASE DATA OUT
for start in 0 8; do
  pids=()
  for ((i=start;i<${#jobs[@]} && i<start+8;i++)); do run_one "${jobs[$i]}" "$((i-start))" & pids+=("$!"); done
  failed=0;for pid in "${pids[@]}";do wait "$pid" || failed=1;done
  [[ $failed -eq 0 ]] || exit 1
done
echo COMPLETE > "$OUT/status.txt"
