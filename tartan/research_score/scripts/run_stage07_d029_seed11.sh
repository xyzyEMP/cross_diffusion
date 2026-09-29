#!/usr/bin/env bash
set -euo pipefail

PROJECT=/zeron-vepfs/tjqc/cross-diffusion
BASE=/tj-share/cross_diffusion_workdir/research_score_v1_2
CACHE="$BASE/02_feature_cache"
D029="$BASE/01_data_audit/d029_window_budgets_seed11_20260927Tcpu01Z_retry01"
OUT="$BASE/05_transfer/formal_seed11_d029"

[[ -f "$D029/target_train_features_d029.pt" ]]
[[ -f "$CACHE/target_val_features.pt" ]]
[[ -f "$CACHE/source_car_features_4096.pt" ]]
mkdir -p "$OUT/logs"

jobs=()
for method in pretrain_finetune pretrain_adapter joint_train emb_cond_diffusion; do
  for budget in 1 10 100; do jobs+=("$method:$budget"); done
done

run_one() {
  local spec=$1 gpu=$2 method budget run
  method=${spec%%:*}; budget=${spec##*:}
  run="$OUT/${method}_${budget}pct_seed11"
  if [[ -s "$run/metrics.json" ]]; then return 0; fi
  [[ ! -e "$run" ]] || { echo "Incomplete run exists: $run" >&2; return 2; }
  CUDA_VISIBLE_DEVICES=$gpu /root/miniconda3/envs/diffusion-planner/bin/python -u \
    -m tartan.research_score.scripts.train_method_formal \
    --method "$method" --train-cache "$D029/target_train_features_d029.pt" \
    --val-cache "$CACHE/target_val_features.pt" --source-cache "$CACHE/source_car_features_4096.pt" \
    --args "$PROJECT/checkpoints/args.json" --checkpoint "$PROJECT/checkpoints/model.pth" \
    --output "$run" --budget "$budget" --max-target-updates 10000 --min-target-updates 5000 \
    --val-every-updates 250 --patience-validations 10 --batch-size 64 --seed 11 --amp \
    >"$OUT/logs/${method}_${budget}pct_seed11.log" 2>&1
}

if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "GPU_PENDING: attach GPUs before running D029 experiment 1" >&2
  exit 3
fi

gpu_count=$(nvidia-smi -L | wc -l)
[[ $gpu_count -ge 1 ]]
for ((start=0; start<${#jobs[@]}; start+=gpu_count)); do
  pids=()
  for ((i=start; i<${#jobs[@]} && i<start+gpu_count; i++)); do
    run_one "${jobs[$i]}" "$((i-start))" & pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ $failed -eq 0 ]] || exit 1
done

[[ $(find "$OUT" -mindepth 2 -maxdepth 2 -name metrics.json | wc -l) -eq 12 ]]
echo COMPLETE > "$OUT/status.txt"
