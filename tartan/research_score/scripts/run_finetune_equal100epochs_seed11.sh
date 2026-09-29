#!/usr/bin/env bash
set -euo pipefail
PROJECT=/zeron-vepfs/tjqc/cross-diffusion
BASE=/tj-share/cross_diffusion_workdir/research_score_v1_2
EXT="$BASE/01_data_audit/d029_extended_20_50_seed11_20260928Tgpu01Z"
CACHE="$BASE/02_feature_cache"
OUT="$BASE/05_transfer/finetune_equal100epochs_seed11"
mkdir -p "$OUT/logs"
budgets=(1 10 20 50 100)
samples=(20 205 410 1024 2048)

run_one(){
 local index=$1 gpu=$2 budget=${budgets[$1]} count=${samples[$1]} batches updates val_every run
 batches=$(( (count + 63) / 64 ))
 updates=$(( batches * 100 ))
 val_every=$(( batches * 10 ))
 run="$OUT/pretrain_finetune_${budget}pct_seed11"
 CUDA_VISIBLE_DEVICES=$gpu /root/miniconda3/envs/diffusion-planner/bin/python -u -m tartan.research_score.scripts.train_method_formal \
  --method pretrain_finetune --train-cache "$EXT/target_train_features_d029_extended.pt" \
  --val-cache "$CACHE/target_val_features.pt" --source-cache "$CACHE/source_car_features_4096.pt" \
  --args "$PROJECT/checkpoints/args.json" --checkpoint "$PROJECT/checkpoints/model.pth" \
  --output "$run" --budget "$budget" --max-target-updates "$updates" --min-target-updates "$updates" \
  --val-every-updates "$val_every" --patience-validations 1000 --batch-size 64 --seed 11 --amp \
  >"$OUT/logs/pretrain_finetune_${budget}pct_seed11.log" 2>&1
}

for ((start=0; start<${#budgets[@]}; start+=2)); do
 pids=()
 for ((i=start; i<${#budgets[@]} && i<start+2; i++)); do run_one "$i" "$((i-start))" & pids+=("$!"); done
 failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; [[ $failed -eq 0 ]]
done
[[ $(find "$OUT" -mindepth 2 -maxdepth 2 -name metrics.json | wc -l) -eq 5 ]]
echo COMPLETE > "$OUT/status.txt"
