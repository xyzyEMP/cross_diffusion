#!/usr/bin/env bash
set -euo pipefail
PROJECT=/zeron-vepfs/tjqc/cross-diffusion
BASE=/tj-share/cross_diffusion_workdir/research_score_v1_2
EXT="$BASE/01_data_audit/d029_extended_20_50_seed11_20260928Tgpu01Z"
CACHE="$BASE/02_feature_cache"
OUT="$BASE/05_transfer/formal_seed11_d029_extended_20_50"
mkdir -p "$OUT/logs"
run_one(){
 local budget=$1 gpu=$2 run="$OUT/pretrain_finetune_${1}pct_seed11"
 CUDA_VISIBLE_DEVICES=$gpu /root/miniconda3/envs/diffusion-planner/bin/python -u -m tartan.research_score.scripts.train_method_formal --method pretrain_finetune --train-cache "$EXT/target_train_features_d029_extended.pt" --val-cache "$CACHE/target_val_features.pt" --source-cache "$CACHE/source_car_features_4096.pt" --args "$PROJECT/checkpoints/args.json" --checkpoint "$PROJECT/checkpoints/model.pth" --output "$run" --budget "$budget" --max-target-updates 10000 --min-target-updates 5000 --val-every-updates 250 --patience-validations 10 --batch-size 64 --seed 11 --amp >"$OUT/logs/pretrain_finetune_${budget}pct_seed11.log" 2>&1
}
run_one 20 0 & p0=$!
run_one 50 1 & p1=$!
failed=0; wait "$p0" || failed=1; wait "$p1" || failed=1; [[ $failed -eq 0 ]]
[[ -s "$OUT/pretrain_finetune_20pct_seed11/metrics.json" ]]
[[ -s "$OUT/pretrain_finetune_50pct_seed11/metrics.json" ]]
echo COMPLETE > "$OUT/status.txt"
