#!/usr/bin/env bash
set -euo pipefail
PROJECT=/zeron-vepfs/tjqc/cross-diffusion
BASE=/tj-share/cross_diffusion_workdir/research_score_v1_2
DATA="$BASE/01_data_audit/stage03_native_v123_remediation_final_20260911T170000Z_nogit/transfer_primary"
CACHE="$BASE/02_feature_cache"
OUT="$BASE/05_transfer/formal_seed11"
mkdir -p "$OUT/logs"
jobs=()
for method in pretrain_finetune pretrain_adapter joint_train emb_cond_diffusion; do
 for budget in 1 10 100; do jobs+=("$method:$budget"); done
done
run_one() {
 local spec=$1 gpu=$2 method budget run
 method=${spec%%:*};budget=${spec##*:}
 run="$OUT/${method}_${budget}pct_seed11"
 if [[ -s "$run/metrics.json" ]]; then return 0; fi
 [[ ! -e "$run" ]] || { echo "Incomplete run exists: $run" >&2; return 2; }
 CUDA_VISIBLE_DEVICES=$gpu /root/miniconda3/envs/diffusion-planner/bin/python -u -m tartan.research_score.scripts.train_method_formal \
  --method "$method" --train-cache "$CACHE/target_train_features.pt" --val-cache "$CACHE/target_val_features.pt" \
  --source-cache "$CACHE/source_car_features_4096.pt" --args "$PROJECT/checkpoints/args.json" --checkpoint "$PROJECT/checkpoints/model.pth" \
  --output "$run" --budget "$budget" --max-target-updates 5000 --min-target-updates 1000 \
  --val-every-updates 250 --patience-validations 5 --batch-size 64 --seed 11 --amp \
  >"$OUT/logs/${method}_${budget}pct_seed11.log" 2>&1
}
for start in 0 8; do
 pids=()
 for ((i=start;i<${#jobs[@]} && i<start+8;i++)); do run_one "${jobs[$i]}" "$((i-start))" & pids+=("$!"); done
 failed=0;for pid in "${pids[@]}";do wait "$pid" || failed=1;done
 [[ $failed -eq 0 ]] || exit 1
done
[[ $(find "$OUT" -mindepth 2 -maxdepth 2 -name metrics.json | wc -l) -eq 15 ]]
echo COMPLETE > "$OUT/status.txt"
