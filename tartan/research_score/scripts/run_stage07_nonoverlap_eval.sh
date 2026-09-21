#!/usr/bin/env bash
set -euo pipefail
PROJECT=/zeron-vepfs/tjqc/cross-diffusion
BASE=/tj-share/cross_diffusion_workdir/research_score_v1_2
DATA="$BASE/01_data_audit/stage03_native_v123_remediation_final_20260911T170000Z_nogit/transfer_primary"
TRAIN="$BASE/05_transfer/formal_seed11_v3"
MANIFEST="$BASE/05_transfer/v124_nonoverlap8m_test_manifest.jsonl"
OUT="$BASE/05_transfer/formal_seed11_v3_eval_nonoverlap8m"
[[ -s "$MANIFEST" ]] || /root/miniconda3/envs/diffusion-planner/bin/python -m tartan.research_score.scripts.build_nonoverlap_segments --input-manifest "$DATA/target_test_windows.jsonl" --output "$MANIFEST"
mkdir -p "$OUT/logs"
jobs=()
for method in pretrain_finetune pretrain_adapter joint_train emb_cond_diffusion;do for budget in 1 10 100;do jobs+=("$method:$budget");done;done
run_one(){
 local spec=$1 gpu=$2 method budget run
 method=${spec%%:*};budget=${spec##*:};run="$OUT/${method}_${budget}pct_seed11"
 [[ -s "$run/summary.json" ]]&&return 0
 [[ ! -e "$run" ]]||{ echo "Incomplete eval exists: $run" >&2;return 2;}
 CUDA_VISIBLE_DEVICES=$gpu /root/miniconda3/envs/diffusion-planner/bin/python -u -m tartan.research_score.scripts.evaluate_nonoverlap_segments --method "$method" --checkpoint "$TRAIN/${method}_${budget}pct_seed11/best.pt" --args "$PROJECT/checkpoints/args.json" --test-manifest "$MANIFEST" --output "$run" >"$OUT/logs/${method}_${budget}pct_seed11.log" 2>&1
}
for ((start=0;start<${#jobs[@]};start+=4));do pids=();for ((i=start;i<${#jobs[@]}&&i<start+4;i++));do run_one "${jobs[$i]}" "$((i-start))"&pids+=("$!");done;failed=0;for p in "${pids[@]}";do wait "$p"||failed=1;done;[[ $failed -eq 0 ]]||exit 1;done
/root/miniconda3/envs/diffusion-planner/bin/python -m tartan.research_score.scripts.summarize_nonoverlap_segments
echo COMPLETE >"$OUT/status.txt"
