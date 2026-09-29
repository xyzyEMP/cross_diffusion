#!/usr/bin/env bash
set -euo pipefail
PROJECT=/zeron-vepfs/tjqc/cross-diffusion
BASE=/tj-share/cross_diffusion_workdir/research_score_v1_2
TRAIN="$BASE/05_transfer/finetune_equal100epochs_seed11"
OUT="$BASE/05_transfer/finetune_equal100epochs_seed11_eval_nonoverlap8m"
MANIFEST="$BASE/05_transfer/v124_nonoverlap8m_test_manifest.jsonl"
mkdir -p "$OUT/logs"
budgets=(1 10 20 50 100)
run_one(){ local budget=$1 gpu=$2; CUDA_VISIBLE_DEVICES=$gpu /root/miniconda3/envs/diffusion-planner/bin/python -u -m tartan.research_score.scripts.evaluate_nonoverlap_segments --method pretrain_finetune --checkpoint "$TRAIN/pretrain_finetune_${budget}pct_seed11/best.pt" --args "$PROJECT/checkpoints/args.json" --test-manifest "$MANIFEST" --output "$OUT/pretrain_finetune_${budget}pct_seed11" >"$OUT/logs/${budget}pct.log" 2>&1; }
for ((start=0;start<${#budgets[@]};start+=2)); do
 pids=(); for ((i=start;i<${#budgets[@]}&&i<start+2;i++)); do run_one "${budgets[$i]}" "$((i-start))" & pids+=("$!"); done
 failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; [[ $failed -eq 0 ]]
done
[[ $(find "$OUT" -mindepth 2 -maxdepth 2 -name summary.json | wc -l) -eq 5 ]]
echo COMPLETE > "$OUT/status.txt"
