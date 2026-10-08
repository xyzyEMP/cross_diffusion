#!/usr/bin/env bash
set -euo pipefail
PROJECT=${PROJECT_ROOT:-/zeron-vepfs/tjqc/cross-diffusion}
BASE=${OUTPUT_ROOT:-/tj-share/cross_diffusion_workdir}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/diffusion-planner/bin/python}
cd "$PROJECT"
RUN_ID=${RUN_ID:?Set RUN_ID to the training run ID printed by run_transfer_training.sh}
[[ "$RUN_ID" =~ ^[0-9]{8}T[0-9]{6}Z_[A-Za-z0-9_-]+$ ]] || { echo 'Invalid RUN_ID' >&2; exit 2; }
if [[ ${PROFILE:-transfer_primary} == proxy_ab ]];then
 RUN="$BASE/runs/$RUN_ID"
 DATA_ID=$("$PYTHON_BIN" - "$RUN/task_matrix.json" <<'PYDATA'
import json,sys
x=json.load(open(sys.argv[1]));assert x['RUN_KIND']=='formal';print(x['DATA_ID'])
PYDATA
 )
 export DATA_ID
 DATA="$BASE/data/$DATA_ID";CACHE="$BASE/cache/$DATA_ID"
 PROXY_CONFIG=${PROXY_CONFIG:-$PROJECT/tartan/research_score/configs/transfer_methods.yaml}
 DEVICE=${DEVICE:-cuda};export CUDA_VISIBLE_DEVICES=${GPU_IDS:-0}
 mkdir -p "$RUN/eval/logs"
 for method in proxy_a proxy_b;do
  checkpoint="$RUN/train/${method}_seed11/navigation_best.pt"
  "$PYTHON_BIN" - "$RUN/train/${method}_seed11/metrics.json" "$checkpoint" <<'PYTRAIN'
import json,sys
from pathlib import Path
m=json.load(open(sys.argv[1]));assert m['status']=='complete' and m['eligible_for_formal_result'];assert Path(sys.argv[2]).is_file()
PYTRAIN
  "$PYTHON_BIN" - "$RUN" <<'PYTESTSTART'
import json,sys
from pathlib import Path
from tartan.research_score.artifacts import publish
root=Path(sys.argv[1]);p=root/'status.json';s=json.loads(p.read_text())
s.update(status='ANYMAL_FINAL_EVALUATION_RUNNING',anymal_final_evaluation_started=True)
publish(s,p,True)
PYTESTSTART
  for kind in offline navigation;do
   output="$RUN/eval/$method/$kind"
   [[ ! -s "$output/summary.json" ]] || continue
   manifest="$DATA/anymal_test.jsonl";extra=(--test-cache "$CACHE/anymal_test.pt" --diagnostic-train-cache "$CACHE/base_train.pt" --diagnostic-val-cache "$CACHE/base_val.pt")
   if [[ "$kind" == navigation ]];then manifest="$DATA/navigation_anymal_test.jsonl";extra=();[[ -s "$manifest" ]] || { echo 'Navigation final test remains unavailable' >&2;continue; };fi
   cmd=("$PYTHON_BIN" -u -m tartan.research_score.scripts.evaluate_navigation --profile proxy_ab --config "$PROXY_CONFIG" --method "$method" --checkpoint "$checkpoint" --args "$PROJECT/checkpoints/args.json" --device "$DEVICE" --evaluation-kind "$kind" --test-manifest "$manifest" --output "$output" "${extra[@]}")
   printf '%q ' "${cmd[@]}" >> "$RUN/command.sh";printf '\n' >> "$RUN/command.sh"
   "${cmd[@]}" > "$RUN/eval/logs/${method}_${kind}.log" 2>&1
  done
 done
 "$PYTHON_BIN" -m tartan.research_score.scripts.summarize_navigation --profile proxy_ab --run-root "$RUN" --task-matrix "$RUN/task_matrix.json"
 "$PYTHON_BIN" - "$RUN" <<'PYEVALSTATUS'
import json,sys
from pathlib import Path
from tartan.research_score.artifacts import publish
root=Path(sys.argv[1]);summary=json.loads((root/'summary.json').read_text());p=root/'status.json';state=json.loads(p.read_text());state['status']=summary['status'];state['final_report']=str(root/'report.md');state['final_summary']=str(root/'summary.json');publish(state,p,True)
PYEVALSTATUS
 exit 0
fi
TRAIN="$BASE/runs/$RUN_ID/train"
MANIFEST=${TEST_NAVIGATION_MANIFEST:-$BASE/data/navigation/test.jsonl}
OUT="$BASE/runs/$RUN_ID/eval"
[[ -s "$TRAIN/status.txt" && -s "$MANIFEST" ]] || { echo 'Completed training and frozen test manifest required' >&2; exit 2; }
mkdir -p "$OUT/logs"
jobs=();for method in pretrain_finetune pretrain_adapter joint_train emb_cond_diffusion;do for budget in 1 10 100;do jobs+=("$method:$budget");done;done
run_one(){
 local spec=$1 gpu=$2 method budget run
 method=${spec%%:*};budget=${spec##*:};run="$OUT/${method}_${budget}pct_seed11"
 [[ -s "$run/summary.json" ]] && return 0
 [[ ! -e "$run" ]] || { echo "Incomplete eval exists: $run" >&2; return 2; }
 CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" -u -m tartan.research_score.scripts.evaluate_navigation \
  --method "$method" --checkpoint "$TRAIN/${method}_${budget}pct_seed11/navigation_best.pt" \
  --args "$PROJECT/checkpoints/args.json" --test-manifest "$MANIFEST" --output "$run" \
  >"$OUT/logs/${method}_${budget}pct_seed11.log" 2>&1
}
for ((start=0;start<${#jobs[@]};start+=4));do
 pids=();for ((i=start;i<${#jobs[@]}&&i<start+4;i++));do run_one "${jobs[$i]}" "$((i-start))" & pids+=("$!");done
 failed=0;for pid in "${pids[@]}";do wait "$pid" || failed=1;done;[[ $failed -eq 0 ]] || exit 1
done
"$PYTHON_BIN" -m tartan.research_score.scripts.summarize_navigation --train "$TRAIN" --eval "$OUT"
echo COMPLETE > "$OUT/status.txt"
