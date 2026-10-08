#!/usr/bin/env bash
set -euo pipefail
PROJECT=${PROJECT_ROOT:-/zeron-vepfs/tjqc/cross-diffusion}
BASE=${OUTPUT_ROOT:-/tj-share/cross_diffusion_workdir}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/diffusion-planner/bin/python}
cd "$PROJECT"
if [[ ${PROFILE:-transfer_primary} == four_groups ]];then
 RUN_ID=${RUN_ID:?Use the saved explicit four-group RUN_ID}
 [[ "$RUN_ID" =~ ^[0-9]{8}T[0-9]{6}Z_[A-Za-z0-9_-]+$ ]] || exit 2
 RUN="$BASE/runs/$RUN_ID";CONFIG="$RUN/config.yaml"
 if [[ -s "$RUN/status.json" ]] && "$PYTHON_BIN" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("status")=="COMPLETE" else 1)' "$RUN/status.json";then exit 0;fi
 mkdir -p "$RUN/logs"
 export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
 "$PYTHON_BIN" - "$RUN" "$PROJECT" <<'PYSETUP'
import json,sys
from pathlib import Path
from tartan.research_score.artifacts import publish,publish_text
run,project=map(Path,sys.argv[1:]);p=run/'status.json'
if p.exists() and json.loads(p.read_text()).get('status')=='COMPLETE':raise SystemExit(0)
status={'DATA_IDS':{g:run.name+'_'+g for g in ('g1','g2','g3','g4')},'CATALOG_DATA_ID':'20261002T045353Z_proxy_cpu','status':'PREPARING_DATA','RUN_ID':run.name,'profile':'four_groups','pending':['g1','g2','g3','g4'],'next_command':'PROFILE=four_groups RUN_ID='+run.name+' bash tartan/research_score/scripts/run_transfer_training.sh'}
publish(status,p,True)
config_path=run/'config.yaml'
if not config_path.exists():publish_text((project/'tartan/research_score/configs/transfer_methods.yaml').read_text(),config_path)
protocol=(project/'TRAINING_PROTOCOL.md').read_text();publish_text(protocol.split('<!-- HISTORICAL_TRANSFER_PROTOCOL -->')[0],run/'experiment_description.md')
publish_text('PROFILE=four_groups RUN_ID='+run.name+' bash '+str(project/'tartan/research_score/scripts/run_transfer_training.sh')+'\n',run/'continue.sh')
PYSETUP
 trap 'code=$?; if [[ $code -ne 0 ]];then "$PYTHON_BIN" - "$RUN" "$code" <<"PYFAIL"
import json,sys
from pathlib import Path
from tartan.research_score.artifacts import publish
run=Path(sys.argv[1]);s=json.loads((run/"status.json").read_text());s.update(status="FAILED",exit_code=int(sys.argv[2]),logs=str(run/"logs"));publish(s,run/"status.json",True)
PYFAIL
fi' EXIT
 run_cmd() {
  local logfile=$1;shift
  { flock 200;printf '%q ' "$@" >> "$RUN/command.sh";printf '\n' >> "$RUN/command.sh"; } 200>"$RUN/.command.lock"
  "$@" >> "$RUN/logs/$logfile" 2>&1
 }
 for group in g1 g2 g3 g4;do
  export DATA_ID="${RUN_ID}_${group}"
  DATA="$BASE/data/$DATA_ID";CACHE="$BASE/cache/$DATA_ID";mkdir -p "$CACHE"
  if [[ ! -s "$DATA/trajectory_summary.json" ]];then run_cmd "${group}_prepare.log" "$PYTHON_BIN" -m tartan.research_score.scripts.build_transfer_manifests --profile four_groups --config "$CONFIG" --group "$group" --stage trajectories --output "$BASE/data" --run-id "$DATA_ID" --resume;fi
  if [[ ! -s "$DATA/window_summary.json" ]];then run_cmd "${group}_prepare.log" "$PYTHON_BIN" -m tartan.research_score.scripts.build_transfer_manifests --profile four_groups --config "$CONFIG" --group "$group" --stage windows --output "$BASE/data" --run-id "$DATA_ID" --resume;fi
  for split in base_train base_val test;do
   [[ -s "$CACHE/$split.pt" ]] || run_cmd "${group}_prepare.log" "$PYTHON_BIN" -m tartan.research_score.scripts.materialize_target_features --profile proxy_ab --manifest "$DATA/$split.jsonl" --args "$PROJECT/checkpoints/args.json" --output "$CACHE/$split.pt"
  done
  for split in val test;do
   input="$DATA/base_val.jsonl";[[ "$split" != test ]] || input="$DATA/test.jsonl"
   [[ -s "$DATA/navigation_$split.jsonl" ]] || run_cmd "${group}_prepare.log" "$PYTHON_BIN" -m tartan.research_score.scripts.build_navigation_tasks --profile proxy_ab --future-stations --input-manifest "$input" --output "$DATA/navigation_$split.jsonl"
  done
  [[ -s "$RUN/preflight/$group/summary.json" ]] || run_cmd "${group}_preflight.log" "$PYTHON_BIN" -m tartan.research_score.scripts.audit_navigation_routes --profile four_groups --manifest "$DATA/navigation_val.jsonl" --output "$RUN/preflight/$group"
  "$PYTHON_BIN" - "$RUN/preflight/$group/summary.json" "$DATA/window_summary.json" <<'PYGATE'
import json,sys
assert all(json.load(open(path))['status']=='PASS' for path in sys.argv[1:])
PYGATE
 done
 run_group() (
  set -e
  local group=$1 gpu=$2;export DATA_ID="${RUN_ID}_${group}" CUDA_VISIBLE_DEVICES=$gpu
  local data="$BASE/data/$DATA_ID" cache="$BASE/cache/$DATA_ID" output="$RUN/train/$group"
  local common=(--profile four_groups --config "$CONFIG" --method proxy_a --disable-history --train-cache "$cache/base_train.pt" --val-cache "$cache/base_val.pt" --val-navigation-manifest "$data/navigation_val.jsonl" --args "$PROJECT/checkpoints/args.json" --checkpoint "$PROJECT/checkpoints/model.pth" --seed 11 --device cuda --amp)
  if [[ ! -s "$RUN/smoke/$group/metrics.json" ]];then
   local smoke_extra=();[[ ! -s "$RUN/smoke/$group/last.pt" ]] || smoke_extra=(--resume "$RUN/smoke/$group/last.pt")
   run_cmd "${group}_smoke.log" "$PYTHON_BIN" -u -m tartan.research_score.scripts.train_transfer "${common[@]}" --output "$RUN/smoke/$group" --smoke gpu "${smoke_extra[@]}"
  fi
  "$PYTHON_BIN" - "$RUN/smoke/$group/metrics.json" <<'PYSMOKE'
import json,sys
s=json.load(open(sys.argv[1]));assert s['status']=='complete' and s['updates']==4 and not s['eligible_for_formal_result']
PYSMOKE
  local done=0
  if [[ -s "$output/metrics.json" ]];then done=$("$PYTHON_BIN" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["status"]=="complete"))' "$output/metrics.json");fi
  if [[ "$done" == 0 ]];then
   local resume=();[[ ! -s "$output/last.pt" ]] || resume=(--resume "$output/last.pt")
   run_cmd "${group}_train.log" "$PYTHON_BIN" -u -m tartan.research_score.scripts.train_transfer "${common[@]}" --output "$output" --smoke none "${resume[@]}"
  fi
  for kind in offline navigation;do
   local evalout="$RUN/eval/$group/$kind" manifest="$data/test.jsonl";local extra=(--test-cache "$cache/test.pt")
   if [[ "$kind" == navigation ]];then manifest="$data/navigation_test.jsonl";extra=();fi
   [[ -s "$evalout/summary.json" ]] || run_cmd "${group}_${kind}.log" "$PYTHON_BIN" -u -m tartan.research_score.scripts.evaluate_navigation --profile four_groups --config "$CONFIG" --method proxy_a --checkpoint "$output/navigation_best.pt" --args "$PROJECT/checkpoints/args.json" --device cuda --evaluation-kind "$kind" --test-manifest "$manifest" --output "$evalout" "${extra[@]}"
  done
 )
 "$PYTHON_BIN" - "$RUN" <<'PYSTART'
import json,sys
from pathlib import Path
from tartan.research_score.artifacts import publish
run=Path(sys.argv[1]);s=json.loads((run/'status.json').read_text());s['status']='GPU_RUNNING';publish(s,run/'status.json',True)
PYSTART
 # Two GPUs, one process per GPU. Complete each pair before the next pair.
 for pair in 'g1 g2' 'g3 g4';do
  read -r first second <<< "$pair"
  run_group "$first" 0 & pid0=$!
  run_group "$second" 1 & pid1=$!
  failed=0;wait "$pid0" || failed=1;wait "$pid1" || failed=1
  [[ "$failed" == 0 ]] || exit 1
 done
 "$PYTHON_BIN" -m tartan.research_score.scripts.summarize_navigation --profile four_groups --run-root "$RUN"
 exit 0
fi
if [[ ${PROFILE:-transfer_primary} == proxy_ab ]]; then
 RUN_ID=${RUN_ID:?Use the saved explicit RUN_ID}
 DATA_ID=${DATA_ID:?Use the frozen DATA_ID}
 RUN_KIND=${RUN_KIND:?Set cpu_smoke, gpu_smoke, or formal}
 [[ "$RUN_ID" =~ ^[0-9]{8}T[0-9]{6}Z_[A-Za-z0-9_-]+$ && "$DATA_ID" =~ ^[0-9]{8}T[0-9]{6}Z_[A-Za-z0-9_-]+$ ]] || exit 2
 case "$RUN_KIND" in
  cpu_smoke) DEVICE=${DEVICE:-cpu};SMOKE=cpu;TRAIN_DIR=cpu_smoke;;
  gpu_smoke) DEVICE=${DEVICE:-cuda};SMOKE=gpu;TRAIN_DIR=train;;
  formal) DEVICE=${DEVICE:-cuda};SMOKE=none;TRAIN_DIR=train;;
  *) echo 'Unknown RUN_KIND' >&2;exit 2;;
 esac
 [[ "$RUN_KIND" == cpu_smoke && "$DEVICE" == cpu || "$RUN_KIND" != cpu_smoke && "$DEVICE" == cuda ]] || { echo 'Phase/device mismatch' >&2;exit 2; }
 RUN="$BASE/runs/$RUN_ID";DATA="$BASE/data/$DATA_ID";CACHE="$BASE/cache/$DATA_ID";PAIRS="$BASE/pairs/$DATA_ID"
 PROXY_CONFIG=${PROXY_CONFIG:-$PROJECT/tartan/research_score/configs/transfer_methods.yaml}
 mkdir -p "$RUN/$TRAIN_DIR" "$RUN/logs"
 printf '%s\n' "$RUN_ID" > "$RUN/run_id.txt"
 export CUDA_VISIBLE_DEVICES=${GPU_IDS:-0}
 run_proxy() {
  local method=$1 output=$2 stop=${3:-} resume=${4:-${RESUME:-0}}
  local cmd=("$PYTHON_BIN" -u -m tartan.research_score.scripts.train_transfer --profile proxy_ab --config "$PROXY_CONFIG" --method "$method" --train-cache "$CACHE/base_train.pt" --val-cache "$CACHE/base_val.pt" --val-navigation-manifest "$DATA/navigation_val.jsonl" --args "$PROJECT/checkpoints/args.json" --checkpoint "${SOURCE_CKPT:-$PROJECT/checkpoints/model.pth}" --output "$output" --seed 11 --device "$DEVICE" --smoke "$SMOKE")
  [[ "$DEVICE" == cpu ]] || cmd+=(--amp)
  if [[ "$method" == proxy_b ]];then cmd+=(--pair-manifest "$PAIRS/train.jsonl" --pair-cache "$CACHE/pair_train.pt" --pair-val-manifest "$PAIRS/val.jsonl" --pair-val-cache "$CACHE/pair_val.pt");fi
  if [[ "$resume" == 1 ]];then [[ -s "$output/last.pt" ]] || { echo "Missing resume checkpoint: $output" >&2;return 2; };cmd+=(--resume "$output/last.pt");fi
  [[ -z "$stop" ]] || cmd+=(--stop-after "$stop")
  printf '%q ' "${cmd[@]}" >> "$RUN/command.sh";printf '\n' >> "$RUN/command.sh"
  "${cmd[@]}" > "$RUN/logs/$(basename "$output")_${method}.log" 2>&1
 }
 "$PYTHON_BIN" - "$RUN" "$DATA_ID" "$RUN_KIND" "$TRAIN_DIR" "$PROXY_CONFIG" <<'PYCONFIG'
import json,sys
from pathlib import Path
from tartan.research_score.artifacts import publish
root=Path(sys.argv[1]);data_id,kind,train_dir,config_file=sys.argv[2:]
tasks=[{'method':m,'seed':11,'directory':str(root/train_dir/(m+'_seed11')),'status':'pending'} for m in ('proxy_a','proxy_b')]
payload={'profile':'proxy_ab','DATA_ID':data_id,'RUN_KIND':kind,'tasks':tasks}
path=root/'task_matrix.json'
if path.exists():
 old=json.loads(path.read_text())
 normalized={**old,'tasks':[{k:v for k,v in t.items() if k in ('method','seed','directory','status')} for t in old['tasks']]}
 for t in normalized['tasks']:t['status']='pending'
 if normalized!=payload:raise ValueError('existing task matrix differs')
else:publish(payload,path,True)
status_path=root/'status.json';status=json.loads(status_path.read_text()) if status_path.exists() else {};status.update({'RUN_ID':root.name,'DATA_ID':data_id,'run_kind':kind,'runner_status':'RUNNING'});publish(status,status_path,True)
from tartan.research_score.scripts.train_transfer import proxy_config
from tartan.research_score.artifacts import publish
config_path=root/'config.json';config=json.loads(config_path.read_text()) if config_path.exists() else {}
execution={'profile':'proxy_ab','RUN_KIND':kind,'DATA_ID':data_id,'proxy_ab':proxy_config(config_file),'tasks':[{k:v for k,v in t.items() if k in ('method','seed','directory')} for t in tasks]}
if 'proxy_execution' in config and config['proxy_execution']!=execution:raise ValueError('run execution config differs')
config['proxy_execution']=execution;publish(config,config_path,True)
PYCONFIG
 trap 'code=$?; if [[ $code -ne 0 ]];then "$PYTHON_BIN" - "$RUN" "$code" <<"PYFAIL"
import json,sys
from pathlib import Path
from tartan.research_score.artifacts import publish
root=Path(sys.argv[1]);p=root/"status.json";s=json.loads(p.read_text()) if p.exists() else {};s["runner_status"]="FAILED";s["runner_exit_code"]=int(sys.argv[2]);s["runner_logs"]=str(root/"logs");publish(s,p,True)
PYFAIL
fi' EXIT
 for method in proxy_a proxy_b;do
  output="$RUN/$TRAIN_DIR/${method}_seed11"
  if [[ "$method" == proxy_b && ! -s "$PAIRS/train.jsonl" ]];then
   [[ -f "$PAIRS/train.jsonl" && -f "$PAIRS/gate_summary.json" ]] || { echo 'Missing pair gate artifacts' >&2;exit 2; }
   [[ "$RUN_KIND" == cpu_smoke ]] || { echo 'BLOCKED_NO_TRAIN_PAIRS' >&2;exit 2; }
   continue
  fi
  if [[ -f "$output/metrics.json" ]];then
   if "$PYTHON_BIN" - "$output" "$SMOKE" <<'PYCOMPLETE'
import json,sys,torch
from pathlib import Path
root=Path(sys.argv[1]);m=json.loads((root/'metrics.json').read_text());z=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
assert m['status']=='complete' and m['updates']==z['training_state']['update']
assert z['config']==json.loads((root/'config.json').read_text())
assert z['config']['cli']['smoke']==sys.argv[2]
if sys.argv[2]!='cpu':assert (root/'navigation_best.pt').is_file()
PYCOMPLETE
   then continue;fi
  fi
  resume_task=0;[[ ! -s "$output/last.pt" ]] || resume_task=1
  run_proxy "$method" "$output" "" "$resume_task"
 done
 if [[ "$RUN_KIND" == cpu_smoke ]];then
  for method in proxy_a proxy_b;do
   [[ -s "$RUN/$TRAIN_DIR/${method}_seed11/last.pt" ]] || continue
   comparison="$RUN/$TRAIN_DIR/${method}_resume_checks.json"
   if [[ -s "$comparison" ]] && "$PYTHON_BIN" - "$comparison" <<'PYALREADYCHECKED'
import json,sys
raise SystemExit(json.load(open(sys.argv[1])).get('status')!='PASS')
PYALREADYCHECKED
   then continue;fi
   resumed="$RUN/$TRAIN_DIR/${method}_resumed"
   if [[ ! -s "$resumed/last.pt" ]];then run_proxy "$method" "$resumed" 1 0;fi
   if [[ ! -s "$resumed/metrics.json" ]] || ! "$PYTHON_BIN" - "$resumed/metrics.json" <<'PYRESUMED'
import json,sys
raise SystemExit(json.load(open(sys.argv[1]))['status']!='complete')
PYRESUMED
   then run_proxy "$method" "$resumed" '' 1;fi
   "$PYTHON_BIN" -m tartan.research_score.scripts.train_transfer --compare-checkpoints "$RUN/$TRAIN_DIR/${method}_seed11/last.pt" "$resumed/last.pt" --comparison-output "$RUN/$TRAIN_DIR/${method}_resume_checks.json"
  done
 fi
 if [[ "$RUN_KIND" == gpu_smoke ]];then
  # One actual stop/restart counterfactual using the same frozen inputs and streams.
  resumed="$RUN/train/resumed"
  if [[ ${RESUME:-0} != 1 ]];then run_proxy proxy_b "$resumed" 2 0;fi
  run_proxy proxy_b "$resumed" '' 1
  "$PYTHON_BIN" -m tartan.research_score.scripts.train_transfer --compare-checkpoints "$RUN/train/proxy_b_seed11/last.pt" "$resumed/last.pt" --comparison-output "$RUN/smoke_checks.json"
 fi
 "$PYTHON_BIN" - "$RUN" "$RUN_KIND" "$PROXY_CONFIG" <<'PYSTATUS'
import json,sys
from pathlib import Path
from tartan.research_score.artifacts import publish
from tartan.research_score.scripts.train_transfer import proxy_config
root=Path(sys.argv[1]);kind=sys.argv[2];matrix=json.loads((root/'task_matrix.json').read_text());cfg=proxy_config(sys.argv[3]);complete=True
for task in matrix['tasks']:
 folder=Path(task['directory'])
 if kind=='cpu_smoke' and task['method']=='proxy_b' and not (folder/'metrics.json').exists():
  task['status']='BLOCKED_NO_TRAIN_PAIRS';task['gate_summary']=str(root.parent.parent/'pairs'/matrix['DATA_ID']/'gate_summary.json');complete=False;continue
 metrics=json.loads((folder/'metrics.json').read_text());task['status']=metrics['status'];task['metrics']=str(folder/'metrics.json');task['last']=str(folder/'last.pt');complete &= metrics['status']=='complete'
 if kind=='cpu_smoke':
  checks=json.loads((folder/'checks.json').read_text());complete &= checks['strict_reload_equal'] and bool(checks['second_step_gradients'])
  if task['method']=='proxy_b':complete &= checks['sampler_finite']
  task['checks']=str(folder/'checks.json')
  resume_checks=root/'cpu_smoke'/(task['method']+'_resume_checks.json');complete &= json.loads(resume_checks.read_text())['status']=='PASS';task['resume_checks']=str(resume_checks)
# Completion comparisons use task identity, not mutable evidence fields.
publish(matrix,root/'task_matrix.json',True)
status=json.loads((root/'status.json').read_text());status['runner_status']='COMPLETE' if complete else 'INTERRUPTED'
if complete:status['runner_exit_code']=0
if any(t['status']=='BLOCKED_NO_TRAIN_PAIRS' for t in matrix['tasks']):status['status']='CPU_A_COMPLETE_B_BLOCKED_NO_TRAIN_PAIRS';status['runner_status']='BLOCKED_NO_TRAIN_PAIRS'
if complete:
 if kind=='cpu_smoke':status['status']='CPU_READY_GPU_PENDING'
 elif kind=='gpu_smoke':
  assert json.loads((root/'smoke_checks.json').read_text())['status']=='PASS';status['status']='GPU_SMOKE_PASSED'
 else:status['status']='TRAINING_COMPLETE_TEST_PENDING'
status['tasks']=matrix['tasks'];publish(status,root/'status.json',True)
config_path=root/'config.json';config=json.loads(config_path.read_text()) if config_path.exists() else {}
execution={'profile':'proxy_ab','RUN_KIND':kind,'DATA_ID':matrix['DATA_ID'],'proxy_ab':cfg,'tasks':[{k:v for k,v in t.items() if k in ('method','seed','directory')} for t in matrix['tasks']]}
if 'proxy_execution' in config and config['proxy_execution']!=execution:raise ValueError('run execution config differs')
config['proxy_execution']=execution;publish(config,config_path,True)
PYSTATUS
 exit 0
fi
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_transfer_seed11}
[[ "$RUN_ID" =~ ^[0-9]{8}T[0-9]{6}Z_[A-Za-z0-9_-]+$ ]] || { echo 'Invalid RUN_ID: use YYYYMMDDTHHMMSSZ_ascii_label' >&2; exit 2; }
TRAIN_CACHE=${TRAIN_CACHE:-$BASE/cache/target_train_features_d029_extended.pt}
VAL_NAVIGATION_MANIFEST=${VAL_NAVIGATION_MANIFEST:-$BASE/data/navigation/val.jsonl}
OUT="$BASE/runs/$RUN_ID/train"
mkdir -p "$OUT/logs"
printf '%s\n' "$RUN_ID" > "$BASE/runs/$RUN_ID/run_id.txt"
echo "Run: $BASE/runs/$RUN_ID"
jobs=()
for method in pretrain_finetune pretrain_adapter joint_train emb_cond_diffusion; do
 for budget in 1 10 100; do jobs+=("$method:$budget"); done
done
run_one() {
 local spec=$1 gpu=$2 method budget run
 method=${spec%%:*};budget=${spec##*:};run="$OUT/${method}_${budget}pct_seed11"
 if [[ -s "$run/metrics.json" && -s "$run/navigation_best.pt" ]]; then return 0; fi
 [[ ! -e "$run" ]] || { echo "Incomplete run exists: $run" >&2; return 2; }
 CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" -u -m tartan.research_score.scripts.train_transfer \
  --method "$method" --train-cache "$TRAIN_CACHE" --val-cache "$BASE/cache/target_val_features.pt" \
  --val-navigation-manifest "$VAL_NAVIGATION_MANIFEST" --source-cache "$BASE/cache/source_car_features_4096.pt" \
  --args "$PROJECT/checkpoints/args.json" --checkpoint "$PROJECT/checkpoints/model.pth" \
  --output "$run" --budget "$budget" --max-target-updates 10000 --min-target-updates 5000 \
  --val-every-updates 250 --patience-validations 10 --batch-size 64 --seed 11 --amp \
  >"$OUT/logs/${method}_${budget}pct_seed11.log" 2>&1
}
for ((start=0;start<${#jobs[@]};start+=4)); do
 pids=();for ((i=start;i<${#jobs[@]} && i<start+4;i++)); do run_one "${jobs[$i]}" "$((i-start))" & pids+=("$!"); done
 failed=0;for pid in "${pids[@]}";do wait "$pid" || failed=1;done
 [[ $failed -eq 0 ]] || exit 1
done
[[ $(find "$OUT" -mindepth 2 -maxdepth 2 -name metrics.json | wc -l) -eq ${#jobs[@]} ]]
echo COMPLETE > "$OUT/status.txt"
