import argparse, hashlib, json, math
from collections import Counter
from pathlib import Path
REV="score-decomp-transfer-v1.2.3";Z=1.6448536269514722
def js(p):return json.loads(p.read_text())
def rows(p):
 with p.open() as h:return [json.loads(x) for x in h if x.strip()]
def wl(k,n):
 if not n:return 0.
 q=k/n;d=1+Z*Z/n;return (q+Z*Z/(2*n)-Z*math.sqrt(q*(1-q)/n+Z*Z/(4*n*n)))/d
def main():
 p=argparse.ArgumentParser();p.add_argument("--profile",choices=["transfer","proxy_ab"],default="transfer");p.add_argument("--input");p.add_argument("--report");p.add_argument("--run");p.add_argument("--output");a=p.parse_args()
 if a.profile=="proxy_ab":
  if not a.input or not a.report:p.error("Proxy validation requires --input and --report")
  validate_proxy_files(Path(a.input),Path(a.report));return
 if not a.run or not a.output:p.error("Transfer validation requires --run and --output")
 run=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);checks=[]
 def ck(name,ok,detail):checks.append({"name":name,"passed":bool(ok),"detail":detail})
 rm=js(run/"run_manifest.json");sel=js(run/"selected_length.json");freeze=js(run/"source_audit_freeze.json");profile=js(run/"transfer_primary/profile_manifest.json");reject=js(run/"rejected_samples.json")
 ids=(run/"source_length_audit_ids.txt").read_text().splitlines();calc=hashlib.sha256(("\n".join(ids)+"\n").encode()).hexdigest();ck("revision",all(x.get("protocol_revision")==REV for x in (rm,sel,freeze,profile)),REV);ck("source_index_population",rm["native_index_rows"]==profile["source_index_rows"]==freeze["population_size"]==1_000_000,rm["native_index_rows"]);ck("audit_frozen_50k",len(ids)==len(set(ids))==freeze["audit_size"]==profile["source_length_audit_rows"]==50_000,{"rows":len(ids),"unique":len(set(ids))});ck("audit_hash",calc==freeze["sample_ids_sha256"]==profile["source_length_audit_sha256"]==rm["length_audit_sha256"],calc);ck("frozen_before_outcomes",freeze["frozen_before_outcomes"] and freeze["algorithm"]=="normalized-path-sha256-smallest-v1",freeze["algorithm"])
 audit=rows(run/"transfer_primary/source_length_audit.jsonl");ck("audit_accounting",len(audit)+len(reject["source_native_npz"])==50_000,{"valid":len(audit),"rejected":len(reject["source_native_npz"])});ck("native_80x3",all(r["trajectory"]["shape"]==[80,3] and abs(r["physical_horizon_s"]-8)<1e-9 for r in audit),len(audit))
 eligible=[]
 for key,v in sel["coverage"].items():
  s=v["source_car"];t=v["target_anymal"];calcwl=wl(s["covered"],s["denominator"]);ck(f"wilson_{key}",abs(calcwl-s["wilson_lower_95_one_sided"])<1e-12,{"stored":s["wilson_lower_95_one_sided"],"calc":calcwl});
  if calcwl>=sel["min_train_coverage"] and t["exact_coverage"]>=sel["min_train_coverage"]:eligible.append(float(key))
 ck("largest_eligible",bool(eligible) and sel["selected_length_m"]==max(eligible),{"eligible":eligible,"selected":sel["selected_length_m"]})
 splitrows={s:rows(run/f"transfer_primary/target_{s}_windows.jsonl") for s in ("train","val","test")};eps={s:{r["episode_id"] for r in z} for s,z in splitrows.items()};ck("episode_disjoint",not(eps["train"]&eps["val"] or eps["train"]&eps["test"] or eps["val"]&eps["test"]),{s:len(x) for s,x in eps.items()});alltarget=sum(splitrows.values(),[]);ck("target_schema",all(r["route_set"]["future_gt_dependency"] is False and "frozen" in r["fixed_goal"]["rule"] and r["branch"] in {"moving_planning","stop_or_short"} for r in alltarget),len(alltarget))
 import numpy as np
 bad_repr=[]
 for r in alltarget:
  x=np.asarray(r["trajectory"].get("fixed_arc_length_80",[]),dtype=float);m=np.asarray(r["trajectory"].get("valid_mask",[]))
  if x.shape!=(80,4) or m.shape!=(80,) or not np.isfinite(x).all() or (x.shape==(80,4) and not np.allclose(x[:,2]**2+x[:,3]**2,1.0,atol=2e-4)):bad_repr.append(r["sample_id"])
 ck("target_fixed_arc_xy_cos_sin",not bad_repr,{"bad_count":len(bad_repr),"examples":bad_repr[:5]})
 summary=js(run/"split_summary.json");bad=[]
 ck("stratified_episode_split",summary.get("algorithm")=="branch-stratified-complete-episode-sha256-v1" and all(any(r["branch"]=="moving_planning" for r in splitrows[s]) for s in ("train","val","test")),summary.get("algorithm"))
 fail=[x for x in checks if not x["passed"]];result={"status":"PASS" if not fail else "FAIL","checks_passed":len(checks)-len(fail),"checks_failed":len(fail),"source_audit":dict(Counter(r["branch"] for r in audit)),"target":dict(Counter(r["branch"] for r in alltarget)),"split_rows":{s:len(v) for s,v in splitrows.items()},"checks":checks,"failures":fail};(out/"validation.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps({k:result[k] for k in ("status","checks_passed","checks_failed","source_audit","target","split_rows")},indent=2));raise SystemExit(bool(fail))



def proxy_checks(trajectories, manifests):
 """Validate frozen Proxy inputs; no model execution or performance selection."""
 import numpy as np
 from tartan.research_score.data.core import proxy_split
 checks=[]
 def ck(name,ok,detail):checks.append({'name':name,'passed':bool(ok),'detail':detail})
 keys=[r.get('trajectory_key') for r in trajectories]
 ck('trajectory_keys_unique',len(keys)==len(set(keys)) and all(keys),len(keys))
 index={r.get('trajectory_key'):r for r in trajectories}
 expected={}
 try:expected={r['trajectory_key']:r['split'] for r in proxy_split(trajectories)}
 except (ValueError,KeyError) as e:ck('split_population',False,str(e))
 bad_split=[]
 for r in trajectories:
  key=r.get('trajectory_key');platform=r.get('embodiment')
  if (key!=f"{r.get('map_id')}/{platform}/{r.get('trajectory_id')}" or r.get('split')!=expected.get(key)
      or r.get('split_seed')!=20260911 or r.get('split_algorithm')!='per-platform-sorted-python-random-shuffle'
      or (platform=='anymal' and r.get('split')!='test') or platform not in ('diff','omni','anymal')):bad_split.append(key)
 ck('frozen_full_trajectory_split',not bad_split and bool(trajectories),bad_split)
 blocked_frame=[r['trajectory_key'] for r in trajectories if not (r.get('gates',{}).get('reference_heading',r.get('gates',{}).get('body_heading',False)) and r.get('gates',{}).get('occupancy_frame') and (r.get('reference_heading_source') or r.get('body_heading_source')) and r.get('frame_convention') and r.get('frame_evidence'))]
 blocked_time=[]
 for r in trajectories:
  rate=r.get('sample_rate_hz')
  if not r.get('gates',{}).get('time') or not r.get('time_source') or rate is None or not np.isfinite(rate) or rate<=0:blocked_time.append(r['trajectory_key'])
 ck('reference_heading_and_occupancy_evidence',not blocked_frame,blocked_frame)
 ck('timestamp_evidence',not blocked_time,blocked_time)
 bad_poses=[r.get('trajectory_key') for r in trajectories if not r.get('pose_path') or not r.get('metadata_path') or not r.get('occupancy_dir') or r.get('pose_count',0)<22]
 ck('trajectory_source_fields',not bad_poses,bad_poses)
 seen=set();bad_windows=[];coverage=Counter();short=Counter()
 for name,windows in manifests.items():
  split={'base_train':'train','base_val':'val','anymal_test':'test'}[name]
  for w in windows:
   sid=w.get('sample_id');key=w.get('trajectory_key');r=index.get(key)
   reasons=[]
   if sid in seen:reasons.append('duplicate_sample_id')
   seen.add(sid)
   if r is None:reasons.append('missing_frozen_trajectory')
   else:
    anchor=w.get('anchor_index',-1)
    if not isinstance(anchor,int):
     bad_windows.append({'sample_id':sid,'reasons':['invalid_anchor_index']});continue
    if (w.get('split')!=split or r['split']!=split or w.get('episode_id')!=key or w.get('embodiment')!=r['embodiment'] or w.get('platform_id')!=r.get('platform_id') or w.get('map_id')!=r['map_id']):reasons.append('source_identity_or_split')
    if sid!=f'{key}:anchor:{anchor:06d}' or anchor not in range(20,r['pose_count']-1,10):reasons.append('anchor_membership')
    if w.get('history_start_frame')!=anchor-20 or w.get('history_end_frame')!=anchor-1 or w.get('history_policy')!='past20_anchor_exclusive':reasons.append('history_range')
    t=w.get('trajectory',{})
    if t.get('raw_reference')!=r['pose_path'] or t.get('source_start_frame')!=anchor or not(anchor<t.get('source_end_frame',-1)<r['pose_count']):reasons.append('raw_provenance')
    expected_map=str(Path(r['occupancy_dir'])/f'occupancy_coarse5_{anchor:06d}_sparse.npy')
    if w.get('route_set',{}).get('map_reference')!=expected_map:reasons.append('anchor_occupancy_provenance')
    if w.get('time_source')!=r.get('time_source') or w.get('body_heading_source')!=r.get('body_heading_source') or w.get('reference_pose_policy')!=r.get('reference_pose_policy') or w.get('reference_heading_source')!=r.get('reference_heading_source') or w.get('frame_convention')!=r.get('frame_convention'):reasons.append('source_evidence_mismatch')
   try:
    t=w['trajectory'];x=np.asarray(t['fixed_arc_length_80'],float);mask=np.asarray(t['valid_mask'])
    if x.shape!=(80,4) or mask.shape!=(80,) or not np.isfinite(x).all() or not np.isin(mask,[0,1]).all() or not mask.any():reasons.append('shape_finite_mask')
    elif not np.allclose((x[:,2:]**2).sum(-1),1,atol=2e-4):reasons.append('heading_unit_vector')
    else:
     if not mask.all():short[name]+=1
     if np.any(np.diff(mask.astype(int))>0):reasons.append('mask_not_prefix')
     measured=t.get('measured_arc_m',-1)
     if not 0<=measured<=8 or not np.array_equal(mask.astype(bool),np.linspace(0,8,80)<=measured+1e-9):reasons.append('arc_coverage_mask')
     if not np.allclose(x[0],[0,0,1,0],atol=1e-4):reasons.append('anchor_origin')
    if t.get('length_m')!=8 or t.get('arc_metric')!='xy' or not np.allclose(t.get('stations_m',[]),np.linspace(0,8,80)):reasons.append('fixed_xy_arc_policy')
    if w['route_set'].get('future_gt_dependency') is not False or not w['route_set'].get('map_reference') or 'frozen' not in w['fixed_goal'].get('rule',''):reasons.append('route_goal_policy')
    h=w['history']
    for field,shape in [('ego_history',(20,4)),('history_mask',(20,)),('history_dt',(19,)),('history_dt_mask',(19,)),('motion_rms',(3,)),('motion_mask',(3,))]:
     value=np.asarray(h[field])
     if value.shape!=shape or not np.isfinite(value).all():reasons.append('history_'+field)
    dt=np.asarray(h['history_dt']);dtmask=np.asarray(h['history_dt_mask'],bool)
    if np.any(dt[dtmask]<=0) or np.any(np.asarray(h['motion_rms'])<0):reasons.append('history_time_or_rms_sign')
    if r is not None and not r.get('gates',{}).get('time') and (np.asarray(h['history_dt_mask']).any() or np.asarray(h['motion_mask']).any()):reasons.append('unknown_time_claims_motion')
   except (KeyError,TypeError,ValueError) as e:reasons.append('schema:'+str(e))
   if reasons:bad_windows.append({'sample_id':sid,'reasons':reasons})
   else:coverage[key]+=1
 ck('window_schema_source_and_history',not bad_windows,bad_windows)
 uncovered=[key for key in keys if not coverage[key]]
 ck('full_trajectory_coverage',not uncovered,uncovered)
 ck('base_train_and_val_nonempty',bool(manifests.get('base_train')) and bool(manifests.get('base_val')),{n:len(v) for n,v in manifests.items()})
 failed=[c for c in checks if not c['passed']]
 status='PASS' if not failed else ('BLOCKED_FRAME' if blocked_frame else 'BLOCKED_TIME' if blocked_time else 'BLOCKED_MANIFEST')
 return {'profile':'proxy_ab','status':status,'checks':checks,'failures':failed,'checks_passed':len(checks)-len(failed),'checks_failed':len(failed),
         'trajectory_counts':dict(Counter(r.get('embodiment') for r in trajectories)), 'split_counts':dict(Counter(r.get('split') for r in trajectories)),
         'window_counts':{n:len(v) for n,v in manifests.items()},'trajectory_window_coverage':dict(coverage),'uncovered_trajectories':uncovered,'short_tail_windows':dict(short),
         'blocked_evidence':{r['trajectory_key']:{'gates':r.get('gates'),'reasons':r.get('gate_reasons'),'time_source':r.get('time_source'),'body_heading_source':r.get('body_heading_source'),'frame_evidence':r.get('frame_evidence')} for r in trajectories if r['trajectory_key'] in blocked_frame+blocked_time}}


def validate_proxy_files(input_dir,report):
 from tartan.research_score.artifacts import publish
 names=['base_train','base_val','anymal_test']
 from tartan.data.pose_utils import read_proxy_trajectories
 trajectories=read_proxy_trajectories(input_dir/'trajectories.jsonl') if (input_dir/'trajectories.jsonl').is_file() else []
 manifests={name:rows(input_dir/(name+'.jsonl')) if (input_dir/(name+'.jsonl')).is_file() else [] for name in names}
 result=proxy_checks(trajectories,manifests)
 missing=[name for name in ['trajectories.jsonl','config.json']+[n+'.jsonl' for n in names] if not (input_dir/name).is_file()]
 source_missing=[{'trajectory_key':r['trajectory_key'],'path':r.get(field)} for r in trajectories for field in ('pose_path','metadata_path','occupancy_dir') if not r.get(field) or not Path(r[field]).exists()]
 if missing or source_missing:
  check={'name':'required_files_exist','passed':False,'detail':{'manifests':missing,'sources':source_missing}}
  result['checks'].append(check);result['failures'].append(check);result['checks_failed']+=1
  if result['status']=='PASS':result['status']='BLOCKED_MANIFEST'
 result.update(input=str(input_dir),report=str(report))
 report.parent.mkdir(parents=True,exist_ok=True)
 publish(result,report,is_json=True)
 print(json.dumps({k:result[k] for k in ('status','checks_passed','checks_failed','window_counts','uncovered_trajectories')},indent=2))
 raise SystemExit(result['status']!='PASS')


if __name__=='__main__':main()
