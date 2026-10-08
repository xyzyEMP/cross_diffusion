"""Build current transfer manifests from native nuPlan NPZ and Tartan poses.

This script does not train. It freezes a deterministic 50k source audit set,
uses a conservative Wilson lower bound, and exactly audits Tartan train windows.
"""
from __future__ import annotations
import argparse, hashlib, heapq, json, math, os, shutil, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import yaml
from tartan.research_score.artifacts import validate_output_name
from tartan.data.pose_utils import load_poses, poses_to_se2, to_local_se2
from tartan.research_score.data.core import RouteSpec, file_sha256, resample_fixed_arc, sha256_json

REV="score-decomp-transfer-v1.2.3"; AUDIT_ALGO="normalized-path-sha256-smallest-v1"; Z95=1.6448536269514722
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
def write_line(h,x):h.write(json.dumps(x,separators=(",",":"),sort_keys=True)+"\n")
def arclen(x):return float(np.linalg.norm(np.diff(x[:,:2],axis=0),axis=1).sum()) if len(x)>1 else 0.
def canonicalize_window(x,length_m,num_points):
 if len(x)>=2:return resample_fixed_arc(x,length_m,num_points)
 # Final anchors can have only one future pose. Retain them in stop_or_short
 # without inventing motion, and expose the measured-token validity explicitly.
 base=np.zeros((num_points,4),dtype=np.float32);base[:,2]=1.0;mask=np.zeros(num_points,dtype=bool)
 if len(x)==1:
  px,py,yaw=np.asarray(x[0],dtype=np.float32)
  base[:]=np.asarray([px,py,np.cos(yaw),np.sin(yaw)],dtype=np.float32);mask[0]=True
 return base,mask
def classify(length,horizon,cfg):return "moving_planning" if length>=cfg["eligibility"]["min_arc_length_m"] and horizon>=cfg["eligibility"]["min_horizon_s"] else "stop_or_short"
def stratified_episode_split(episodes, moving_counts, cfg):
 out={};seed=int(cfg["split"]["seed"]);minimum=int(cfg["split"]["min_moving_windows_per_episode"])
 strata={"moving_capable":[],"stop_dominant":[]}
 for ep in episodes:strata["moving_capable" if moving_counts.get(ep,0)>=minimum else "stop_dominant"].append(ep)
 for name,items in strata.items():
  ordered=sorted(items,key=lambda ep:hashlib.sha256(f"{seed}|{name}|{ep}".encode()).digest());n=len(ordered)
  ntrain=int(round(n*float(cfg["split"]["train"])));nval=int(round(n*float(cfg["split"]["val"])))
  if n>=3:ntrain=min(max(1,ntrain),n-2);nval=min(max(1,nval),n-ntrain-1)
  for ep in ordered[:ntrain]:out[ep]="train"
  for ep in ordered[ntrain:ntrain+nval]:out[ep]="val"
  for ep in ordered[ntrain+nval:]:out[ep]="test"
 return out,strata
def wilson_lower(k,n,z=Z95):
 if n<=0:return 0.
 p=k/n;d=1+z*z/n
 return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/d
def norm_path(value,root):
 p=Path(value).expanduser()
 # The authority index contains canonical filenames under an absolute root.
 # Avoid Path.resolve(): on a million-row shared filesystem it performs a
 # metadata lookup per row and does not change the deterministic path identity.
 return root/p if not p.is_absolute() else p
def sample_id(path):return "nuplan-native:"+hashlib.sha256(path.as_posix().encode()).hexdigest()
def cache_key(sid,route_hash,checkpoint_hash):return hashlib.sha256(f"{sid}|native-npz-v1|{route_hash}|{checkpoint_hash}|native-dp-npz-v1".encode()).hexdigest()
def infer_map_token(path):
 stem=path.stem
 if "_" not in stem:return "UNKNOWN",stem
 return tuple(stem.rsplit("_",1))
def interp_goal(se2,distance):
 seg=np.linalg.norm(np.diff(se2[:,:2],axis=0),axis=1);s=np.r_[0,np.cumsum(seg)];q=min(distance,float(s[-1]));return np.array([np.interp(q,s,se2[:,0]),np.interp(q,s,se2[:,1])],np.float32),q
def read_npz_traj(item):
 rank,path=item
 try:
  with np.load(path,allow_pickle=False) as z:
   x=np.asarray(z["ego_agent_future"],np.float32)
   if x.shape!=(80,3):raise ValueError(f"ego_agent_future shape {x.shape}")
   if not np.isfinite(x).all():raise ValueError("nonfinite ego_agent_future")
  return rank,path,x,None
 except Exception as e:return rank,path,None,{"path":str(path),"reason":type(e).__name__,"detail":str(e)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--profile",default="transfer_primary",choices=("transfer_primary","proxy_ab","four_groups"));ap.add_argument("--group",choices=("g1","g2","g3","g4"));ap.add_argument("--stage",choices=("trajectories","windows"));ap.add_argument("--trajectory-manifest");ap.add_argument("--resume",action="store_true");ap.add_argument("--config",required=True);ap.add_argument("--output",required=True);ap.add_argument("--run-id",required=True);a=ap.parse_args();cfg=yaml.safe_load(Path(a.config).read_text());
 if a.profile=="four_groups":return build_four_group(a,cfg)
 if a.profile=="proxy_ab":return build_proxy(a,cfg["proxy_ab"])
 assert cfg["protocol_revision"]==REV
 validate_output_name(a.run_id)
 final_out=Path(a.output)/a.run_id
 if final_out.exists():raise FileExistsError(final_out)
 out=final_out.with_name("."+final_out.name+".building")
 out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists():raise FileExistsError(out)
 out.mkdir();(out/"transfer_primary").mkdir();start=time.time();native_root=Path(cfg["nuplan_native"]["root"]);index=Path(cfg["nuplan_native"]["training_index"]);paths=[norm_path(x,native_root) for x in json.loads(index.read_text())];route_spec=RouteSpec(max_candidates=6,points_per_candidate=80);checkpoint=Path(os.environ["SOURCE_CKPT"]);checkpoint_hash=file_sha256(checkpoint);normalizer=Path(os.environ["OUTPUT_ROOT"])/"data/source_baseline/source_normalizers.json";normalizer_hash=file_sha256(normalizer);config_hash=file_sha256(Path(a.config));source_hash=file_sha256(Path(__file__))
 if len(paths)!=cfg["nuplan_native"]["expected_index_rows"]:raise RuntimeError(f"native index rows {len(paths)}")
 # Freeze sample IDs before reading any trajectory outcome.
 ranked=heapq.nsmallest(cfg["nuplan_native"]["length_audit_size"],((hashlib.sha256(p.as_posix().encode()).digest(),i,p) for i,p in enumerate(paths)))
 audit=[(rank,p) for rank,(_,_,p) in enumerate(ranked)];audit_ids=[sample_id(p) for _,p in audit];sample_hash=hashlib.sha256(("\n".join(audit_ids)+"\n").encode()).hexdigest()
 with (out/"source_length_audit_ids.txt").open("x") as h:h.write("\n".join(audit_ids)+"\n");h.flush();os.fsync(h.fileno())
 dump(out/"source_audit_freeze.json",{"protocol_revision":REV,"algorithm":AUDIT_ALGO,"population_index":str(index),"population_size":len(paths),"audit_size":len(audit),"sample_ids_sha256":sample_hash,"frozen_before_outcomes":True,"selection_unit":"normalized_npz_path"})
 # Provenance-only population manifest; it does not claim per-file content validation.
 with (out/"transfer_primary/source_index.jsonl").open("x",buffering=1024*1024) as h:
  for p in paths:
   sid=sample_id(p);write_line(h,{"sample_id":sid,"npz_reference":str(p),"split":"train","domain":"nuplan","embodiment":"car","protocol_revision":REV,"content_status":"INDEXED_NOT_INDIVIDUALLY_VALIDATED","provenance":{"schema_version":"native-dp-npz-v1","source_preprocess_version":"native-npz-v1","cache_key":cache_key(sid,route_spec.hash,checkpoint_hash),"checkpoint_sha256":checkpoint_hash,"normalizer_sha256":normalizer_hash}})
  h.flush();os.fsync(h.fileno())
 # Read only frozen 50k source audit trajectories.
 results=[];rejected=[]
 with ThreadPoolExecutor(max_workers=cfg["nuplan_native"]["workers"]) as pool:
  for done,row in enumerate(pool.map(read_npz_traj,audit,chunksize=64),1):
   if row[-1] is None:results.append(row)
   else:rejected.append(row[-1])
   if done%1000==0:
    with (out/"native_audit_progress.log").open("a") as h:h.write(f"{datetime.now(timezone.utc).isoformat()} {done}/{len(audit)}\n")
 results.sort();source_lengths=np.asarray([arclen(x[2]) for x in results],np.float32);source_branches=[classify(float(v),8.0,cfg) for v in source_lengths];src_moving=source_lengths[np.asarray(source_branches)=="moving_planning"]
 # Full-tensor schema/meta validation is separate from the 50k trajectory audit.
 schema_idx=np.linspace(0,len(audit)-1,min(1000,len(audit)),dtype=int);signatures=Counter();meta_mismatch=[];schema_reject=[]
 for j in schema_idx:
  _,p=audit[int(j)]
  try:
   with np.load(p,allow_pickle=False) as z:
    signature=tuple((k,str(z[k].dtype),tuple(z[k].shape)) for k in sorted(z.files));signatures[str(signature)]+=1;imap,itoken=infer_map_token(p);amap=str(np.asarray(z["map_name"]).item());atoken=str(np.asarray(z["token"]).item())
    if (imap,itoken)!=(amap,atoken):meta_mismatch.append({"path":str(p),"inferred":[imap,itoken],"actual":[amap,atoken]})
  except Exception as e:schema_reject.append({"path":str(p),"reason":type(e).__name__,"detail":str(e)})
 dump(out/"native_schema_validation.json",{"protocol_revision":REV,"sample_size":len(schema_idx),"selection":"evenly_spaced_over_frozen_hash_rank_50k","signature_counts":dict(signatures),"schema_rejected":schema_reject,"filename_meta_mismatches":meta_mismatch})
 with (out/"transfer_primary/source_length_audit.jsonl").open("x",buffering=1024*1024) as h:
  for (rank,p,x,_),length,branch in zip(results,source_lengths,source_branches):
   sid=sample_id(p);map_name,token=infer_map_token(p);write_line(h,{"sample_id":sid,"audit_rank":rank,"npz_reference":str(p),"map_id":map_name,"anchor_index":token,"split":"train","domain":"nuplan","embodiment":"car","branch":branch,"future_arc_length_m":float(length),"physical_horizon_s":8.0,"trajectory":{"npz_key":"ego_agent_future","shape":[80,3],"representation":"source_temporal_xy_yaw","timestamps_s":[float(i+1)/10 for i in range(80)]},"provenance":{"protocol_revision":REV,"audit_sample_sha256":sample_hash,"schema_version":"native-dp-npz-v1","cache_key":cache_key(sid,route_spec.hash,checkpoint_hash),"checkpoint_sha256":checkpoint_hash,"normalizer_sha256":normalizer_hash,"map_token_source":"filename_rsplit_validated_on_fixed_1000"}})
  h.flush();os.fsync(h.fileno())
 # Exact target anchor/window audit and stratified same-map unseen-episode split.
 tartan=Path(os.environ["TARTAN_ROOT"]);episodes=[]
 for pose in sorted((tartan/"Data_anymal").glob("*/pose_lcam_front.txt")):episodes.append((pose.parent.name,pose,poses_to_se2(load_poses(pose))))
 windows=[]
 for ep,pose,se2 in episodes:
  for anchor in range(cfg["tartan"]["history_steps"],len(se2)-1,cfg["tartan"]["anchor_stride"]):
   local=to_local_se2(se2[anchor+1:min(len(se2),anchor+1+cfg["tartan"]["future_steps"])],se2[anchor]);length=arclen(local);horizon=len(local)/cfg["tartan"]["sample_rate_hz"];goal,goal_d=interp_goal(local,cfg["goal"]["distance_m"])
   windows.append({"episode":ep,"pose":pose,"anchor":anchor,"local":local,"length":length,"horizon":horizon,"branch":classify(length,horizon,cfg),"goal":goal,"goal_d":goal_d,"occ":pose.parent/"coarse_occ"/f"occupancy_coarse5_{anchor:06d}_sparse.npy"})
 moving_counts=Counter(w["episode"] for w in windows if w["branch"]=="moving_planning")
 split,strata=stratified_episode_split([e[0] for e in episodes],moving_counts,cfg)
 for w in windows:w["split"]=split[w["episode"]]
 tgt_moving=np.asarray([w["length"] for w in windows if w["split"]=="train" and w["branch"]=="moving_planning"]);threshold=float(cfg["trajectory"]["min_train_coverage"]);coverage={}
 for c in map(float,cfg["trajectory"]["length_candidates_m"]):
  sk=int(np.sum(src_moving>=c));sn=len(src_moving);tk=int(np.sum(tgt_moving>=c));tn=len(tgt_moving)
  coverage[str(int(c))]={"source_car":{"covered":sk,"denominator":sn,"estimate":sk/sn if sn else 0.,"wilson_lower_95_one_sided":wilson_lower(sk,sn)},"target_anymal":{"covered":tk,"denominator":tn,"exact_coverage":tk/tn if tn else 0.}}
 eligible=[c for c in map(float,[cfg["trajectory"]["length_m"]]) if coverage[str(int(c))]["source_car"]["wilson_lower_95_one_sided"]>=threshold and coverage[str(int(c))]["target_anymal"]["exact_coverage"]>=threshold];selected=float(cfg["trajectory"]["length_m"]) if eligible else None
 selection={"protocol_revision":REV,"status":"SELECTED" if selected is not None else "BLOCKED_NO_ELIGIBLE_LENGTH","selection_source":"train_only","formal_branches":["source_car_native_npz_50k_wilson","target_anymal_all_train_moving_exact"],"source_audit_sample_sha256":sample_hash,"candidates_m":cfg["trajectory"]["length_candidates_m"],"min_train_coverage":threshold,"coverage":coverage,"selected_length_m":selected};selection["statistics_sha256"]=sha256_json(selection);dump(out/"selected_length.json",selection)
 target_files={s:out/f"transfer_primary/target_{s}_windows.jsonl" for s in ("train","val","test")};handles={s:p.open("x",buffering=1024*1024) for s,p in target_files.items()}
 try:
  for i,w in enumerate(windows):
   eid=f"anymal:{w['episode']}";membership={};sid=f"tartan:anymal:{w['episode']}:{w['anchor']:06d}"
   rec={"sample_id":sid,"episode_id":eid,"map_id":tartan.name,"split":w["split"],"budget_membership":membership,"anchor_index":w["anchor"],"domain":"tartanground","embodiment":"anymal","branch":w["branch"],"current_state":{"global_se2_index":w["anchor"],"se2_local":[0.,0.,0.]},"fixed_goal":{"xy_local":w["goal"].tolist(),"rule":"offline_future_path_arc_distance_clamped_then_frozen","requested_distance_m":cfg["goal"]["distance_m"],"actual_distance_m":w["goal_d"]},"route_set":{"source":"current_coarse_occupancy_plus_fixed_goal","map_reference":str(w["occ"]),"cache_key":sha256_json({"sample_id":sid,"goal":w["goal"].tolist(),"map":str(w["occ"]),"route_spec":route_spec.hash,"revision":REV}),"max_candidates":6,"future_gt_dependency":False},"trajectory":{"raw_reference":str(w["pose"]),"raw_future_xy_yaw":w["local"].tolist(),"raw_points":len(w["local"]),"representation":"source_temporal_xy_yaw","timestamps_s":[float(j+1)/cfg["tartan"]["sample_rate_hz"] for j in range(len(w["local"]))]},"provenance":{"protocol_revision":REV,"same_map_unseen_episode":True,"unseen_map_claim":False}}
   if selected is not None:
    can,mask=canonicalize_window(w["local"],selected,80);rec["trajectory"]["fixed_arc_length_80"]=can.tolist();rec["trajectory"]["valid_mask"]=mask.astype(int).tolist()
   write_line(handles[w["split"]],rec)
 finally:
  for h in handles.values():h.flush();os.fsync(h.fileno());h.close()
 branch={"source_audit":dict(Counter(source_branches)),"target_all":dict(Counter(w["branch"] for w in windows)),"target_by_split":{s:dict(Counter(w["branch"] for w in windows if w["split"]==s)) for s in ("train","val","test")}};dump(out/"branch_counts.json",branch);dump(out/"rejected_samples.json",{"source_native_npz":rejected,"target":[]})
 dump(out/"split_summary.json",{"claim_scope":"same-map unseen-episode","algorithm":"branch-stratified-complete-episode-sha256-v1","stratification_threshold_moving_windows":cfg["split"]["min_moving_windows_per_episode"],"strata":strata,"episode_split":split,"window_counts":dict(Counter(w["split"] for w in windows)),"moving_window_counts":dict(moving_counts)})
 transfer_status="COMPLETE" if selected is not None else "BLOCKED_LENGTH_SELECTION";dump(out/"transfer_primary/profile_manifest.json",{"profile":"transfer_primary","protocol_revision":REV,"status":transfer_status,"source":"nuplan/car/native_diffusion_planner_npz","target":"tartanground/anymal","claim_scope":"same-map unseen-episode transfer","unseen_map_claim":False,"source_index_rows":len(paths),"source_length_audit_rows":len(audit),"source_length_audit_sha256":sample_hash,"selected_length_sha256":file_sha256(out/"selected_length.json"),"route_spec_hash":route_spec.hash,"checkpoint_sha256":checkpoint_hash,"normalizer_sha256":normalizer_hash,"config_sha256":config_hash,"source_sha256":source_hash})
 dump(out/"run_manifest.json",{"run_id":a.run_id,"task":"build_transfer_manifests","status":transfer_status,"protocol_revision":REV,"created_at":datetime.now(timezone.utc).isoformat(),"duration_s":time.time()-start,"native_index":str(index),"native_index_sha256":file_sha256(index),"native_index_rows":len(paths),"index_coverage":len(paths),"length_audit_sample":len(audit),"length_audit_sha256":sample_hash,"content_validation_scope":"50k ego trajectory audit plus fixed 1000 full tensor schema/meta validation","schema_validation_sample":len(schema_idx),"schema_validation_rejected":len(schema_reject),"filename_meta_mismatches":len(meta_mismatch),"source_rejected":len(rejected),"selected_length_sha256":file_sha256(out/"selected_length.json"),"checkpoint_sha256":checkpoint_hash,"normalizer_sha256":normalizer_hash,"config_sha256":config_hash,"source_sha256":source_hash,"route_spec_hash":route_spec.hash})
 os.replace(out,final_out)
 print(json.dumps({"status":transfer_status,"source_index":len(paths),"source_audit_valid":len(results),"source_rejected":len(rejected),"source_branches":branch["source_audit"],"target_windows":len(windows),"target_branches":branch["target_all"],"coverage":coverage,"selected":selected,"published":str(final_out)},indent=2))
 raise SystemExit(0 if selected is not None else 2)
def build_proxy(a,cfg):
 from tartan.data.pose_utils import discover_trajectories,proxy_trajectory_evidence,read_proxy_trajectories,load_proxy_se2
 from tartan.research_score.data.core import proxy_split,proxy_window
 from tartan.research_score.artifacts import publish
 validate_output_name(a.run_id)
 out=Path(a.output)/a.run_id;out.mkdir(parents=True,exist_ok=True)
 if a.stage not in ('trajectories','windows'):raise ValueError('Proxy requires explicit --stage')
 def atomic_lines(path,items):
  import tempfile
  text=''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in items)
  if path.exists():
   if a.resume and path.read_text()==text:return
   raise FileExistsError(path)
  fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
  with os.fdopen(fd,'w') as h:h.write(text);h.flush();os.fsync(h.fileno())
  os.replace(tmp,path)
 if a.stage=='trajectories':
  if (out/'trajectories.jsonl').exists():
   if not a.resume:raise FileExistsError(out/'trajectories.jsonl')
   print(json.dumps({'status':'REUSED_FROZEN','path':str(out/'trajectories.jsonl')}));return
  rows=[]
  for record in discover_trajectories(Path(cfg['root'])):
   if record.robot_type not in cfg['platforms']:continue
   platform=cfg['platforms'][record.robot_type]
   row={'trajectory_key':f'{record.environment}/{record.robot_type}/{record.trajectory_id}',
     'map_id':record.environment,'embodiment':record.robot_type,'trajectory_id':record.trajectory_id,
     'pose_path':str(record.pose_path),'dataset_relative_pose':str(record.pose_path.relative_to(Path(cfg['root']))),
     'platform_id':platform['platform_id'],'robot_radius_m':platform['robot_radius_m'],
     **proxy_trajectory_evidence(record,platform)}
   rows.append(row)
  rows=proxy_split(rows,cfg['split']['seed'])
  atomic_lines(out/'trajectories.jsonl',rows)
  summary={'profile':'proxy_ab','DATA_ID':a.run_id,'counts':dict(Counter(r['split'] for r in rows)),
    'platform_counts':dict(Counter(r['embodiment'] for r in rows)),
    'membership':{r['trajectory_key']:r['split'] for r in rows},
    'gates_passed':all(all(r['gates'].values()) for r in rows),
    'blocked_trajectories':{r['trajectory_key']:r['gate_reasons'] for r in rows if r['gate_reasons']}}
  publish(summary,out/'trajectory_summary.json',is_json=True);publish(cfg,out/'config.json',is_json=True)
  print(json.dumps(summary,indent=2));return
 if not a.trajectory_manifest:raise ValueError('--trajectory-manifest required')
 rows=read_proxy_trajectories(a.trajectory_manifest)
 frozen=json.loads((Path(a.trajectory_manifest).parent/'config.json').read_text())
 if frozen!=cfg:raise ValueError('frozen trajectory config conflict')
 windows={'train':[],'val':[],'test':[]};rejected=[];index=[];navigation_index=[]
 for r in rows:
  se2=poses_to_se2(load_poses(Path(r['pose_path'])))
  cumulative=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(se2[:,:2],axis=0),axis=1))]
  for anchor in range(20,len(se2)-1,10):
   available=float(cumulative[-1]-cumulative[anchor]);complete=available>=8.
   end=int(np.searchsorted(cumulative,cumulative[anchor]+8.,side='left')) if complete else len(se2)-1
   index.append({'sample_id':f"{r['trajectory_key']}:anchor:{anchor:06d}",'trajectory_key':r['trajectory_key'],'split':r['split'],'embodiment':r['embodiment'],'anchor_index':anchor,'source_end_frame':end,'covered_arc_m':min(8.,available),'full_8m':complete,'valid_mask':(np.linspace(0,8,80)<=min(8.,available)+1e-9).tolist(),'pose_path':r['pose_path'],'status':'SOURCE_INDEX_ONLY_NO_BODY_FEATURES'})
  anchor=20;segment=0
  while anchor<len(se2)-1 and cumulative[-1]-cumulative[anchor]>=8.:
   end=int(np.searchsorted(cumulative,cumulative[anchor]+8.,side='left'))
   goal=[float(np.interp(cumulative[anchor]+8.,cumulative,se2[:,i])) for i in (0,1)]
   navigation_index.append({'sample_id':f"{r['trajectory_key']}:nonoverlap8m:{segment:06d}",'trajectory_key':r['trajectory_key'],'split':r['split'],'embodiment':r['embodiment'],'anchor_index':anchor,'segment_end_index':end,'goal_world_xy':goal,'pose_path':r['pose_path'],'status':'SOURCE_INDEX_ONLY_NO_BODY_FEATURES'})
   segment+=1;anchor=20+int(np.ceil((end-20)/10.))*10
  if not r['gates'].get('reference_heading',r['gates'].get('body_heading',False)) or not r['gates']['occupancy_frame']:
   rejected.append({'trajectory_key':r['trajectory_key'],'reasons':r['gate_reasons'],'candidate_anchors':len(range(20,r['pose_count']-1,10))});continue
  se2=load_proxy_se2(r)
  for anchor in range(20,len(se2)-1,10):
   try:windows[r['split']].append(proxy_window(r,se2,anchor,a.trajectory_manifest))
   except (ValueError,FileNotFoundError) as e:rejected.append({'trajectory_key':r['trajectory_key'],'anchor_index':anchor,'reason':str(e)})
 if not (out/'window_index.jsonl').exists():atomic_lines(out/'window_index.jsonl',index)
 if not (out/'navigation_index.jsonl').exists():atomic_lines(out/'navigation_index.jsonl',navigation_index)
 for split,name in [('train','base_train'),('val','base_val'),('test','anymal_test')]:
  if windows[split]:atomic_lines(out/(name+'.jsonl'),windows[split])
 summary={'status':'BLOCKED_FRAME' if any(not r['gates'].get('reference_heading',r['gates'].get('body_heading',False)) or not r['gates']['occupancy_frame'] for r in rows) else ('BLOCKED_TIME' if any(not r['gates']['time'] for r in rows) else 'PASS'),
  'counts':{k:len(v) for k,v in windows.items()},'source_index_counts':dict(Counter(r['split'] for r in index)),'navigation_index_counts':dict(Counter(r['split'] for r in navigation_index)),'navigation_index_coverage':len({r['trajectory_key'] for r in navigation_index}),'index_policy':'world_xy_suffix_ranges_only_no_body_heading_or_model_inputs','rejected':rejected,
  'uncovered_trajectories':[r['trajectory_key'] for r in rows if not any(w['trajectory_key']==r['trajectory_key'] for w in windows[r['split']])]}
 publish(summary,out/'window_summary.json',is_json=True)
 evidence_summary=json.loads((out/'trajectory_summary.json').read_text())
 evidence_summary.update(gates_passed=all(all(r['gates'].values()) for r in rows),blocked_trajectories={r['trajectory_key']:r['gate_reasons'] for r in rows if r['gate_reasons']})
 publish(evidence_summary,out/'trajectory_summary.json',is_json=True)
 print(json.dumps(summary,indent=2))
 return

def build_four_group(a,config):
 from tartan.research_score.data.core import four_group_split,four_group_window
 from tartan.data.pose_utils import read_proxy_trajectories,load_proxy_se2
 from tartan.research_score.artifacts import publish,publish_text
 from tartan.research_score.scripts.train_transfer import proxy_config
 cfg=proxy_config(a.config,'four_groups');out=Path(a.output)/a.run_id;out.mkdir(parents=True,exist_ok=True)
 validate_output_name(a.run_id)
 if a.group not in cfg['groups']:raise ValueError('--group required')
 def lines(path,rows):
  text=''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows)
  if path.exists():
   if a.resume and path.read_text()==text:return
   raise FileExistsError(path)
  publish_text(text,path)
 if a.stage=='trajectories':
  rows=four_group_split(read_proxy_trajectories(cfg['catalog']),a.group,cfg['split_seed'])
  if any(not all(r['gates'].values()) for r in rows):raise ValueError('catalog evidence gate not passed')
  lines(out/'trajectories.jsonl',rows)
  publish(cfg,out/'config.json',True)
  publish({'profile':'four_groups','group':a.group,'gates_passed':True,'counts':dict(Counter(r['split'] for r in rows)),'membership':{r['trajectory_key']:r['split'] for r in rows}},out/'trajectory_summary.json',True)
  return
 if a.stage!='windows':raise ValueError('explicit --stage required')
 rows=read_proxy_trajectories(out/'trajectories.jsonl')
 if json.loads((out/'config.json').read_text())!=cfg:raise ValueError('frozen config conflict')
 windows={k:[] for k in ('train','val','test')};rejected=[]
 for row in rows:
  se2=load_proxy_se2(row)
  for anchor in range(20,len(se2)-1,10):
   try:windows[row['split']].append(four_group_window(row,se2,anchor,out/'trajectories.jsonl'))
   except (ValueError,FileNotFoundError) as e:rejected.append({'trajectory_key':row['trajectory_key'],'anchor':anchor,'reason':str(e)})
 uncovered=[r['trajectory_key'] for r in rows if not any(w['trajectory_key']==r['trajectory_key'] for w in windows[r['split']])]
 for split,name in (('train','base_train'),('val','base_val'),('test','test')):lines(out/(name+'.jsonl'),windows[split])
 publish({'counts':{k:len(v) for k,v in windows.items()},'rejected':rejected,'uncovered_trajectories':uncovered,'status':'PASS' if not uncovered and all(windows.values()) else 'BLOCKED_COVERAGE'},out/'window_summary.json',True)
 if uncovered or not all(windows.values()):raise ValueError('incomplete four-group data coverage')

if __name__=="__main__":main()
