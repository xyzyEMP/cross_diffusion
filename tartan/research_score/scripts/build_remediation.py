from __future__ import annotations
import argparse,gzip,hashlib,json,os,pickletools,time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd,yaml
from tartan.data.pose_utils import load_poses,poses_to_se2,to_local_se2
from tartan.research_score.data.core import RouteSpec,cache_key,file_sha256,group_split,nested_budgets,resample_fixed_arc,sha256_json

REV="score-decomp-transfer-v1.2.1"
def dump(path,x):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
def write_line(handle,x):
 handle.write(json.dumps(x,separators=(",",":"),sort_keys=True)+"\n")
def safe_traj(path):
 raw=gzip.open(path,"rb").read();blobs=[a for o,a,_ in pickletools.genops(raw) if o.name in {"SHORT_BINBYTES","BINBYTES","BINBYTES8"} and isinstance(a,bytes) and len(a)>=24 and len(a)%12==0]
 if not blobs:raise ValueError("no ndarray payload")
 a=np.frombuffer(max(blobs,key=len),dtype="<f4")
 if a.size%3:raise ValueError("payload not Nx3")
 a=a.reshape(-1,3).astype(np.float32)
 if not np.isfinite(a).all():raise ValueError("nonfinite")
 return a
def arclen(a):return float(np.linalg.norm(np.diff(a[:,:2],axis=0),axis=1).sum()) if len(a)>1 else 0.
def interp_goal(se2,distance):
 seg=np.linalg.norm(np.diff(se2[:,:2],axis=0),axis=1);s=np.r_[0,np.cumsum(seg)];q=min(distance,float(s[-1]));return np.array([np.interp(q,s,se2[:,0]),np.interp(q,s,se2[:,1])],np.float32),q
def classify(length,horizon,cfg):return "moving_planning" if length>=cfg["eligibility"]["min_arc_length_m"] and horizon>=cfg["eligibility"]["min_horizon_s"] else "stop_or_short"
def source_scan(paths,cfg,raw_path,log):
 n=len(paths);future_steps=int(cfg["source_scan"]["future_steps"]);raw=np.lib.format.open_memmap(raw_path,mode="w+",dtype="float32",shape=(n,future_steps,3));lengths=np.zeros(n,np.float32);valid=np.zeros(n,np.uint8);rejected=[]
 def one(item):
  i,p=item
  try:return i,safe_traj(p),None
  except Exception as e:return i,None,{"path":str(p),"reason":type(e).__name__,"detail":str(e)}
 with ThreadPoolExecutor(max_workers=cfg["source_scan"]["workers"]) as pool:
  for done,(i,a,e) in enumerate(pool.map(one,enumerate(paths),chunksize=128),1):
   if a is not None:
    k=min(future_steps,len(a));raw[i,:k]=a[:k];raw[i,k:]=a[k-1] if k else 0;lengths[i]=arclen(a[:k]);valid[i]=k
   else:rejected.append(e)
   if done%5000==0:
    with log.open("a") as f:f.write(f"{datetime.now(timezone.utc).isoformat()} {done}/{n}\n")
 raw.flush();return lengths,valid,rejected
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True);p.add_argument("--run-id",required=True);a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text());assert cfg["protocol_revision"]==REV
 out=Path(a.output)/a.run_id;out.mkdir(parents=True,exist_ok=False);tmp=Path("/tmp")/a.run_id;tmp.mkdir(exist_ok=False);start=time.time();log=out/"source_scan_progress.log"
 tartan=Path(os.environ["TARTAN_ROOT"]);nuplan=Path(os.environ["NUPLAN_ROOT"]);ckpt=Path(os.environ["SOURCE_CKPT"]);index=nuplan/"dataset/nuplan-v1.1/exp/pluto/cache_mini/metadata/cache_mini_metadata_node_0.csv"
 values=pd.read_csv(index,usecols=["file_name"])["file_name"];vals=values[values.str.endswith("/trajectory")].tolist();paths=[Path(x+".gz") for x in vals]
 lengths,valid,rejected=source_scan(paths,cfg,tmp/"source_raw.npy",log)
 source_meta=[]
 for i,value in enumerate(vals):
  q=Path(value);source_meta.append((q.parts[-4],q.parts[-3],q.parts[-2]))
 # Tartan complete-episode split, then actual anchor/window extraction.
 episodes=[]
 for pose in sorted((tartan/"Data_anymal").glob("*/pose_lcam_front.txt")):
  se2=poses_to_se2(load_poses(pose));episodes.append((pose.parent.name,pose,se2))
 split=group_split([x[0] for x in episodes],cfg["split"]["seed"],cfg["split"]["train"],cfg["split"]["val"])
 tartan_windows=[]
 for ep,pose,se2 in episodes:
  for anchor in range(cfg["tartan"]["history_steps"],len(se2)-1,cfg["tartan"]["anchor_stride"]):
   fut=se2[anchor+1:min(len(se2),anchor+1+cfg["tartan"]["future_steps"])]
   local=to_local_se2(fut,se2[anchor]);length=arclen(local);horizon=len(local)/cfg["tartan"]["sample_rate_hz"];branch=classify(length,horizon,cfg);goal,goal_d=interp_goal(local,cfg["goal"]["distance_m"])
   occ=pose.parent/"coarse_occ"/f"occupancy_coarse5_{anchor:06d}_sparse.npy"
   tartan_windows.append({"episode":ep,"pose":pose,"anchor":anchor,"local":local,"length":length,"horizon":horizon,"branch":branch,"goal":goal,"goal_d":goal_d,"occ":occ,"split":split[ep]})
 # Train-only, formal branches source Car + target ANYmal. Proxy excluded.
 candidates=list(map(float,cfg["trajectory"]["length_candidates_m"]));threshold=float(cfg["trajectory"]["min_train_coverage"])
 source_hz=float(cfg["source_scan"]["sample_rate_hz"])
 src_train=[float(x) for x,v in zip(lengths,valid) if v and classify(float(x),float(v)/source_hz,cfg)=="moving_planning"]
 tgt_train=[w["length"] for w in tartan_windows if w["split"]=="train" and w["branch"]=="moving_planning"]
 coverage={str(int(c)):{"source_car":float(np.mean(np.asarray(src_train)>=c)) if src_train else 0.,"target_anymal":float(np.mean(np.asarray(tgt_train)>=c)) if tgt_train else 0.} for c in candidates}
 eligible=[c for c in candidates if min(coverage[str(int(c))].values())>=threshold]
 if not eligible:raise RuntimeError("no candidate satisfies v1.2.1 moving-window coverage")
 selected=max(eligible);selection={"protocol_revision":REV,"selection_source":"train_moving_windows_only","formal_branches":["source_car","target_anymal"],"proxy_excluded":True,"candidates_m":candidates,"min_train_coverage":threshold,"coverage":coverage,"selected_length_m":selected,"counts":{"source_car_moving_train":len(src_train),"target_anymal_moving_train":len(tgt_train)}};selection["statistics_sha256"]=sha256_json(selection);dump(out/"selected_length.json",selection)
 route_spec=RouteSpec(max_candidates=6,points_per_candidate=80);normalizer=Path(os.environ["OUTPUT_ROOT"])/"00_baseline_freeze/stage02_remediation_final_transfer_primary_20260911T021214Z_nogit/source_normalizers.json";normalizer_hash=file_sha256(normalizer);ckpt_hash=file_sha256(ckpt)
 # Canonical source manifest references raw/canonical arrays; every original sample retained.
 src_values=np.lib.format.open_memmap(tmp/"source_canonical.npy",mode="w+",dtype="float32",shape=(len(paths),80,4));src_masks=np.lib.format.open_memmap(tmp/"source_mask.npy",mode="w+",dtype="uint8",shape=(len(paths),80));sm=out/"transfer_primary/source_windows.jsonl";sm.parent.mkdir(parents=True)
 raw=np.load(tmp/"source_raw.npy",mmap_mode="r")
 counts={"source":{"moving_planning":0,"stop_or_short":0},"target":{"moving_planning":0,"stop_or_short":0}}
 with sm.open("x",buffering=1024*1024) as smh:
  for i,(path,(scene,stype,token)) in enumerate(zip(paths,source_meta)):
   k=int(valid[i]);traj=raw[i,:k];length=float(lengths[i]);branch=classify(length,float(k)/source_hz,cfg) if k else "stop_or_short";counts["source"][branch]+=1
   if k:
    goal,goal_d=interp_goal(traj, cfg["goal"]["distance_m"]);can,mask=resample_fixed_arc(traj,selected,80)
   else:goal=np.zeros(2,np.float32);goal_d=0.;can=np.zeros((80,4),np.float32);mask=np.zeros(80,bool)
   src_values[i]=can;src_masks[i]=mask
   sid=f"nuplan:{scene}:{stype}:{token}";feature=path.parent/"feature.gz"
   rec={"sample_id":sid,"episode_id":scene,"map_id":"nuplan_source_map_from_feature","split":"train","budget_membership":"source_all","anchor_index":token,"domain":"nuplan","embodiment":"car","branch":branch,"current_state":{"se2_local":[0.,0.,0.]},"fixed_goal":{"xy_local":goal.tolist(),"rule":"path_arc_distance_clamped","requested_distance_m":cfg["goal"]["distance_m"],"actual_distance_m":goal_d},"route_set":{"source":"nuplan_map_route_feature","reference":str(feature),"max_candidates":6,"future_gt_dependency":False},"trajectory":{"values_ref":"source_canonical.npy","mask_ref":"source_mask.npy","row":i,"representation":"xy_cos_sin","raw_reference":str(path),"raw_points":k,"timestamps_s":[float(j+1)/source_hz for j in range(k)]},"provenance":{"protocol_revision":REV,"source_preprocess_version":"source-loader-v1","cache_key":cache_key(sid,"source-loader-v1",route_spec.hash,ckpt_hash),"normalizer_sha256":normalizer_hash}}
   write_line(smh,rec)
  smh.flush();os.fsync(smh.fileno())
 src_values.flush();src_masks.flush()
 # Target canonical arrays and manifests.
 tv=np.lib.format.open_memmap(tmp/"target_canonical.npy",mode="w+",dtype="float32",shape=(len(tartan_windows),80,4));tm=np.lib.format.open_memmap(tmp/"target_mask.npy",mode="w+",dtype="uint8",shape=(len(tartan_windows),80));budgets={str(seed):nested_budgets([f"anymal:{ep}" for ep,_,_ in episodes if split[ep]=="train"],seed) for seed in cfg["budgets"]["joint_seeds"]}
 target_files={s:out/f"transfer_primary/target_{s}_windows.jsonl" for s in ("train","val","test")}
 target_handles={s:target_files[s].open("x",buffering=1024*1024) for s in target_files}
 try:
  for i,w in enumerate(tartan_windows):
   can,mask=resample_fixed_arc(w["local"],selected,80);tv[i]=can;tm[i]=mask;counts["target"][w["branch"]]+=1;eid=f"anymal:{w['episode']}";membership={seed:[b for b,ids in bs.items() if eid in ids] for seed,bs in budgets.items()} if w["split"]=="train" else {}
   sid=f"tartan:anymal:{w['episode']}:{w['anchor']:06d}";route_key=sha256_json({"sample_id":sid,"goal":w["goal"].tolist(),"map":str(w["occ"]),"route_spec":route_spec.hash,"revision":REV})
   rec={"sample_id":sid,"episode_id":eid,"map_id":tartan.name,"split":w["split"],"budget_membership":membership,"anchor_index":w["anchor"],"domain":"tartanground","embodiment":"anymal","branch":w["branch"],"current_state":{"global_se2_index":w["anchor"],"se2_local":[0.,0.,0.]},"fixed_goal":{"xy_local":w["goal"].tolist(),"rule":"offline_future_path_arc_distance_clamped_then_frozen","requested_distance_m":cfg["goal"]["distance_m"],"actual_distance_m":w["goal_d"]},"route_set":{"source":"current_coarse_occupancy_plus_fixed_goal","map_reference":str(w["occ"]),"cache_key":route_key,"max_candidates":6,"future_gt_dependency":False},"trajectory":{"values_ref":"target_canonical.npy","mask_ref":"target_mask.npy","row":i,"representation":"xy_cos_sin","raw_reference":str(w["pose"]),"raw_points":len(w["local"]),"timestamps_s":[float((j+1)/cfg["tartan"]["sample_rate_hz"]) for j in range(len(w["local"]))]},"provenance":{"protocol_revision":REV,"same_map_unseen_episode":True,"unseen_map_claim":False,"normalizer_sha256":normalizer_hash}}
   write_line(target_handles[w["split"]],rec)
 finally:
  for h in target_handles.values():h.flush();os.fsync(h.fileno());h.close()
 tv.flush();tm.flush()
 # Fsync-copy array shards from local disk to shared output.
 for name in ("source_canonical.npy","source_mask.npy","target_canonical.npy","target_mask.npy"):
  dest=out/"transfer_primary"/name
  with open(tmp/name,"rb") as fi,open(dest,"xb") as fo:
   while True:
    b=fi.read(8*1024*1024)
    if not b:break
    fo.write(b)
   fo.flush();os.fsync(fo.fileno())
 profile={"profile":"transfer_primary","protocol_revision":REV,"status":"AWAITING_REVIEW","claim_scope":"same-map unseen-episode transfer","unseen_map_claim":False,"source":"nuplan/car","target":"tartanground/anymal","selected_length_artifact":"selected_length.json","selected_length_sha256":file_sha256(out/"selected_length.json"),"source_normalizer_sha256":normalizer_hash,"route_spec_hash":route_spec.hash,"budgets":budgets};dump(out/"transfer_primary/profile_manifest.json",profile)
 proxy=out/"proxy_pair_auxiliary";proxy.mkdir();dump(proxy/"profile_manifest.json",{"profile":"proxy_pair_auxiliary","protocol_revision":REV,"status":"DIAGNOSTIC_ONLY_NOT_SCIENTIFICALLY_PAIRED","included_in_formal_training":False,"included_in_length_selection":False,"included_in_main_table":False,"real_data_audit_reference":str(Path(os.environ["OUTPUT_ROOT"])/"01_data_audit/stage03_index_20260911T100714Z_nogit/proxy_pair_auxiliary/pairs_retry2"),"synthetic_interface_fixture":{"pair_level":"proxy_fixture","platform_a":"diff","platform_b":"anymal","supports_claims":{"pipeline_validation":True,"car_dog_scientific_claim":False}}})
 strict=out/"strict_pair";strict.mkdir();dump(strict/"profile_manifest.json",{"profile":"strict_pair","protocol_revision":REV,"status":"BLOCKED_WAITING_DATA","cf_pair_root":os.environ.get("CF_PAIR_ROOT","")})
 detailed={}
 for s in ("train","val","test"):
  sw=[w for w in tartan_windows if w["split"]==s]
  detailed[s]={b:sum(w["branch"]==b for w in sw) for b in ("moving_planning","stop_or_short")}
 for seed,bs in budgets.items():
  for budget,ids in bs.items():
   key=f"train_seed_{seed}_budget_{budget}";chosen=set(ids);sw=[w for w in tartan_windows if w["split"]=="train" and f"anymal:{w['episode']}" in chosen]
   detailed[key]={b:sum(w["branch"]==b for w in sw) for b in ("moving_planning","stop_or_short")}
 counts["target_by_split_and_budget"]=detailed
 dump(out/"branch_counts.json",counts);dump(out/"rejected_samples.json",rejected);dump(out/"split_summary.json",{"claim_scope":"same-map unseen-episode","map_ids":[tartan.name],"episode_split":split,"window_counts":{s:sum(w["split"]==s for w in tartan_windows) for s in ("train","val","test")},"budgets":budgets,"branch_counts":detailed})
 index_hash=file_sha256(index);dump(out/"run_manifest.json",{"run_id":a.run_id,"stage":"03_remediation","status":"AWAITING_REVIEW","protocol_revision":REV,"created_at":datetime.now(timezone.utc).isoformat(),"duration_s":time.time()-start,"config_sha256":file_sha256(Path(a.config)),"checkpoint_sha256":ckpt_hash,"normalizer_sha256":normalizer_hash,"nuplan_index_sha256":index_hash,"nuplan_index_rows":len(paths),"source_content_validation":"full trajectory geometry scan; feature content not rehashed","source_scan_rejected":len(rejected),"selected_length_sha256":file_sha256(out/"selected_length.json"),"next_stage_started":False})
 print(json.dumps({"source_rows":len(paths),"target_windows":len(tartan_windows),"counts":counts,"coverage":coverage,"selected":selected,"rejected":len(rejected)},indent=2))
if __name__=="__main__":main()
