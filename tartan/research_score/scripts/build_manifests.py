from __future__ import annotations
import argparse,gzip,json,os,pickletools
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd,yaml
from tartan.data.pose_utils import load_poses,poses_to_se2
from tartan.research_score.data.core import RouteSpec,arc_length,cache_key,file_sha256,group_split,nested_budgets,select_length,sha256_json

def atomic_json(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");tmp.replace(path)
def write_jsonl(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("x") as f:
  for r in rows:f.write(json.dumps(r,sort_keys=True)+"\n")
def safe_xy(path):
 raw=gzip.open(path,"rb").read();blobs=[a for o,a,_ in pickletools.genops(raw) if o.name in {"SHORT_BINBYTES","BINBYTES","BINBYTES8"} and isinstance(a,bytes) and len(a)>=24 and len(a)%12==0]
 if not blobs:return None
 a=np.frombuffer(max(blobs,key=len),dtype="<f4");return a.reshape(-1,3)[:,:2] if a.size%3==0 and np.isfinite(a).all() else None
def tartan_rows(root):
 rows=[];reject=[]
 for robot in ("anymal","diff","omni"):
  for pose in sorted((root/f"Data_{robot}").glob("*/pose_lcam_front.txt")):
   try:
    se2=poses_to_se2(load_poses(pose));meta=pose.parent/f"{pose.parent.name}_metadata.json"
    rows.append({"sample_id":f"tartan:{robot}:{pose.parent.name}","domain":"tartanground","embodiment":robot,"scene_id":root.name,"map_id":root.name,"trajectory_id":pose.parent.name,"episode_id":f"{robot}:{pose.parent.name}","pose_path":str(pose),"num_poses":len(se2),"arc_length_m":arc_length(se2[:,:2]),"metadata_path":str(meta),"metadata_sha256":file_sha256(meta) if meta.exists() else None,"route_source":"map_goal","source_temporal_space":False})
   except Exception as e:reject.append({"path":str(pose),"reason":type(e).__name__,"detail":str(e)})
 return rows,reject
def source_rows(root,ckpt_hash,spec):
 base=root/"dataset/nuplan-v1.1/exp/pluto/cache_mini";index=base/"metadata/cache_mini_metadata_node_0.csv";reject=[]
 s=pd.read_csv(index,usecols=["file_name"])["file_name"];values=s[s.str.endswith("/trajectory")].tolist();rows=[]
 for value in values:
  p=Path(value);scenario,stype,token=p.parts[-4],p.parts[-3],p.parts[-2];sid=f"nuplan:{scenario}:{stype}:{token}"
  rows.append({"sample_id":sid,"domain":"nuplan","embodiment":"car","scene_id":scenario,"map_id":"source_map_not_materialized","trajectory_id":token,"episode_id":scenario,"scenario_type":stype,"feature_path":value.rsplit("/",1)[0]+"/feature.gz","trajectory_path":value+".gz","cache_index_path":str(index),"arc_length_m":None,"route_source":"source_cached_route","source_temporal_space":True,"source_horizon_s":8,"num_points":80,"cache_key":cache_key(sid,"source-loader-v1",spec.hash,ckpt_hash)})
 # deterministic bounded content validation: first two rows per scenario type
 sampled=[]
 for stype,grp in pd.DataFrame(rows).groupby("scenario_type"):
  for rec in grp.iloc[:2].to_dict("records"):
   try:
    xy=safe_xy(Path(rec["trajectory_path"]));
    if xy is None:raise ValueError("no finite Nx3 ndarray")
    sampled.append({"sample_id":rec["sample_id"],"scenario_type":stype,"arc_length_m":arc_length(xy),"status":"PASS"})
   except Exception as e:sampled.append({"sample_id":rec["sample_id"],"scenario_type":stype,"status":"FAIL","reason":str(e)});reject.append({"path":rec["trajectory_path"],"reason":"sampled_content_validation","detail":str(e)})
 return rows,reject,sampled,index
def main():
 a=argparse.ArgumentParser();a.add_argument("--config",required=True);a.add_argument("--profile",default="all_available");a.add_argument("--output",required=True);a.add_argument("--run-id",required=True);z=a.parse_args();cfg=yaml.safe_load(Path(z.config).read_text());out=Path(z.output)/z.run_id;out.mkdir(parents=True,exist_ok=False)
 tartan=Path(os.environ["TARTAN_ROOT"]);nuplan=Path(os.environ["NUPLAN_ROOT"]);ckpt=Path(os.environ["SOURCE_CKPT"]);spec=RouteSpec(max_candidates=cfg["route"]["max_candidates"],points_per_candidate=cfg["trajectory"]["num_points"],dedup_overlap_threshold=cfg["route"]["dedup_overlap_threshold"])
 tr,trej=tartan_rows(tartan);sr,srej,sampled,index=source_rows(nuplan,file_sha256(ckpt),spec);spl=group_split([r["trajectory_id"] for r in tr if r["embodiment"]=="anymal"],cfg["splits"]["seed"])
 for r in tr:r["split"]=spl.get(r["trajectory_id"],"train")
 for r in sr:r["split"]="train"
 train_ids=[r["episode_id"] for r in tr if r["embodiment"]=="anymal" and r["split"]=="train"];budgets={str(seed):nested_budgets(train_ids,seed) for seed in cfg["budgets"]["joint_seeds"]}
 branches={"target_anymal":[r["arc_length_m"] for r in tr if r["embodiment"]=="anymal" and r["split"]=="train"],"proxy_diff":[r["arc_length_m"] for r in tr if r["embodiment"]=="diff"],"proxy_anymal":[r["arc_length_m"] for r in tr if r["embodiment"]=="anymal" and r["split"]=="train"]};non_source=select_length(branches,cfg["trajectory"]["length_candidates_m"],cfg["trajectory"]["min_train_coverage"])
 stationary=sum(r["scenario_type"]=="stationary" for r in sr);upper=1-stationary/len(sr)
 selected={"status":"BLOCKED","reason":"source_car_stationary_fraction_makes_90_percent_rule_impossible","selection_source":"train_manifest_only","selected_length_m":None,"candidates_m":cfg["trajectory"]["length_candidates_m"],"min_train_coverage":cfg["trajectory"]["min_train_coverage"],"source_cache_rows":len(sr),"source_stationary_rows":stationary,"source_coverage_upper_bound_for_every_candidate":upper,"proof":"all candidates >=8m; stationary samples cannot cover 8m; stationary fraction alone exceeds 10%","non_source_exact_audit":non_source};selected["train_statistics_sha256"]=sha256_json(selected);atomic_json(out/"selected_length.json",selected);atomic_json(out/"nuplan_sampled_content_validation.json",sampled)
 contracts={"transfer_primary":{"status":"BLOCKED_LENGTH_SELECTION","source":"nuplan/car","target":"tartanground/anymal","pair_level":"none","supports_claims":{"ordinary_transfer":True,"car_dog_scientific_claim":False}},"proxy_pair_auxiliary":{"status":"READY_FOR_PAIR_AUDIT","platform_a":"diff","platform_b":"anymal","pair_level":"proxy","supports_claims":{"pipeline_validation":True,"car_dog_scientific_claim":False}},"strict_pair":{"status":"BLOCKED_WAITING_DATA","pair_level":"strict","cf_pair_root":os.environ.get("CF_PAIR_ROOT",""),"supports_claims":{"pipeline_validation":False,"car_dog_scientific_claim":False}}}
 for name,c in contracts.items():
  d=out/name;d.mkdir();atomic_json(d/"profile_manifest.json",{"profile":name,"schema_version":"canonical-v1.2","route_spec":spec.__dict__,"route_spec_hash":spec.hash,"selected_length_artifact":str(out/"selected_length.json"),"selected_length_sha256":file_sha256(out/"selected_length.json"),"profile_contract":c,"budgets":budgets if name=="transfer_primary" else {},"source_normalizer_policy":"frozen_stage02","source_temporal_semantics":"8s/80 source regression only; not equivalent to fixed-arc-length/80"})
 write_jsonl(out/"transfer_primary/source_train.jsonl",sr)
 for split in ("train","val","test"):write_jsonl(out/f"transfer_primary/target_{split}.jsonl",[r for r in tr if r["embodiment"]=="anymal" and r["split"]==split])
 write_jsonl(out/"proxy_pair_auxiliary/diff_episodes.jsonl",[r for r in tr if r["embodiment"]=="diff"]);write_jsonl(out/"proxy_pair_auxiliary/anymal_episodes.jsonl",[r for r in tr if r["embodiment"]=="anymal"]);write_jsonl(out/"rejected_samples.jsonl",trej+srej)
 summary={"stage_status":"BLOCKED_LENGTH_SELECTION","tartan":{"total":len(tr),"by_embodiment":{x:sum(r["embodiment"]==x for r in tr) for x in ("anymal","diff","omni")},"target_split":{x:sum(r["embodiment"]=="anymal" and r["split"]==x for r in tr) for x in ("train","val","test")}},"nuplan_source_cache_rows":len(sr),"rejected":{"tartan":len(trej),"nuplan_sampled":len(srej)},"selected_length_m":None,"budgets":budgets};atomic_json(out/"data_summary.json",summary)
 atomic_json(out/"run_manifest.json",{"run_id":z.run_id,"stage":"03","status":"BLOCKED_LENGTH_SELECTION","created_at":datetime.now(timezone.utc).isoformat(),"profile_selector":z.profile,"config":str(Path(z.config).resolve()),"config_sha256":file_sha256(Path(z.config)),"checkpoint_sha256":file_sha256(ckpt),"data_roots":{"tartan":str(tartan),"nuplan":str(nuplan)},"data_index_sha256":sha256_json({"tartan":[r["metadata_sha256"] for r in tr],"nuplan_official_index_sha256":file_sha256(index)}),"data_hash_policy":"full official index SHA256 + bounded stratified content validation; not full gzip content audit","source_snapshot":"pending"});print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
