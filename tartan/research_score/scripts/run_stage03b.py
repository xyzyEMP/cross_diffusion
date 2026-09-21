from __future__ import annotations
import argparse,hashlib,json,os,shutil
from pathlib import Path
import numpy as np,pandas as pd,yaml
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tartan.research_score.evaluation.goal_benchmark import analytic_cases,evaluate_policy,no_future_check
from tartan.research_score.evaluation.metrics_navigation import aggregate

def publish_tree(src,dst):
 if dst.exists():raise FileExistsError(dst)
 staging=dst.with_name("."+dst.name+".publishing");staging.mkdir(parents=True,exist_ok=False)
 for source in src.rglob("*"):
  rel=source.relative_to(src);target=staging/rel
  if source.is_dir():target.mkdir(exist_ok=True);continue
  target.parent.mkdir(parents=True,exist_ok=True);ok=False
  for attempt in range(2):
   part=target.with_name(target.name+f".part{attempt}")
   with source.open("rb") as fi,part.open("xb") as fo:shutil.copyfileobj(fi,fo,8*1024*1024);fo.flush();os.fsync(fo.fileno())
   if source.stat().st_size==part.stat().st_size:os.replace(part,target);ok=True;break
   part.unlink()
  if not ok:raise IOError(f"publish size mismatch: {rel}")
 os.replace(staging,dst)

def load_rows(run,include_test=False):
    z=[]
    for s in (("train","val","test") if include_test else ("train","val")):
        with (run/f"transfer_primary/target_{s}_windows.jsonl").open() as h:z += [json.loads(x) for x in h]
    return z
def terrain_score(r):
 x=np.load(r["route_set"]["map_reference"],allow_pickle=False)
 z=np.asarray(x[:,2],float);sem=np.asarray(x[:,3],int)
 return float(np.ptp(z)+5*np.mean(np.isin(sem,[2,4])))
def choose_smoke(rows,audit):
 by={r["sample_id"]:r for r in rows};dev=[x for x in audit if x["split"] in {"train","val"} and x["sample_id"] in by]
 used=set();chosen=[]
 def take(label,seq):
  got=[]
  for x in seq:
   if x["sample_id"] not in used:got.append(x);used.add(x["sample_id"])
   if len(got)==5:break
  if len(got)<5:raise RuntimeError(f"insufficient {label} smoke cases")
  chosen.extend((label,by[x["sample_id"]],x) for x in got)
 normal=sorted(dev,key=lambda x:(-x["candidate_count"],x["unknown_fraction"],x["sample_id"]))
 narrow=sorted(dev,key=lambda x:(x["candidate_count"] if x["candidate_count"] else 99,-int(x["goal_adjusted"]),-x["unknown_fraction"],x["sample_id"]))
 terrain=sorted(dev,key=lambda x:(-terrain_score(by[x["sample_id"]]),x["sample_id"]))
 take("normal",normal);take("narrow",narrow);take("terrain",terrain);return chosen
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--stage03-run",required=True);p.add_argument("--output",required=True);p.add_argument("--run-id",required=True);a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text());root=Path(a.output)/a.run_id
 if root.exists():raise FileExistsError(root)
 tmp=Path("/tmp")/a.run_id
 if tmp.exists():raise FileExistsError(tmp)
 tmp.mkdir();(tmp/"visualizations").mkdir();run=Path(a.stage03_run);rows=load_rows(run,True)
 audit_path=Path(cfg["stage03_route_audit_jsonl"]);audit=[json.loads(x) for x in audit_path.open()]
 selected_meta=choose_smoke(rows,audit);selected=[r for _,r,_ in selected_meta]
 policies=("shortest_path_follower","intentional_collision_policy","stop_policy","single_plan_map_reference");out=[]
 from tartan.research_score.evaluation.goal_benchmark import build_from_record,prepare_episode
 prepared={}
 for r in selected:
  base=build_from_record(r)
  for robot in ("diff","anymal"):
   prepared[(r["sample_id"],robot)]=prepare_episode(r,robot,base)
   for policy in policies:out.append(evaluate_policy(r,robot,policy,prepared[(r["sample_id"],robot)]))
 future=[{"sample_id":r["sample_id"],**no_future_check(r)} for r in selected]
 pd.DataFrame(out).to_parquet(tmp/"per_episode_metrics.parquet",index=False);pd.DataFrame(out).to_csv(tmp/"per_episode_metrics.csv",index=False)
 summary={f"{robot}/{policy}":aggregate([x for x in out if x["robot"]==robot and x["policy"]==policy and x["included_in_denominator"]]) for robot in ("diff","anymal") for policy in policies}
 (tmp/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");(tmp/"analytic_cases.json").write_text(json.dumps(analytic_cases(),indent=2)+"\n");(tmp/"no_future_invariant.json").write_text(json.dumps({"passed":all(x["passed"] for x in future),"cases":future},indent=2)+"\n")
 pd.DataFrame([x for x in out if x["route_failure"]]).to_parquet(tmp/"route_failure_episodes.parquet",index=False)
 pd.DataFrame([x for x in out if not x["included_in_denominator"]]).to_parquet(tmp/"rejected_episodes.parquet",index=False)
 for i,(category,r,_) in enumerate(selected_meta):
  route,_,goal=build_from_record(r);fig,ax=plt.subplots(figsize=(5,5));
  for k in np.flatnonzero(route["route_candidate_mask"]):ax.plot(route["route_candidates_xy"][k,:,0],route["route_candidates_xy"][k,:,1],lw=1)
  ax.scatter([0,goal[0]],[0,goal[1]],c=["black","red"]);ax.set_aspect("equal");ax.set_title(f"{category}: {r['sample_id']}");fig.tight_layout();fig.savefig(tmp/"visualizations"/f"{category}_{i:02d}.png",dpi=120);plt.close(fig)
 manifest=[{"category":c,"sample_id":r["sample_id"],"split":m["split"],"stage03_reason":m["reason"]} for c,r,m in selected_meta]
 pd.DataFrame(manifest).to_parquet(tmp/"episode_manifest.parquet",index=False)
 (tmp/"episode_manifest.json").write_text(json.dumps({"selection":"stage03_frozen_route_audit_train_val_stratified_5x3","test_excluded_from_metrics":True,"episodes":manifest,"policies":policies,"robots":["diff","anymal"],"route_failure_rule":"D024_safe_stop_in_denominator_not_collision"},indent=2)+"\n")
 # Audit all frozen 30, including held-out, only for uniform D024 boundary handling—not model metrics.
 boundary=[];all_by={r["sample_id"]:r for r in rows}
 for m in audit:
  if m["reason"]=="astar_disconnected":
   for robot in ("diff","anymal"):
    for policy in policies:boundary.append({"sample_id":m["sample_id"],"split":m["split"],"robot":robot,"policy":policy,"termination_reason":"route_failure","success":0,"collision":0,"spl":0.0,"included_in_denominator":True,"decision":"D024"})
 pd.DataFrame(boundary).to_parquet(tmp/"astar_disconnected_boundary.parquet",index=False)
 shutil.copy2(a.config,tmp/"config.resolved.yaml")
 (tmp/"oracle_reference.json").write_text(json.dumps({"status":"historical_boundary_only","source":cfg.get("historical_oracle_reference"),"included_in_main_metrics":False,"reason":"oracle route reads recorded future and is not comparable to the no-leak benchmark"},indent=2)+"\n")
 root.parent.mkdir(parents=True,exist_ok=True);publish_tree(tmp,root)
 print(json.dumps({"status":"AWAITING_REVIEW","cases":len(selected),"rollouts":len(out),"no_future":all(x["passed"] for x in future),"summary":summary},indent=2))
if __name__=="__main__":main()
