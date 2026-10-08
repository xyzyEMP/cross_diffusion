import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tartan.research_score.data.route_builder import OccupancyRouteSetBuilder,RouteSpec,overlap_ratio
def read(path):return [json.loads(x) for x in path.read_text().splitlines() if x]
def digest(x):return hashlib.sha256(x["route_candidates_xy"].tobytes()+x["route_candidate_mask"].tobytes()).hexdigest()
def build_from_record(builder,record):
 # This is the production manifest boundary: route construction is allowed to
 # consume only the frozen map and goal. The trajectory remains in the record
 # solely for supervision/evaluation and must not affect the route-set.
 sparse=np.load(Path(record["route_set"]["map_reference"]),allow_pickle=False)
 goal=np.asarray(record["fixed_goal"]["xy_local"],float)
 return builder.build(sparse,goal),sparse,goal
def main():
 p=argparse.ArgumentParser();p.add_argument("--profile",choices=("transfer_primary","four_groups"),default="transfer_primary");p.add_argument("--manifest");p.add_argument("--run");p.add_argument("--output",required=True);a=p.parse_args();
 if a.profile=="four_groups":return audit_four_groups(a)
 run=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);viz=out/"visualizations";viz.mkdir();cache=out/"route_cache";cache.mkdir()
 rows=[]
 for split in ("train","val","test"):rows+=read(run/f"transfer_primary/target_{split}_windows.jsonl")
 # Deterministic, map-present sampling across every split and both branches.
 chosen=[]
 for split in ("train","val","test"):
  pool=[]
  for branch in ("moving_planning","stop_or_short"):
   z=[r for r in rows if r["split"]==split and r["branch"]==branch and Path(r["route_set"]["map_reference"]).exists()]
   step=max(1,len(z)//5);pool+=z[::step][:5]
  if len(pool)<10:
   used={r["sample_id"] for r in pool};z=[r for r in rows if r["split"]==split and r["sample_id"] not in used and Path(r["route_set"]["map_reference"]).exists()]
   step=max(1,len(z)//max(1,10-len(pool)));pool+=z[::step][:10-len(pool)]
  chosen+=pool[:10]
 builder=OccupancyRouteSetBuilder(RouteSpec(max_candidates=6,points_per_candidate=80,dedup_overlap_threshold=.90));aud=[]
 for i,r in enumerate(chosen):
  mp=Path(r["route_set"]["map_reference"]);goal=np.asarray(r["fixed_goal"]["xy_local"],float)
  if not mp.exists():
   aud.append({"sample_id":r["sample_id"],"split":r["split"],"branch":r["branch"],"status":"REJECT","reason":"map_missing","candidate_count":0});continue
  route,sparse,goal=build_from_record(builder,r);before=digest(route)
  # Mutate the actual future-supervision field in a deep-copied manifest row,
  # while keeping the frozen task definition unchanged, then rebuild through
  # the same manifest boundary used above.
  mutated=copy.deepcopy(r)
  raw=np.asarray(mutated["trajectory"]["raw_future_xy_yaw"],dtype=float)
  raw[:,:2]=raw[::-1,:2]*-17.0+31.0
  mutated["trajectory"]["raw_future_xy_yaw"]=raw.tolist()
  mutated_route,_,_=build_from_record(builder,mutated)
  after=digest(mutated_route);invariant=before==after
  d=route["diagnostics"];count=int(route["route_candidate_mask"].sum());status="PASS" if count and invariant else "REJECT";reason=d.get("reason","unknown") if count==0 else ("future_mutation_changed_input" if not invariant else "ok")
  valid=[route["route_candidates_xy"][k] for k in np.flatnonzero(route["route_candidate_mask"])];ovs=[overlap_ratio(valid[x],valid[y]) for x in range(len(valid)) for y in range(x)]
  aud.append({"sample_id":r["sample_id"],"split":r["split"],"branch":r["branch"],"status":status,"reason":reason,"candidate_count":count,"future_gt_mutation_invariant":invariant,"goal_rule":r["fixed_goal"]["rule"],"goal_xy":json.dumps(goal.tolist()),"map_reference":str(mp),"start_occupied":d.get("start_occupied"),"goal_occupied":d.get("goal_occupied"),"start_adjusted":d.get("start_adjusted"),"goal_adjusted":d.get("goal_adjusted"),"out_of_bounds":d.get("out_of_bounds",False),"unknown_fraction":d.get("unknown_fraction"),"max_pair_overlap":max(ovs) if ovs else 0.0,"dedup_threshold":.90})
  np.savez_compressed(cache/f"{i:03d}.npz",route_candidates_xy=route["route_candidates_xy"],route_candidate_mask=route["route_candidate_mask"],fixed_goal_xy=goal)
  fig,ax=plt.subplots(figsize=(5,5));st=np.asarray(route["grid_start"]);obs=sparse[np.isin(sparse[:,3],[3,5]),:2];local=(obs-st)*.5
  if len(local)>5000:local=local[::max(1,len(local)//5000)]
  ax.scatter(local[:,0],local[:,1],s=1,c="0.75")
  for k,q in enumerate(valid):ax.plot(q[:,0],q[:,1],lw=1,label=f"r{k}")
  ax.scatter([0,goal[0]],[0,goal[1]],c=["black","red"],s=25);ax.set_xlim(-35,35);ax.set_ylim(-35,35);ax.set_aspect("equal");ax.set_title(f"{r['sample_id']}\n{status}/{reason}, n={count}");fig.tight_layout();fig.savefig(viz/f"route_{i:03d}.png",dpi=120);plt.close(fig)
 (out/"route_audit.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in aud));pd.DataFrame(aud).to_parquet(out/"route_audit.parquet",index=False)
 sb={f"{s}/{b}":int(n) for (s,b),n in pd.DataFrame(aud).groupby(["split","branch"]).size().items()}
 summary={"audited":len(aud),"passed":sum(x["status"]=="PASS" for x in aud),"rejected":sum(x["status"]!="PASS" for x in aud),"reason_counts":pd.Series([x["reason"] for x in aud]).value_counts().to_dict(),"candidate_count_distribution":pd.Series([x["candidate_count"] for x in aud]).value_counts().sort_index().to_dict(),"split_branch_counts":sb,"future_mutation_failures":sum(not x.get("future_gt_mutation_invariant",False) for x in aud if x["reason"]!="map_missing"),"visualizations":len(list(viz.glob("*.png")))};(out/"route_audit_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
def audit_four_groups(a):
 """Real validation tasks only; collision conflict is reported without filtering."""
 from tartan.data.pose_utils import load_occupancy_record,load_proxy_se2,to_local_se2,local_xy_to_global,read_proxy_trajectories
 from tartan.research_score.evaluation.goal_benchmark import prepare_episode
 from tartan.research_score.evaluation.collision import path_collision
 from tartan.research_score.artifacts import publish
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);source=Path(a.manifest)
 records=read(source);trajectories={r['trajectory_key']:r for r in read_proxy_trajectories(source.parent/'trajectories.jsonl')};checks=[];plotted=set()
 for r in records:
  se2=load_proxy_se2(trajectories[r['trajectory_key']]);anchor=r['anchor_index'];end=r['segment_end_index']
  local=to_local_se2(se2[anchor:end+1],se2[anchor]);roundtrip=local_xy_to_global(local[:,:2],se2[anchor]);error=float(np.max(np.abs(roundtrip-se2[anchor:end+1,:2])))
  target=np.asarray(r['trajectory']['fixed_arc_length_80']);shape_ok=target.shape==(80,4) and np.isfinite(target).all() and all(r['trajectory']['valid_mask'])
  sparse=load_occupancy_record(r);route,blocked,start,goal,cells,_,_,invalid=prepare_episode(r,r['embodiment'])
  mutated=dict(r);mutated['trajectory']={**r['trajectory'],'fixed_arc_length_80':np.zeros((80,4)).tolist()}
  alternative=OccupancyRouteSetBuilder(RouteSpec(max_candidates=6,points_per_candidate=80),grid_size=101).build(load_occupancy_record(mutated),goal)
  invariant=np.array_equal(route['route_candidates_xy'],alternative['route_candidates_xy']) and np.array_equal(route['route_candidate_mask'],alternative['route_candidate_mask'])
  hit,_=path_collision(np.vstack(([0.,0.],target[:,:2])),blocked,start)
  checks.append({'sample_id':r['sample_id'],'roundtrip_error_m':error,'full_8m_supervision':bool(shape_ok),'future_input_invariant':bool(invariant),'recorded_path_collision':bool(hit),'invalid_start':bool(invalid),'route_failure':cells is None,'passed':bool(shape_ok and invariant and error<1e-4)})
  if r['embodiment'] not in plotted:
   plotted.add(r['embodiment']);fig,ax=plt.subplots(figsize=(6,6));obs=np.argwhere(blocked);xy=(obs-start)*.5
   ax.scatter(xy[:,0],xy[:,1],s=2,c='0.7');ax.plot(target[:,0],target[:,1],label='recorded 8m');ax.scatter([0.,goal[0]],[0.,goal[1]],c=['black','red']);ax.set(xlim=(-10,10),ylim=(-10,10),aspect='equal',title=r['sample_id']);ax.legend();fig.tight_layout();fig.savefig(out/(r['embodiment']+'_alignment.png'),dpi=120);plt.close(fig)
 status='PASS' if checks and all(x['passed'] for x in checks) else 'BLOCKED_GEOMETRY'
 publish({'status':status,'task_count':len(checks),'recorded_path_collision_count':sum(x['recorded_path_collision'] for x in checks),'route_failure_count':sum(x['route_failure'] for x in checks),'invalid_start_count':sum(x['invalid_start'] for x in checks),'policy':'report conflicts; never filter tasks or loosen thresholds','checks':checks},out/'summary.json',True)
 if status!='PASS':raise ValueError(status)

if __name__=="__main__":main()
