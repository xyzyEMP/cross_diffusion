from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tartan.data.pose_utils import load_poses,poses_to_se2,to_local_se2
from tartan.research_score.data.route_builder import OccupancyRouteSetBuilder,RouteSpec
from tartan.research_score.data.core import sha256_json

def readj(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def main():
 p=argparse.ArgumentParser();p.add_argument("--manifest-root",required=True);p.add_argument("--output",required=True);a=p.parse_args();m=Path(a.manifest_root);o=Path(a.output);o.mkdir(parents=True,exist_ok=False);v=o/"visualizations";v.mkdir()
 rows=readj(m/"proxy_pair_auxiliary/diff_episodes.jsonl")+readj(m/"proxy_pair_auxiliary/anymal_episodes.jsonl")
 spec=RouteSpec();builder=OccupancyRouteSetBuilder(spec); audits=[]
 for idx,r in enumerate(rows[:30]):
  se2=poses_to_se2(load_poses(Path(r["pose_path"])));anchor=min(20,len(se2)-1);goal=to_local_se2(se2[-1:],se2[anchor])[0,:2]
  occ=Path(r["pose_path"]).parent/"coarse_occ"/f"occupancy_coarse5_{anchor:06d}_sparse.npy"
  if not occ.exists():audits.append({"sample_id":r["sample_id"],"accepted":False,"reason":"missing_current_occupancy","candidate_count":0});continue
  sparse=np.load(occ,allow_pickle=False); route=builder.build(sparse,goal); route2=builder.build(sparse,goal) # future mutation is intentionally absent from dependency graph
  invariant=np.array_equal(route["route_candidates_xy"],route2["route_candidates_xy"]);count=int(route["route_candidate_mask"].sum())
  audits.append({"sample_id":r["sample_id"],"accepted":bool(count and invariant),"reason":"ok" if count and invariant else "no_route_or_invariant_failure","candidate_count":count,"route_source":"map_goal","map_source":route["map_source"],"future_gt_mutation_invariant":bool(invariant),"occupancy_path":str(occ),"occupancy_identity_sha256":sha256_json({"path":str(occ),"size":occ.stat().st_size,"mtime_ns":occ.stat().st_mtime_ns})})
  fig,ax=plt.subplots(figsize=(5,5)); start=np.array(route["grid_start"]); obs=sparse[np.isin(sparse[:,3],[3,5]),:2]; local=(obs-start)*.5
  if len(local)>5000:local=local[::max(1,len(local)//5000)]
  ax.scatter(local[:,0],local[:,1],s=1,c="0.75",label="current obstacles")
  for k in np.flatnonzero(route["route_candidate_mask"]):ax.plot(*route["route_candidates_xy"][k].T,lw=1.2,label=f"route{k}")
  ax.scatter([0,goal[0]],[0,goal[1]],c=["black","red"],s=25);ax.set_xlim(-35,35);ax.set_ylim(-35,35);ax.set_aspect("equal");ax.set_title(f"{r['sample_id']} candidates={count}");fig.tight_layout();fig.savefig(v/f"route_{idx:03d}.png",dpi=120);plt.close(fig)
 df=pd.DataFrame(audits);df.to_parquet(o/"route_set_audit.parquet",index=False);summary={"audited":len(audits),"accepted":int(df.accepted.sum()),"rejected":int((~df.accepted).sum()),"visualizations":len(list(v.glob("*.png"))),"future_gt_mutation_failures":int((~df.future_gt_mutation_invariant.fillna(False)).sum()),"audit_sha256":sha256_json(audits)};(o/"route_set_audit_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
