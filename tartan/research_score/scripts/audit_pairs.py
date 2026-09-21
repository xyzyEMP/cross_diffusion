from __future__ import annotations
import argparse,json,os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tartan.data.pose_utils import load_poses,poses_to_se2
from tartan.research_score.data.core import arc_length,sha256_json

def readj(path): return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def normpath(x,n=120):
    s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(x[:,:2],axis=0),axis=1))]; q=np.linspace(0,s[-1],n)
    return np.column_stack([np.interp(q,s,x[:,i]) for i in range(2)])
def main():
    p=argparse.ArgumentParser();p.add_argument("--profile",required=True);p.add_argument("--manifest-root",required=True);p.add_argument("--output",required=True);p.add_argument("--max-start-m",type=float,default=20);p.add_argument("--max-goal-m",type=float,default=25);a=p.parse_args()
    if a.profile!="proxy_pair_auxiliary": raise SystemExit("only proxy_pair_auxiliary is available")
    root=Path(a.manifest_root); out=Path(a.output); out.mkdir(parents=True,exist_ok=False); viz=out/"visualizations";viz.mkdir()
    ds=readj(root/"diff_episodes.jsonl"); anys=readj(root/"anymal_episodes.jsonl")
    trajectories={r["episode_id"]:poses_to_se2(load_poses(Path(r["pose_path"]))) for r in ds+anys}
    candidates=[]
    for d in ds:
      xd=trajectories[d["episode_id"]]; nd=normpath(xd)
      for b in anys:
        xb=trajectories[b["episode_id"]]; nb=normpath(xb)
        direct=float(np.linalg.norm(nd-nb,axis=1).mean()); reverse=float(np.linalg.norm(nd-nb[::-1],axis=1).mean())
        rev=reverse<direct; score=min(direct,reverse); bb=nb[::-1] if rev else nb
        candidates.append((score,d,b,float(np.linalg.norm(nd[0]-bb[0])),float(np.linalg.norm(nd[-1]-bb[-1])),rev,nd,bb))
    candidates.sort(key=lambda x:x[0]); rows=[]
    # One-to-one episode matching prevents either platform trajectory from
    # appearing in multiple splits. Six deterministic progress anchors provide
    # the requested 30 visual audits without pretending they are 30 episodes.
    selected=[]; used_a=set();used_b=set()
    for item in candidates:
      d,b=item[1],item[2]
      if d["trajectory_id"] in used_a or b["trajectory_id"] in used_b:continue
      selected.append(item);used_a.add(d["trajectory_id"]);used_b.add(b["trajectory_id"])
      if len(selected)==min(len(ds),5):break
    split_names=["train","train","train","val","test"]
    i=0
    for pair_idx,(score,d,b,start,goal,rev,nd,nb) in enumerate(selected):
      valid=start<=a.max_start_m and goal<=a.max_goal_m; split=split_names[pair_idx]
      for anchor in range(6):
        frac=(anchor+1)/6; upto=max(2,int(round(len(nd)*frac)))
        row={"pair_id":f"proxy:{d['trajectory_id']}:{b['trajectory_id']}:{anchor:02d}","pair_episode_id":f"proxy:{d['trajectory_id']}:{b['trajectory_id']}","anchor_index":anchor,"platform_a_sample_id":d["sample_id"],"platform_b_sample_id":b["sample_id"],"platform_a_embodiment":"diff","platform_b_embodiment":"anymal","common_scene_id":d["scene_id"],"common_goal_id":f"endpoint:{d['trajectory_id']}:{b['trajectory_id']}","T_a_to_common":[[1,0,0],[0,1,0],[0,0,1]],"T_b_to_common":[[1,0,0],[0,1,0],[0,0,1]],"pair_valid":valid,"pair_level":"proxy","pair_source":"tartan_cross_robot","supports_claims":{"pipeline_validation":True,"car_dog_scientific_claim":False},"pair_quality_flags":[] if valid else ["endpoint_alignment_threshold"],"provenance":{"mean_shape_distance_m":score,"start_error_m":start,"goal_error_m":goal,"reverse_b":rev,"selection":"geometry_only_not_outcome","progress_fraction":frac},"split":split}
        rows.append(row)
        fig,ax=plt.subplots(figsize=(5,5));ax.plot(nd[:upto,0],nd[:upto,1],"b-",label="Diff");ax.plot(nb[:upto,0],nb[:upto,1],"r-",label="ANYmal");ax.scatter([nd[0,0],nb[0,0]],[nd[0,1],nb[0,1]],c=["navy","darkred"],s=20);ax.set_aspect("equal");ax.legend();ax.set_title(f"proxy {i:02d} {split} valid={valid}\nmean={score:.1f} start={start:.1f} goal={goal:.1f}");fig.tight_layout();fig.savefig(viz/f"pair_{i:03d}.png",dpi=120);plt.close(fig);i+=1
    with (out/"pair_audit.jsonl").open("x") as f:
      for r in rows:f.write(json.dumps(r,sort_keys=True)+"\n")
    flat=[]
    for r in rows:
      q=dict(r);q["supports_claims"]=json.dumps(q["supports_claims"],sort_keys=True);q["pair_quality_flags"]=json.dumps(q["pair_quality_flags"]);q["provenance"]=json.dumps(q["provenance"],sort_keys=True);q["T_a_to_common"]=json.dumps(q["T_a_to_common"]);q["T_b_to_common"]=json.dumps(q["T_b_to_common"]);flat.append(q)
    pd.DataFrame(flat).to_parquet(out/"pair_audit.parquet",index=False)
    summary={"candidate_episode_pairs":len(candidates),"selected_disjoint_episode_pairs":len(selected),"audited_pair_windows":len(rows),"split_windows":{s:sum(r["split"]==s for r in rows) for s in ("train","val","test")},"valid":sum(r["pair_valid"] for r in rows),"rejected":sum(not r["pair_valid"] for r in rows),"visualizations":len(list(viz.glob("*.png"))),"claim_scope":"proxy pipeline only; never Car-Dog scientific evidence","audit_sha256":sha256_json(rows)}
    (out/"pair_audit_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
