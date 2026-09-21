"""Recover completed source raw scan and diagnose v1.2.1 length selection."""
import argparse, json, os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, yaml
from tartan.data.pose_utils import load_poses, poses_to_se2, to_local_se2
from tartan.research_score.scripts.build_remediation import arclen, classify, safe_traj
from tartan.research_score.data.core import group_split


def main():
 p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--raw",required=True);p.add_argument("--failed-log",required=True);p.add_argument("--output",required=True);a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 log=Path(a.failed_log).read_text(errors="replace")
 if "no candidate satisfies v1.2.1 moving-window coverage" not in log:raise RuntimeError("failed log does not prove execution passed source_scan")
 nuplan=Path(os.environ["NUPLAN_ROOT"]);index=nuplan/"dataset/nuplan-v1.1/exp/pluto/cache_mini/metadata/cache_mini_metadata_node_0.csv";vals=pd.read_csv(index,usecols=["file_name"])["file_name"];vals=vals[vals.str.endswith("/trajectory")].tolist();paths=[Path(x+".gz") for x in vals]
 raw=np.load(a.raw,mmap_mode="r");expected=(len(paths),int(cfg["source_scan"]["future_steps"]),3)
 if raw.shape!=expected or raw.dtype!=np.float32:raise RuntimeError(f"bad raw shard {raw.shape}/{raw.dtype}, expected {expected}/float32")
 lengths=np.zeros(len(paths),np.float32)
 for lo in range(0,len(paths),4096):
  x=np.asarray(raw[lo:lo+4096]);lengths[lo:lo+len(x)]=np.linalg.norm(np.diff(x[:,:,:2],axis=1),axis=2).sum(axis=1)
 valid=np.full(len(paths),expected[1],np.uint8);zeros=np.flatnonzero(~np.any(raw,axis=(1,2)));shape_counts=Counter();reject=[]
 def one(i):
  try:return i,safe_traj(paths[i]),None
  except Exception as e:return i,None,f"{type(e).__name__}:{e}"
 with ThreadPoolExecutor(max_workers=cfg["source_scan"]["workers"]) as pool:
  for i,x,e in pool.map(one,zeros.tolist(),chunksize=128):
   if e:valid[i]=0;reject.append({"row":i,"path":str(paths[i]),"reason":e})
   else:
    k=min(expected[1],len(x));valid[i]=k;lengths[i]=arclen(x[:k]);shape_counts[str(tuple(x.shape))]+=1
 # Verify 1000 nonzero rows and record actual shape distribution.
 nz=np.flatnonzero(np.any(raw,axis=(1,2)));sample=nz[np.linspace(0,len(nz)-1,min(1000,len(nz)),dtype=int)]
 mismatch=[]
 for i in sample:
  x=safe_traj(paths[int(i)]);k=min(expected[1],len(x));shape_counts[str(tuple(x.shape))]+=1
  if not np.array_equal(np.asarray(raw[int(i),:k]),x[:k]):mismatch.append(int(i))
 source_hz=float(cfg["source_scan"]["sample_rate_hz"]);src_moving=np.asarray([lengths[i] for i in range(len(paths)) if valid[i] and classify(float(lengths[i]),float(valid[i])/source_hz,cfg)=="moving_planning"])
 tartan=Path(os.environ["TARTAN_ROOT"]);episodes=[]
 for pose in sorted((tartan/"Data_anymal").glob("*/pose_lcam_front.txt")):episodes.append((pose.parent.name,pose,poses_to_se2(load_poses(pose))))
 split=group_split([e[0] for e in episodes],cfg["split"]["seed"],cfg["split"]["train"],cfg["split"]["val"]);target=[]
 for ep,pose,se2 in episodes:
  for anchor in range(cfg["tartan"]["history_steps"],len(se2)-1,cfg["tartan"]["anchor_stride"]):
   local=to_local_se2(se2[anchor+1:min(len(se2),anchor+1+cfg["tartan"]["future_steps"])],se2[anchor]);length=arclen(local);horizon=len(local)/cfg["tartan"]["sample_rate_hz"]
   target.append((split[ep],length,horizon,classify(length,horizon,cfg),len(local)))
 tgt_moving=np.asarray([x[1] for x in target if x[0]=="train" and x[3]=="moving_planning"]);cands=cfg["trajectory"]["length_candidates_m"]
 coverage={str(c):{"source_car":float(np.mean(src_moving>=c)) if len(src_moving) else 0.,"target_anymal":float(np.mean(tgt_moving>=c)) if len(tgt_moving) else 0.} for c in cands}
 result={"completion_evidence":"failed build reached post-scan length selection","source":{"rows":len(paths),"moving":len(src_moving),"stop_or_short":int(len(paths)-len(src_moving)),"zero_rows_reparsed":len(zeros),"rejected":len(reject),"sampled_nonzero_validation":len(sample),"mismatches":mismatch,"shape_counts_zero_plus_sample":dict(shape_counts),"arc_percentiles_m":np.percentile(lengths[valid>0],[0,10,25,50,75,90,100]).tolist()},"target":{"episodes":len(episodes),"windows":len(target),"split":dict(Counter(x[0] for x in target)),"moving":sum(x[3]=="moving_planning" for x in target),"stop_or_short":sum(x[3]=="stop_or_short" for x in target),"train_moving":len(tgt_moving),"future_point_counts":dict(Counter(x[4] for x in target)),"horizon_s":dict(Counter(x[2] for x in target))},"coverage":coverage,"threshold":cfg["trajectory"]["min_train_coverage"],"eligible":[c for c in cands if min(coverage[str(c)].values())>=cfg["trajectory"]["min_train_coverage"]],"rejected_rows":reject}
 (out/"length_diagnostic.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps({k:result[k] for k in ("source","target","coverage","eligible")},indent=2))
if __name__=="__main__":main()
