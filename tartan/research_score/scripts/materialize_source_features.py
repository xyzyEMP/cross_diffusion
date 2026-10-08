from __future__ import annotations
from tartan.research_score.artifacts import publish
import argparse,json,random
from pathlib import Path
import numpy as np,torch
from tartan.research_score.data.source_representation import TrajectoryRepresentationBridge

KEYS=("ego_current_state","neighbor_agents_past","lanes","lanes_speed_limit","lanes_has_speed_limit","route_lanes","route_lanes_speed_limit","route_lanes_has_speed_limit","static_objects")
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--samples',type=int,default=4096);p.add_argument('--seed',type=int,default=20260914);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
 if out.exists():raise FileExistsError(out)
 rows=[json.loads(x)['npz_reference'] for x in Path(a.manifest).open()];rng=random.Random(a.seed);indices=sorted(rng.sample(range(len(rows)),min(a.samples,len(rows))));bridge=TrajectoryRepresentationBridge();data={k:[] for k in KEYS};ys=[];ms=[];paths=[]
 for n,i in enumerate(indices,1):
  z=np.load(rows[i],allow_pickle=False)
  for k in KEYS:data[k].append(torch.from_numpy(z[k]))
  y,m=bridge(z['ego_agent_future']);ys.append(torch.from_numpy(y));ms.append(torch.from_numpy(m));paths.append(rows[i])
  if n%250==0:print(json.dumps({'built':n,'total':len(indices)}),flush=True)
 obj={k:torch.stack(v) for k,v in data.items()};obj.update({'trajectory':torch.stack(ys),'valid_mask':torch.stack(ms),'source_paths':paths,'manifest_pool_size':len(rows),'selection_seed':a.seed})
 publish(obj,out)
 print(json.dumps({'status':'complete','samples':len(indices),'manifest_pool_size':len(rows),'bytes':out.stat().st_size}))
if __name__=='__main__':main()
