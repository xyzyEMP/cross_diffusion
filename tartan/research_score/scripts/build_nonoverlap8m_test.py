from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from tartan.data.pose_utils import load_poses,poses_to_se2,to_local_se2

def interp_goal(local,distance):
 seg=np.linalg.norm(np.diff(local[:,:2],axis=0),axis=1);s=np.r_[0.,np.cumsum(seg)]
 return np.array([np.interp(distance,s,local[:,0]),np.interp(distance,s,local[:,1])],np.float32)

def main():
 p=argparse.ArgumentParser();p.add_argument('--input-manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--distance-m',type=float,default=8.);p.add_argument('--anchor-stride',type=int,default=10);a=p.parse_args()
 by={}
 for line in Path(a.input_manifest).open():
  r=json.loads(line);by.setdefault(r['episode_id'],[]).append(r)
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);rows=[]
 for episode,records in sorted(by.items()):
  template=min(records,key=lambda r:r['anchor_index']);pose=Path(template['trajectory']['raw_reference']);se2=poses_to_se2(load_poses(pose));anchors=list(range(0,len(se2),a.anchor_stride));anchor=0;segment=0
  while anchor<len(se2)-1:
   local=to_local_se2(se2[anchor+1:],se2[anchor]);seg=np.linalg.norm(np.diff(np.vstack(([0.,0.],local[:,:2])),axis=0),axis=1);cum=np.cumsum(seg);hit=np.flatnonzero(cum>=a.distance_m)
   if not len(hit):break
   end=anchor+1+int(hit[0]);goal=interp_goal(np.vstack((np.array([[0.,0.,0.]],np.float32),local[:int(hit[0])+1])),a.distance_m)
   occ=pose.parent/'coarse_occ'/f'occupancy_coarse5_{anchor:06d}_sparse.npy'
   if not occ.exists():raise FileNotFoundError(occ)
   r=dict(template);r.update({'sample_id':f'{episode}:nonoverlap8m:{segment:03d}','segment_index':segment,'anchor_index':anchor,'segment_end_index':end,'branch':'moving_planning'})
   r['current_state']={'global_se2_index':anchor,'se2_local':[0.,0.,0.]};r['fixed_goal']={'xy_local':goal.tolist(),'rule':'nonoverlap_fixed_arc_8m','requested_distance_m':a.distance_m,'actual_distance_m':a.distance_m};r['route_set']={**r['route_set'],'map_reference':str(occ),'future_gt_dependency':False};r['trajectory']={'raw_reference':str(pose),'raw_points':end-anchor,'representation':'evaluation_segment_only'};r['provenance']={**r.get('provenance',{}),'protocol_revision':'score-decomp-transfer-v1.2.4','nonoverlap':True,'anchor_stride':a.anchor_stride};rows.append(r);segment+=1
   candidates=[q for q in anchors if q>=end]
   if not candidates:break
   anchor=min(candidates)
 with out.open('x') as h:
  for r in rows:h.write(json.dumps(r,separators=(',',':'),sort_keys=True)+'\n')
 counts={}
 for r in rows:counts[r['episode_id']]=counts.get(r['episode_id'],0)+1
 print(json.dumps({'segments':len(rows),'episodes':len(counts),'counts':counts,'output':str(out)},indent=2))
if __name__=='__main__':main()
