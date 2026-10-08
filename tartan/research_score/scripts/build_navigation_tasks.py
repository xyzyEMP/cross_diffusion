from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from tartan.data.pose_utils import load_poses,poses_to_se2,to_local_se2

def interp_goal(local,distance):
 seg=np.linalg.norm(np.diff(local[:,:2],axis=0),axis=1);s=np.r_[0.,np.cumsum(seg)]
 return np.array([np.interp(distance,s,local[:,0]),np.interp(distance,s,local[:,1])],np.float32)

def main():
 p=argparse.ArgumentParser();p.add_argument('--future-stations',action='store_true');p.add_argument('--profile',default='transfer_primary',choices=('transfer_primary','proxy_ab'));p.add_argument('--input-manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--distance-m',type=float,default=8.);p.add_argument('--anchor-stride',type=int,default=10);a=p.parse_args()
 if a.profile=='proxy_ab':return build_proxy_tasks(a)
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
def build_proxy_tasks(a):
 from tartan.research_score.data.core import proxy_window,four_group_window
 from tartan.data.pose_utils import read_proxy_trajectories,load_proxy_se2
 import os,tempfile
 if a.distance_m!=8. or a.anchor_stride!=10:raise ValueError('Proxy fixed 8m/stride10 config conflict')
 source=Path(a.input_manifest);data=source.parent
 trajectories={r['trajectory_key']:r for r in read_proxy_trajectories(data/'trajectories.jsonl')}
 base=[json.loads(x) for x in source.read_text().splitlines() if x];keys=sorted({r['trajectory_key'] for r in base});rows=[];counts={}
 for key in keys:
  tr=trajectories[key];se2=load_proxy_se2(tr);anchor=20;segment=0
  while anchor<len(se2)-1:
   try:window=(four_group_window if a.future_stations else proxy_window)(tr,se2,anchor,data/'trajectories.jsonl')
   except ValueError as e:
    if str(e)=='incomplete_8m':break
    raise
   if not all(window['trajectory']['valid_mask']):break
   end=window['trajectory']['source_end_frame']
   window.update(sample_id=f'{key}:nonoverlap8m:{segment:06d}',segment_index=segment,segment_end_index=end)
   window['provenance'].update(nonoverlap=True,anchor_stride=10)
   rows.append(window);segment+=1;anchor=20+int(np.ceil((end-20)/10.))*10
  counts[key]=segment
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists():raise FileExistsError(out)
 fd,tmp=tempfile.mkstemp(prefix='.'+out.name+'.',dir=out.parent)
 with os.fdopen(fd,'w') as h:
  for r in rows:h.write(json.dumps(r,sort_keys=True)+'\n')
  h.flush();os.fsync(h.fileno())
 os.replace(tmp,out)
 print(json.dumps({'segments':len(rows),'counts':counts,'status':'PREPARED' if rows else 'BLOCKED_NO_VERIFIED_WINDOWS','output':str(out)}))

if __name__=='__main__':main()
