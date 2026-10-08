from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
import numpy as np,torch
from torch.utils.data import Dataset
from tartan.data.pose_utils import load_occupancy_record
from tartan.data.features import _polyline_features,_route_segments
from tartan.research_score.data.route_builder import OccupancyRouteSetBuilder,RouteSpec

class TartanTargetDataset(Dataset):
 def __init__(self,manifest,config,budget=None,repeat_id=11,limit=None):
  rows=[json.loads(x) for x in Path(manifest).open()]
  tag=str(budget);seed=str(repeat_id)
  self.rows=rows if budget is None else [r for r in rows if tag in r.get("budget_membership",{}).get(seed,[])]
  if limit:self.rows=self.rows[:limit]
  self.c=config;self.builder=OccupancyRouteSetBuilder(RouteSpec(max_candidates=6,points_per_candidate=80))
 def __len__(self):return len(self.rows)
 @lru_cache(maxsize=None)
 def _features(self,i):
  r=self.rows[i];s=load_occupancy_record(r);g=np.asarray(r["fixed_goal"]["xy_local"],np.float32)
  builder=OccupancyRouteSetBuilder(self.builder.spec,grid_size=101) if r.get("profile")=="proxy_ab" else self.builder
  q=builder.build(s,g);paths=q["route_candidates_xy"];mask=q["route_candidate_mask"]
  lanes=np.zeros((self.c.lane_num,self.c.lane_len,self.c.lane_state_dim),np.float32);routes=np.zeros((self.c.route_num,self.c.route_len,self.c.route_state_dim),np.float32)
  n=0
  for k in np.flatnonzero(mask):
   for seg in _polyline_features(_route_segments(paths[k],4,self.c.lane_len),.8):
    if n<self.c.route_num:routes[n]=seg[:,:self.c.route_state_dim]
    if n<self.c.lane_num:lanes[n]=seg[:,:self.c.lane_state_dim]
    n+=1
  z=lambda *x:np.zeros(x,np.float32)
  inputs={"ego_current_state":np.array([0,0,1,0,0,0,0,0,0,0],np.float32),"neighbor_agents_past":z(self.c.agent_num,self.c.time_len,self.c.agent_state_dim),"static_objects":z(self.c.static_objects_num,self.c.static_objects_state_dim),"lanes":lanes,"lanes_speed_limit":z(self.c.lane_num,1),"lanes_has_speed_limit":np.zeros((self.c.lane_num,1),bool),"route_lanes":routes,"route_lanes_speed_limit":z(self.c.route_num,1),"route_lanes_has_speed_limit":np.zeros((self.c.route_num,1),bool)}
  y=np.asarray(r["trajectory"]["fixed_arc_length_80"],np.float32);valid=np.asarray(r["trajectory"]["valid_mask"],bool)
  if r.get('profile')=='proxy_ab':
   history=r['history']
   for k in ('ego_history','history_mask','history_dt','history_dt_mask','motion_rms','motion_mask'):
    inputs[k]=np.asarray(history[k],dtype=bool if k.endswith('mask') else np.float32)
   metadata={k:r[k] for k in ('sample_id','trajectory_key','platform_id','split','embodiment','history_start_frame','history_end_frame','time_source','body_heading_source','frame_convention')}
   if r.get('experiment_profile')=='four_groups':metadata.update(experiment_profile='four_groups',study_group=r['study_group'])
   metadata.update({k:r.get(k) for k in ('reference_pose_policy','reference_heading_source','reference_approval_record')})
   return inputs,y,valid,metadata
  return inputs,y,valid,r["sample_id"]
 def __getitem__(self,i):
  x,y,m,s=self._features(i);return {k:torch.from_numpy(v) for k,v in x.items()},torch.from_numpy(y),torch.from_numpy(m),s
