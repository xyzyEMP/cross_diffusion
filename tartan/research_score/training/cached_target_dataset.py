from __future__ import annotations
import torch
from torch.utils.data import Dataset

class CachedTargetDataset(Dataset):
 def __init__(self,path,config,budget=None,repeat_id=11):
  self.z=torch.load(path,map_location='cpu',weights_only=False);tag=str(budget);seed=str(repeat_id)
  self.indices=list(range(len(self.z['sample_ids']))) if budget is None else [i for i,m in enumerate(self.z['budget_membership']) if tag in m.get(seed,[])]
  self.c=config
 def __len__(self):return len(self.indices)
 def __getitem__(self,j):
  i=self.indices[j];z=lambda *s:torch.zeros(s,dtype=torch.float32)
  x={'ego_current_state':self.z['ego_current_state'][i],
     'neighbor_agents_past':z(self.c.agent_num,self.c.time_len,self.c.agent_state_dim),
     'static_objects':z(self.c.static_objects_num,self.c.static_objects_state_dim),
     'lanes':self.z['lanes'][i],
     'lanes_speed_limit':z(self.c.lane_num,1),'lanes_has_speed_limit':torch.zeros(self.c.lane_num,1,dtype=torch.bool),
     'route_lanes':self.z['route_lanes'][i],
     'route_lanes_speed_limit':z(self.c.route_num,1),'route_lanes_has_speed_limit':torch.zeros(self.c.route_num,1,dtype=torch.bool)}
  return x,self.z['trajectory'][i],self.z['valid_mask'][i],self.z['sample_ids'][i]
