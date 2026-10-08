from __future__ import annotations
import torch
from torch.utils.data import Dataset

class CachedTargetDataset(Dataset):
 def __init__(self,path,config,budget=None,repeat_id=11):
  self.z=torch.load(path,map_location='cpu',weights_only=False);tag=str(budget);seed=str(repeat_id)
  self.indices=list(range(len(self.z['sample_ids']))) if budget is None else [i for i,m in enumerate(self.z['budget_membership']) if tag in m.get(seed,[])]
  self.c=config
  if self.z.get('profile')=='proxy_ab':
   required=('metadata','platform_id','ego_history','history_mask','history_dt','history_dt_mask','motion_rms','motion_mask','representation_policy','normalizer_reference','manifest_path')
   if any(k not in self.z for k in required):raise ValueError('incomplete Proxy physical-history cache')
   if 'proxy_context' in self.z or 'latent' in self.z:raise ValueError('learned Proxy context may not be cached')
   if self.z['representation_policy']!='anchor-inclusive-xy-8m-80-first-duplicate':raise ValueError('Proxy representation mismatch')
   if len(self.z['metadata'])!=len(self.z['sample_ids']) or any(m['sample_id']!=sid for m,sid in zip(self.z['metadata'],self.z['sample_ids'])):raise ValueError('Proxy cache metadata IDs mismatch')
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
  if self.z.get('profile')=='proxy_ab':
   for k in ('ego_history','history_mask','history_dt','history_dt_mask','motion_rms','motion_mask'):x[k]=self.z[k][i]
   return x,self.z['trajectory'][i],self.z['valid_mask'][i],{k:v for k,v in self.z['metadata'][i].items() if v is not None}
  return x,self.z['trajectory'][i],self.z['valid_mask'][i],self.z['sample_ids'][i]


class CachedPairDataset(Dataset):
 def __init__(self,manifest,cache,config,split='train'):
  import json
  from pathlib import Path
  self.rows=[json.loads(x) for x in Path(manifest).read_text().splitlines() if x]
  self.dataset=CachedTargetDataset(cache,config)
  if self.dataset.z.get('profile')!='proxy_ab':raise ValueError('Proxy pair cache schema required')
  self.lookup={sid:i for i,sid in enumerate(self.dataset.z['sample_ids'])}
  if len(self.lookup)!=len(self.dataset.z['sample_ids']):raise ValueError('duplicate cache sample IDs')
  for r in self.rows:
   if r['split']!=split or not r['pair_valid']:raise ValueError('invalid pair split/validity')
   for side in ('a','b'):
    sid=r[side]['sample_id']
    if sid not in self.lookup:raise ValueError('pair references missing sample '+sid)
    meta=self.dataset.z['metadata'][self.lookup[sid]]
    if meta['split']!=split or meta['embodiment'] not in ('diff','omni'):raise ValueError('cross split/test pair')
    if meta['trajectory_key']!=r[side]['trajectory_key']:raise ValueError('pair/cache identity mismatch')
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  return {'a':self.dataset[self.lookup[r['a']['sample_id']]],'b':self.dataset[self.lookup[r['b']['sample_id']]],
   'pair_id':r['pair_id'],'pair_valid':torch.tensor(r['pair_valid'],dtype=torch.bool),
   'T_a':torch.tensor(r['T_a'],dtype=torch.float32),'T_b':torch.tensor(r['T_b'],dtype=torch.float32),'quality':r['metrics_8m']}
