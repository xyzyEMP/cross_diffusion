from __future__ import annotations
from tartan.research_score.artifacts import publish
import argparse,json
from pathlib import Path
import torch
from diffusion_planner.utils.config import Config
from tartan.research_score.training.tartan_target_dataset import TartanTargetDataset

def main():
 p=argparse.ArgumentParser();p.add_argument('--profile',default='transfer_primary',choices=('transfer_primary','proxy_ab'));p.add_argument('--manifest',required=True);p.add_argument('--args',required=True);p.add_argument('--output',required=True);p.add_argument('--reuse-cache',nargs='+',default=[]);a=p.parse_args();out=Path(a.output)
 if out.exists():raise FileExistsError(out)
 out.parent.mkdir(parents=True,exist_ok=True)
 if a.profile=='proxy_ab':return materialize_proxy(a,out)
 c=Config(a.args,None);ds=TartanTargetDataset(a.manifest,c,None,11);ego=[];lanes=[];routes=[];ys=[];masks=[];ids=[];members=[];platform_ids=[];abilities=[];ability_masks=[];pair_valid=[];evidence=[]
 raw=[json.loads(x) for x in Path(a.manifest).open()]
 for i in range(len(ds)):
  x,y,m,s=ds[i];ego.append(x['ego_current_state']);lanes.append(x['lanes']);routes.append(x['route_lanes']);ys.append(y);masks.append(m);ids.append(s);members.append(raw[i].get('budget_membership',{}));platform_ids.append(int(raw[i].get('platform_id',1)));abilities.append(raw[i].get('ability',[0.0]*10));ability_masks.append(raw[i].get('ability_mask',[0.0]*10));pair_valid.append(bool(raw[i].get('pair_valid',False)));evidence.append(raw[i].get('evidence_kind','recorded_unpaired'))
  if (i+1)%100==0:print(json.dumps({'built':i+1,'total':len(ds)}),flush=True)
 obj={'ego_current_state':torch.stack(ego),'lanes':torch.stack(lanes),'route_lanes':torch.stack(routes),'trajectory':torch.stack(ys),'valid_mask':torch.stack(masks),'sample_ids':ids,'budget_membership':members,'platform_id':torch.tensor(platform_ids,dtype=torch.long),'ability':torch.tensor(abilities,dtype=torch.float32),'ability_mask':torch.tensor(ability_masks,dtype=torch.float32),'pair_valid':torch.tensor(pair_valid,dtype=torch.bool),'evidence_kind':evidence}
 publish(obj,out)
 print(json.dumps({'status':'complete','samples':len(ds),'bytes':out.stat().st_size,'output':str(out)}))
def materialize_proxy(a,out):
 c=Config(a.args,None);raw=[json.loads(x) for x in Path(a.manifest).read_text().splitlines() if x]
 if any(r.get('profile')!='proxy_ab' for r in raw):raise ValueError('not a Proxy manifest')
 lookup={}
 for source in a.reuse_cache:
  z=torch.load(source,map_location='cpu',weights_only=False)
  if z.get('profile')!='proxy_ab' or z.get('normalizer_reference')!=str(Path(a.args).absolute()):raise ValueError('reuse cache profile/normalizer differs')
  for j,sid in enumerate(z['sample_ids']):lookup[sid]=(z,j)
 ds=TartanTargetDataset(a.manifest,c);items={k:[] for k in ('ego_current_state','lanes','route_lanes','trajectory','valid_mask','ego_history','history_mask','history_dt','history_dt_mask','motion_rms','motion_mask')};metadata=[]
 for i in range(len(ds)):
  if raw[i]['sample_id'] in lookup:
   z,j=lookup[raw[i]['sample_id']];meta=z['metadata'][j]
   if any(meta[k]!=raw[i][k] for k in ('trajectory_key','split','history_start_frame','history_end_frame','reference_pose_policy')):raise ValueError('reuse cache identity differs')
   x={k:z[k][j] for k in items if k not in ('trajectory','valid_mask')};y=z['trajectory'][j];m=z['valid_mask'][j]
  else:x,y,m,meta=ds[i]
  if not m.any():raise ValueError('empty validity '+meta['sample_id'])
  for k in items:
   items[k].append(y if k=='trajectory' else m if k=='valid_mask' else x[k])
  metadata.append(meta)
  if (i+1)%100==0:print(json.dumps({'built':i+1,'total':len(ds)}),flush=True)
 shapes={'ego_current_state':(10,),'lanes':(c.lane_num,c.lane_len,c.lane_state_dim),'route_lanes':(c.route_num,c.route_len,c.route_state_dim),'trajectory':(80,4),'valid_mask':(80,),'ego_history':(20,4),'history_mask':(20,),'history_dt':(19,),'history_dt_mask':(19,),'motion_rms':(3,),'motion_mask':(3,)}
 obj={k:torch.stack(v) if v else torch.empty((0,)+shapes[k],dtype=torch.bool if k.endswith('mask') else torch.float32) for k,v in items.items()}
 obj.update(profile='proxy_ab',schema_version='proxy-history-physical-v1',manifest_path=str(Path(a.manifest).absolute()),normalizer_reference=str(Path(a.args).absolute()),representation_policy='anchor-inclusive-xy-8m-80-first-duplicate',metadata=metadata,sample_ids=[m['sample_id'] for m in metadata],platform_id=torch.tensor([m['platform_id'] for m in metadata],dtype=torch.long),budget_membership=[{} for m in metadata])
 publish(obj,out)
 print(json.dumps({'status':'complete' if raw else 'EMPTY_SCHEMA_ONLY','samples':len(ds),'bytes':out.stat().st_size,'output':str(out),'reused_samples':sum(r['sample_id'] in lookup for r in raw)}))

if __name__=='__main__':main()
