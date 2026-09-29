from __future__ import annotations
import argparse,json,tempfile,os,subprocess
from pathlib import Path
import torch
from diffusion_planner.utils.config import Config
from tartan.research_score.training.tartan_target_dataset import TartanTargetDataset

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--args',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
 if out.exists():raise FileExistsError(out)
 c=Config(a.args,None);ds=TartanTargetDataset(a.manifest,c,None,11);ego=[];lanes=[];routes=[];ys=[];masks=[];ids=[];members=[];platform_ids=[];abilities=[];ability_masks=[];pair_valid=[];evidence=[]
 raw=[json.loads(x) for x in Path(a.manifest).open()]
 for i in range(len(ds)):
  x,y,m,s=ds[i];ego.append(x['ego_current_state']);lanes.append(x['lanes']);routes.append(x['route_lanes']);ys.append(y);masks.append(m);ids.append(s);members.append(raw[i].get('budget_membership',{}));platform_ids.append(int(raw[i].get('platform_id',1)));abilities.append(raw[i].get('ability',[0.0]*10));ability_masks.append(raw[i].get('ability_mask',[0.0]*10));pair_valid.append(bool(raw[i].get('pair_valid',False)));evidence.append(raw[i].get('evidence_kind','legacy_target'))
  if (i+1)%100==0:print(json.dumps({'built':i+1,'total':len(ds)}),flush=True)
 obj={'ego_current_state':torch.stack(ego),'lanes':torch.stack(lanes),'route_lanes':torch.stack(routes),'trajectory':torch.stack(ys),'valid_mask':torch.stack(masks),'sample_ids':ids,'budget_membership':members,'platform_id':torch.tensor(platform_ids,dtype=torch.long),'ability':torch.tensor(abilities,dtype=torch.float32),'ability_mask':torch.tensor(ability_masks,dtype=torch.float32),'pair_valid':torch.tensor(pair_valid,dtype=torch.bool),'evidence_kind':evidence}
 fd,tmp=tempfile.mkstemp(prefix='target_features_',suffix='.pt');os.close(fd)
 try:torch.save(obj,tmp);subprocess.run(['dd',f'if={tmp}',f'of={out}','conv=fsync','status=none'],check=True)
 finally:Path(tmp).unlink(missing_ok=True)
 print(json.dumps({'status':'complete','samples':len(ds),'bytes':out.stat().st_size,'output':str(out)}))
if __name__=='__main__':main()
