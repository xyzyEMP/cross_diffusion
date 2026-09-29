from __future__ import annotations
import argparse, json, math, os, random, subprocess, tempfile, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from tartan.research_score.model.score_decomposition import ScoreDecompositionPlanner
from tartan.research_score.training.cached_target_dataset import CachedTargetDataset

METHODS={"pretrain_finetune","pretrain_adapter","joint_train","emb_cond_diffusion"}
ABILITIES=torch.tensor([
 [.95,.85,.90,.35,.80,.20,.15,.25,.20,.90],
 [.35,.30,.30,.85,.55,.80,.75,.70,.80,.65],
],dtype=torch.float32)

def publish(obj,path,is_json=False):
 fd,name=tempfile.mkstemp(prefix="cross_diffusion_",suffix=".json" if is_json else ".pt");os.close(fd)
 try:
  if is_json:Path(name).write_text(json.dumps(obj,indent=2)+"\n")
  else:torch.save(obj,name)
  subprocess.run(["dd",f"if={name}",f"of={path}","conv=fsync","status=none"],check=True)
 finally:Path(name).unlink(missing_ok=True)

def load_ckpt(model,path):
 x=torch.load(path,map_location="cpu",weights_only=False);s=x.get("ema_state_dict",x.get("model",x));s={k.removeprefix("module."):v for k,v in s.items()};model.load_state_dict(s,strict=True)

def source_batch(cache,batch):
 idx=torch.tensor(random.choices(range(len(cache['trajectory'])),k=batch),dtype=torch.long)
 keys=("ego_current_state","neighbor_agents_past","lanes","lanes_speed_limit","lanes_has_speed_limit","route_lanes","route_lanes_speed_limit","route_lanes_has_speed_limit","static_objects")
 return {k:cache[k].index_select(0,idx) for k in keys},cache['trajectory'].index_select(0,idx),cache['valid_mask'].index_select(0,idx)

def loss_step(model,c,x,y,m,wrapped,eid,amp):
 x={k:v.cuda(non_blocking=True) for k,v in x.items()};x=c.observation_normalizer(x);y=y.cuda();m=m.cuda();B=len(y)
 cur=torch.zeros(B,11,1,4,device="cuda");cur[:,0,0]=x["ego_current_state"][:,:4];raw=torch.zeros(B,11,80,4,device="cuda");raw[:,0]=y;target=c.state_normalizer(raw)
 t=torch.rand(B,device="cuda")*.999+.001;base=model.backbone if wrapped else model;mean,std=base.sde.marginal_prob(target,t);std=std.view(B,1,1,1)
 inp={**x,"sampled_trajectories":torch.cat([cur,mean+std*torch.randn_like(mean)],2),"diffusion_time":t}
 with torch.cuda.amp.autocast(enabled=amp):
  if wrapped:_,o=model(inp,torch.full((B,),eid,dtype=torch.long,device="cuda"),ABILITIES[eid:eid+1].cuda().expand(B,-1))
  else:_,o=model(inp)
  pred=o["score"][:,0,1:];return (((pred-target[:,0]).square().sum(-1))*m).sum()/m.sum().clamp_min(1)

def validate(model,c,loader,wrapped,amp):
 cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state();torch.manual_seed(20260914);torch.cuda.manual_seed(20260914);was_training=model.training;model.eval();vals=[]
 try:
  with torch.no_grad():
   for x,y,m,_ in loader:vals.append(float(loss_step(model,c,x,y,m,wrapped,1,amp)))
 finally:
  model.train(was_training);torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
 return float(np.mean(vals))

def main():
 p=argparse.ArgumentParser();p.add_argument("--method",choices=sorted(METHODS),required=True);p.add_argument("--train-cache",required=True);p.add_argument("--val-cache",required=True);p.add_argument("--source-cache",required=True);p.add_argument("--args",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True);p.add_argument("--budget",type=int,choices=[1,10,20,50,100],required=True);p.add_argument("--max-target-updates",type=int,default=10000);p.add_argument("--min-target-updates",type=int,default=5000);p.add_argument("--val-every-updates",type=int,default=250);p.add_argument("--snapshot-every-updates",type=int,default=0);p.add_argument("--patience-validations",type=int,default=10);p.add_argument("--batch-size",type=int,default=64);p.add_argument("--seed",type=int,required=True);p.add_argument("--amp",action="store_true");a=p.parse_args()
 torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);random.seed(a.seed);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 c=Config(a.args,None);c.device="cuda";backbone=Diffusion_Planner(c)
 load_ckpt(backbone,a.checkpoint)
 wrapped=a.method in {"pretrain_adapter","emb_cond_diffusion"};model=ScoreDecompositionPlanner(backbone).cuda() if wrapped else backbone.cuda()
 if a.method=="pretrain_adapter":
  for q in model.backbone.parameters():q.requires_grad=False
  model.backbone.eval()
 params=[q for q in model.parameters() if q.requires_grad];lr=2e-4 if a.method=="pretrain_adapter" else 1e-4
 opt=torch.optim.AdamW(params,lr=lr,weight_decay=1e-4);scaler=torch.cuda.amp.GradScaler(enabled=a.amp)
 ds=CachedTargetDataset(a.train_cache,c,a.budget,a.seed);vd=CachedTargetDataset(a.val_cache,c,None,a.seed)
 dl=DataLoader(ds,batch_size=min(a.batch_size,len(ds)),shuffle=True,num_workers=0,drop_last=False);vl=DataLoader(vd,batch_size=a.batch_size,shuffle=False,num_workers=0)
 src=torch.load(a.source_cache,map_location='cpu',weights_only=False) if a.method in {"joint_train","emb_cond_diffusion"} else None
 sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=2,min_lr=5e-6)
 best=float("inf");best_update=0;best_state=None;history=[];global_step=0;target_updates=0;source_updates=0;epoch=0;next_val=a.val_every_updates;next_snapshot=a.snapshot_every_updates if a.snapshot_every_updates>0 else None;stale=0;stop=False;t0=time.time();train_target=[];train_source=[]
 if next_snapshot is not None:(out/"snapshots").mkdir(exist_ok=False)
 while target_updates<a.max_target_updates and not stop:
  epoch+=1;model.train()
  if a.method=="pretrain_adapter":model.backbone.eval()
  for x,y,m,_ in dl:
   if target_updates>=a.max_target_updates:break
   loss=loss_step(model,c,x,y,m,wrapped,1,a.amp);opt.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(params,5.0);scaler.step(opt);scaler.update();global_step+=1;target_updates+=1;train_target.append(float(loss))
   if src is not None:
    sx,sy,sm=source_batch(src,len(y));sl=loss_step(model,c,sx,sy,sm,wrapped,0,a.amp);opt.zero_grad(set_to_none=True);scaler.scale(sl).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(params,5.0);scaler.step(opt);scaler.update();global_step+=1;source_updates+=1;train_source.append(float(sl))
   if next_snapshot is not None and (target_updates>=next_snapshot or target_updates>=a.max_target_updates):
    publish({"model":model.state_dict(),"epoch":epoch,"global_step":global_step,"target_updates":target_updates,"method":a.method,"budget":a.budget,"seed":a.seed},out/"snapshots"/f"epoch_{epoch:03d}_update_{target_updates:06d}.pt")
    next_snapshot+=a.snapshot_every_updates
   if target_updates>=next_val or target_updates>=a.max_target_updates:
    val=validate(model,c,vl,wrapped,a.amp)
    if a.method=="pretrain_adapter":model.backbone.eval()
    sched.step(val);row={"epoch":epoch,"global_step":global_step,"target_updates":target_updates,"source_updates":source_updates,"target_loss":float(np.mean(train_target)),"source_loss":float(np.mean(train_source)) if train_source else None,"val_loss":val,"lr":opt.param_groups[0]["lr"],"seconds":time.time()-t0};history.append(row);print(json.dumps(row),flush=True);train_target=[];train_source=[]
    if val<best-1e-5:
     best=val;best_update=target_updates;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    else:stale+=1
    next_val+=a.val_every_updates
    if target_updates>=a.min_target_updates and stale>=a.patience_validations:stop=True;break
 publish({"model":best_state,"target_updates":best_update,"method":a.method,"budget":a.budget,"seed":a.seed,"val_loss":best},out/"best.pt")
 publish({"model":model.state_dict(),"optimizer":opt.state_dict(),"scaler":scaler.state_dict(),"scheduler":sched.state_dict(),"epoch":epoch,"global_step":global_step},out/"last.pt")
 result={"status":"complete","protocol":"formal_fixed_target_updates_v3_eval_mode","method":a.method,"budget":a.budget,"seed":a.seed,"target_samples":len(ds),"validation_samples":len(vd),"source_cache_samples":len(src['trajectory']) if src is not None else 0,"source_manifest_pool_samples":int(src['manifest_pool_size']) if src is not None else 0,"batch_size":a.batch_size,"max_target_updates":a.max_target_updates,"snapshot_every_updates":a.snapshot_every_updates,"completed_epochs":epoch,"best_target_update":best_update,"best_validation_loss":best,"target_updates":target_updates,"source_updates":source_updates,"trainable_parameters":sum(q.numel() for q in params),"seconds":time.time()-t0,"history":history}
 publish(result,out/"metrics.json",True)

if __name__=="__main__":main()
