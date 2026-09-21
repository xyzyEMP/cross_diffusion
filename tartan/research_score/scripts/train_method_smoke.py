from __future__ import annotations
import argparse,json,random,time,subprocess,tempfile,os
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from tartan.research_score.model.score_decomposition import ScoreDecompositionPlanner
from tartan.research_score.model.representation_bridge import TrajectoryRepresentationBridge
from tartan.research_score.training.tartan_target_dataset import TartanTargetDataset

METHODS={"pretrain_finetune","pretrain_adapter","joint_train","emb_cond_diffusion"}
def publish(obj,path,is_json=False):
 fd,name=tempfile.mkstemp(prefix="cross_diffusion_",suffix=".json" if is_json else ".pt");os.close(fd)
 try:
  if is_json:Path(name).write_text(json.dumps(obj,indent=2)+"\n")
  else:torch.save(obj,name)
  subprocess.run(["dd",f"if={name}",f"of={path}","conv=fsync","status=none"],check=True)
 finally:Path(name).unlink(missing_ok=True)
def load_ckpt(model,path):
 x=torch.load(path,map_location="cpu",weights_only=False);s=x.get("ema_state_dict",x.get("model",x));s={k.removeprefix("module."):v for k,v in s.items()};model.load_state_dict(s,strict=True)
def source_rows(path,n=None):
 out=[]
 with open(path) as h:
  for line in h:
   r=json.loads(line)
   # Stage 03 already validates this frozen manifest.  Re-statting 50k files
   # independently in every GPU worker only amplifies shared-filesystem I/O.
   out.append(r["npz_reference"])
   if n is not None and len(out)>=n:break
 return out
def source_batch(paths,c,batch,bridge):
 chosen=random.choices(paths,k=batch);items=[];ys=[];ms=[]
 for p in chosen:
  z=np.load(p,allow_pickle=False);x={k:torch.from_numpy(z[k]) for k in ("ego_current_state","neighbor_agents_past","lanes","lanes_speed_limit","lanes_has_speed_limit","route_lanes","route_lanes_speed_limit","route_lanes_has_speed_limit","static_objects")};y,m=bridge(z["ego_agent_future"]);items.append(x);ys.append(torch.from_numpy(y));ms.append(torch.from_numpy(m))
 return {k:torch.stack([x[k] for x in items]) for k in items[0]},torch.stack(ys),torch.stack(ms)
def forward_loss(model,c,x,y,m,wrapped,eid,amp):
 x={k:v.cuda() for k,v in x.items()};x=c.observation_normalizer(x);y=y.cuda();m=m.cuda();B=len(y);cur=torch.zeros(B,11,1,4,device="cuda");cur[:,0,0]=x["ego_current_state"][:,:4];raw=torch.zeros(B,11,80,4,device="cuda");raw[:,0]=y;target=c.state_normalizer(raw);t=torch.rand(B,device="cuda")*.999+.001;mean,std=model.backbone.sde.marginal_prob(target,t) if wrapped else model.sde.marginal_prob(target,t);std=std.view(B,1,1,1);inp={**x,"sampled_trajectories":torch.cat([cur,mean+std*torch.randn_like(mean)],2),"diffusion_time":t}
 with torch.cuda.amp.autocast(enabled=amp):
  if wrapped:
   # Normalized, platform-specific capability descriptors.  Keeping these
   # distinct is essential for embodiment conditioning in joint training.
   abilities=torch.tensor([
    [.95,.85,.90,.35,.80,.20,.15,.25,.20,.90], # road car
    [.35,.30,.30,.85,.55,.80,.75,.70,.80,.65], # ANYmal
   ],device="cuda")
   _,o=model(inp,torch.full((B,),eid,dtype=torch.long,device="cuda"),abilities[eid:eid+1].expand(B,-1))
  else:_,o=model(inp)
  pred=o["score"][:,0,1:];loss=(((pred-target[:,0]).square().sum(-1))*m).sum()/m.sum().clamp_min(1)
 return loss
def train_step(model,c,opt,x,y,m,wrapped,eid,amp,scaler):
 loss=forward_loss(model,c,x,y,m,wrapped,eid,amp);opt.zero_grad();scaler.scale(loss).backward();scaler.step(opt);scaler.update();return float(loss)
def main():
 p=argparse.ArgumentParser();p.add_argument("--method",choices=sorted(METHODS),required=True);p.add_argument("--manifest",required=True);p.add_argument("--val-manifest");p.add_argument("--source-manifest",required=True);p.add_argument("--args",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True);p.add_argument("--budget",type=int,choices=[1,10,100],default=1);p.add_argument("--steps",type=int,default=20);p.add_argument("--val-every",type=int,default=100);p.add_argument("--batch-size",type=int,default=2);p.add_argument("--seed",type=int,default=44);p.add_argument("--resume");p.add_argument("--amp",action="store_true");a=p.parse_args();torch.manual_seed(a.seed);random.seed(a.seed);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);c=Config(a.args,None);c.device="cuda";backbone=Diffusion_Planner(c)
 load_ckpt(backbone,a.checkpoint)
 wrapped=a.method in {"pretrain_adapter","emb_cond_diffusion"};model=ScoreDecompositionPlanner(backbone).cuda() if wrapped else backbone.cuda()
 if a.method=="pretrain_adapter":
  for p0 in model.backbone.parameters():p0.requires_grad=False
 params=[p0 for p0 in model.parameters() if p0.requires_grad];opt=torch.optim.AdamW(params,lr=2e-4);start=0
 scaler=torch.cuda.amp.GradScaler(enabled=a.amp)
 if a.resume:
  state=torch.load(a.resume,map_location="cpu");model.load_state_dict(state["model"]);opt.load_state_dict(state["optimizer"]);start=int(state["step"]);torch.set_rng_state(state["torch_rng"]);random.setstate(state["python_rng"]);scaler.load_state_dict(state.get("scaler",{}))
 ds=TartanTargetDataset(a.manifest,c,a.budget,a.seed);dl=DataLoader(ds,batch_size=a.batch_size,shuffle=True,num_workers=0,drop_last=True);it=iter(dl);vds=TartanTargetDataset(a.val_manifest or a.manifest,c,None,a.seed,32);vdl=DataLoader(vds,batch_size=a.batch_size,shuffle=False,num_workers=0);src=source_rows(a.source_manifest);bridge=TrajectoryRepresentationBridge();losses=[];vals=[];best=float("inf");t0=time.time()
 for step in range(start,a.steps):
  use_source=a.method in {"joint_train","emb_cond_diffusion"} and step%2==1
  if use_source:x,y,m=source_batch(src,c,a.batch_size,bridge);eid=0
  else:
   try:x,y,m,_=next(it)
   except StopIteration:it=iter(dl);x,y,m,_=next(it)
   eid=1
  loss=train_step(model,c,opt,x,y,m,wrapped,eid,a.amp,scaler);losses.append(loss);print(json.dumps({"method":a.method,"step":step+1,"domain":"car" if use_source else "anymal","loss":loss}),flush=True)
  if (step+1)%a.val_every==0 or step+1==a.steps:
   cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state();torch.manual_seed(100000+step);torch.cuda.manual_seed(100000+step);model.train();vv=[]
   with torch.no_grad():
    for vx,vy,vm,_ in vdl:vv.append(float(forward_loss(model,c,vx,vy,vm,wrapped,1,a.amp)))
   torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng);model.train();v=sum(vv)/len(vv);vals.append({"step":step+1,"loss":v});print(json.dumps({"method":a.method,"step":step+1,"val_loss":v}),flush=True)
   if v<best:best=v;publish({"model":model.state_dict(),"step":step+1,"method":a.method,"budget":a.budget,"seed":a.seed},out/"best.pt")
 publish({"model":model.state_dict(),"optimizer":opt.state_dict(),"scaler":scaler.state_dict(),"step":a.steps,"method":a.method,"torch_rng":torch.get_rng_state(),"python_rng":random.getstate()},out/"last.pt");publish({"status":"complete","method":a.method,"budget":a.budget,"seed":a.seed,"start_step":start,"steps":a.steps,"amp":a.amp,"target_samples":len(ds),"validation_samples":len(vds),"validation":vals,"best_validation_loss":best,"source_samples":len(src) if a.method in {"joint_train","emb_cond_diffusion"} else 0,"loss_first":losses[0],"loss_last":losses[-1],"trainable_parameters":sum(p.numel() for p in params),"seconds":time.time()-t0},out/"metrics.json",True)
if __name__=="__main__":main()
