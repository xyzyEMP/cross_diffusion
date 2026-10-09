from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

def publish(obj,path,is_json=False):
 fd,name=tempfile.mkstemp(prefix="cross_diffusion_",suffix=".json" if is_json else ".pt");os.close(fd)
 try:
  if is_json:Path(name).write_text(json.dumps(obj,indent=2)+"\n")
  else:torch.save(obj,name)
  subprocess.run(["dd",f"if={name}",f"of={path}","conv=fsync","status=none"],check=True)
 finally:Path(name).unlink(missing_ok=True)

def load_ckpt(model,path):
 x=torch.load(path,map_location="cpu",weights_only=False);s=x.get("ema_state_dict",x.get("model",x));s={k.removeprefix("module."):v for k,v in s.items()};model.load_state_dict(s,strict=True)

def loss_step(model,c,x,y,m,amp):
 x={k:v.cuda(non_blocking=True) for k,v in x.items()};x=c.observation_normalizer(x);y=y.cuda();m=m.cuda();B=len(y)
 cur=torch.zeros(B,11,1,4,device="cuda");cur[:,0,0]=x["ego_current_state"][:,:4];raw=torch.zeros(B,11,80,4,device="cuda");raw[:,0]=y;target=c.state_normalizer(raw)
 t=torch.rand(B,device="cuda")*.999+.001;mean,std=model.sde.marginal_prob(target,t);std=std.view(B,1,1,1)
 inp={**x,"sampled_trajectories":torch.cat([cur,mean+std*torch.randn_like(mean)],2),"diffusion_time":t}
 with torch.cuda.amp.autocast(enabled=amp):
  _,o=model(inp)
  pred=o["score"][:,0,1:];return (((pred-target[:,0]).square().sum(-1))*m).sum()/m.sum().clamp_min(1)

def validate(model,c,loader,amp):
 cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state();torch.manual_seed(20260914);torch.cuda.manual_seed(20260914);was_training=model.training;model.eval();vals=[]
 try:
  with torch.no_grad():
   for x,y,m,_ in loader:vals.append(float(loss_step(model,c,x,y,m,amp)))
 finally:
  model.train(was_training);torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
 return float(np.mean(vals))
