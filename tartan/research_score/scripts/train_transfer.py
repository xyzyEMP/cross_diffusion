from __future__ import annotations
import shlex, sys
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from tartan.research_score.model.score_decomposition import ScoreDecompositionPlanner
from tartan.research_score.artifacts import publish, validate_output_name
from tartan.research_score.training.cached_target_dataset import CachedTargetDataset
from tartan.research_score.scripts.evaluate_navigation import load_segments, rollout, episode_macro
from tartan.research_score.evaluation.metrics_navigation import checkpoint_selection_key

from tartan.research_score.training.methods import METHODS, method_spec
ABILITIES=torch.tensor([
 [.95,.85,.90,.35,.80,.20,.15,.25,.20,.90],
 [.35,.30,.30,.85,.55,.80,.75,.70,.80,.65],
],dtype=torch.float32)

def load_ckpt(model,path):
 x=torch.load(path,map_location="cpu",weights_only=False);s=x.get("ema_state_dict",x.get("model",x));s={k.removeprefix("module."):v for k,v in s.items()};model.load_state_dict(s,strict=True);return "ema_state_dict" if "ema_state_dict" in x else ("model" if "model" in x else "raw_state_dict")

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


def navigation_validate(model, config, records, wrapped):
 cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state();was_training=model.training
 model.eval()
 try:
  with torch.no_grad():
   rows=[rollout(record,config,model,wrapped,index,inference_seed=20260914) for index,record in enumerate(records)]
 finally:
  model.train(was_training);torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
 valid=[row for row in rows if row["included_in_denominator"]]
 if not valid:raise ValueError("navigation validation has no valid map tasks")
 return episode_macro(valid)

def main():
 p=argparse.ArgumentParser();p.add_argument("--method",choices=sorted(METHODS),required=True);p.add_argument("--train-cache",required=True);p.add_argument("--val-cache",required=True);p.add_argument("--val-navigation-manifest",required=True);p.add_argument("--source-cache",required=True);p.add_argument("--args",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True);p.add_argument("--budget",type=int,choices=[1,10,20,50,100],required=True);p.add_argument("--max-target-updates",type=int,default=10000);p.add_argument("--min-target-updates",type=int,default=5000);p.add_argument("--val-every-updates",type=int,default=250);p.add_argument("--snapshot-every-updates",type=int,default=0);p.add_argument("--patience-validations",type=int,default=10);p.add_argument("--batch-size",type=int,default=64);p.add_argument("--seed",type=int,required=True);p.add_argument("--amp",action="store_true");a=p.parse_args()
 torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);random.seed(a.seed);np.random.seed(a.seed);out=Path(a.output);validate_output_name(out.name);out.mkdir(parents=True,exist_ok=False);(out/"command.txt").write_text(shlex.join([sys.executable,"-m",__spec__.name,*sys.argv[1:]])+"\n")
 c=Config(a.args,None);c.device="cuda";backbone=Diffusion_Planner(c)
 load_ckpt(backbone,a.checkpoint)
 spec=method_spec(a.method,"transfer_primary");wrapped=spec["wrapper"];model=ScoreDecompositionPlanner(backbone).cuda() if wrapped else backbone.cuda()
 if a.method=="pretrain_adapter":
  for q in model.backbone.parameters():q.requires_grad=False
  model.backbone.eval()
 params=[q for q in model.parameters() if q.requires_grad];lr=2e-4 if a.method=="pretrain_adapter" else 1e-4
 opt=torch.optim.AdamW(params,lr=lr,weight_decay=1e-4);scaler=torch.cuda.amp.GradScaler(enabled=a.amp)
 publish({"cli":vars(a),"model_args":json.loads(Path(a.args).read_text()),"optimizer":{"name":"AdamW","lr":lr,"weight_decay":1e-4,"gradient_clip":5.0},"selection":"episode_macro_sr_spl_low_cr_progress_early_update"},out/"config.json",True)
 ds=CachedTargetDataset(a.train_cache,c,a.budget,a.seed);vd=CachedTargetDataset(a.val_cache,c,None,a.seed)
 if not len(ds) or not len(vd):raise ValueError("training budget and validation cache must be nonempty")
 dl=DataLoader(ds,batch_size=min(a.batch_size,len(ds)),shuffle=True,num_workers=0,drop_last=False);vl=DataLoader(vd,batch_size=a.batch_size,shuffle=False,num_workers=0)
 src=torch.load(a.source_cache,map_location='cpu',weights_only=False) if spec["source"] else None
 sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=2,min_lr=5e-6)
 best=None;best_sr=-float("inf");best_update=0;best_state=None;history=[];global_step=0;target_updates=0;source_updates=0;epoch=0;next_val=a.val_every_updates;next_snapshot=a.snapshot_every_updates if a.snapshot_every_updates>0 else None;stale=0;stop=False;t0=time.time();train_target=[];train_source=[]
 records=load_segments(a.val_navigation_manifest)
 if not records or any(r.get("split")!="val" for r in records):raise ValueError("validation navigation manifest must contain only val tasks")
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
    macro=navigation_validate(model,c,records,wrapped)
    if a.method=="pretrain_adapter":model.backbone.eval()
    sched.step(macro["sr"]);row={"epoch":epoch,"global_step":global_step,"target_updates":target_updates,"source_updates":source_updates,"target_loss":float(np.mean(train_target)),"source_loss":float(np.mean(train_source)) if train_source else None,"val_loss":val,"navigation":macro,"lr":opt.param_groups[0]["lr"],"seconds":time.time()-t0};history.append(row);print(json.dumps(row),flush=True);train_target=[];train_source=[]
    if best is None or checkpoint_selection_key(row)>checkpoint_selection_key(best):
     best=row;best_update=target_updates;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if macro["sr"]>best_sr:best_sr=macro["sr"];stale=0
    else:stale+=1
    next_val+=a.val_every_updates
    if target_updates>=a.min_target_updates and stale>=a.patience_validations:stop=True;break
 publish({"model":best_state,"target_updates":best_update,"method":a.method,"budget":a.budget,"seed":a.seed,"navigation_metrics":best["navigation"],"selection":"episode_macro_sr_spl_low_cr_progress_early_update"},out/"navigation_best.pt")
 publish({"model":model.state_dict(),"optimizer":opt.state_dict(),"scaler":scaler.state_dict(),"scheduler":sched.state_dict(),"epoch":epoch,"global_step":global_step,"target_updates":target_updates,"source_updates":source_updates},out/"last.pt")
 result={"status":"complete","protocol":"score-decomp-transfer-v1.2.3-navigation-sr","method":a.method,"budget":a.budget,"seed":a.seed,"target_samples":len(ds),"validation_samples":len(vd),"source_cache_samples":len(src['trajectory']) if src is not None else 0,"source_manifest_pool_samples":int(src['manifest_pool_size']) if src is not None else 0,"batch_size":a.batch_size,"max_target_updates":a.max_target_updates,"snapshot_every_updates":a.snapshot_every_updates,"completed_epochs":epoch,"best_target_update":best_update,"best_navigation":best["navigation"],"best_validation_loss":min(r["val_loss"] for r in history),"validation_navigation_manifest":a.val_navigation_manifest,"validation_navigation_tasks":len(records),"selection":"episode_macro_sr_spl_low_cr_progress_early_update","target_updates":target_updates,"source_updates":source_updates,"trainable_parameters":sum(q.numel() for q in params),"seconds":time.time()-t0,"history":history}
 publish(result,out/"metrics.json",True)




def capture_rng(generators):
 return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],'streams':{k:g.get_state() for k,g in generators.items()}}


def restore_rng(state,generators):
 random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
 if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])
 for k,g in generators.items():g.set_state(state['streams'][k])


def proxy_config(path,profile="proxy_ab"):
 import yaml
 from tartan.research_score.preflight.cli import _resolve
 cfg=_resolve(yaml.safe_load(Path(path).read_text())[profile])
 if cfg['seed']!=11 or cfg['trajectory']['length_m']!=8 or cfg['trajectory']['points']!=80:raise ValueError('conflicting Proxy frozen settings')
 return cfg


def proxy_forward(model,c,batch,generator,id_generator,wrapped,device,shared_noise=None,allowed_platforms=(2,3)):
 from tartan.research_score.training.losses import proxy_masked_mse
 x,y,mask,meta=batch;x=c.observation_normalizer({k:v.to(device) for k,v in x.items()});y=y.to(device);mask=mask.to(device);n=len(y)
 if any(s not in ('train','val') for s in meta['split']) or any(int(v) not in allowed_platforms for v in meta['platform_id']):raise ValueError('ANYmal/test is forbidden in Proxy training')
 raw=torch.zeros(n,11,80,4,device=device);raw[:,0]=y;target=c.state_normalizer(raw)
 if shared_noise is None:t=torch.rand(n,generator=generator)*.999+.001;noise=torch.randn(raw.shape,generator=generator)
 else:t,noise=shared_noise
 t=t.to(device);base=model.backbone if wrapped else model;mean,std=base.sde.marginal_prob(target,t);noisy=mean+std.reshape(-1,1,1,1)*noise.to(device)
 noisy[:,1:]=0
 cur=torch.zeros(n,11,1,4,device=device);cur[:,0,0]=x['ego_current_state'][:,:4]
 inp={**x,'sampled_trajectories':torch.cat([cur,noisy],2),'diffusion_time':t}
 if wrapped:
  ids=meta['platform_id'].to(device);id_mask=(torch.rand(n,generator=id_generator)>=.2).float().to(device) if model.training else torch.ones(n,device=device)
  _,out=model(inp,ids,id_mask=id_mask)
 else:_,out=model(inp)
 loss=proxy_masked_mse(out['score'][:,0,1:],target[:,0],mask)
 return loss,out,target[:,0],mask,(t.cpu(),noise),meta


def validation_due(state, cfg, epoch_clock=False):
 """Run only after a full fifth epoch; a resumed partial epoch is not a check."""
 if epoch_clock:
  return state['cursor']==len(state['permutation']) and state['epoch']%cfg['val_every_epochs']==0
 return state['update']>=state['next_val']


def proxy_train():
 from torch.utils.data import default_collate
 from tartan.research_score.training.cached_target_dataset import CachedPairDataset
 from tartan.research_score.training.losses import proxy_pair_losses
 p=argparse.ArgumentParser();p.add_argument('--profile',choices=['proxy_ab','four_groups'],required=True);p.add_argument('--config',required=True);p.add_argument('--method',choices=['proxy_a','proxy_b'],required=True)
 for key in ('train-cache','val-cache','val-navigation-manifest','args','checkpoint','output'):p.add_argument('--'+key,required=True)
 for key in ('pair-manifest','pair-cache','pair-val-manifest','pair-val-cache','resume'):p.add_argument('--'+key)
 p.add_argument('--disable-history',action='store_true');p.add_argument('--stop-after',type=int);p.add_argument('--seed',type=int,default=11);p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--smoke',choices=['none','cpu','gpu'],default='none');p.add_argument('--amp',action='store_true');a=p.parse_args();cfg=proxy_config(a.config,a.profile)
 study=a.profile=="four_groups"
 if study and (a.method!="proxy_a" or not a.disable_history):raise ValueError("four groups require plain no-history backbone")
 from functools import partial
 forward=partial(proxy_forward,allowed_platforms=(1,2,3) if study else (2,3))
 if a.seed!=cfg['seed']:raise ValueError('Proxy seed must be 11')
 if a.device=='cpu' and (a.smoke!='cpu' or a.amp):raise ValueError('CPU is engineering smoke only, AMP forbidden')
 if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('GPU_PENDING')
 if a.device=='cuda' and not a.amp:raise ValueError('GPU Proxy phase requires approved AMP')
 wrapped=a.method=='proxy_b'
 if wrapped and a.disable_history:raise ValueError('history ablation is A-only')
 pairs_args=(a.pair_manifest,a.pair_cache,a.pair_val_manifest,a.pair_val_cache)
 if wrapped and not all(pairs_args):raise ValueError('B requires all frozen pair inputs')
 if not wrapped and any(pairs_args):raise ValueError('A forbids pair inputs')
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if any(out.glob('*.pt')) and not a.resume:raise ValueError('existing weights require explicit same-run resume')
 if a.resume and Path(a.resume).resolve()!= (out/'last.pt').resolve():raise ValueError('resume must use this output last.pt')
 frozen={'cli':{k:v for k,v in vars(a).items() if k not in ('resume','stop_after')},'proxy_ab':cfg,'purpose':cfg.get(a.smoke+'_smoke',{}).get('purpose','formal'),'eligible_for_formal_result':a.smoke=='none'}
 if not a.disable_history:frozen['cli'].pop('disable_history',None)
 if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=frozen:raise ValueError('run config differs; no implicit restart')
 publish(frozen,out/'config.json',True);(out/'command.sh').write_text(shlex.join([sys.executable,'-m',__spec__.name,*sys.argv[1:]])+'\n')
 c=Config(a.args,None);c.device=a.device
 ds=CachedTargetDataset(a.train_cache,c);vd=CachedTargetDataset(a.val_cache,c)
 if not len(ds) or not len(vd):raise ValueError('base train and val must be nonempty')
 for dataset,split in ((ds,'train'),(vd,'val')):
  if (dataset.z.get('experiment_profile')=='four_groups')!=study:raise ValueError('cache experiment profile mismatch')
  for i in range(len(dataset)):
   meta=dataset[i][3]
   if meta['split']!=split or int(meta['platform_id']) not in ((1,2,3) if study else (2,3)):raise ValueError('base cache split/platform leak')
 pairs=CachedPairDataset(a.pair_manifest,a.pair_cache,c,split='train') if wrapped else None
 valpairs=CachedPairDataset(a.pair_val_manifest,a.pair_val_cache,c,split='val') if wrapped else None
 if wrapped and not len(pairs):raise RuntimeError('BLOCKED_NO_TRAIN_PAIR')
 data=Path(cfg['paths']['output_root'])/'data'/cfg['paths']['data_id']
 gate=json.loads((data/'trajectory_summary.json').read_text())
 from tartan.data.pose_utils import read_proxy_trajectories
 trajectories=read_proxy_trajectories(data/'trajectories.jsonl')
 if not gate.get('gates_passed') or any(not r.get('gates') or not all(r['gates'].values()) for r in trajectories):raise RuntimeError('BLOCKED_DATA_TIME_REFERENCE_OCCUPANCY: P1 evidence gate failed')
 if any(r.get('reference_pose_policy') != cfg.get('reference_pose_policy') for r in trajectories):raise ValueError('reference policy differs from approved config')
 # Path/strict-loading are deliberately insufficient evidence of original nuPlan provenance.
 init=cfg['initialization']
 if not a.resume and (not init['provenance_verified'] or not init['provenance_record'] or not Path(init['provenance_record']).is_file()):raise RuntimeError('BLOCKED_SOURCE_PROVENANCE: original nuPlan source record required')
 if Path(a.args).resolve()!=Path(init['args']).resolve():raise ValueError('args differ from frozen source normalizer')
 if not a.resume and Path(a.checkpoint).resolve()!=Path(init['checkpoint']).resolve():raise ValueError('source checkpoint differs from approved original source')
 torch.manual_seed(11);random.seed(11);np.random.seed(11);base=Diffusion_Planner(c)
 if not a.resume:
  selected_key=load_ckpt(base,a.checkpoint);publish({'original_source':a.checkpoint,'provenance_record':init['provenance_record'],'selected_state_key':selected_key,'strict_load':True,'new_optimizer':True},out/'source_initialization.json',True)
 if not a.disable_history:base.encoder.encoder.enable_proxy_history(seed=11)
 model=ScoreDecompositionPlanner(base,proxy=True,seed=11) if wrapped else base;model.to(a.device)
 opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);scaler=torch.cuda.amp.GradScaler(enabled=a.amp);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,**cfg['scheduler'])
 streams={k:torch.Generator().manual_seed(cfg['rng'][k]) for k in ('base_sampler','base_noise','pair_sampler','pair_noise','id_dropout')}
 pool=sorted(range(len(ds)),key=lambda i:ds[i][3]['sample_id']);bs=cfg['batch_size'];ps=cfg['pair_batch_size'];max_updates=cfg['max_updates'];min_updates=cfg['min_updates'];val_every=cfg['val_every']
 if a.smoke=='cpu':
  pool=[min((i for i in range(len(ds)) if int(ds[i][3]['platform_id'])==eid),key=lambda i:ds[i][3]['sample_id']) for eid in (sorted(set(int(v) for v in ds.z['platform_id'])) if study else (2,3))];bs=2;ps=1;max_updates=2;min_updates=0;val_every=2
 elif a.smoke=='gpu':max_updates=4;min_updates=0;val_every=2
 def cache_identity(dataset):
  z=dataset.z
  return {'profile':z.get('profile'),'schema_version':z.get('schema_version'),'sample_ids':z['sample_ids'],'count':len(z['sample_ids']),'representation_policy':z.get('representation_policy'),'normalizer_reference':z.get('normalizer_reference'),'manifest_path':z.get('manifest_path'),'metadata':z.get('metadata')}
 input_identity={'train':cache_identity(ds),'val':cache_identity(vd),'normalizers':{'state':c.state_normalizer.to_dict(),'observation':c.observation_normalizer.to_dict()},'navigation_records':load_segments(a.val_navigation_manifest)}
 if wrapped:input_identity.update(pair_train=cache_identity(pairs.dataset),pair_val=cache_identity(valpairs.dataset),pair_train_records=pairs.rows,pair_val_records=valpairs.rows)
 for dataset,name in ((ds,'base_train'),(vd,'base_val')):
  expected=[json.loads(line)['sample_id'] for line in (data/(name+'.jsonl')).open() if line.strip()]
  if dataset.z.get('profile')!='proxy_ab' or dataset.z['sample_ids']!=expected:raise ValueError('cache identity differs from frozen manifest '+name)
 state={'update':0,'epoch':0,'cursor':0,'permutation':[],'stale':0,'best_sr':-float('inf'),'best':None,'best_model':None,'history':[],'base_exposures':0,'pair_exposures':0,'pair_repeated_draws':0,'pair_draw_counts':{},'elapsed_seconds':0.,'next_val':val_every}
 if a.resume:
  saved=torch.load(a.resume,map_location='cpu',weights_only=False)
  if saved['config']!=frozen or saved.get('input_identity')!=input_identity:raise ValueError('checkpoint frozen config/manifest/schema/normalizer identity mismatch')
  model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler']);scheduler.load_state_dict(saved['scheduler']);state=saved['training_state'];restore_rng(saved['rng'],streams)
  if state['best_model'] is not None:publish({'model':state['best_model'],'method':a.method,'profile':'proxy_ab','target_updates':state['best']['target_updates'],'config':frozen,'navigation_metrics':state['best']['navigation']},out/'navigation_best.pt')
 session_started=time.time();prior_elapsed=state['elapsed_seconds']
 def checkpoint():
  state['elapsed_seconds']=prior_elapsed+time.time()-session_started
  publish({'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'scheduler':scheduler.state_dict(),'training_state':state,'rng':capture_rng(streams),'config':frozen,'method':a.method,'profile':'proxy_ab','input_identity':input_identity},out/'last.pt')
 records=load_segments(a.val_navigation_manifest)
 robots=sorted({ds[i][3]['embodiment'] for i in range(len(ds))}) if study else ['diff','omni']
 if study:
  if ds.z.get('experiment_profile')!='four_groups' or vd.z.get('experiment_profile')!='four_groups':raise ValueError('four-group cache marker required')
  if set(robots)!=set(r['embodiment'] for r in trajectories if r['split']=='train'):raise ValueError('training platform membership differs from frozen split')
 if not records or any(r['split']!='val' or r['embodiment'] not in robots for r in records):raise ValueError('validation platform/split differs from frozen training group')
 if a.smoke!='none':
  from tartan.research_score.evaluation.goal_benchmark import prepare_episode
  selected=[];input_checks=[]
  for robot in robots:
   candidates=sorted((r for r in records if r['embodiment']==robot),key=lambda r:r['sample_id'])
   for record in candidates:
    prepared=prepare_episode(record,robot);route,_,_,_,cells,_,_,invalid=prepared
    valid=cells is not None and not invalid and bool(route['route_candidate_mask'].any())
    if a.smoke=='gpu' and not valid:continue
    history=record.get('history',record)
    for key,shape in (('ego_history',(20,4)),('history_mask',(20,)),('history_dt',(19,)),('history_dt_mask',(19,)),('motion_rms',(3,)),('motion_mask',(3,))):
     if np.asarray(history[key]).shape!=shape:raise ValueError('navigation history shape '+key)
    selected.append(record);input_checks.append({'sample_id':record['sample_id'],'embodiment':robot,'map_valid':not invalid,'route_available':bool(route['route_candidate_mask'].any()),'D024_safe_stop':cells is None});break
   else:raise ValueError('no compatible smoke validation task for '+robot)
  records=selected

 if not a.resume:checkpoint()
 t0=time.time();checks=json.loads((out/'checks.json').read_text()) if a.resume and (out/'checks.json').is_file() else {'selected_base_ids':[ds[i][3]['sample_id'] for i in pool] if a.smoke=='cpu' else None,'purpose':frozen['purpose'],'eligible_for_formal_result':False}
 while state['update']<max_updates and (a.stop_after is None or state['update']<a.stop_after):
  attempt_rng=capture_rng(streams);attempt_sampler={k:state[k] for k in ('cursor','epoch','permutation')}
  if state['cursor']>=len(state['permutation']):
   state['permutation']=[pool[i] for i in torch.randperm(len(pool),generator=streams['base_sampler']).tolist()];state['cursor']=0;state['epoch']+=1
  step_started=time.perf_counter();pair_seconds=0.
  ids=state['permutation'][state['cursor']:state['cursor']+bs];state['cursor']+=len(ids);batch=default_collate([ds[i] for i in ids]);opt.zero_grad(set_to_none=True);model.train();u=state['update']
  def sliced(batch,start,end):
   x,y,m,meta=batch
   return ({k:v[start:end] for k,v in x.items()},y[start:end],m[start:end],{k:v[start:end] for k,v in meta.items()})
  def draw_noise(n,generator):return torch.rand(n,generator=generator)*.999+.001,torch.randn(n,11,80,4,generator=generator)
  noise=draw_noise(len(ids),streams['base_noise']);base_den=batch[2].sum()
  if base_den<=0:raise ValueError('empty base point denominator')
  losses={'base':0.};micro=cfg['microbatch_size']
  with torch.random.fork_rng(devices=[torch.cuda.current_device()] if a.device=='cuda' else []):
   torch.manual_seed(11+u)
   for start in range(0,len(ids),micro):
    end=min(start+micro,len(ids));piece=sliced(batch,start,end)
    with torch.autocast(device_type=a.device,enabled=a.amp):
     term,_,_,_,_,_=forward(model,c,piece,streams['base_noise'],streams['id_dropout'],wrapped,a.device,(noise[0][start:end],noise[1][start:end]));term=term*piece[2].sum().to(a.device)/base_den.to(a.device)
    scaler.scale(term).backward();losses['base']+=float(term.detach())
  pairids=[]
  if wrapped:
   pair_started=time.perf_counter()
   indices=([min(range(len(pairs)),key=lambda i:(pairs.rows[i].get('candidate_kind')!='native8m_frozen_base_window',pairs.rows[i]['pair_id']))] if a.smoke=='cpu' else torch.randint(len(pairs),(ps,),generator=streams['pair_sampler']).tolist());pairbatch=default_collate([pairs[i] for i in indices]);pairids=pairbatch['pair_id'];pn=draw_noise(len(indices),streams['pair_noise'])
   pa,pb=pairbatch['a'],pairbatch['b'];common=pa[2]&pb[2]&pairbatch['pair_valid'][:,None]
   den={'a':pa[2].sum(),'b':pb[2].sum(),'common':common.sum(),'classifier':2*len(indices)}
   if any(float(v)<=0 for v in den.values()):raise ValueError('empty pair full-batch denominator')
   losses['pair_weighted']=0.;pmicro=cfg['pair_microbatch_size']
   with torch.random.fork_rng(devices=[torch.cuda.current_device()] if a.device=='cuda' else []):
    torch.manual_seed(100000+u)
    for start in range(0,len(indices),pmicro):
     end=min(start+pmicro,len(indices));shared_noise=(pn[0][start:end],pn[1][start:end])
     with torch.autocast(device_type=a.device,enabled=a.amp):
      _,oa,ta,ma,_,_=forward(model,c,sliced(pa,start,end),streams['pair_noise'],streams['id_dropout'],True,a.device,shared_noise)
      _,ob,tb,mb,_,_=forward(model,c,sliced(pb,start,end),streams['pair_noise'],streams['id_dropout'],True,a.device,shared_noise)
      terms=proxy_pair_losses(oa['decomposition'],ob['decomposition'],ta,tb,ma,mb,pairbatch['pair_valid'][start:end].to(a.device),pairbatch['T_a'][start:end].to(a.device),pairbatch['T_b'][start:end].to(a.device),c.state_normalizer,model.classifier,denominators={k:v.to(a.device) if torch.is_tensor(v) else v for k,v in den.items()})
      q=cfg['objective'];pair_loss=q.get('pair_denoising_weight',1.)*terms['pair_diff']+q['lambda_inv']*terms['inv']+q['lambda_swap']*terms['swap']+q['lambda_sep']*terms['sep']
     scaler.scale(pair_loss).backward()
     for k,v in terms.items():
      if torch.is_tensor(v):losses[k]=losses.get(k,0.)+float(v.detach())
     losses['pair_weighted']+=float(pair_loss.detach())
   pair_seconds=time.perf_counter()-pair_started
  if not all(np.isfinite(v) for v in losses.values()):raise FloatingPointError('nonfinite Proxy loss')
  scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'])
  if not torch.isfinite(norm) and not a.amp:raise FloatingPointError('nonfinite Proxy gradient')
  if a.smoke=='cpu' and u==1:
   checks['second_step_gradients']={k:float(p.grad.norm()) for k,p in model.named_parameters() if p.grad is not None and ('proxy_history' in k or 'residual' in k or 'classifier' in k)}
   for module in (() if a.disable_history else ('proxy_history',))+(('residual','classifier') if wrapped else ()):
    if not any(module in k and v>0 for k,v in checks['second_step_gradients'].items()):raise AssertionError('missing second-step gradient: '+module)
  old_scale=scaler.get_scale();scaler.step(opt);scaler.update()
  if scaler.get_scale()<old_scale:
   restore_rng(attempt_rng,streams);state.update(attempt_sampler);checkpoint();continue
  state['update']+=1;state['base_exposures']+=len(ids);state['pair_exposures']+=2*len(pairids);
  for pair_id in pairids:
   if pair_id in state['pair_draw_counts']:state['pair_repeated_draws']+=1
   state['pair_draw_counts'][pair_id]=state['pair_draw_counts'].get(pair_id,0)+1
  row={'target_updates':state['update'],'base_ids':batch[3]['sample_id'],'pair_ids':pairids,'losses':losses,'lr':opt.param_groups[0]['lr'],'step_s':time.perf_counter()-step_started,'pair_s':pair_seconds}
  if a.smoke!='none':state['last_noise']={'base_t':noise[0].clone(),'base_noise':noise[1].clone(),**({'pair_t':pn[0].clone(),'pair_noise':pn[1].clone()} if wrapped else {})}
  epoch_end=state['cursor']==len(state['permutation'])
  validate_now=validation_due(state,cfg,study and a.smoke=='none')
  if study:row.update(epoch=state['epoch'],completed_epochs=state['epoch'] if epoch_end else state['epoch']-1)
  if validate_now and a.smoke!='cpu':
   validation_started=time.perf_counter();rng=capture_rng(streams);model.eval()
   try:
    with torch.no_grad():
     nav=[{**rollout(r,c,model,wrapped,i,inference_seed=seed),'inference_seed':seed} for seed in (cfg['inference_seeds'] if study else [20260914]) for i,r in enumerate(records)]
     vg_base=torch.Generator().manual_seed(20260914);vid_base=torch.Generator().manual_seed(14);val_sse=0.;val_points=0
     for start in range(0,len(vd),cfg['batch_size']):
      vb=default_collate([vd[i] for i in range(start,min(start+cfg['batch_size'],len(vd)))])
      value,_,_,vm,_,_=forward(model,c,vb,vg_base,vid_base,wrapped,a.device)
      count=int(vm.sum());val_sse+=float(value)*count;val_points+=count
     row['val_denoising']={'sse':val_sse,'valid_points':val_points,'mse':val_sse/val_points}

     pair_diagnostic=None
     if wrapped and len(valpairs):
      allpairs=default_collate([valpairs[i] for i in range(len(valpairs))]);va,vb=allpairs['a'],allpairs['b'];valid_common=va[2]&vb[2]&allpairs['pair_valid'][:,None];den={'a':va[2].sum(),'b':vb[2].sum(),'common':valid_common.sum(),'classifier':2*len(valpairs)};pair_diagnostic={};vg=torch.Generator().manual_seed(20260914);vid=torch.Generator().manual_seed(14)
      for start in range(0,len(valpairs),cfg['pair_microbatch_size']):
       end=min(start+cfg['pair_microbatch_size'],len(valpairs));noise=draw_noise(end-start,vg)
       _,oa,ta,ma,_,_=forward(model,c,sliced(va,start,end),vg,vid,True,a.device,noise)
       _,ob,tb,mb,_,_=forward(model,c,sliced(vb,start,end),vg,vid,True,a.device,noise)
       terms=proxy_pair_losses(oa['decomposition'],ob['decomposition'],ta,tb,ma,mb,allpairs['pair_valid'][start:end].to(a.device),allpairs['T_a'][start:end].to(a.device),allpairs['T_b'][start:end].to(a.device),c.state_normalizer,model.classifier,denominators={k:v.to(a.device) if torch.is_tensor(v) else v for k,v in den.items()})
       for key,value in terms.items():pair_diagnostic[key]=pair_diagnostic.get(key,0.)+float(value)
     row['val_pair_diagnostic']=pair_diagnostic

   finally:restore_rng(rng,streams);model.train()
   valid=[r for r in nav if r['included_in_denominator']]
   if not valid:raise ValueError('no valid validation tasks')
   platforms={robot:episode_macro([r for r in valid if r['embodiment']==robot]) for robot in robots}
   if any(v['episodes']==0 for v in platforms.values()):raise ValueError('all frozen validation platforms required')
   macro={key:float(np.mean([v[key] for v in platforms.values()])) for key in ('sr','spl','cr','goal_progress')};row['navigation']=macro;row['platform_navigation']=platforms;scheduler.step(macro['sr'])
   if state['best'] is None or checkpoint_selection_key(row)>checkpoint_selection_key(state['best']):
    state['best']=row;state['best_model']={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};publish({'model':state['best_model'],'method':a.method,'profile':'proxy_ab','target_updates':state['update'],'config':frozen,'navigation_metrics':macro},out/'navigation_best.pt')
   if macro['sr']>state['best_sr']:state['best_sr']=macro['sr'];state['stale']=0
   else:state['stale']+=1
   state['next_val']+=val_every;row['validation_s']=time.perf_counter()-validation_started
  state['history'].append(row);print(json.dumps(row),flush=True)
  if a.smoke!='none' or 'navigation' in row:checkpoint()
  if a.smoke=='none' and state['update']>=min_updates and state['stale']>=cfg['patience'] and (not study or 'navigation' in row):break
 if a.smoke=='none':checkpoint()
 if a.smoke=='cpu':
  model.eval();probe=default_collate([ds[i] for i in pool]);rng=capture_rng(streams)
  with torch.no_grad():_,before,_,_,_,_=forward(model,c,probe,streams['base_noise'],streams['id_dropout'],wrapped,a.device)
  saved=torch.load(out/'last.pt',map_location=a.device,weights_only=False);model.load_state_dict(saved['model'],strict=True);restore_rng(rng,streams)
  with torch.no_grad():_,after,_,_,_,_=forward(model,c,probe,streams['base_noise'],streams['id_dropout'],wrapped,a.device)
  checks['strict_reload_equal']=torch.equal(before['score'],after['score'])
  if not checks['strict_reload_equal']:raise AssertionError('strict reload mismatch')
  if wrapped:
   x,_,_,meta=probe;x=c.observation_normalizer({k:v.to(a.device) for k,v in x.items()})
   with torch.no_grad():_,sample=model(x,meta['platform_id'].to(a.device),id_mask=torch.ones(len(pool),device=a.device))
   checks['sampler_finite']=bool(torch.isfinite(sample['prediction']).all())
   if not checks['sampler_finite']:raise AssertionError('nonfinite B sampler')
  checks['validation_input_only']=input_checks
  publish(checks,out/'checks.json',True)
 result={'completed_epochs':state['epoch'] if state['cursor']==len(state['permutation']) else state['epoch']-1,'validation_clock':'epoch5' if study and a.smoke=='none' else 'updates','status':'complete' if state['update']>=max_updates or (a.smoke=='none' and state['update']>=min_updates and state['stale']>=cfg['patience']) else 'interrupted','method':a.method,'purpose':frozen['purpose'],'eligible_for_formal_result':a.smoke=='none','updates':state['update'],'stop_reason':'max_updates' if state['update']==max_updates else ('interrupted' if a.stop_after is not None and state['update']==a.stop_after else 'sr_patience'),'best_update':state['best']['target_updates'] if state['best'] else None,'base_exposures':state['base_exposures'],'pair_exposures':state['pair_exposures'],'pair_repeated_draws':state['pair_repeated_draws'],'seconds':state['elapsed_seconds'],'peak_memory_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else None,'history':state['history']};publish(result,out/'metrics.json',True)

def compare_checkpoints():
 p=argparse.ArgumentParser();p.add_argument('--compare-checkpoints',nargs=2,required=True);p.add_argument('--comparison-output',required=True);a=p.parse_args()
 left,right=[torch.load(path,map_location='cpu',weights_only=False) for path in a.compare_checkpoints]
 def compare(x,y,path=''):
  if torch.is_tensor(x):
   if x.dtype.is_floating_point:torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-6,msg=path)
   elif not torch.equal(x,y):raise AssertionError(path)
  elif isinstance(x,dict):
   if x.keys()!=y.keys():raise AssertionError(path)
   for k in x:
    if k not in ('step_s','pair_s','validation_s','elapsed_seconds'):compare(x[k],y[k],path+'/'+str(k))
  elif isinstance(x,(list,tuple)):
   if len(x)!=len(y):raise AssertionError(path)
   for i,(u,v) in enumerate(zip(x,y)):compare(u,v,path+'/'+str(i))
  elif isinstance(x,np.ndarray):np.testing.assert_equal(x,y,err_msg=path)
  elif isinstance(x,float):
   if not np.isclose(x,y,rtol=1e-4,atol=1e-6,equal_nan=True):raise AssertionError(path)
  elif x!=y:raise AssertionError(path)
 lc,rc=left['config'],right['config']
 lc={**lc,'cli':{k:v for k,v in lc['cli'].items() if k!='output'}};rc={**rc,'cli':{k:v for k,v in rc['cli'].items() if k!='output'}}
 compare(lc,rc,'config')
 for key in ('model','optimizer','scheduler','scaler','training_state','rng','input_identity'):compare(left[key],right[key],key)
 publish({'status':'PASS','rtol':1e-4,'atol':1e-6,'continuous':a.compare_checkpoints[0],'resumed':a.compare_checkpoints[1],'verified':['model','optimizer','scheduler','scaler','training_state','rng','batch_ids','losses','selector','last_diffusion_time_and_noise']},Path(a.comparison_output),True)


if __name__=='__main__':
 if '--compare-checkpoints' in sys.argv:compare_checkpoints()
 elif '--profile' in sys.argv:proxy_train()
 else:main()
