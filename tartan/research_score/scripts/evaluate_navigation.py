from __future__ import annotations
import shlex, sys
import argparse, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from tartan.research_score.artifacts import publish, validate_output_name
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from tartan.data.pose_utils import load_occupancy_record
from tartan.data.features import _polyline_features,_route_segments
from tartan.research_score.data.route_builder import OccupancyRouteSetBuilder,RouteSpec
from tartan.research_score.evaluation.goal_benchmark import prepare_episode
from tartan.research_score.evaluation.collision import path_collision
from tartan.research_score.evaluation.metrics_navigation import aggregate,episode_metrics
from tartan.research_score.model.score_decomposition import ScoreDecompositionPlanner

ABILITY_ANYMAL=torch.tensor([[.35,.30,.30,.85,.55,.80,.75,.70,.80,.65]],dtype=torch.float32)

def load_segments(manifest):
 rows=[json.loads(line) for line in Path(manifest).open() if line.strip()]
 return sorted(rows,key=lambda r:(r['episode_id'],r['segment_index']))

def route_inputs(c,sparse,goal,current,yaw,proxy=False):
 shift=np.rint(np.asarray(current)/.5).astype(int);x=np.asarray(sparse).copy();n=101 if proxy else int(max(250,x[:,:2].max()+1));x[:,:2]-=shift
 keep=(x[:,0]>=0)&(x[:,1]>=0)&(x[:,0]<n)&(x[:,1]<n);x=x[keep];goal_rel=np.asarray(goal)-np.asarray(current)
 builder=OccupancyRouteSetBuilder(RouteSpec(max_candidates=6,points_per_candidate=80),grid_size=101) if proxy else OccupancyRouteSetBuilder(RouteSpec(max_candidates=6,points_per_candidate=80))
 q=builder.build(x,goal_rel)
 lanes=np.zeros((c.lane_num,c.lane_len,c.lane_state_dim),np.float32);routes=np.zeros((c.route_num,c.route_len,c.route_state_dim),np.float32);j=0
 for k in np.flatnonzero(q['route_candidate_mask']):
  for seg in _polyline_features(_route_segments(q['route_candidates_xy'][k] @ np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]]) if proxy else q['route_candidates_xy'][k],4,c.lane_len),.8):
   if j<c.route_num:routes[j]=seg[:,:c.route_state_dim]
   if j<c.lane_num:lanes[j]=seg[:,:c.lane_state_dim]
   j+=1
 z=lambda *s:np.zeros(s,np.float32)
 inp={'ego_current_state':np.array([0,0,1. if proxy else math.cos(yaw),0. if proxy else math.sin(yaw),0,0,0,0,0,0],np.float32),'neighbor_agents_past':z(c.agent_num,c.time_len,c.agent_state_dim),'static_objects':z(c.static_objects_num,c.static_objects_state_dim),'lanes':lanes,'lanes_speed_limit':z(c.lane_num,1),'lanes_has_speed_limit':np.zeros((c.lane_num,1),bool),'route_lanes':routes,'route_lanes_speed_limit':z(c.route_num,1),'route_lanes_has_speed_limit':np.zeros((c.route_num,1),bool)}
 return {k:torch.from_numpy(v)[None].to(c.device) for k,v in inp.items()},q

def controller_segment(pred,max_distance=1.0,step=.1):
 p=np.asarray(pred,float)[:,:2];p=p[np.isfinite(p).all(1)]
 if not len(p):return np.zeros((1,2))
 p=np.vstack(([0.,0.],p));d=np.linalg.norm(np.diff(p,axis=0),axis=1);keep=np.r_[True,d>1e-5];p=p[keep]
 if len(p)<2:return np.zeros((1,2))
 arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];end=min(max_distance,arc[-1]);s=np.arange(step,end+1e-6,step)
 if not len(s):return np.zeros((1,2))
 return np.column_stack([np.interp(s,arc,p[:,0]),np.interp(s,arc,p[:,1])])

def controller_heading(pred, distance):
 """Interpolate predicted observation yaw at the executed spatial station."""
 p=np.asarray(pred,float);xy=np.vstack(([0.,0.],p[:,:2]));angles=np.r_[0.,np.arctan2(p[:,3],p[:,2])]
 arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
 _,first=np.unique(arc,return_index=True)
 angle=float(np.interp(distance,arc[first],np.unwrap(angles[first])))
 return math.atan2(math.sin(angle),math.cos(angle))


def load_model(method,checkpoint,args,device='cuda',profile='transfer_primary'):
 c=Config(args,None);c.device=device;base=Diffusion_Planner(c)
 z=torch.load(checkpoint,map_location='cpu',weights_only=False)
 disable_history=z.get('config',{}).get('cli',{}).get('disable_history',False)
 if profile=='proxy_ab' and not disable_history:base.encoder.encoder.enable_proxy_history()
 wrapped=method in {'pretrain_adapter','emb_cond_diffusion','proxy_b'};m=ScoreDecompositionPlanner(base,proxy=True).to(device) if profile=='proxy_ab' and wrapped else (ScoreDecompositionPlanner(base).to(device) if wrapped else base.to(device));state=z.get('ema_state_dict',z.get('model',z));state={k.removeprefix('module.'):v for k,v in state.items()};m.load_state_dict(state,strict=True);m.eval();return c,m,wrapped

def rollout(record,c,model,wrapped,episode_index,max_replans=16,inference_seed=10000):
 sparse=load_occupancy_record(record);goal=np.asarray(record['fixed_goal']['xy_local'],float);prepared=prepare_episode(record,record.get('embodiment','anymal'));_,blocked,start,_,cells,shortest,_,invalid=prepared
 if cells is None:return finish(record,np.zeros((1,2)),goal,shortest,False,False,'route_failure',0,0.)
 if invalid:return finish(record,np.zeros((1,2)),goal,shortest,False,False,'invalid_map_episode',0,0.)
 current=np.zeros(2);yaw=0.;executed=[current.copy()];stalled=0;lat=[];reason='timeout';collision=False
 for rep in range(max_replans):
  proxy=record.get('profile')=='proxy_ab' or hasattr(model.backbone.encoder.encoder if wrapped else model.encoder.encoder,'proxy_history')
  inp,q=route_inputs(c,sparse,goal,current,yaw,proxy)
  if not q['route_candidate_mask'].any():reason='route_failure';break
  if proxy:
   history=record.get('history',record)
   for key in ('ego_history','history_mask','history_dt','history_dt_mask','motion_rms','motion_mask'):
    inp[key]=torch.as_tensor(history[key],device=c.device)[None]
  inp=c.observation_normalizer(inp);torch.manual_seed(inference_seed+episode_index*100+rep);torch.cuda.manual_seed(inference_seed+episode_index*100+rep);t=time.perf_counter()
  with torch.no_grad():
   if wrapped and proxy:
    eid=torch.tensor([record['platform_id']],device=c.device);_,o=model(inp,eid,id_mask=(eid!=1).float())
   elif wrapped:_,o=model(inp,torch.ones(1,dtype=torch.long,device=c.device),ABILITY_ANYMAL.to(c.device))
   else:_,o=model(inp)
  if str(c.device).startswith('cuda'):torch.cuda.synchronize()
  lat.append(time.perf_counter()-t);prediction=o['prediction'][0,0].float().cpu().numpy();seg=controller_segment(prediction)
  predicted_yaw_delta=controller_heading(prediction,.1*len(seg)) if record.get('experiment_profile')=='four_groups' else None
  if proxy:seg=seg @ np.array([[math.cos(yaw),math.sin(yaw)],[-math.sin(yaw),math.cos(yaw)]])
  absolute=current[None]+seg;hit,idx=path_collision(absolute,blocked,start)
  if hit:
   if idx:absolute=absolute[:idx[0]+1]
   executed.extend(absolute.tolist());collision=True;reason='collision';break
  travelled=float(np.linalg.norm(absolute[-1]-current)) if len(absolute) else 0.;stalled=stalled+1 if travelled<.05 else 0
  if len(absolute):
   delta=absolute[-1]-current;current=absolute[-1];executed.extend(absolute.tolist())
   if np.linalg.norm(delta)>1e-4:yaw=(yaw+predicted_yaw_delta) if predicted_yaw_delta is not None else math.atan2(delta[1],delta[0])
  if np.linalg.norm(current-goal)<=.75:reason='success';break
  if stalled>=3:reason='stuck';break
 return finish(record,np.asarray(executed),goal,shortest,reason=='success',collision,reason,rep+1,float(np.mean(lat)) if lat else 0.)

def finish(r,path,goal,shortest,success,collision,reason,replans,latency):
 m=episode_metrics(success=success,collision=collision,shortest_path_m=shortest,executed_path=path,initial_goal_distance_m=float(np.linalg.norm(goal)),final_goal_distance_m=float(np.linalg.norm(path[-1]-goal)),termination_reason=reason)
 return {**m,'embodiment':r.get('embodiment','anymal'),'trajectory_key':r.get('trajectory_key',r['episode_id']),'sample_id':r['sample_id'],'episode_id':r['episode_id'],'segment_index':r['segment_index'],'anchor_index':r['anchor_index'],'segment_end_index':r['segment_end_index'],'map_id':r['map_id'],'branch':r['branch'],'goal_distance_m':float(np.linalg.norm(goal)),'replans':replans,'mean_inference_s':latency,'included_in_denominator':reason!='invalid_map_episode'}

def episode_macro(rows):
 groups={}
 for r in rows:groups.setdefault(r['episode_id'],[]).append(r)
 per={k:aggregate(v) for k,v in groups.items()}
 keys=('sr','cr','spl','stuck_rate','route_failure_rate','goal_progress')
 return {'episodes':len(per),**{k:float(np.mean([v[k] for v in per.values()])) for k in keys},'per_episode':per}

def main():
 p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--args',required=True);p.add_argument('--test-manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--inference-seed',type=int,default=10000);a=p.parse_args();out=Path(a.output);validate_output_name(out.name);out.mkdir(parents=True,exist_ok=False);(out/"command.txt").write_text(shlex.join([sys.executable,"-m",__spec__.name,*sys.argv[1:]])+"\n");publish(vars(a),out/"config.json",True);c,m,w=load_model(a.method,a.checkpoint,a.args);records=load_segments(a.test_manifest);rows=[rollout(r,c,m,w,i,inference_seed=a.inference_seed) for i,r in enumerate(records)];pd.DataFrame(rows).to_csv(out/'per_segment.csv',index=False);valid=[r for r in rows if r['included_in_denominator']];summary=aggregate(valid);summary.update({'method':a.method,'segments_selected':len(rows),'episode_macro':episode_macro(valid),'mean_inference_s':float(np.mean([r['mean_inference_s'] for r in rows]))});publish(summary,out/'summary.json',True);print(json.dumps(summary))



def offline_metrics(prediction, target, mask, normalized_prediction=None, normalized_target=None):
 """Physical XY metrics; FDE is the final valid spatial station."""
 keep=mask.bool()
 if not keep.any():raise ValueError('offline window has no valid points')
 error=torch.linalg.vector_norm(prediction[keep,:2]-target[keep,:2],dim=-1)
 sse=((normalized_prediction[keep]-normalized_target[keep]).square().sum().item() if normalized_prediction is not None else None)
 return {'sse':sse,'valid_count':int(keep.sum()),'ade':float(error.mean()),'fde':float(error[-1]),'distance_sum':float(error.sum())}


def proxy_evaluate():
 from tartan.research_score.training.cached_target_dataset import CachedTargetDataset
 from torch.utils.data import default_collate
 p=argparse.ArgumentParser();p.add_argument('--profile',choices=['proxy_ab','four_groups'],required=True);p.add_argument('--config',required=True);p.add_argument('--method',choices=['proxy_a','proxy_b'],required=True)
 for k in ('checkpoint','args','test-manifest','output'):p.add_argument('--'+k,required=True)
 p.add_argument('--diagnostic-train-cache');p.add_argument('--diagnostic-val-cache');p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--evaluation-kind',choices=['offline','navigation'],required=True);p.add_argument('--test-cache');a=p.parse_args()
 import yaml
 from tartan.research_score.preflight.cli import _resolve
 cfg=_resolve(yaml.safe_load(Path(a.config).read_text())[a.profile]);study=a.profile=='four_groups'
 if a.device!='cuda' or not torch.cuda.is_available():raise RuntimeError('final ANYmal model evaluation requires the authorized GPU phase')
 checkpoint=Path(a.checkpoint)
 if checkpoint.name!='navigation_best.pt':raise ValueError('final evaluation requires navigation_best.pt')
 saved=torch.load(checkpoint,map_location='cpu',weights_only=False);frozen=saved['config']
 if frozen['proxy_ab']!=cfg or saved['method']!=a.method or frozen['cli']['args']!=a.args:raise ValueError('evaluation configuration differs from frozen checkpoint')
 if frozen['purpose']!='formal' or not frozen['eligible_for_formal_result']:raise ValueError('smoke checkpoint is ineligible for final test')
 train_metrics=json.loads((checkpoint.parent/'metrics.json').read_text())
 if train_metrics['status']!='complete' or not train_metrics['eligible_for_formal_result']:raise ValueError('training must complete before final test')
 data=Path(cfg['paths']['output_root'])/'data'/cfg['paths']['data_id']
 expected_manifest=data/(('test.jsonl' if a.evaluation_kind=='offline' else 'navigation_test.jsonl') if study else ('anymal_test.jsonl' if a.evaluation_kind=='offline' else 'navigation_anymal_test.jsonl'))
 if Path(a.test_manifest).resolve()!=expected_manifest.resolve():raise ValueError('test manifest does not belong to frozen DATA_ID')
 if a.evaluation_kind=='offline':
  z=torch.load(a.test_cache,map_location='cpu',weights_only=False)
  manifest_ids=[json.loads(line)['sample_id'] for line in expected_manifest.open() if line.strip()]
  if set(z['sample_ids'])!=set(manifest_ids) or len(z['sample_ids'])!=len(manifest_ids):raise ValueError('test cache/manifest IDs mismatch')
 out=Path(a.output);out.mkdir(parents=True,exist_ok=study);evaluation_config={'cli':vars(a),'proxy_ab':cfg,'selected_update':saved['target_updates']}
 if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=evaluation_config:raise ValueError('existing evaluation config differs')
 publish(evaluation_config,out/'config.json',True)
 c,m,w=load_model(a.method,a.checkpoint,a.args,a.device,'proxy_ab')
 if a.evaluation_kind=='navigation':
  records=load_segments(a.test_manifest)
  if any(r['split']!='test' for r in records):raise ValueError('final evaluation requires test-only manifest')
  rows=[{**rollout(r,c,m,w,i,inference_seed=seed),'inference_seed':seed} for seed in (cfg['inference_seeds'] if study else [10000]) for i,r in enumerate(records)];pd.DataFrame(rows).to_csv(out/'per_segment.csv',index=False)
  valid=[r for r in rows if r['included_in_denominator']];summary=aggregate(valid);summary['episode_macro']=episode_macro(valid)
  summary.update({'status':'complete' if valid else 'NO_VALID_TASKS','segments_selected':len(rows),'included_segments':len(valid),'excluded_invalid_map_ids':[r['sample_id'] for r in rows if not r['included_in_denominator']],'raw_trajectory_coverage':sorted({r['trajectory_key'] for r in rows})})
  pd.DataFrame([{'trajectory_key':k,**v} for k,v in summary['episode_macro']['per_episode'].items()]).to_csv(out/'per_trajectory.csv',index=False)
 else:
  if not a.test_cache:raise ValueError('offline requires --test-cache')
  ds=CachedTargetDataset(a.test_cache,c);rows=[];g=torch.Generator().manual_seed(20260914)
  for seed,i in [(seed,i) for seed in (cfg['inference_seeds'] if study else [10000]) for i in sorted(range(len(ds)),key=lambda i:ds[i][3]['sample_id'])]:
   x,y,mask,meta=default_collate([ds[i]])
   if meta['split'][0]!='test' or int(meta['platform_id'][0]) not in ((1,2,3) if study else (1,)):raise ValueError('final offline requires ANYmal test')
   x={k:v.to(a.device) for k,v in x.items()};x=c.observation_normalizer(x);raw=torch.zeros(1,11,80,4,device=a.device);raw[:,0]=y.to(a.device);target=c.state_normalizer(raw)
   t=(torch.rand(1,generator=g)*.999+.001).to(a.device);eps=torch.randn(target.shape,generator=g).to(a.device);mean,std=(m.backbone if w else m).sde.marginal_prob(target,t)
   mean[:,1:]=0;eps[:,1:]=0
   cur=torch.zeros(1,11,1,4,device=a.device);cur[:,0,0]=x['ego_current_state'][:,:4]
   noisy={**x,'sampled_trajectories':torch.cat([cur,mean+std.reshape(-1,1,1,1)*eps],2),'diffusion_time':t}
   with torch.no_grad():
    _,den=m(noisy,meta['platform_id'].to(a.device),id_mask=torch.zeros(1,device=a.device)) if w else m(noisy)
    torch.manual_seed(seed+i if study else 10000+len(rows));_,generated=m(x,meta['platform_id'].to(a.device),id_mask=torch.zeros(1,device=a.device)) if w else m(x)
   metrics=offline_metrics(generated['prediction'][0,0].cpu(),y[0],mask[0],den['score'][0,0,1:].cpu(),target[0,0].cpu())
   rows.append({'inference_seed':seed,'sample_id':meta['sample_id'][0],'trajectory_key':meta['trajectory_key'][0],**metrics})
  frame=pd.DataFrame(rows);frame['mse']=frame.sse/frame.valid_count;frame.to_csv(out/'per_window.csv',index=False)
  trajectories=frame.groupby('trajectory_key')[['mse','ade','fde']].mean();trajectories.to_csv(out/'per_trajectory.csv')
  declared={json.loads(line)['trajectory_key'] for line in Path(a.test_manifest).open() if line.strip()}
  summary={'trajectory_macro':trajectories.mean().to_dict(),'window_micro':frame[['mse','ade','fde']].mean().to_dict(),'point_weighted_mse':float(frame.sse.sum()/frame.valid_count.sum()),'point_weighted_ade':float(frame.distance_sum.sum()/frame.valid_count.sum()),'covered_trajectories':sorted(trajectories.index),'uncovered_trajectories':sorted(declared-set(trajectories.index))}
 if a.diagnostic_train_cache or a.diagnostic_val_cache:
  if not all((a.diagnostic_train_cache,a.diagnostic_val_cache)):raise ValueError('both diagnostic caches required')
  summary['frozen_physical_diagnostics']=proxy_diagnostics(m,c,a.diagnostic_train_cache,a.diagnostic_val_cache,a.device,w)
 publish(summary,out/'summary.json',True)

def heading_diagnostics(trajectory,mask):
 p=np.asarray(trajectory,float);valid=np.asarray(mask,bool);d=np.diff(p[:,:2],axis=0);keep=valid[:-1]&valid[1:]&(np.linalg.norm(d,axis=1)>1e-8);heading=p[:-1,2:4];hn=np.linalg.norm(heading,axis=1);ok=keep&(hn>1e-8)
 angles=np.arccos(np.clip(np.sum(d[ok]*heading[ok],axis=1)/(np.linalg.norm(d[ok],axis=1)*hn[ok]),-1,1))
 turnkeep=keep[:-1]&keep[1:];turn=np.arccos(np.clip(np.sum(d[:-1][turnkeep]*d[1:][turnkeep],axis=1)/(np.linalg.norm(d[:-1][turnkeep],axis=1)*np.linalg.norm(d[1:][turnkeep],axis=1)),-1,1))
 return {'heading_sum':float(angles.sum()),'heading_count':len(angles),'turn_sum':float(turn.sum()),'turn_count':len(turn),'zero_displacement_count':int((valid[:-1]&valid[1:]&~keep).sum())}


def proxy_diagnostics(model,c,train_path,val_path,device,wrapped):
 from tartan.research_score.training.cached_target_dataset import CachedTargetDataset
 from torch.utils.data import default_collate
 encoder=(model.backbone if wrapped else model).encoder.encoder
 if not hasattr(encoder,'proxy_history'):raise ValueError('latent diagnostics require history-enabled model')
 collected=[];ids=[];geometry={'shared':[],'total':[]};g=torch.Generator().manual_seed(20260914)
 for path,split in ((train_path,'train'),(val_path,'val')):
  ds=CachedTargetDataset(path,c);z=[];rms=[];masks=[];sample_ids=[]
  for seed,i in [(seed,i) for seed in (cfg['inference_seeds'] if study else [10000]) for i in sorted(range(len(ds)),key=lambda i:ds[i][3]['sample_id'])]:
   x,y,mask,meta=default_collate([ds[i]])
   if meta['split'][0]!=split or int(meta['platform_id'][0]) not in (2,3):raise ValueError('diagnostics forbid ANYmal/test')
   x={k:v.to(device) for k,v in x.items()}
   with torch.no_grad():context=encoder.encode_proxy_history(x)
   z.append(context[0,:16].cpu().numpy());rms.append(x['motion_rms'][0].cpu().numpy());masks.append(x['motion_mask'][0].cpu().numpy());sample_ids.append(meta['sample_id'][0])
   if split=='val':
    inp=c.observation_normalizer(x);raw=torch.zeros(1,11,80,4,device=device);raw[:,0]=y.to(device);target=c.state_normalizer(raw);t=(torch.rand(1,generator=g)*.999+.001).to(device);eps=torch.randn(raw.shape,generator=g).to(device);mean,std=(model.backbone if wrapped else model).sde.marginal_prob(target,t);mean[:,1:]=0;eps[:,1:]=0;cur=torch.zeros(1,11,1,4,device=device);cur[:,0,0]=inp['ego_current_state'][:,:4]
    with torch.no_grad():
     noisy={**inp,'sampled_trajectories':torch.cat([cur,mean+std.reshape(-1,1,1,1)*eps],2),'diffusion_time':t};_,out=model(noisy,meta['platform_id'].to(device),id_mask=torch.ones(1,device=device)) if wrapped else model(noisy)
    shared=out['decomposition'].x0_shared if wrapped else out['score']
    for key,pred in (('shared',shared),('total',out['score'])):geometry[key].append(heading_diagnostics(c.state_normalizer.inverse(pred[:,:,1:])[0,0].cpu().numpy(),mask[0].numpy()))
  collected.append((np.asarray(z),np.asarray(rms),np.asarray(masks,bool)));ids.append(sample_ids)
 train_z,train_y,train_mask=collected[0];val_z,val_y,val_mask=collected[1];readout=[]
 for j in range(3):
  tr=train_mask[:,j];va=val_mask[:,j]
  if not tr.any() or not va.any():readout.append({'dimension':j,'status':'missing_valid_motion'});continue
  weights=np.linalg.lstsq(np.column_stack((train_z[tr],np.ones(tr.sum()))),train_y[tr,j],rcond=None)[0];pred=np.column_stack((val_z[va],np.ones(va.sum())))@weights;truth=val_y[va,j];den=((truth-truth.mean())**2).sum()
  readout.append({'dimension':j,'mae':float(np.abs(pred-truth).mean()),'r2':float(1-((pred-truth)**2).sum()/den) if den>0 else None,'train_mean_baseline_mae':float(np.abs(train_y[tr,j].mean()-truth).mean()),'coefficients':weights.tolist()})
 diagnostic={}
 for key,rows in geometry.items():
  sums={k:sum(r[k] for r in rows) for k in ('heading_sum','heading_count','turn_sum','turn_count','zero_displacement_count')};diagnostic[key]={**sums,'heading_angle_mean_rad':sums['heading_sum']/sums['heading_count'] if sums['heading_count'] else None,'path_turn_mean_rad':sums['turn_sum']/sums['turn_count'] if sums['turn_count'] else None}
 return {'train_ids':ids[0],'val_ids':ids[1],'linear_readout':readout,'val_geometry':diagnostic,'interpretation':'readability only; no causal identification or ability ceiling'}


if __name__=='__main__':
 if '--profile' in sys.argv:proxy_evaluate()
 else:main()
