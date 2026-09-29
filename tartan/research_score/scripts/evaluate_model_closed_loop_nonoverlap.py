from __future__ import annotations
import argparse, json, math, time, os, subprocess, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
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

def route_inputs(c,sparse,goal,current,yaw):
 shift=np.rint(np.asarray(current)/.5).astype(int);x=np.asarray(sparse).copy();n=int(max(250,x[:,:2].max()+1));x[:,:2]-=shift
 keep=(x[:,0]>=0)&(x[:,1]>=0)&(x[:,0]<n)&(x[:,1]<n);x=x[keep];goal_rel=np.asarray(goal)-np.asarray(current)
 q=OccupancyRouteSetBuilder(RouteSpec(max_candidates=6,points_per_candidate=80)).build(x,goal_rel)
 lanes=np.zeros((c.lane_num,c.lane_len,c.lane_state_dim),np.float32);routes=np.zeros((c.route_num,c.route_len,c.route_state_dim),np.float32);j=0
 for k in np.flatnonzero(q['route_candidate_mask']):
  for seg in _polyline_features(_route_segments(q['route_candidates_xy'][k],4,c.lane_len),.8):
   if j<c.route_num:routes[j]=seg[:,:c.route_state_dim]
   if j<c.lane_num:lanes[j]=seg[:,:c.lane_state_dim]
   j+=1
 z=lambda *s:np.zeros(s,np.float32)
 inp={'ego_current_state':np.array([0,0,math.cos(yaw),math.sin(yaw),0,0,0,0,0,0],np.float32),'neighbor_agents_past':z(c.agent_num,c.time_len,c.agent_state_dim),'static_objects':z(c.static_objects_num,c.static_objects_state_dim),'lanes':lanes,'lanes_speed_limit':z(c.lane_num,1),'lanes_has_speed_limit':np.zeros((c.lane_num,1),bool),'route_lanes':routes,'route_lanes_speed_limit':z(c.route_num,1),'route_lanes_has_speed_limit':np.zeros((c.route_num,1),bool)}
 return {k:torch.from_numpy(v)[None].cuda() for k,v in inp.items()},q

def controller_segment(pred,max_distance=1.0,step=.1):
 p=np.asarray(pred,float)[:,:2];p=p[np.isfinite(p).all(1)]
 if not len(p):return np.zeros((1,2))
 p=np.vstack(([0.,0.],p));d=np.linalg.norm(np.diff(p,axis=0),axis=1);keep=np.r_[True,d>1e-5];p=p[keep]
 if len(p)<2:return np.zeros((1,2))
 arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];end=min(max_distance,arc[-1]);s=np.arange(step,end+1e-6,step)
 if not len(s):return np.zeros((1,2))
 return np.column_stack([np.interp(s,arc,p[:,0]),np.interp(s,arc,p[:,1])])

def load_model(method,checkpoint,args):
 c=Config(args,None);c.device='cuda';base=Diffusion_Planner(c);wrapped=method in {'pretrain_adapter','emb_cond_diffusion'};m=ScoreDecompositionPlanner(base).cuda() if wrapped else base.cuda();z=torch.load(checkpoint,map_location='cpu',weights_only=False);m.load_state_dict(z['model'],strict=True);m.eval();return c,m,wrapped

def rollout(record,c,model,wrapped,episode_index,max_replans=16):
 sparse=np.load(record['route_set']['map_reference'],allow_pickle=False);goal=np.asarray(record['fixed_goal']['xy_local'],float);prepared=prepare_episode(record,'anymal');_,blocked,start,_,cells,shortest,_,invalid=prepared
 if cells is None:return finish(record,np.zeros((1,2)),goal,shortest,False,False,'route_failure',0,0.)
 if invalid:return finish(record,np.zeros((1,2)),goal,shortest,False,False,'invalid_map_episode',0,0.)
 current=np.zeros(2);yaw=0.;executed=[current.copy()];stalled=0;lat=[];reason='timeout';collision=False
 for rep in range(max_replans):
  inp,q=route_inputs(c,sparse,goal,current,yaw)
  if not q['route_candidate_mask'].any():reason='route_failure';break
  inp=c.observation_normalizer(inp);torch.manual_seed(10000+episode_index*100+rep);torch.cuda.manual_seed(10000+episode_index*100+rep);t=time.perf_counter()
  with torch.no_grad():
   if wrapped:_,o=model(inp,torch.ones(1,dtype=torch.long,device='cuda'),ABILITY_ANYMAL.cuda())
   else:_,o=model(inp)
  torch.cuda.synchronize();lat.append(time.perf_counter()-t);seg=controller_segment(o['prediction'][0,0].float().cpu().numpy())
  absolute=current[None]+seg;hit,idx=path_collision(absolute,blocked,start)
  if hit:
   if idx:absolute=absolute[:idx[0]+1]
   executed.extend(absolute.tolist());collision=True;reason='collision';break
  travelled=float(np.linalg.norm(absolute[-1]-current)) if len(absolute) else 0.;stalled=stalled+1 if travelled<.05 else 0
  if len(absolute):
   delta=absolute[-1]-current;current=absolute[-1];executed.extend(absolute.tolist())
   if np.linalg.norm(delta)>1e-4:yaw=math.atan2(delta[1],delta[0])
  if np.linalg.norm(current-goal)<=.75:reason='success';break
  if stalled>=3:reason='stuck';break
 return finish(record,np.asarray(executed),goal,shortest,reason=='success',collision,reason,rep+1,float(np.mean(lat)) if lat else 0.)

def finish(r,path,goal,shortest,success,collision,reason,replans,latency):
 m=episode_metrics(success=success,collision=collision,shortest_path_m=shortest,executed_path=path,initial_goal_distance_m=float(np.linalg.norm(goal)),final_goal_distance_m=float(np.linalg.norm(path[-1]-goal)),termination_reason=reason)
 return {**m,'sample_id':r['sample_id'],'episode_id':r['episode_id'],'segment_index':r['segment_index'],'anchor_index':r['anchor_index'],'segment_end_index':r['segment_end_index'],'map_id':r['map_id'],'branch':r['branch'],'goal_distance_m':float(np.linalg.norm(goal)),'replans':replans,'mean_inference_s':latency,'included_in_denominator':reason!='invalid_map_episode'}

def episode_macro(rows):
 groups={}
 for r in rows:groups.setdefault(r['episode_id'],[]).append(r)
 per={k:aggregate(v) for k,v in groups.items()}
 keys=('sr','cr','spl','stuck_rate','route_failure_rate','goal_progress')
 return {'episodes':len(per),**{k:float(np.mean([v[k] for v in per.values()])) for k in keys},'per_episode':per}

def publish_json(obj,path):
 fd,tmp=tempfile.mkstemp(prefix='closed_loop_',suffix='.json');os.close(fd)
 try:Path(tmp).write_text(json.dumps(obj,indent=2)+'\n');subprocess.run(['dd',f'if={tmp}',f'of={path}','conv=fsync','status=none'],check=True)
 finally:Path(tmp).unlink(missing_ok=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--args',required=True);p.add_argument('--test-manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);c,m,w=load_model(a.method,a.checkpoint,a.args);records=load_segments(a.test_manifest);rows=[rollout(r,c,m,w,i) for i,r in enumerate(records)];pd.DataFrame(rows).to_csv(out/'per_segment.csv',index=False);valid=[r for r in rows if r['included_in_denominator']];summary=aggregate(valid);summary.update({'method':a.method,'segments_selected':len(rows),'episode_macro':episode_macro(valid),'mean_inference_s':float(np.mean([r['mean_inference_s'] for r in rows]))});publish_json(summary,out/'summary.json');print(json.dumps(summary))
if __name__=='__main__':main()
