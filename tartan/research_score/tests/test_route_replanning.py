import numpy as np
from tartan.research_score.evaluation.shortest_path import astar
from tartan.research_score.evaluation.goal_benchmark import rollout_shortest

def test_replanning_changes_with_goal_and_is_deterministic():
 g=np.zeros((10,10),bool);g[4,1:9]=True;g[4,5]=False
 p1,d1=astar(g,(1,1),(8,1));p2,d2=astar(g,(1,1),(8,8));p2b,d2b=astar(g,(1,1),(8,8))
 assert not np.array_equal(p1,p2) and np.array_equal(p2,p2b) and d2==d2b

def test_disconnected_is_explicit():
 g=np.zeros((8,8),bool);g[3,:]=True
 p,d=astar(g,(1,1),(6,6));assert p is None and np.isinf(d)

def test_rollout_replans_and_reaches_fixed_goal():
 g=np.zeros((30,30),bool);g[10,2:25]=True;g[10,12]=False
 p,reason,n=rollout_shortest(g,(2,2),(20,20),np.array([9.,9.]))
 assert reason=="success" and n>=2 and np.linalg.norm(p[-1]-[9,9])<=.75


def test_proxy_routes_rotate_into_current_body_at_ninety_degrees(monkeypatch):
 from types import SimpleNamespace
 from tartan.research_score.scripts import evaluate_navigation as evaluator
 class MathRoutes:
  def __init__(self,*args,**kwargs):pass
  def build(self,*args):
   path=np.column_stack((np.linspace(0,8,80),np.zeros(80))).astype(np.float32)
   return {'route_candidates_xy':path[None],'route_candidate_mask':np.array([True])}
 monkeypatch.setattr(evaluator,'OccupancyRouteSetBuilder',MathRoutes)
 c=SimpleNamespace(lane_num=70,lane_len=20,lane_state_dim=12,route_num=25,route_len=20,route_state_dim=12,agent_num=32,time_len=21,agent_state_dim=11,static_objects_num=5,static_objects_state_dim=10,device='cpu')
 inputs,_=evaluator.route_inputs(c,np.array([[0,0,0],[249,249,0]]),[8,0],[0,0],np.pi/2,proxy=True)
 route=inputs['route_lanes'][0,0].numpy()
 assert np.abs(route[:,0]).max()<1e-5 and route[:,1].min()<-.5
 assert inputs['ego_current_state'][0,:4].tolist()==[0.,0.,1.,0.]
 local=np.array([[0.,-1.]])
 rotation=np.array([[0.,1.],[-1.,0.]])
 assert np.allclose(local@rotation,[[1.,0.]])
