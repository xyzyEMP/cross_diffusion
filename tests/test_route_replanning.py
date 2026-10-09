import numpy as np
import copy
import evaluation.goal_benchmark as gb
from evaluation.shortest_path import astar, inflate
from evaluation.goal_benchmark import rollout_shortest

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

def test_straight_and_disconnected():
    g=np.zeros((5,5),bool);p,d=astar(g,(0,0),(0,4),.5);assert d==2 and tuple(p[-1])==(0,4)
    g[2,:]=1;p,d=astar(g,(0,0),(4,4));assert p is None and np.isinf(d)

def test_footprint_inflation():
    g=np.zeros((7,7),bool);g[3,3]=1
    assert inflate(g,.7).sum()==1 and inflate(g,1).sum()==5 and inflate(g,2).sum()==13

def test_future_mutation_invariant_and_goal_sensitive(monkeypatch):
 def fake(rec):
  g=np.asarray(rec["fixed_goal"]["xy_local"],np.float32)
  route={"route_candidates_xy":np.tile(np.linspace([0,0],g,80),(6,1,1)),"route_candidate_mask":np.ones(6,bool)}
  return route,None,g
 monkeypatch.setattr(gb,"build_from_record",fake)
 r={"fixed_goal":{"xy_local":[8.,1.]},"trajectory":{"raw_future_xy_yaw":np.zeros((80,3)).tolist()}}
 assert gb.no_future_check(r)["passed"]
 a=gb.route_digest(fake(r)[0]);z=copy.deepcopy(r);z["fixed_goal"]["xy_local"]=[2.,7.]
 assert a!=gb.route_digest(fake(z)[0])
