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
