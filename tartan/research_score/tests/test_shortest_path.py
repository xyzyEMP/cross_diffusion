import numpy as np
from tartan.research_score.evaluation.shortest_path import astar,inflate

def test_straight_and_disconnected():
    g=np.zeros((5,5),bool);p,d=astar(g,(0,0),(0,4),.5);assert d==2 and tuple(p[-1])==(0,4)
    g[2,:]=1;p,d=astar(g,(0,0),(4,4));assert p is None and np.isinf(d)

def test_footprint_inflation():
    g=np.zeros((7,7),bool);g[3,3]=1
    assert inflate(g,.7).sum()==1 and inflate(g,1).sum()==5 and inflate(g,2).sum()==13
