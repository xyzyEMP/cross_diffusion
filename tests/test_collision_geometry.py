import numpy as np
from evaluation.collision import path_collision

def test_collision_and_outside():
    g=np.zeros((5,5),bool);g[2,3]=1
    assert path_collision([[0,0],[0,.5]],g,[2,2])[0]
    assert not path_collision([[0,0],[-.5,0]],g,[2,2])[0]
    assert path_collision([[99,0]],g,[2,2])[0]
