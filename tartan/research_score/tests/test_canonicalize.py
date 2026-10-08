import numpy as np
from tartan.research_score.data.core import canonical_trajectory
def test_canonical_fields():
    x=np.column_stack([np.linspace(0,10,20),np.zeros(20),np.zeros(20)]);o=canonical_trajectory(x,np.arange(20)/10,8)
    assert o["values"].shape==(80,4) and o["valid_mask"].all()
