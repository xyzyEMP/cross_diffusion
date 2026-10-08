import numpy as np
from tartan.research_score.data.core import resample_fixed_arc
def test_bridge_and_padding_mask():
    x=np.column_stack([np.linspace(0,5,10),np.zeros(10),np.zeros(10)]); y,m=resample_fixed_arc(x,10,80)
    assert y.shape==(80,4) and 0<m.sum()<80
    pred=y.copy();pred[~m]=999
    assert np.mean((pred[m]-y[m])**2)==0
