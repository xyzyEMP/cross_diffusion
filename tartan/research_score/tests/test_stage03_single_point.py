import numpy as np
from tartan.research_score.scripts.build_native_v122 import canonicalize_window

def test_single_point_short_window_uses_xy_cos_sin_and_mask():
    x=np.asarray([[1.25,-0.5,np.pi/3]],dtype=np.float32)
    out,mask=canonicalize_window(x,8.0,80)
    assert out.shape==(80,4)
    assert mask.shape==(80,) and mask.sum()==1 and mask[0]
    assert np.allclose(out[:,0],1.25) and np.allclose(out[:,1],-0.5)
    assert np.allclose(out[:,2]**2+out[:,3]**2,1.0,atol=1e-6)

def test_empty_short_window_is_finite_xy_cos_sin_padding():
    out,mask=canonicalize_window(np.empty((0,3),dtype=np.float32),8.0,80)
    assert out.shape==(80,4) and np.isfinite(out).all() and not mask.any()
    assert np.allclose(out[:,2]**2+out[:,3]**2,1.0)
