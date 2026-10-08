import numpy as np
from tartan.research_score.scripts.build_transfer_manifests import canonicalize_window

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


def test_proxy_duplicate_xy_keeps_first_heading_and_short_tail_mask():
    # Mathematical representation check, never a saved training fixture.
    from tartan.research_score.data.core import resample_fixed_arc
    x=np.array([[0.,0.,0.],[0.,0.,1.],[2.,0.,2.]])
    distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(x[:,:2],axis=0),axis=1))]
    _,first=np.unique(distance,return_index=True)
    out,mask=resample_fixed_arc(x[first],8.,80)
    assert np.allclose(out[0],[0.,0.,1.,0.])
    assert mask.sum()==20 and not mask[-1]


def test_proxy_history_body_velocity_and_unknown_time_masks():
    from tartan.data.pose_utils import proxy_history
    se2=np.column_stack((np.zeros(21),np.arange(21)*.1,np.full(21,np.pi/2)))
    known=proxy_history(se2,20,10.)
    assert np.allclose(known['motion_rms'],[1.,0.,0.],atol=1e-6)
    assert known['motion_mask']==[True]*3
    assert len(known['ego_history'])==20 and np.allclose(known['ego_history'][-1][:2],[-.1,0.],atol=1e-6)
    unknown=proxy_history(se2,20,None)
    assert not any(unknown['history_dt_mask']) and not any(unknown['motion_mask'])
    assert unknown['motion_rms']==[0.,0.,0.]


def test_complete_8m_future_stations_exclude_current_and_reject_short_suffix():
    import pytest
    from tartan.research_score.data.core import resample_future_8m
    values=resample_future_8m(np.array([[0.,0.,0.],[0.,0.,1.],[4.,0.,0.],[9.,0.,0.]]))
    assert np.allclose(values[:,0],np.arange(1,81)/10.)
    assert np.allclose(values[:,1],0.) and np.allclose(values[:,2],1.)
    assert values.shape==(80,4)
    with pytest.raises(ValueError,match='incomplete_8m'):
        resample_future_8m(np.array([[0.,0.,0.],[7.9,0.,0.]]))
