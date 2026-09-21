import numpy as np

def canonical_trajectory(se2,timestamps,length_m):
    from .core import resample_fixed_arc
    values,mask=resample_fixed_arc(np.asarray(se2),length_m,80)
    return {"values":values,"valid_mask":mask,"raw_se2":np.asarray(se2),"timestamps_s":np.asarray(timestamps)}
