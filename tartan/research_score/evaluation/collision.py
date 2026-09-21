import numpy as np

def path_collision(path_local,blocked,grid_start,resolution_m=0.5):
    p=np.rint(np.asarray(path_local)[:,:2]/resolution_m+np.asarray(grid_start)).astype(int)
    outside=(p[:,0]<0)|(p[:,1]<0)|(p[:,0]>=blocked.shape[0])|(p[:,1]>=blocked.shape[1])
    hit=outside.copy();valid=~outside
    hit[valid]=blocked[p[valid,0],p[valid,1]]
    return bool(hit.any()),np.flatnonzero(hit).tolist()
