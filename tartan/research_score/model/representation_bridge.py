import numpy as np
class TrajectoryRepresentationBridge:
    def __init__(self,length_m=8.,points=80):self.length_m=length_m;self.points=points
    def __call__(self,xy_yaw):
        x=np.asarray(xy_yaw,float);d=np.linalg.norm(np.diff(x[:,:2],axis=0),axis=1);s=np.r_[0,np.cumsum(d)];end=min(self.length_m,s[-1]);q=np.linspace(0,end,self.points);xy=np.column_stack([np.interp(q,s,x[:,i]) for i in range(2)]);yaw=np.unwrap(x[:,2]);a=np.interp(q,s,yaw);valid=q<=s[-1]+1e-6;return np.c_[xy,np.cos(a),np.sin(a)].astype("float32"),valid
