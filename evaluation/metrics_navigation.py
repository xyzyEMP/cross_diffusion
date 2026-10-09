from __future__ import annotations
import numpy as np

def path_length(path):
    x=np.asarray(path,float)
    return float(np.linalg.norm(np.diff(x[:,:2],axis=0),axis=1).sum()) if len(x)>1 else 0.0

def episode_metrics(*,success,collision,shortest_path_m,executed_path,initial_goal_distance_m,final_goal_distance_m,termination_reason):
    travelled=path_length(executed_path)
    spl=(float(shortest_path_m)/max(float(shortest_path_m),travelled,1e-9)
         if bool(success) and np.isfinite(shortest_path_m) else 0.0)
    progress=(float(initial_goal_distance_m)-float(final_goal_distance_m))/max(float(initial_goal_distance_m),1e-9)
    return {"success":int(bool(success)),"collision":int(bool(collision)),"spl":spl,"path_length_m":travelled,"shortest_path_m":float(shortest_path_m),"goal_progress":float(progress),"stuck":int(termination_reason=="stuck"),"route_failure":int(termination_reason=="route_failure"),"termination_reason":termination_reason}

def aggregate(rows):
    n=len(rows)
    if not n:return {"episodes":0,"sr":0.,"cr":0.,"spl":0.,"stuck_rate":0.,"route_failure_rate":0.,"goal_progress":0.}
    mean=lambda k:float(np.mean([r[k] for r in rows]))
    return {"episodes":n,"sr":mean("success"),"cr":mean("collision"),"spl":mean("spl"),"stuck_rate":mean("stuck"),"route_failure_rate":mean("route_failure"),"goal_progress":mean("goal_progress")}
