from __future__ import annotations
import copy,hashlib,json
from pathlib import Path
import numpy as np
from datasets.route_builder import OccupancyRouteSetBuilder,RouteSpec
from .collision import path_collision
from .metrics_navigation import aggregate,episode_metrics,path_length
from .shortest_path import astar,inflate,nearest_free,sparse_grid

ROBOT_RADIUS={"anymal":0.35,"diff":0.50}
REPLAN_STEPS=10;MAX_STEPS=80;GOAL_RADIUS_M=.75
def route_digest(route):return hashlib.sha256(route["route_candidates_xy"].tobytes()+route["route_candidate_mask"].tobytes()).hexdigest()
def build_from_record(record):
    sparse=np.load(record["route_set"]["map_reference"],allow_pickle=False);goal=np.asarray(record["fixed_goal"]["xy_local"],float)
    return OccupancyRouteSetBuilder(RouteSpec(max_candidates=6,points_per_candidate=80)).build(sparse,goal),sparse,goal

def prepare_episode(record,robot,base=None):
    route,sparse,goal=base if base is not None else build_from_record(record)
    blocked=inflate(sparse_grid(sparse),ROBOT_RADIUS[robot]/0.5);start=np.asarray(route["grid_start"])
    goal_cell=np.rint(goal/0.5+start).astype(int);s=nearest_free(blocked,start);g=nearest_free(blocked,goal_cell)
    cells,shortest=(None,float("inf")) if s is None or g is None else astar(blocked,s,g,0.5)
    return route,blocked,start,goal,cells,shortest,s,bool(blocked[tuple(start)])

def analytic_cases():
    straight=np.column_stack((np.linspace(0,10,11),np.zeros(11)))
    detour=np.array([[0,0],[0,5],[10,5],[10,0]],float)
    cases={
      "shortest_success":episode_metrics(success=True,collision=False,shortest_path_m=10,executed_path=straight,initial_goal_distance_m=10,final_goal_distance_m=0,termination_reason="success"),
      "double_detour_success":episode_metrics(success=True,collision=False,shortest_path_m=10,executed_path=detour,initial_goal_distance_m=10,final_goal_distance_m=0,termination_reason="success"),
      "collision":episode_metrics(success=False,collision=True,shortest_path_m=10,executed_path=straight[:5],initial_goal_distance_m=10,final_goal_distance_m=6,termination_reason="collision"),
      "stuck":episode_metrics(success=False,collision=False,shortest_path_m=10,executed_path=np.zeros((11,2)),initial_goal_distance_m=10,final_goal_distance_m=10,termination_reason="stuck"),
      "route_failure":episode_metrics(success=False,collision=False,shortest_path_m=10,executed_path=np.zeros((1,2)),initial_goal_distance_m=10,final_goal_distance_m=10,termination_reason="route_failure")}
    return cases

def rollout_shortest(blocked,start,goal_cell,goal_xy,oracle=False):
    current=np.asarray(start,int);executed=[np.zeros(2)];replans=0
    for step in range(0,MAX_STEPS,REPLAN_STEPS):
        cells,_=astar(blocked,tuple(current),tuple(goal_cell),.5);replans+=1
        if cells is None:return np.asarray(executed),"route_failure",replans
        n=min(len(cells)-1,MAX_STEPS-step if oracle else REPLAN_STEPS)
        if n<=0:return np.asarray(executed),"success",replans
        for c in cells[1:n+1]:executed.append((np.asarray(c)-np.asarray(start))*.5)
        current=np.asarray(cells[n])
        if np.linalg.norm(executed[-1]-goal_xy)<=GOAL_RADIUS_M:return np.asarray(executed),"success",replans
        if oracle:break
    return np.asarray(executed),"timeout",replans

def evaluate_policy(record,robot,policy,prepared=None):
    route,blocked,start,goal,cells,shortest,s,invalid_start=prepared or prepare_episode(record,robot)
    diag=route["diagnostics"];mask=route["route_candidate_mask"]
    route_failure=cells is None or not mask.any()
    if route_failure:
        # Frozen D024: every method gets exactly the same safe-stop outcome.
        path=np.zeros((1,2));reason="route_failure";collision=False;replans=1
    elif invalid_start:
        path=np.zeros((1,2));reason="invalid_map_episode";collision=False;replans=0
    else:
        shortest_path=(cells-start)*0.5
        if policy in {"shortest_path_follower","single_plan_map_reference"}:
            path,reason,replans=rollout_shortest(blocked,start,cells[-1],goal,policy=="single_plan_map_reference")
        elif policy=="stop_policy":path=np.zeros((80,2));replans=8
        elif policy=="intentional_collision_policy":
            obstacles=np.argwhere(blocked);nearest=obstacles[np.argmin(np.linalg.norm(obstacles-start,axis=1))];end=(nearest-start)*0.5;path=np.linspace([0.,0.],end,80);replans=1
        collision,_=path_collision(path,blocked,start)
        final_d=float(np.linalg.norm(path[-1]-goal));success=final_d<=0.75 and not collision
        if policy not in {"shortest_path_follower","single_plan_map_reference"}:reason="collision" if collision else ("success" if success else ("stuck" if policy=="stop_policy" else "timeout"))
    final_d=float(np.linalg.norm(path[-1]-goal));success=reason=="success"
    m=episode_metrics(success=success,collision=collision,shortest_path_m=shortest,executed_path=path,initial_goal_distance_m=float(np.linalg.norm(goal)),final_goal_distance_m=final_d,termination_reason=reason)
    included=reason!="invalid_map_episode"
    return {**m,"sample_id":record["sample_id"],"robot":robot,"policy":policy,"route_reason":diag.get("reason"),"included_in_denominator":included,"boundary_policy":"D024_safe_stop_retained" if route_failure else ("invalid_map_rejected_all_methods" if not included else "normal"),"replans":replans,"fixed_goal_x":float(goal[0]),"fixed_goal_y":float(goal[1])}

def no_future_check(record):
    before=route_digest(build_from_record(record)[0]);mut=copy.deepcopy(record);x=np.asarray(mut["trajectory"]["raw_future_xy_yaw"],float);x[:,:2]=x[::-1,:2]*-9+13;mut["trajectory"]["raw_future_xy_yaw"]=x.tolist();after=route_digest(build_from_record(mut)[0]);return {"passed":before==after,"before":before,"after":after}
