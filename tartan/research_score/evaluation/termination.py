from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class TerminationConfig:
    goal_radius_m: float=0.5
    no_progress_window: int=10
    no_progress_epsilon_m: float=0.05
    max_steps: int=80

def termination_reason(distances,step,cfg,collision=False,terrain_failure=False,route_failure=False):
    d=np.asarray(distances,float)
    if collision:return "collision"
    if terrain_failure:return "terrain_failure"
    if route_failure:return "route_failure"
    if len(d) and d[-1]<=cfg.goal_radius_m:return "success"
    if len(d)>=cfg.no_progress_window and d[-cfg.no_progress_window]-d[-1]<cfg.no_progress_epsilon_m:return "stuck"
    if step>=cfg.max_steps:return "timeout"
    return None
