import numpy as np
from tartan.research_score.evaluation.metrics_navigation import aggregate,episode_metrics

def test_spl_exact():
    p=np.array([[0,0],[0,5],[10,5],[10,0]],float)
    m=episode_metrics(success=True,collision=False,shortest_path_m=10,executed_path=p,initial_goal_distance_m=10,final_goal_distance_m=0,termination_reason="success")
    assert m["spl"] == .5 and m["goal_progress"] == 1.

def test_failed_episode_spl_zero_and_denominator():
    a=episode_metrics(success=True,collision=False,shortest_path_m=1,executed_path=[[0,0],[1,0]],initial_goal_distance_m=1,final_goal_distance_m=0,termination_reason="success")
    b=episode_metrics(success=False,collision=False,shortest_path_m=float("inf"),executed_path=[[0,0]],initial_goal_distance_m=1,final_goal_distance_m=1,termination_reason="route_failure")
    z=aggregate([a,b]);assert z["sr"]==.5 and z["route_failure_rate"]==.5 and b["spl"]==0
