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


def test_proxy_offline_fde_uses_last_valid_point_and_xy_only():
    import torch
    from tartan.research_score.scripts.evaluate_navigation import offline_metrics
    pred = torch.tensor([[3., 4., 1., 0.], [6., 8., 0., 1.], [999., 999., 0., 0.]])
    target = torch.zeros_like(pred)
    mask = torch.tensor([True, True, False])
    result = offline_metrics(pred, target, mask, pred, target)
    assert result['ade'] == 7.5 and result['fde'] == 10.
    assert result['valid_count'] == 2 and result['sse'] == 127.
