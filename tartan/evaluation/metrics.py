from __future__ import annotations

from typing import Dict

import numpy as np

from tartan.config import ROBOT_LIMITS
from tartan.data.pose_utils import wrap_angle


def _point_to_polyline_distance(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    starts, ends = polyline[:-1], polyline[1:]
    segment = ends - starts
    denom = np.sum(segment * segment, axis=1)
    delta = points[:, None, :] - starts[None, :, :]
    t = np.sum(delta * segment[None, :, :], axis=-1) / np.maximum(denom[None, :], 1e-9)
    projection = starts[None] + np.clip(t, 0.0, 1.0)[..., None] * segment[None]
    return np.linalg.norm(points[:, None] - projection, axis=-1).min(axis=1)


def trajectory_metrics(pred: np.ndarray, gt: np.ndarray, robot_type: str, dt: float) -> Dict[str, float]:
    limit = ROBOT_LIMITS[robot_type]
    position_error = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1)
    heading_error = np.abs(wrap_angle(pred[:, 2] - gt[:, 2]))
    delta = np.diff(np.vstack((np.zeros((1, 2)), pred[:, :2])), axis=0)
    speed = np.linalg.norm(delta, axis=1) / dt
    acceleration = np.diff(speed, prepend=speed[0]) / dt
    jerk = np.diff(acceleration, prepend=acceleration[0]) / dt
    yaw = np.unwrap(np.concatenate(([0.0], pred[:, 2])))
    yaw_rate = np.diff(yaw) / dt
    curvature = np.abs(yaw_rate) / np.maximum(speed, 0.1)
    forward = np.column_stack((np.cos(pred[:, 2]), np.sin(pred[:, 2])))
    reverse_fraction = np.mean(np.sum(delta * forward, axis=1) < -0.01)
    route = np.vstack((np.zeros((1, 2)), gt[:, :2]))
    route_deviation = _point_to_polyline_distance(pred[:, :2], route)
    goal_norm = max(float(np.linalg.norm(gt[-1, :2])), 1e-6)
    progress = float(np.dot(pred[-1, :2], gt[-1, :2]) / (goal_norm * goal_norm))
    path_length = float(np.linalg.norm(delta, axis=1).sum())
    gt_length = float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
    return {
        "ade_m": float(position_error.mean()),
        "fde_m": float(position_error[-1]),
        "position_error_1s_m": float(position_error[min(9, len(position_error) - 1)]),
        "position_error_2s_m": float(position_error[min(19, len(position_error) - 1)]),
        "position_error_4s_m": float(position_error[min(39, len(position_error) - 1)]),
        "position_error_8s_m": float(position_error[-1]),
        "heading_mae_rad": float(heading_error.mean()),
        "final_heading_error_rad": float(heading_error[-1]),
        "route_deviation_mean_m": float(route_deviation.mean()),
        "route_deviation_max_m": float(route_deviation.max()),
        "corridor_violation_rate": float(np.mean(route_deviation > limit.corridor_half_width)),
        "goal_progress_ratio": progress,
        "path_length_ratio": path_length / max(gt_length, 1e-6),
        "mean_speed_mps": float(speed.mean()),
        "max_speed_mps": float(speed.max()),
        "max_abs_acceleration_mps2": float(np.abs(acceleration).max()),
        "mean_abs_jerk_mps3": float(np.abs(jerk).mean()),
        "max_abs_yaw_rate_rps": float(np.abs(yaw_rate).max()),
        "max_curvature_inv_m": float(curvature.max()),
        "speed_violation_rate": float(np.mean(speed > limit.max_speed)),
        "acceleration_violation_rate": float(np.mean(np.abs(acceleration) > limit.max_acceleration)),
        "yaw_rate_violation_rate": float(np.mean(np.abs(yaw_rate) > limit.max_yaw_rate)),
        "curvature_violation_rate": float(np.mean(curvature > limit.max_curvature)),
        "reverse_fraction": float(reverse_fraction),
    }
