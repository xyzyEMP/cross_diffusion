"""Weight-free robot-domain adaptation for Diffusion-Planner trajectories."""

from __future__ import annotations

import numpy as np

from tartan.config import ROBOT_LIMITS
from tartan.data.pose_utils import wrap_angle


def estimate_history_motion(history: np.ndarray, dt: float, window: int = 5) -> dict[str, float]:
    """Estimate a robust current motion scale from recent TartanGround poses."""
    delta = np.diff(history[-(window + 1) :, :2], axis=0)
    speeds = np.linalg.norm(delta, axis=1) / dt
    yaw = np.unwrap(history[-(window + 1) :, 2])
    yaw_rates = np.diff(yaw) / dt
    return {
        "current_speed_mps": float(speeds[-1]),
        "target_speed_mps": float(np.median(speeds)),
        "yaw_rate_rps": float(np.median(yaw_rates)),
    }


def adapt_trajectory(raw: np.ndarray, history: np.ndarray, robot_type: str, dt: float) -> np.ndarray:
    """Retimes raw path geometry using observed robot speed and hard motion limits.

    The checkpoint weights and predicted spatial path remain unchanged.  Only its
    time parameterization and yaw sequence are projected to a continuous,
    robot-specific envelope.
    """
    limits = ROBOT_LIMITS[robot_type]
    motion = estimate_history_motion(history, dt)
    speed_limit = limits.max_speed * (1.0 - 1e-5)
    target_speed = min(motion["target_speed_mps"], speed_limit)
    speed = min(motion["current_speed_mps"], speed_limit)

    distance_query = np.empty(len(raw), dtype=np.float64)
    distance = 0.0
    for index in range(len(raw)):
        speed += np.clip(target_speed - speed, -limits.max_acceleration * dt, limits.max_acceleration * dt)
        speed = float(np.clip(speed, 0.0, speed_limit))
        distance += speed * dt
        distance_query[index] = distance

    path_xy = np.vstack((np.zeros((1, 2), dtype=np.float64), raw[:, :2]))
    segment = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(segment)))
    keep = np.concatenate(([True], np.diff(arc) > 1e-5))
    arc, path_xy = arc[keep], path_xy[keep]
    if arc[-1] < 1e-5:
        xy = np.zeros((len(raw), 2), dtype=np.float64)
    else:
        query = np.minimum(distance_query, arc[-1])
        xy = np.column_stack([np.interp(query, arc, path_xy[:, dim]) for dim in range(2)])
        beyond = distance_query > arc[-1]
        if beyond.any():
            tangent = path_xy[-1] - path_xy[-2]
            tangent /= max(float(np.linalg.norm(tangent)), 1e-6)
            xy[beyond] = path_xy[-1] + (distance_query[beyond] - arc[-1])[:, None] * tangent[None]

    # Arc-length sampling can still create scalar-speed spikes at sharp corners
    # because chord length differs from arc length. Project the discrete samples
    # once more so the metric-level speed and acceleration envelope is exact.
    projected_xy = np.empty_like(xy)
    previous_xy = np.zeros(2, dtype=np.float64)
    previous_speed = min(motion["current_speed_mps"], speed_limit)
    for index, desired_xy in enumerate(xy):
        direction = desired_xy - previous_xy
        norm = float(np.linalg.norm(direction))
        desired_speed = min(norm / dt, speed_limit)
        lower = max(0.0, previous_speed - limits.max_acceleration * dt * (1.0 - 1e-5))
        upper = min(speed_limit, previous_speed + limits.max_acceleration * dt * (1.0 - 1e-5))
        actual_speed = float(np.clip(desired_speed, lower, upper))
        if norm > 1e-8:
            previous_xy = previous_xy + direction / norm * actual_speed * dt
        projected_xy[index] = previous_xy
        previous_speed = actual_speed
    xy = projected_xy

    raw_yaw = np.unwrap(np.concatenate(([0.0], raw[:, 2])))[keep]
    desired_yaw = np.interp(np.minimum(distance_query, arc[-1]), arc, raw_yaw)
    yaw = np.empty(len(raw), dtype=np.float64)
    previous = 0.0
    for index, desired in enumerate(desired_yaw):
        change = float(wrap_angle(desired - previous))
        yaw_step_limit = limits.max_yaw_rate * dt * (1.0 - 1e-5)
        change = float(np.clip(change, -yaw_step_limit, yaw_step_limit))
        previous += change
        yaw[index] = previous
    return np.column_stack((xy, wrap_angle(yaw))).astype(np.float32)
