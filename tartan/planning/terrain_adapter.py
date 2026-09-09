"""Terrain-aware, weight-free trajectory selection and safety projection."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from tartan.planning.adaptation import adapt_trajectory, estimate_history_motion
from tartan.config import ROBOT_LIMITS
from tartan.evaluation.metrics import trajectory_metrics
from tartan.data.pose_utils import local_xy_to_global, wrap_angle
from tartan.data.terrain import ElevationMap, terrain_feasibility_metrics


def _profile(trajectory: np.ndarray, anchor: np.ndarray, terrain: ElevationMap, robot_type: str) -> dict:
    limit = ROBOT_LIMITS[robot_type]
    global_xy = local_xy_to_global(trajectory[:, :2], anchor)
    z = terrain.sample(global_xy)
    ds = np.linalg.norm(np.diff(global_xy, axis=0), axis=1)
    dz = np.abs(np.diff(z))
    slope = np.zeros(len(trajectory), dtype=np.float32)
    valid = np.isfinite(dz) & (ds > 0.05)
    slope[1:][valid] = np.degrees(np.arctan2(dz[valid], ds[valid]))
    step = np.zeros(len(trajectory), dtype=np.float32)
    step[1:] = np.where(np.isfinite(dz), dz, np.inf)

    half_l, half_w = limit.footprint_length / 2, limit.footprint_width / 2
    offsets = np.array([[0, 0], [half_l, half_w], [half_l, -half_w], [-half_l, half_w], [-half_l, -half_w]])
    footprint = []
    for xy, yaw in zip(global_xy, trajectory[:, 2] + anchor[2]):
        c, s = np.cos(yaw), np.sin(yaw)
        footprint.append(xy + offsets @ np.array([[c, s], [-s, c]]))
    footprint_z = terrain.sample(np.concatenate(footprint)).reshape(len(trajectory), 5)
    support = np.isfinite(footprint_z).mean(axis=1)
    roughness = np.zeros(len(trajectory), dtype=np.float32)
    for index, values in enumerate(footprint_z):
        values = values[np.isfinite(values)]
        roughness[index] = values.max() - values.min() if len(values) >= 3 else np.inf
    # Hard-stop thresholds deliberately exceed the reporting thresholds: a
    # single conservative 0.5 m PCD cell should be reported, but must not halt
    # a robot unless the geometry is persistently severe.
    danger = (
        (support < 0.2) | (slope > 1.25 * limit.max_slope_deg)
        | (step > 1.25 * limit.max_step_height) | (roughness > 1.5 * limit.max_roughness)
    )
    normalized = np.maximum.reduce((
        slope / max(limit.max_slope_deg, 1e-6),
        step / max(limit.max_step_height, 1e-6),
        roughness / max(limit.max_roughness, 1e-6),
        1.0 - support,
    ))
    normalized = np.nan_to_num(normalized, nan=2.0, posinf=3.0, neginf=0.0)
    return {"danger": danger, "risk": np.clip(normalized, 0.0, 3.0), "slope": slope, "step": step, "support": support, "roughness": roughness}


def candidate_cost(candidate: np.ndarray, gt_route: np.ndarray, anchor: np.ndarray, terrain: ElevationMap, robot_type: str, dt: float) -> float:
    motion = trajectory_metrics(candidate, gt_route, robot_type, dt)
    feasibility, _ = terrain_feasibility_metrics(candidate, anchor, terrain, robot_type)
    return float(
        motion["route_deviation_mean_m"]
        + 2.0 * motion["corridor_violation_rate"]
        + 8.0 * (1.0 - feasibility["geometric_contact_feasibility_score"])
        + 2.0 * feasibility["slope_violation_rate"]
        + 2.0 * feasibility["step_violation_rate"]
        + 2.0 * feasibility["support_failure_rate"]
    )


def _smooth_and_project(trajectory: np.ndarray, history: np.ndarray, robot_type: str, dt: float) -> np.ndarray:
    result = trajectory.copy()
    joined = np.vstack((np.zeros((1, 2)), result[:, :2]))
    if len(joined) >= 9:
        joined[:, 0] = savgol_filter(joined[:, 0], 9, 2, mode="interp")
        joined[:, 1] = savgol_filter(joined[:, 1], 9, 2, mode="interp")
        joined -= joined[0]
    result[:, :2] = joined[1:]
    result = adapt_trajectory(result, history, robot_type, dt)
    if robot_type == "diff":
        limit = ROBOT_LIMITS[robot_type]
        projected = result.copy()
        position = np.zeros(2)
        yaw = 0.0
        for index, target in enumerate(result[:, :2]):
            delta = target - position
            desired_yaw = np.arctan2(delta[1], delta[0]) if np.linalg.norm(delta) > 1e-6 else yaw
            yaw += np.clip(float(wrap_angle(desired_yaw - yaw)), -limit.max_yaw_rate * dt, limit.max_yaw_rate * dt)
            distance = min(float(np.linalg.norm(delta)), limit.max_speed * dt)
            position += distance * np.array([np.cos(yaw), np.sin(yaw)])
            projected[index] = [position[0], position[1], wrap_angle(yaw)]
        result = adapt_trajectory(projected, history, robot_type, dt)
    return result


def _terrain_retime(
    path: np.ndarray, history: np.ndarray, risk_factor: np.ndarray,
    robot_type: str, dt: float, hazard_index: int | None,
) -> np.ndarray:
    """Retimes along a fixed path with acceleration-limited terrain braking."""
    limit = ROBOT_LIMITS[robot_type]
    motion = estimate_history_motion(history, dt)
    xy_path = np.vstack((np.zeros((1, 2)), path[:, :2]))
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy_path, axis=0), axis=1))))
    keep = np.concatenate(([True], np.diff(arc) > 1e-6))
    arc_unique, xy_unique = arc[keep], xy_path[keep]
    yaw_unique = np.unwrap(np.concatenate(([0.0], path[:, 2])))[keep]
    raw_speed = np.linalg.norm(np.diff(xy_path, axis=0), axis=1) / dt
    speed = min(motion["current_speed_mps"], limit.max_speed * (1 - 1e-5))
    progress = 0.0
    stop_s = None
    if hazard_index is not None:
        hazard_s = arc[min(hazard_index + 1, len(arc) - 1)]
        # Reserve both continuous braking distance and one discrete-time step.
        stop_s = max(0.0, hazard_s - speed * speed / (2 * limit.max_acceleration) - speed * dt)
    queries = []
    for index in range(len(path)):
        target = min(raw_speed[index] * risk_factor[index], limit.max_speed * (1 - 1e-5))
        if stop_s is not None:
            remaining = max(stop_s - progress, 0.0)
            target = min(target, np.sqrt(max(0.0, 2 * limit.max_acceleration * remaining)) * 0.90)
        step = limit.max_acceleration * dt * (1 - 1e-5)
        speed += float(np.clip(target - speed, -step, step))
        distance = max(speed, 0.0) * dt
        if stop_s is not None and progress + distance > stop_s:
            distance = max(stop_s - progress, 0.0)
            speed = distance / dt
        progress += distance
        queries.append(min(progress, arc_unique[-1]))
    queries = np.asarray(queries)
    xy = np.column_stack([np.interp(queries, arc_unique, xy_unique[:, dim]) for dim in range(2)])
    desired_yaw = np.interp(queries, arc_unique, yaw_unique)
    yaw = np.empty(len(path))
    previous = 0.0
    yaw_step = limit.max_yaw_rate * dt * (1 - 1e-5)
    for index, desired in enumerate(desired_yaw):
        previous += np.clip(float(wrap_angle(desired - previous)), -yaw_step, yaw_step)
        yaw[index] = wrap_angle(previous)
    return np.column_stack((xy, yaw)).astype(np.float32)


def _enforce_discrete_limits(path: np.ndarray, history: np.ndarray, robot_type: str, dt: float) -> np.ndarray:
    """Final metric-level projection, including braking through a requested stop."""
    limit = ROBOT_LIMITS[robot_type]
    motion = estimate_history_motion(history, dt)
    previous_xy = np.zeros(2)
    previous_speed = min(motion["current_speed_mps"], limit.max_speed * (1 - 1e-5))
    previous_yaw = 0.0
    last_direction = np.array([1.0, 0.0])
    output = np.empty_like(path)
    accel_step = limit.max_acceleration * dt * (1 - 1e-3)
    for index, desired in enumerate(path):
        vector = desired[:2] - previous_xy
        norm = float(np.linalg.norm(vector))
        if norm > 1e-7:
            last_direction = vector / norm
        desired_speed = min(norm / dt, limit.max_speed * (1 - 1e-5))
        speed = float(np.clip(desired_speed, max(0.0, previous_speed - accel_step), previous_speed + accel_step))
        speed = min(speed, limit.max_speed * (1 - 1e-3))
        previous_xy = previous_xy + last_direction * speed * dt
        yaw_bound = min(limit.max_yaw_rate, limit.max_curvature * max(speed, 0.1)) * dt * (1 - 1e-3)
        previous_yaw += np.clip(float(wrap_angle(desired[2] - previous_yaw)), -yaw_bound, yaw_bound)
        output[index] = [previous_xy[0], previous_xy[1], wrap_angle(previous_yaw)]
        previous_speed = speed
    return output.astype(np.float32)


def terrain_aware_adapt(
    trajectory: np.ndarray,
    history: np.ndarray,
    anchor: np.ndarray,
    terrain: ElevationMap,
    robot_type: str,
    dt: float,
) -> tuple[np.ndarray, dict[str, float]]:
    result = _smooth_and_project(trajectory, history, robot_type, dt)
    profile = _profile(result, anchor, terrain, robot_type)

    # Look ahead 0.6 s and reduce progress before rough/steep terrain.
    future_risk = np.array([
        np.nanmax(profile["risk"][i : min(i + 6, len(result))]) for i in range(len(result))
    ])
    factor = np.clip(1.0 - 0.25 * np.clip(future_risk - 0.8, 0.0, 2.0), 0.50, 1.0)
    # Require three consecutive unsafe samples and ignore the already-observed
    # first second. Persistent initial risk is treated as map calibration noise.
    repeated = profile["danger"][:-2] & profile["danger"][1:-1] & profile["danger"][2:]
    # Stop only on a newly entered hazard. Persistent risk already present in
    # the observed neighborhood is usually a map/footprint calibration issue.
    entries = []
    for index in np.flatnonzero(repeated):
        if index >= 10 and not profile["danger"][max(0, index - 5) : index].all():
            entries.append(index)
    hazard = np.asarray(entries, dtype=np.int64)
    hazard_index = int(hazard[0]) if hazard.size else None
    result = _terrain_retime(result, history, factor, robot_type, dt, hazard_index)
    result = _enforce_discrete_limits(result, history, robot_type, dt)
    stop_index = len(result)
    if hazard.size:
        speed = np.linalg.norm(np.diff(np.vstack((np.zeros((1, 2)), result[:, :2])), axis=0), axis=1) / dt
        stopped = np.flatnonzero(speed < 1e-3)
        stop_index = int(stopped[0]) if stopped.size else len(result)
    return result.astype(np.float32), {
        "safety_stop_applied": float(stop_index < len(result)),
        "safety_stop_time_s": float(stop_index * dt) if stop_index < len(result) else float("nan"),
        "pre_stop_max_risk": float(np.nanmax(profile["risk"])),
    }
