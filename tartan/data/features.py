from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from tartan.config import ROBOT_LIMITS


def _resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points, count, axis=0)
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(segment)))
    if arc[-1] < 1e-6:
        return np.repeat(points[:1], count, axis=0)
    query = np.linspace(0.0, arc[-1], count)
    return np.column_stack([np.interp(query, arc, points[:, d]) for d in range(2)]).astype(np.float32)


def _route_segments(points: np.ndarray, segment_count: int, points_per_segment: int) -> np.ndarray:
    dense = _resample_polyline(points, segment_count * (points_per_segment - 1) + 1)
    result = np.zeros((segment_count, points_per_segment, 2), dtype=np.float32)
    stride = points_per_segment - 1
    for i in range(segment_count):
        result[i] = dense[i * stride : i * stride + points_per_segment]
    return result


def _polyline_features(center: np.ndarray, half_width: float) -> np.ndarray:
    vectors = np.zeros_like(center)
    vectors[:, :-1] = center[:, 1:] - center[:, :-1]
    vectors[:, -1] = vectors[:, -2]
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    tangent = vectors / np.maximum(norms, 1e-6)
    normal = np.stack((-tangent[..., 1], tangent[..., 0]), axis=-1)
    left_delta = normal * half_width
    right_delta = -normal * half_width
    traffic_unknown = np.zeros((*center.shape[:-1], 4), dtype=np.float32)
    traffic_unknown[..., 3] = 1.0
    return np.concatenate((center, vectors, left_delta, right_delta, traffic_unknown), axis=-1)


def build_model_features(config, future_gt: np.ndarray, robot_type: str) -> Dict[str, torch.Tensor]:
    """Create checkpoint-shaped vector features using an oracle reference corridor."""
    route_xy = np.vstack((np.zeros((1, 2), dtype=np.float32), future_gt[:, :2]))
    active_segments = min(4, config.route_num, config.lane_num)
    centers = _route_segments(route_xy, active_segments, config.lane_len)
    encoded = _polyline_features(centers, ROBOT_LIMITS[robot_type].corridor_half_width)

    lanes = np.zeros((config.lane_num, config.lane_len, config.lane_state_dim), dtype=np.float32)
    route_lanes = np.zeros((config.route_num, config.route_len, config.route_state_dim), dtype=np.float32)
    lanes[:active_segments] = encoded[:, :, : config.lane_state_dim]
    route_lanes[:active_segments] = encoded[:, :, : config.route_state_dim]

    arrays = {
        "ego_current_state": np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "neighbor_agents_past": np.zeros((config.agent_num, config.time_len, config.agent_state_dim), dtype=np.float32),
        "static_objects": np.zeros((config.static_objects_num, config.static_objects_state_dim), dtype=np.float32),
        "lanes": lanes,
        "lanes_speed_limit": np.zeros((config.lane_num, 1), dtype=np.float32),
        "lanes_has_speed_limit": np.zeros((config.lane_num, 1), dtype=np.bool_),
        "route_lanes": route_lanes,
        "route_lanes_speed_limit": np.zeros((config.route_num, 1), dtype=np.float32),
        "route_lanes_has_speed_limit": np.zeros((config.route_num, 1), dtype=np.bool_),
    }
    return {key: torch.from_numpy(value) for key, value in arrays.items()}


def stack_features(samples: list[Dict[str, torch.Tensor]], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: torch.stack([sample[key] for sample in samples]).to(device) for key in samples[0]}

