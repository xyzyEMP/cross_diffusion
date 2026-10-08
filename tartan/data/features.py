from __future__ import annotations


import numpy as np



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
