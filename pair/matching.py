from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

import numpy as np

from pair.trajectory import LocalSegment


@dataclass(frozen=True)
class MiningConfig:
    segment_length: float
    segment_stride: float
    region_search_radius: float
    entry_threshold: float
    exit_threshold: float
    resample_points: int
    mean_path_distance_threshold: float
    max_path_distance_threshold: float
    path_relation: str
    visualization_count: int
    random_seed: int

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "MiningConfig":
        config = cls(**values)
        positive = (
            config.segment_length,
            config.segment_stride,
            config.region_search_radius,
            config.entry_threshold,
            config.exit_threshold,
            config.mean_path_distance_threshold,
            config.max_path_distance_threshold,
        )
        if not all(np.isfinite(value) and value > 0 for value in positive):
            raise ValueError("All distance and length thresholds must be positive and finite")
        if config.resample_points < 2 or config.visualization_count < 0:
            raise ValueError("resample_points must be >= 2 and visualization_count >= 0")
        if config.path_relation not in {"similar", "different"}:
            raise ValueError("path_relation must be 'similar' or 'different'")
        return config


@dataclass(frozen=True)
class CandidatePair:
    pair_id: str
    segment_a: LocalSegment
    segment_b: LocalSegment
    center_distance: float
    spatial_overlap: float
    entry_distance: float
    exit_distance: float
    mean_path_distance: float
    max_path_distance: float
    chamfer_distance: float
    path_length_ratio: float

    def to_record(self) -> Dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "map_id": self.segment_a.map_id,
            "trajectory_a": self.segment_a.trajectory_id,
            "trajectory_b": self.segment_b.trajectory_id,
            "segment_a": self.segment_a.segment_id,
            "segment_b": self.segment_b.segment_id,
            "embodiment_a": self.segment_a.embodiment,
            "embodiment_b": self.segment_b.embodiment,
            "entry_a": self.segment_a.entry_position.tolist(),
            "entry_b": self.segment_b.entry_position.tolist(),
            "exit_a": self.segment_a.exit_position.tolist(),
            "exit_b": self.segment_b.exit_position.tolist(),
            "center_distance": self.center_distance,
            "spatial_overlap": self.spatial_overlap,
            "entry_distance": self.entry_distance,
            "exit_distance": self.exit_distance,
            "mean_path_distance": self.mean_path_distance,
            "max_path_distance": self.max_path_distance,
            "chamfer_distance": self.chamfer_distance,
            "path_length_a": self.segment_a.path_length,
            "path_length_b": self.segment_b.path_length,
            "path_length_ratio": self.path_length_ratio,
        }


def _resample_xy(points: np.ndarray, count: int) -> np.ndarray:
    xy = np.asarray(points[:, :2], dtype=np.float64)
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))
    keep = np.concatenate(([True], np.diff(arc) > 1e-12))
    xy, arc = xy[keep], arc[keep]
    if len(xy) == 1:
        return np.repeat(xy, count, axis=0)
    query = np.linspace(0.0, arc[-1], count)
    return np.column_stack([np.interp(query, arc, xy[:, axis]) for axis in range(2)])


def _path_arrays(
    segment_a: LocalSegment, segment_b: LocalSegment, count: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path_a = _resample_xy(segment_a.points, count)
    path_b = _resample_xy(segment_b.points, count)
    pairwise = np.linalg.norm(path_a[:, None, :] - path_b[None, :, :], axis=2)
    return path_a, path_b, pairwise


def is_same_local_region(
    segment_a: LocalSegment, segment_b: LocalSegment, config: MiningConfig
) -> Dict[str, object]:
    center_distance = float(
        np.linalg.norm(segment_a.center_position[:2] - segment_b.center_position[:2])
    )
    _, _, pairwise = _path_arrays(segment_a, segment_b, config.resample_points)
    spatial_overlap = float(
        0.5
        * (
            np.mean(np.min(pairwise, axis=1) <= config.region_search_radius)
            + np.mean(np.min(pairwise, axis=0) <= config.region_search_radius)
        )
    )
    return {
        "matched": center_distance <= config.region_search_radius,
        "center_distance": center_distance,
        "spatial_overlap": spatial_overlap,
    }


def compare_paths(
    segment_a: LocalSegment, segment_b: LocalSegment, config: MiningConfig
) -> Dict[str, float]:
    path_a, path_b, pairwise = _path_arrays(
        segment_a, segment_b, config.resample_points
    )
    correspondence = np.linalg.norm(path_a - path_b, axis=1)
    shorter = min(segment_a.path_length, segment_b.path_length)
    return {
        "mean_correspondence_distance": float(np.mean(correspondence)),
        "max_correspondence_distance": float(np.max(correspondence)),
        "chamfer_distance": float(
            0.5 * (np.mean(np.min(pairwise, axis=1)) + np.mean(np.min(pairwise, axis=0)))
        ),
        "path_length_a": segment_a.path_length,
        "path_length_b": segment_b.path_length,
        "path_length_ratio": (
            float(max(segment_a.path_length, segment_b.path_length) / shorter)
            if shorter > 0
            else float("inf")
        ),
    }


def _cell(position: np.ndarray, cell_size: float) -> Tuple[int, int]:
    return tuple(np.floor(position[:2] / cell_size).astype(int))


def mine_candidate_pairs(
    segments_a: List[LocalSegment],
    segments_b: List[LocalSegment],
    config: MiningConfig,
) -> List[CandidatePair]:
    grid = defaultdict(list)
    for segment in segments_b:
        grid[_cell(segment.center_position, config.region_search_radius)].append(segment)

    candidates = []
    for segment_a in segments_a:
        cell_x, cell_y = _cell(segment_a.center_position, config.region_search_radius)
        nearby = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nearby.extend(grid.get((cell_x + dx, cell_y + dy), ()))
        for segment_b in nearby:
            if (
                segment_a.map_id != segment_b.map_id
                or segment_a.embodiment == segment_b.embodiment
            ):
                continue
            if (
                np.linalg.norm(
                    segment_a.center_position[:2] - segment_b.center_position[:2]
                )
                > config.region_search_radius
            ):
                continue
            entry_distance = float(
                np.linalg.norm(segment_a.entry_position[:2] - segment_b.entry_position[:2])
            )
            exit_distance = float(
                np.linalg.norm(segment_a.exit_position[:2] - segment_b.exit_position[:2])
            )
            if (
                entry_distance >= config.entry_threshold
                or exit_distance >= config.exit_threshold
            ):
                continue
            region = is_same_local_region(segment_a, segment_b, config)
            metrics = compare_paths(segment_a, segment_b, config)
            mean_distance = metrics["mean_correspondence_distance"]
            max_distance = metrics["max_correspondence_distance"]
            if config.path_relation == "similar" and (
                mean_distance >= config.mean_path_distance_threshold
                or max_distance >= config.max_path_distance_threshold
            ):
                continue
            if config.path_relation == "different" and (
                mean_distance <= config.mean_path_distance_threshold
                or max_distance <= config.max_path_distance_threshold
            ):
                continue
            candidates.append(
                CandidatePair(
                    pair_id=f"{segment_a.segment_id}__{segment_b.segment_id}",
                    segment_a=segment_a,
                    segment_b=segment_b,
                    center_distance=region["center_distance"],
                    spatial_overlap=region["spatial_overlap"],
                    entry_distance=entry_distance,
                    exit_distance=exit_distance,
                    mean_path_distance=metrics["mean_correspondence_distance"],
                    max_path_distance=metrics["max_correspondence_distance"],
                    chamfer_distance=metrics["chamfer_distance"],
                    path_length_ratio=metrics["path_length_ratio"],
                )
            )
    return sorted(candidates, key=lambda pair: pair.pair_id)
