from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    map_id: str
    embodiment: str
    timestamps: Optional[np.ndarray]
    positions: np.ndarray
    quaternions: np.ndarray


@dataclass(frozen=True)
class LocalSegment:
    segment_id: str
    trajectory_id: str
    map_id: str
    embodiment: str
    start_frame: int
    end_frame: int
    points: np.ndarray
    entry_position: np.ndarray
    exit_position: np.ndarray
    center_position: np.ndarray
    path_length: float


def _interpolate_position(
    positions: np.ndarray, arc_lengths: np.ndarray, distance: float
) -> np.ndarray:
    right = min(
        int(np.searchsorted(arc_lengths, distance, side="right")), len(positions) - 1
    )
    left = max(0, right - 1)
    span = arc_lengths[right] - arc_lengths[left]
    if span == 0:
        return positions[left].copy()
    alpha = (distance - arc_lengths[left]) / span
    return positions[left] + alpha * (positions[right] - positions[left])


def _load_poses(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"Expected Nx7 poses in {path}, got {poses.shape}")
    if not np.isfinite(poses).all():
        raise ValueError(f"Non-finite pose value in {path}")
    return poses


def load_trajectory(path: Union[str, Path], embodiment: str) -> Trajectory:
    sequence_dir = Path(path)
    if not sequence_dir.is_dir():
        raise ValueError(f"Expected a trajectory directory, got {sequence_dir}")

    embodiment = embodiment[5:] if embodiment.startswith("Data_") else embodiment
    directory_name = sequence_dir.parent.name
    directory_embodiment = (
        directory_name[5:] if directory_name.startswith("Data_") else directory_name
    )
    if embodiment != directory_embodiment:
        raise ValueError(
            f"Embodiment {embodiment!r} does not match {sequence_dir.parent.name!r}"
        )

    poses = _load_poses(sequence_dir / "pose_lcam_front.txt")
    metadata_path = sequence_dir / f"{sequence_dir.name}_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    time_step = metadata.get("time_step")
    timestamps = None
    if time_step is not None:
        time_step = float(time_step)
        if not np.isfinite(time_step) or time_step <= 0:
            raise ValueError(f"Invalid time_step in {metadata_path}: {time_step}")
        timestamps = np.arange(len(poses), dtype=np.float64) * time_step

    return Trajectory(
        trajectory_id=sequence_dir.name,
        map_id=sequence_dir.parent.parent.name,
        embodiment=embodiment,
        timestamps=timestamps,
        positions=poses[:, :3],
        quaternions=poses[:, 3:7],
    )


def extract_local_segments(
    trajectory: Trajectory,
    segment_length: float,
    segment_stride: Optional[float] = None,
) -> list[LocalSegment]:
    """Extract full arc-length windows with linearly interpolated endpoints."""
    if not np.isfinite(segment_length) or segment_length <= 0:
        raise ValueError("segment_length must be positive and finite")
    if segment_stride is None:
        segment_stride = segment_length / 2.0
    if not np.isfinite(segment_stride) or segment_stride <= 0:
        raise ValueError("segment_stride must be positive and finite")
    if len(trajectory.positions) < 2:
        return []

    step_lengths = np.linalg.norm(np.diff(trajectory.positions, axis=0), axis=1)
    arc_lengths = np.concatenate(([0.0], np.cumsum(step_lengths)))
    total_length = float(arc_lengths[-1])
    if total_length < segment_length:
        return []

    tolerance = max(total_length, 1.0) * 1e-12
    start_distances = np.arange(
        0.0,
        total_length - segment_length + tolerance,
        segment_stride,
    )
    segments: list[LocalSegment] = []
    for segment_index, start_distance in enumerate(start_distances):
        end_distance = start_distance + segment_length
        start_frame = max(
            0, int(np.searchsorted(arc_lengths, start_distance, side="right")) - 1
        )
        end_frame = int(np.searchsorted(arc_lengths, end_distance, side="left"))
        start_point = _interpolate_position(
            trajectory.positions, arc_lengths, start_distance
        )
        end_point = _interpolate_position(
            trajectory.positions, arc_lengths, end_distance
        )
        center_point = _interpolate_position(
            trajectory.positions, arc_lengths, start_distance + segment_length / 2.0
        )
        interior = trajectory.positions[start_frame + 1 : end_frame]
        points = np.vstack((start_point, interior, end_point))
        segments.append(
            LocalSegment(
                segment_id=(
                    f"{trajectory.map_id}/{trajectory.embodiment}/"
                    f"{trajectory.trajectory_id}:{segment_index:06d}"
                ),
                trajectory_id=trajectory.trajectory_id,
                map_id=trajectory.map_id,
                embodiment=trajectory.embodiment,
                start_frame=start_frame,
                end_frame=end_frame,
                points=points,
                entry_position=points[0].copy(),
                exit_position=points[-1].copy(),
                center_position=center_point,
                path_length=float(segment_length),
            )
        )
    return segments
