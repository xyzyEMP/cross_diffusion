from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np


@dataclass(frozen=True)
class TrajectoryRecord:
    environment: str
    robot_type: str
    trajectory_id: str
    pose_path: Path

    @property
    def key(self) -> str:
        return f"{self.environment}_{self.robot_type}_{self.trajectory_id}"


def discover_trajectories(data_root: Path) -> List[TrajectoryRecord]:
    records: List[TrajectoryRecord] = []
    for pose_path in sorted(data_root.glob("*/Data_*/*/pose_lcam_front.txt")):
        version_dir = pose_path.parents[1].name
        records.append(
            TrajectoryRecord(
                environment=pose_path.parents[2].name,
                robot_type=version_dir.removeprefix("Data_"),
                trajectory_id=pose_path.parent.name,
                pose_path=pose_path,
            )
        )
    return records


def load_poses(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"Expected Nx7 poses in {path}, got {poses.shape}")
    if not np.isfinite(poses).all():
        raise ValueError(f"Non-finite pose value in {path}")
    return poses


def quaternion_to_yaw(quaternion_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion_xyzw, dtype=np.float64)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def poses_to_se2(poses: np.ndarray) -> np.ndarray:
    return np.column_stack((poses[:, 0], poses[:, 1], quaternion_to_yaw(poses[:, 3:7])))


def to_local_se2(se2: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    delta = np.asarray(se2[:, :2] - anchor[None, :2], dtype=np.float64)
    c, s = np.cos(anchor[2]), np.sin(anchor[2])
    rotation_global_to_local = np.array([[c, s], [-s, c]], dtype=np.float64)
    xy = delta @ rotation_global_to_local.T
    heading = wrap_angle(se2[:, 2] - anchor[2])
    return np.column_stack((xy, heading)).astype(np.float32)


def local_xy_to_global(local_xy: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    c, s = np.cos(anchor[2]), np.sin(anchor[2])
    rotation_local_to_global = np.array([[c, -s], [s, c]], dtype=np.float64)
    return np.asarray(local_xy, dtype=np.float64) @ rotation_local_to_global.T + anchor[None, :2]


def select_anchor_indices(
    length: int,
    history_steps: int,
    future_steps: int,
    count: int,
) -> np.ndarray:
    first = history_steps
    last = length - future_steps - 1
    if last < first:
        return np.empty(0, dtype=np.int64)
    available = np.arange(first, last + 1, dtype=np.int64)
    if count <= 0 or count >= len(available):
        return available
    positions = np.linspace(0, len(available) - 1, count)
    return np.unique(available[np.round(positions).astype(np.int64)])


def current_state_from_history(history_local: np.ndarray, dt: float) -> np.ndarray:
    """Build the checkpoint-compatible 10-D ego state in the current frame."""
    state = np.zeros(10, dtype=np.float32)
    state[2] = 1.0
    # The released online adapter deliberately zeros dynamics. Retain that behavior
    # for a clean zero-shot comparison while computing dynamics as evaluation metrics.
    return state


def constant_velocity_baseline(history_local: np.ndarray, future_steps: int, dt: float) -> np.ndarray:
    window = min(5, len(history_local) - 1)
    velocity = (history_local[-1, :2] - history_local[-1 - window, :2]) / (window * dt)
    yaw_rate = float(wrap_angle(history_local[-1:, 2] - history_local[-1 - window:-window, 2])[0]) / (window * dt)
    times = np.arange(1, future_steps + 1, dtype=np.float32) * dt
    xy = velocity[None] * times[:, None]
    yaw = wrap_angle(yaw_rate * times)
    return np.column_stack((xy, yaw)).astype(np.float32)
