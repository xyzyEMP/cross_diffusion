from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import hashlib

import numpy as np
from scipy.ndimage import distance_transform_edt

from tartan.config import ROBOT_LIMITS
from tartan.data.pose_utils import TrajectoryRecord, load_poses


def _pcd_binary_layout(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        points = None
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"Missing DATA header in {path}")
            if line.startswith(b"POINTS"):
                points = int(line.split()[1])
            if line.startswith(b"DATA"):
                if b"binary" not in line:
                    raise ValueError(f"Only binary PCD is supported: {path}")
                if points is None:
                    raise ValueError(f"Missing POINTS header in {path}")
                return stream.tell(), points


@dataclass
class ElevationMap:
    elevation: np.ndarray
    origin_xy: np.ndarray
    resolution: float
    environment: str

    def sample(self, global_xy: np.ndarray) -> np.ndarray:
        index = np.rint((np.asarray(global_xy) - self.origin_xy[None]) / self.resolution).astype(np.int64)
        valid = (
            (index[:, 0] >= 0)
            & (index[:, 1] >= 0)
            & (index[:, 0] < self.elevation.shape[1])
            & (index[:, 1] < self.elevation.shape[0])
        )
        result = np.full(len(index), np.nan, dtype=np.float32)
        result[valid] = self.elevation[index[valid, 1], index[valid, 0]]
        return result

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            elevation=self.elevation,
            origin_xy=self.origin_xy,
            resolution=np.float32(self.resolution),
            environment=self.environment,
        )

    @classmethod
    def load(cls, path: Path) -> "ElevationMap":
        data = np.load(path)
        return cls(data["elevation"], data["origin_xy"], float(data["resolution"]), str(data["environment"]))


def build_elevation_map(
    environment_dir: Path,
    records: Iterable[TrajectoryRecord],
    cache_dir: Path,
    resolution: float = 0.5,
    margin: float = 70.0,
    chunk_points: int = 2_000_000,
) -> ElevationMap:
    environment = environment_dir.name
    records = list(records)
    record_signature = hashlib.sha1(
        "\n".join(str(record.pose_path.resolve()) for record in records).encode("utf-8")
    ).hexdigest()[:10]
    cache_path = cache_dir / f"{environment}_surface_v3_r{resolution:.2f}_m{margin:.1f}_{record_signature}.npz"
    if cache_path.exists():
        return ElevationMap.load(cache_path)
    pcd_path = environment_dir / f"{environment}_sem.pcd"
    if not pcd_path.exists():
        raise FileNotFoundError(f"Semantic point cloud not found: {pcd_path}")
    poses = np.concatenate([load_poses(record.pose_path)[:, :3] for record in records], axis=0)
    xy_min = poses[:, :2].min(axis=0) - margin
    xy_max = poses[:, :2].max(axis=0) + margin
    width, height = np.ceil((xy_max - xy_min) / resolution).astype(np.int64) + 1
    elevation = np.full((height, width), -np.inf, dtype=np.float32)
    offset, point_count = _pcd_binary_layout(pcd_path)
    cloud = np.memmap(pcd_path, dtype=np.float32, mode="r", offset=offset, shape=(point_count, 4))
    accepted = 0
    for start in range(0, point_count, chunk_points):
        points = np.asarray(cloud[start : min(start + chunk_points, point_count), :3])
        inside = (
            (points[:, 0] >= xy_min[0]) & (points[:, 0] <= xy_max[0])
            & (points[:, 1] >= xy_min[1]) & (points[:, 1] <= xy_max[1])
        )
        points = points[inside]
        if not len(points):
            continue
        # Pose z and semantic-PCD z have platform-dependent offsets in TartanGround.
        # A pose-relative height filter would therefore remove the actual ground.
        # The highest visible surface per XY cell is a conservative traversability
        # surface: roofs, rocks, vegetation and wall tops remain potential obstacles.
        grid = np.floor((points[:, :2] - xy_min[None]) / resolution).astype(np.int64)
        flat = grid[:, 1] * width + grid[:, 0]
        np.maximum.at(elevation.ravel(), flat, points[:, 2])
        accepted += len(points)
        if start % (chunk_points * 10) == 0:
            print(f"Terrain {environment}: {min(start + chunk_points, point_count)}/{point_count} PCD points", flush=True)
    elevation[~np.isfinite(elevation)] = np.nan
    missing = ~np.isfinite(elevation)
    if (~missing).any():
        distance, nearest = distance_transform_edt(missing, return_indices=True)
        fill = missing & (distance * resolution <= 1.0)
        elevation[fill] = elevation[nearest[0, fill], nearest[1, fill]]
    result = ElevationMap(elevation, xy_min.astype(np.float64), resolution, environment)
    result.save(cache_path)
    print(f"Terrain cache built: {cache_path} ({accepted} accepted points)", flush=True)
    return result


def terrain_feasibility_metrics(
    local_trajectory: np.ndarray,
    anchor_se2: np.ndarray,
    terrain: ElevationMap,
    robot_type: str,
) -> tuple[dict[str, float], np.ndarray]:
    limits = ROBOT_LIMITS[robot_type]
    c, s = np.cos(anchor_se2[2]), np.sin(anchor_se2[2])
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    center_global = local_trajectory[:, :2] @ rotation.T + anchor_se2[None, :2]
    headings = local_trajectory[:, 2] + anchor_se2[2]
    half_l, half_w = limits.footprint_length / 2.0, limits.footprint_width / 2.0
    local_offsets = np.array([[0, 0], [half_l, half_w], [half_l, -half_w], [-half_l, half_w], [-half_l, -half_w]])
    all_xy = []
    for xy, heading in zip(center_global, headings):
        ch, sh = np.cos(heading), np.sin(heading)
        footprint_rotation = np.array([[ch, -sh], [sh, ch]])
        all_xy.append(xy[None] + local_offsets @ footprint_rotation.T)
    elevations = terrain.sample(np.concatenate(all_xy)).reshape(len(local_trajectory), len(local_offsets))
    valid = np.isfinite(elevations)
    support_fraction = valid.mean(axis=1)
    center_z = elevations[:, 0]
    roughness = np.full(len(elevations), np.nan, dtype=np.float32)
    for i, row in enumerate(elevations):
        values = row[np.isfinite(row)]
        if len(values) >= 3:
            roughness[i] = values.max() - values.min()
    ds = np.linalg.norm(np.diff(center_global, axis=0), axis=1)
    dz = np.abs(np.diff(center_z))
    valid_step = np.isfinite(dz) & (ds > 0.05)
    slope_deg = np.full_like(ds, np.nan, dtype=np.float64)
    slope_deg[valid_step] = np.degrees(np.arctan2(dz[valid_step], ds[valid_step]))
    terrain_coverage = float(np.isfinite(center_z).mean())
    support_failure = support_fraction < 0.8
    slope_violation = slope_deg > limits.max_slope_deg
    step_violation = dz > limits.max_step_height
    roughness_violation = roughness > limits.max_roughness
    components = [
        terrain_coverage,
        1.0 - float(np.nanmean(support_failure)),
        1.0 - float(np.nanmean(slope_violation)),
        1.0 - float(np.nanmean(step_violation)),
        1.0 - float(np.nanmean(roughness_violation)),
    ]
    metrics = {
        "terrain_coverage_rate": terrain_coverage,
        "support_coverage_mean": float(support_fraction.mean()),
        "support_failure_rate": float(np.mean(support_failure)),
        "max_slope_deg": float(np.nanmax(slope_deg)) if np.isfinite(slope_deg).any() else float("nan"),
        "slope_violation_rate": float(np.nanmean(slope_violation)),
        "max_step_height_m": float(np.nanmax(dz)) if np.isfinite(dz).any() else float("nan"),
        "step_violation_rate": float(np.nanmean(step_violation)),
        "mean_roughness_m": float(np.nanmean(roughness)) if np.isfinite(roughness).any() else float("nan"),
        "max_roughness_m": float(np.nanmax(roughness)) if np.isfinite(roughness).any() else float("nan"),
        "roughness_violation_rate": float(np.nanmean(roughness_violation)),
        "geometric_contact_feasibility_score": float(np.mean(components)),
    }
    return metrics, center_z
