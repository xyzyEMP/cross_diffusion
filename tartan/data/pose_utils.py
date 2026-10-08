from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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
    # Accept either a dataset root containing environment directories or one
    # environment directory itself (the server commonly mounts the latter).
    patterns = ("*/Data_*/*/pose_lcam_front.txt", "Data_*/*/pose_lcam_front.txt")
    pose_paths = sorted({path for pattern in patterns for path in data_root.glob(pattern)})
    for pose_path in pose_paths:
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


def proxy_trajectory_evidence(record, platform):
    """Record evidence without treating camera filename or a nominal Hz as body truth."""
    import json
    directory = record.pose_path.parent
    metadata_path = directory / f"{record.trajectory_id}_metadata.json"
    meta = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    poses = load_poses(record.pose_path)
    declared_dt = meta.get('time_step')
    synchronized = meta.get('num_poses') == len(poses)
    time_ok = declared_dt is not None and synchronized and np.isfinite(float(declared_dt)) and float(declared_dt) > 0
    # Generator/extrinsic evidence must be explicitly supplied, never inferred
    # from motion direction (omnidirectional motion is an immediate counterexample).
    body = platform.get('body_heading_source')
    frame = platform.get('frame_convention')
    frame_source = platform.get('frame_evidence')
    gates = {'time': bool(time_ok), 'body_heading': bool(body), 'occupancy_frame': bool(frame and frame_source)}
    reasons = [f'unverified_{k}' for k, ok in gates.items() if not ok]
    return {'pose_count':len(poses), 'metadata_path':str(metadata_path),
            'occupancy_dir':str(directory/'coarse_occ'),
            'sample_rate_hz':1./float(declared_dt) if time_ok else None,
            'time_source':f'{metadata_path}:time_step+num_poses' if time_ok else None,
            'timestamp_path_or_null':None, 'frame_time_policy':'metadata_fixed_dt_pose_count_matched' if time_ok else 'unknown_zero_mask',
            'frame_convention':frame, 'body_heading_source':body,
            'frame_evidence':frame_source, 'gates':gates, 'gate_reasons':reasons,
            'metadata_declared':meta, 'pose_source_kind':'camera_pose_body_extrinsic_unverified' if not body else 'evidence_verified_body_pose'}


def proxy_history(se2, anchor, sample_rate_hz=None):
    if anchor < 20 or anchor >= len(se2):
        raise ValueError('insufficient_history')
    history = np.asarray(se2[anchor-20:anchor], dtype=np.float64)
    if history.shape != (20,3) or not np.isfinite(history).all():
        raise ValueError('invalid_history_pose')
    local = to_local_se2(history, se2[anchor])
    encoded = np.column_stack((local[:,:2], np.cos(local[:,2]), np.sin(local[:,2]))).astype(np.float32)
    dt = np.zeros(19,np.float32); dt_mask = np.zeros(19,bool)
    rms = np.zeros(3,np.float32); motion_mask = np.zeros(3,bool)
    if sample_rate_hz is not None:
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0: raise ValueError('invalid_time_base')
        dt[:] = 1./sample_rate_hz; dt_mask[:] = True
        displacement = np.diff(history[:,:2],axis=0)
        c,s = np.cos(history[:-1,2]),np.sin(history[:-1,2])
        velocities = np.column_stack((c*displacement[:,0]+s*displacement[:,1], -s*displacement[:,0]+c*displacement[:,1], np.diff(np.unwrap(history[:,2]))))/dt[:,None]
        rms = np.sqrt(np.mean(velocities**2,axis=0)).astype(np.float32); motion_mask[:] = True
    return {'ego_history':encoded.tolist(),'history_mask':[True]*20,'history_dt':dt.tolist(),
            'history_dt_mask':dt_mask.tolist(),'motion_rms':rms.tolist(),'motion_mask':motion_mask.tolist()}


def load_proxy_se2(row):
    """Use the approved observed reference or evidenced base extrinsic in NWU."""
    if row.get("reference_pose_policy") == "observed_lcam_front_reference" and (row.get("camera_to_body_se3") is not None or row.get("camera_to_body_se2") is not None):
        raise ValueError("observed reference must not apply a base extrinsic")
    poses = load_poses(Path(row['pose_path']))
    if row.get('camera_to_body_se3') is not None:
        from scipy.spatial.transform import Rotation
        extrinsic = np.asarray(row['camera_to_body_se3'], dtype=float)
        rotation = Rotation.from_quat(poses[:, 3:7])
        poses[:, :3] += rotation.apply(extrinsic[:3])
        poses[:, 3:7] = (rotation * Rotation.from_quat(extrinsic[3:])).as_quat()
    se2 = poses_to_se2(poses)
    extrinsic = row.get('camera_to_body_se2')
    if extrinsic is not None:
        dx,dy,dyaw = np.asarray(extrinsic,dtype=float)
        c,s=np.cos(se2[:,2]),np.sin(se2[:,2])
        se2[:,:2] += np.column_stack((c*dx-s*dy,s*dx+c*dy))
        se2[:,2] = wrap_angle(se2[:,2]+dyaw)
    if row.get('world_frame_policy') == 'ned_to_nwu':
        se2[:, 1:] *= -1
    return se2


@lru_cache(maxsize=35)
def _proxy_camera_poses(path):
    return load_poses(Path(path))


def proxy_occupancy_geometry(row, anchor):
    """Freeze the observed camera pose and known voxel quantization per anchor."""
    geometry = dict(row['occupancy_geometry'])
    geometry['camera_pose_world'] = _proxy_camera_poses(row['pose_path'])[anchor].tolist()
    geometry['world_frame_policy'] = row['world_frame_policy']
    return geometry


def proxy_sparse_to_body(sparse, geometry, body_anchor_se2, grid_size=101, resolution_m=.5):
    """Dequantize 3D camera voxels before projecting to the body-aligned BEV."""
    from scipy.spatial.transform import Rotation
    if geometry['source_frame'] != 'camera_quaternion_local':
        raise ValueError('unsupported occupancy source frame')
    x = np.asarray(sparse)
    bounds = np.asarray(geometry['bounds_m'], dtype=float)
    camera = np.asarray(geometry['camera_pose_world'], dtype=float)
    centers = bounds[[0,2,4]] + (x[:, :3] + .5) * geometry['resolution_m']
    world = Rotation.from_quat(camera[3:]).apply(centers) + camera[:3]
    if geometry['world_frame_policy'] == 'ned_to_nwu':
        world[:, 1:] *= -1
    elif geometry['world_frame_policy'] != 'identity':
        raise ValueError('unsupported world frame policy')
    anchor = np.asarray(body_anchor_se2, dtype=float)
    c,s = np.cos(anchor[2]),np.sin(anchor[2])
    xy = (world[:, :2] - anchor[:2]) @ np.array([[c,-s],[s,c]])
    ij = np.rint(xy / resolution_m).astype(np.int64) + grid_size // 2
    keep = ((ij >= 0) & (ij < grid_size)).all(axis=1)
    return np.column_stack((ij[keep], np.zeros(keep.sum(),dtype=np.int64), x[keep,3])).astype(np.int64)


def load_occupancy_record(record):
    sparse = np.load(record['route_set']['map_reference'], allow_pickle=False)
    if record.get('profile') == 'proxy_ab':
        sparse = proxy_sparse_to_body(sparse, record['route_set']['occupancy_geometry'],
                                      record['current_state']['anchor_world_se2'])
    return sparse


def read_proxy_trajectories(path):
    """Membership stays frozen; verified source evidence is a separate artifact."""
    import json
    path=Path(path)
    rows=[json.loads(x) for x in path.read_text().splitlines() if x]
    evidence=path.parent/'trajectory_evidence.json'
    if not evidence.exists():return rows
    supplied=json.loads(evidence.read_text())
    if supplied.get('policy')!='verified_sources_preserve_frozen_membership':raise ValueError('invalid evidence policy')
    updates=supplied['trajectories'];keys={r['trajectory_key'] for r in rows};approvals={}
    if not set(updates)<=keys:raise ValueError('evidence contains unknown trajectory')
    allowed={'sample_rate_hz','time_source','frame_time_policy','body_heading_source','frame_convention','frame_evidence','camera_to_body_se2','camera_to_body_se3','body_pose_policy','world_frame_policy','occupancy_geometry','reference_pose_policy','reference_heading_source','reference_approval_record'}
    for r in rows:
        update=updates.get(r['trajectory_key'],{})
        if set(update)-allowed:raise ValueError('evidence may not alter frozen identities/splits')
        if update.get('body_heading_source') and update.get('body_pose_policy') not in ('pose_is_body_se2','camera_to_body_se2','camera_to_body_se3'):raise ValueError('explicit body pose policy required')
        if update.get('body_pose_policy')=='camera_to_body_se2' and len(update.get('camera_to_body_se2',[]))!=3:raise ValueError('camera body extrinsic required')
        if update.get('body_pose_policy')=='camera_to_body_se3' and len(update.get('camera_to_body_se3',[]))!=7:raise ValueError('camera body SE3 extrinsic required')
        r.update(update)
        hz=r.get('sample_rate_hz')
        observed = r.get('reference_pose_policy') == 'observed_lcam_front_reference'
        if observed and (r.get('camera_to_body_se2') is not None or r.get('camera_to_body_se3') is not None or r.get('body_heading_source')):
            raise ValueError('observed reference cannot claim verified body or apply extrinsics')
        heading = bool(r.get('body_heading_source'))
        if observed:
            record = r.get('reference_approval_record')
            if record and record not in approvals and Path(record).is_file():approvals[record]=json.loads(Path(record).read_text())
            approval = approvals.get(record,{})
            heading = bool(r.get('reference_heading_source') and approval.get('status')=='APPROVED' and approval.get('policy')==r['reference_pose_policy'])
        r['gates']={'time':bool(r.get('time_source') and hz is not None and np.isfinite(hz) and hz>0),('reference_heading' if observed else 'body_heading'):heading,'occupancy_frame':bool(r.get('frame_convention') and r.get('frame_evidence') and r.get('occupancy_geometry') and r.get('world_frame_policy'))}
        r['gate_reasons']=[f'unverified_{k}' for k,v in r['gates'].items() if not v]
    return rows
