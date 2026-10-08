from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = "canonical-v1.2"
LENGTH_CANDIDATES_M = (8.0, 10.0, 12.0, 15.0, 20.0)


def sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class RouteSpec:
    condition_mode: str = "route_set"
    source: str = "map_goal"
    max_candidates: int = 6
    points_per_candidate: int = 80
    dedup_overlap_threshold: float = 0.90

    @property
    def hash(self) -> str:
        return sha256_json(asdict(self))


def arc_length(xy: np.ndarray) -> float:
    xy = np.asarray(xy, dtype=np.float64)
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0


def resample_fixed_arc(se2: np.ndarray, length_m: float, count: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(se2, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 3 or len(p) < 2:
        raise ValueError("se2 must be Nx3 with N>=2")
    seg = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    covered = min(float(s[-1]), float(length_m))
    query = np.linspace(0.0, float(length_m), count)
    valid = query <= covered + 1e-9
    q = np.minimum(query, covered)
    xy = np.column_stack([np.interp(q, s, p[:, i]) for i in range(2)])
    yaw_u = np.unwrap(p[:, 2])
    yaw = np.interp(q, s, yaw_u)
    out = np.column_stack([xy, np.cos(yaw), np.sin(yaw)]).astype(np.float32)
    return out, valid.astype(bool)








def cache_key(sample_id: str, preprocess_version: str, route_spec_hash: str, checkpoint_hash: str, schema_version: str = SCHEMA_VERSION) -> str:
    return sha256_json({"sample_id": sample_id, "source_preprocess_version": preprocess_version, "route_spec_hash": route_spec_hash, "source_checkpoint_hash": checkpoint_hash, "schema_version": schema_version})


def overlap_ratio(a: np.ndarray, b: np.ndarray, tolerance=0.5) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if not len(a) or not len(b): return 0.0
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float((d.min(axis=1) <= tolerance).mean())


def resample_polyline(xy, count):
    xy = np.asarray(xy, float)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    arc = np.r_[0, np.cumsum(seg)]
    if arc[-1] < 1e-9:
        return np.repeat(xy[:1], count, axis=0).astype(np.float32)
    query = np.linspace(0, arc[-1], count)
    return np.column_stack([np.interp(query, arc, xy[:, i]) for i in range(2)]).astype(np.float32)


def assert_no_split_leak(rows: Sequence[Mapping[str, object]], keys=("map_id", "trajectory_id", "pair_id")) -> None:
    for key in keys:
        seen: Dict[str,str] = {}
        for r in rows:
            v=r.get(key); split=r.get("split")
            if not v: continue
            if v in seen and seen[v] != split: raise AssertionError(f"{key}={v} leaks {seen[v]}->{split}")
            seen[str(v)] = str(split)


def canonical_trajectory(se2, timestamps, length_m):
    """Canonical values/mask with raw provenance; timestamps describe raw poses."""
    values, mask = resample_fixed_arc(np.asarray(se2), length_m, 80)
    return {"values": values, "valid_mask": mask, "raw_se2": np.asarray(se2),
            "timestamps_s": np.asarray(timestamps)}


PROXY_REPRESENTATION = 'anchor-inclusive-xy-8m-80-first-duplicate'


def proxy_split(rows, seed=20260911):
    import random
    assigned = []
    for platform in ('diff','omni','anymal'):
        group = sorted((dict(r) for r in rows if r['embodiment']==platform), key=lambda r:r['trajectory_key'])
        if platform != 'anymal' and len(group)<2: raise ValueError(f'{platform}: requires at least two trajectories')
        order = list(range(len(group))); random.Random(seed).shuffle(order)
        val = set(order[:max(1,round(.2*len(group)))]) if platform!='anymal' else set()
        for i,r in enumerate(group):
            r.update(split='test' if platform=='anymal' else ('val' if i in val else 'train'), split_seed=seed, split_algorithm='per-platform-sorted-python-random-shuffle')
            assigned.append(r)
    assert_no_split_leak(assigned, keys=('trajectory_key',))
    return sorted(assigned,key=lambda r:r['trajectory_key'])


def proxy_window(row, se2, anchor, manifest_path):
    from tartan.data.pose_utils import to_local_se2, proxy_history, proxy_occupancy_geometry
    gates=row.get('gates',{})
    if not gates.get('reference_heading',gates.get('body_heading',False)) or not gates.get('occupancy_frame'):
        raise ValueError('unverified_reference_or_occupancy_frame')
    if anchor < 20: raise ValueError('insufficient_history')
    suffix=se2[anchor:];cum=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(suffix[:,:2],axis=0),axis=1))]
    hits=np.flatnonzero(cum>=8.)
    end=anchor+int(hits[0]) if len(hits) else len(se2)-1
    local=to_local_se2(se2[anchor:end+1],se2[anchor])
    # First pose at each arc station; repeated XY never creates spatial motion.
    distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(local[:,:2],axis=0),axis=1))]
    _,indices=np.unique(distance,return_index=True)
    unique=local[indices]
    if len(unique)<2:
        values=np.repeat(np.array([[0.,0.,1.,0.]],np.float32),80,axis=0);mask=np.zeros(80,bool);mask[0]=True
    else: values,mask=resample_fixed_arc(unique,8.,80)
    measured=min(float(cum[-1]),8.)
    goal=np.asarray([np.interp(measured,distance,local[:,i]) for i in range(2)],np.float32)
    occ=Path(row['occupancy_dir'])/f'occupancy_coarse5_{anchor:06d}_sparse.npy'
    if not occ.is_file():raise FileNotFoundError(occ)
    key=row['trajectory_key'];sid=f'{key}:anchor:{anchor:06d}'
    return {'profile':'proxy_ab','sample_id':sid,'episode_id':key,'trajectory_key':key,'map_id':row['map_id'],
      'trajectory_id':row['trajectory_id'],'domain':'tartanground','embodiment':row['embodiment'],'platform_id':row['platform_id'],
      'robot_radius_m':row['robot_radius_m'],'split':row['split'],'anchor_index':anchor,'branch':'moving_planning' if mask.all() else 'stop_or_short',
      'history_start_frame':anchor-20,'history_end_frame':anchor-1,'history_policy':'past20_anchor_exclusive',
      'time_source':row['time_source'],'body_heading_source':row['body_heading_source'],'reference_pose_policy':row.get('reference_pose_policy'),'reference_heading_source':row.get('reference_heading_source'),'reference_approval_record':row.get('reference_approval_record'),'frame_convention':row['frame_convention'],
      'history':proxy_history(se2,anchor,row['sample_rate_hz']),
      'current_state':{'global_se2_index':anchor,'anchor_world_se2':se2[anchor].tolist(),'se2_local':[0.,0.,0.],'frame_convention':row['frame_convention']},
      'fixed_goal':{'xy_local':goal.tolist(),'actual_distance_m':measured,'requested_distance_m':8.,'rule':'true_xy_arc_8m_clamped_then_frozen'},
      'route_set':{'map_reference':str(occ),'future_gt_dependency':False,'source':'current_coarse_occupancy_plus_fixed_goal','max_candidates':6,'occupancy_geometry':proxy_occupancy_geometry(row,anchor)},
      'trajectory':{'raw_reference':row['pose_path'],'source_start_frame':anchor,'source_end_frame':end,'raw_points':end-anchor+1,'fixed_arc_length_80':values.tolist(),'valid_mask':mask.tolist(),'arc_metric':'xy','length_m':8.,'stations_m':np.linspace(0,8,80).tolist(),'measured_arc_m':measured},
      'provenance':{'trajectory_manifest':str(manifest_path),'representation_policy':PROXY_REPRESENTATION,'time_source':row['time_source'],'body_heading_source':row['body_heading_source'],'frame_evidence':row['frame_evidence']}}


def torch_transform_trajectory(value, transform, delta=False):
    """Physical XY/direction transform; displacements have no translation."""
    import torch
    t=torch.as_tensor(transform,dtype=value.dtype,device=value.device)
    if t.ndim==2:t=t.unsqueeze(0)
    r=t[:,:2,:2]
    xy=torch.einsum('bij,bnj->bni',r,value[...,:2])
    if not delta:xy=xy+t[:,:2,2].unsqueeze(1)
    direction=torch.einsum('bij,bnj->bni',r,value[...,2:])
    return torch.cat((xy,direction),dim=-1)


def four_group_split(rows, group, seed=20260911):
    """Freeze full trajectories; tiny platforms keep nonempty validation/test."""
    import random
    spec = {'g1':('anymal',), 'g2':('omni',), 'g3':('diff',), 'g4':('diff','omni')}
    if group not in spec: raise ValueError('unknown four-group task')
    assigned=[]
    for platform in spec[group]:
        items=sorted((dict(r) for r in rows if r['embodiment']==platform),key=lambda r:r['trajectory_key'])
        if len(items)<(2 if group=='g4' else 3):raise ValueError('insufficient complete trajectories')
        order=list(range(len(items)));random.Random(seed).shuffle(order)
        nv=max(1,round((.2 if group=='g4' else .1)*len(items)))
        nt=0 if group=='g4' else max(1,round(.2*len(items)))
        val=set(order[:nv]);test=set(order[nv:nv+nt])
        for i,r in enumerate(items):
            r.update(split='val' if i in val else ('test' if i in test else 'train'),split_seed=seed,split_algorithm='four-group-per-platform-sorted-python-random-shuffle',study_group=group)
            assigned.append(r)
    if group=='g4':
        for r in rows:
            if r['embodiment']=='anymal':assigned.append({**r,'split':'test','split_seed':seed,'split_algorithm':'four-group-all-anymal-final-test','study_group':group})
    assert_no_split_leak(assigned,keys=('trajectory_key',))
    return sorted(assigned,key=lambda r:r['trajectory_key'])


def resample_future_8m(local):
    """80 future stations, excluding the separately supplied current-state token."""
    distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(local[:,:2],axis=0),axis=1))]
    if distance[-1]<8.-1e-6:raise ValueError('incomplete_8m')
    _,first=np.unique(distance,return_index=True);x=np.asarray(local)[first];d=distance[first]
    stations=np.arange(1,81,dtype=float)/10.;yaw=np.interp(stations,d,np.unwrap(x[:,2]))
    return np.column_stack((np.interp(stations,d,x[:,0]),np.interp(stations,d,x[:,1]),np.cos(yaw),np.sin(yaw))).astype(np.float32)


def four_group_window(row,se2,anchor,manifest_path):
    from tartan.data.pose_utils import to_local_se2
    window=proxy_window(row,se2,anchor,manifest_path)
    if window['trajectory']['measured_arc_m']<8.-1e-6:raise ValueError('incomplete_8m')
    end=window['trajectory']['source_end_frame']
    window['trajectory'].update(fixed_arc_length_80=resample_future_8m(to_local_se2(se2[anchor:end+1],se2[anchor])).tolist(),valid_mask=[True]*80,stations_m=(np.arange(1,81)/10.).tolist())
    window['provenance']['representation_policy']='future-xy-8m-80-stations-0.1-to-8'
    window.update(study_group=row['study_group'],experiment_profile='four_groups',branch='moving_planning')
    return window
