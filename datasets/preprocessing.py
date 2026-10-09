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


def select_length(train_lengths: Mapping[str, Sequence[float]], candidates=LENGTH_CANDIDATES_M, threshold=0.90) -> Dict[str, object]:
    if not train_lengths or any(len(v) == 0 for v in train_lengths.values()):
        raise RuntimeError("each active training branch must have non-empty train lengths")
    coverage = {str(int(c) if float(c).is_integer() else c): {k: float(np.mean(np.asarray(v) + 1e-9 >= c)) for k, v in train_lengths.items()} for c in candidates}
    eligible = [float(c) for c in candidates if all(coverage[str(int(c) if float(c).is_integer() else c)][k] >= threshold for k in train_lengths)]
    if not eligible:
        raise RuntimeError("BLOCKED: no fixed arc length meets min_train_coverage for every active branch")
    selected = max(eligible)
    payload = {"selection_source": "train_manifest_only", "candidates_m": list(map(float, candidates)), "min_train_coverage": threshold, "coverage": coverage, "selected_length_m": selected, "branches": sorted(train_lengths)}
    payload["train_statistics_sha256"] = sha256_json(payload)
    return payload


def group_split(ids: Sequence[str], seed: int = 20260911, train=0.70, val=0.15) -> Dict[str, str]:
    unique = sorted(set(ids))
    rng = np.random.default_rng(seed)
    order = [unique[i] for i in rng.permutation(len(unique))]
    n = len(order)
    nt = max(1, int(math.floor(n * train))) if n else 0
    nv = max(1, int(math.floor(n * val))) if n >= 3 else 0
    if nt + nv >= n and n >= 3:
        nt = n - nv - 1
    return {x: ("train" if i < nt else "val" if i < nt + nv else "test") for i, x in enumerate(order)}


def nested_budgets(train_episode_ids: Sequence[str], seed: int) -> Dict[str, List[str]]:
    ids = sorted(set(train_episode_ids))
    rng = np.random.default_rng(seed)
    order = [ids[i] for i in rng.permutation(len(ids))]
    def take(frac: float) -> List[str]:
        if not order:
            return []
        return sorted(order[:max(1, int(math.ceil(len(order) * frac)))])
    return {"1": take(.01), "10": take(.10), "100": sorted(order)}


def cache_key(sample_id: str, preprocess_version: str, route_spec_hash: str, checkpoint_hash: str, schema_version: str = SCHEMA_VERSION) -> str:
    return sha256_json({"sample_id": sample_id, "source_preprocess_version": preprocess_version, "route_spec_hash": route_spec_hash, "source_checkpoint_hash": checkpoint_hash, "schema_version": schema_version})


def overlap_ratio(a: np.ndarray, b: np.ndarray, tolerance=0.5) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if not len(a) or not len(b): return 0.0
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float((d.min(axis=1) <= tolerance).mean())


class RouteSetBuilder:
    """Map/goal-only builder. The API intentionally has no future trajectory argument."""
    dependency_fields = ("current_xy", "goal_xy", "map_polylines", "route_spec")

    def __init__(self, spec: RouteSpec): self.spec = spec

    def build(self, current_xy: Sequence[float], goal_xy: Sequence[float], map_polylines: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
        start, goal = np.asarray(current_xy, dtype=float), np.asarray(goal_xy, dtype=float)
        proposals: List[np.ndarray] = []
        direct = np.linspace(start, goal, self.spec.points_per_candidate)
        proposals.append(direct)
        for line in map_polylines:
            line = np.asarray(line, dtype=float)
            if line.ndim != 2 or line.shape[1] != 2 or len(line) < 2: continue
            oriented = line if np.linalg.norm(line[0]-start) <= np.linalg.norm(line[-1]-start) else line[::-1]
            raw = np.vstack([start, oriented, goal])
            proposals.append(self._resample(raw))
        kept: List[np.ndarray] = []
        for p in proposals:
            p = self._resample(p)
            if all(overlap_ratio(p, q) < self.spec.dedup_overlap_threshold or overlap_ratio(q, p) < self.spec.dedup_overlap_threshold for q in kept):
                kept.append(p)
            if len(kept) == self.spec.max_candidates: break
        arr = np.zeros((self.spec.max_candidates, self.spec.points_per_candidate, 2), np.float32)
        mask = np.zeros(self.spec.max_candidates, bool)
        for i,p in enumerate(kept): arr[i], mask[i] = p, True
        return {"route_candidates_xy": arr, "route_candidate_mask": mask, "route_source": "map_goal", "route_spec_hash": self.spec.hash}

    def _resample(self, xy: np.ndarray) -> np.ndarray:
        xy=np.asarray(xy,float); seg=np.linalg.norm(np.diff(xy,axis=0),axis=1); s=np.r_[0,np.cumsum(seg)]
        if s[-1] < 1e-9: return np.repeat(xy[:1], self.spec.points_per_candidate, axis=0).astype(np.float32)
        q=np.linspace(0,s[-1],self.spec.points_per_candidate)
        return np.column_stack([np.interp(q,s,xy[:,i]) for i in range(2)]).astype(np.float32)


def assert_no_split_leak(rows: Sequence[Mapping[str, object]], keys=("map_id", "trajectory_id", "pair_id")) -> None:
    for key in keys:
        seen: Dict[str,str] = {}
        for r in rows:
            v=r.get(key); split=r.get("split")
            if not v: continue
            if v in seen and seen[v] != split: raise AssertionError(f"{key}={v} leaks {seen[v]}->{split}")
            seen[str(v)] = str(split)


def masked_mse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    m=np.asarray(mask,bool)
    if not m.any(): return 0.0
    return float(np.mean((np.asarray(pred)[m]-np.asarray(target)[m])**2))
