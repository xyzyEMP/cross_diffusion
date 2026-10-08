from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, TypedDict

import torch


class ModelBatch(TypedDict):
    ego_current_state: torch.Tensor
    neighbor_agents_past: torch.Tensor
    static_objects: torch.Tensor
    lanes: torch.Tensor
    lanes_speed_limit: torch.Tensor
    lanes_has_speed_limit: torch.Tensor
    route_lanes: torch.Tensor
    route_lanes_speed_limit: torch.Tensor
    route_lanes_has_speed_limit: torch.Tensor


@dataclass(frozen=True)
class CanonicalObservation:
    scene_id: str
    goal_xy: List[float]
    history_se2: List[List[float]]
    route_candidates_xy: List[List[List[float]]]
    route_candidate_mask: List[bool]
    platform: str
    local_map: Dict[str, Any] = None
    route_source: str = "map_goal"


@dataclass(frozen=True)
class CanonicalSample:
    sample_id: str
    domain: str
    embodiment: str
    scene_id: str
    trajectory_id: str
    anchor_index: int
    timestamp: float
    observation: CanonicalObservation
    target: "CanonicalTrajectory"
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class CanonicalPair:
    pair_id: str
    platform_a_sample_id: str
    platform_b_sample_id: str
    platform_a_embodiment: str
    platform_b_embodiment: str
    common_scene_id: str
    common_goal_id: str
    T_a_to_common: List[List[float]]
    T_b_to_common: List[List[float]]
    pair_valid: bool
    pair_level: str
    pair_source: str
    supports_claims: Dict[str, bool]
    pair_quality_flags: List[str]
    provenance: Dict[str, Any]


@dataclass(frozen=True)
class CanonicalTrajectory:
    xy: List[List[float]]
    cos_yaw: List[float]
    sin_yaw: List[float]
    valid_mask: List[bool]
    timestamps_s: List[float]


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    profile: str
    dataset_version: str
    split: str
    method: str
    budget: str
    seed: int


def source_input_schema(config, normalizer_reference: Dict[str, str] = None) -> Dict[str, object]:
    fields = {
        "ego_current_state": (["B", 10], "float32", "observed/current state"),
        "neighbor_agents_past": (["B", config.agent_num, config.time_len, config.agent_state_dim], "float32", "observed history; target zero-pads absent actors"),
        "static_objects": (["B", config.static_objects_num, config.static_objects_state_dim], "float32", "observed; target zero-pads absent objects"),
        "lanes": (["B", config.lane_num, config.lane_len, config.lane_state_dim], "float32", "observed native nuPlan map lanes; target occupancy map"),
        "lanes_speed_limit": (["B", config.lane_num, 1], "float32", "map; Tartan zero-pads"),
        "lanes_has_speed_limit": (["B", config.lane_num, 1], "bool", "map; Tartan zero-pads"),
        "route_lanes": (["B", config.route_num, config.route_len, config.route_state_dim], "float32", "native nuPlan route; target observed occupancy plus fixed goal"),
        "route_lanes_speed_limit": (["B", config.route_num, 1], "float32", "route/map; Tartan zero-pads"),
        "route_lanes_has_speed_limit": (["B", config.route_num, 1], "bool", "route/map; Tartan zero-pads"),
    }
    return {
        "schema_version": "source-input-v1",
        "fields": {k: {"shape": s, "dtype": d, "provenance": p} for k, (s, d, p) in fields.items()},
        "normalization": {
            "policy": "frozen_from_checkpoint_args",
            "observation": config.observation_normalizer.to_dict(),
            "state": config.state_normalizer.to_dict(),
            "artifact": normalizer_reference,
        },
        "future_gt_as_input": False,
        "canonical_types": [asdict(RunIdentity("example", "transfer_primary", "unset", "unset", "source_regression", "na", 0))],
    }
