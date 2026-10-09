from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RobotLimits:
    max_speed: float
    max_acceleration: float
    max_yaw_rate: float
    max_curvature: float
    corridor_half_width: float
    footprint_length: float
    footprint_width: float
    max_slope_deg: float
    max_step_height: float
    max_roughness: float


ROBOT_LIMITS: Dict[str, RobotLimits] = {
    "omni": RobotLimits(5.0, 5.0, 2.0, 2.0, 1.0, 0.9, 0.8, 20.0, 0.15, 0.10),
    "diff": RobotLimits(5.0, 4.0, 1.5, 1.5, 0.9, 1.1, 0.7, 25.0, 0.20, 0.12),
    "anymal": RobotLimits(2.5, 4.0, 2.0, 2.5, 0.8, 0.9, 0.55, 35.0, 0.35, 0.25),
}


