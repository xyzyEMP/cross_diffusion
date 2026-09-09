from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
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


@dataclass
class EvaluationConfig:
    data_root: str
    args_file: str
    checkpoint: str
    output_dir: str
    device: str = "auto"
    seed: int = 42
    sample_rate_hz: float = 10.0
    history_steps: int = 20
    future_steps: int = 80
    samples_per_trajectory: int = 5
    batch_size: int = 1
    route_mode: str = "oracle"
    save_visualizations: bool = True
    enable_terrain_metrics: bool = True
    terrain_resolution: float = 0.5
    terrain_margin: float = 70.0
    terrain_cache_dir: str = "tartan/cache/elevation"
    diffusion_repeats: int = 3
    enable_zero_shot_adapter: bool = True

    def to_dict(self) -> dict:
        result = asdict(self)
        result["robot_limits"] = {k: asdict(v) for k, v in ROBOT_LIMITS.items()}
        return result

    def resolve(self, project_root: Path) -> "EvaluationConfig":
        for name in ("data_root", "args_file", "checkpoint", "output_dir", "terrain_cache_dir"):
            path = Path(getattr(self, name)).expanduser()
            if not path.is_absolute():
                path = project_root / path
            setattr(self, name, str(path.resolve()))
        return self
