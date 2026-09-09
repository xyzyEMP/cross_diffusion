"""Differentiable TartanGround terrain guidance for DPM-Solver sampling."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from tartan.config import ROBOT_LIMITS
from tartan.data.pose_utils import local_xy_to_global
from tartan.data.terrain import ElevationMap


GRID_RADIUS_M = 45.0
GRID_RESOLUTION_M = 0.5


def build_local_risk_grid(terrain: ElevationMap, anchor: np.ndarray, robot_type: str) -> np.ndarray:
    coordinates = np.arange(-GRID_RADIUS_M, GRID_RADIUS_M + GRID_RESOLUTION_M, GRID_RESOLUTION_M)
    xx, yy = np.meshgrid(coordinates, coordinates)
    local_xy = np.column_stack((xx.ravel(), yy.ravel()))
    elevation = terrain.sample(local_xy_to_global(local_xy, anchor)).reshape(xx.shape)
    valid = np.isfinite(elevation)
    filled = np.where(valid, elevation, 0.0)
    # Central differences are used only where both neighboring cells exist.
    dz_dy, dz_dx = np.gradient(filled, GRID_RESOLUTION_M)
    neighbor_valid = valid.copy()
    neighbor_valid[1:-1, 1:-1] &= valid[:-2, 1:-1] & valid[2:, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:]
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    limit = ROBOT_LIMITS[robot_type]
    slope_risk = np.clip((slope - 0.65 * limit.max_slope_deg) / max(0.35 * limit.max_slope_deg, 1.0), 0.0, 3.0)
    risk = np.where(neighbor_valid, slope_risk, 2.0).astype(np.float32)
    return risk[None]


class TerrainGuidance:
    """Classifier-style reward whose gradient pushes denoising away from terrain risk."""

    def __call__(self, x_in, t_input, cond, **kwargs):
        state_normalizer = kwargs["state_normalizer"]
        inputs = kwargs["inputs"]
        batch, agents, _ = x_in.shape
        physical = state_normalizer.inverse(x_in.reshape(batch, agents, -1, 4))
        xy = physical[:, 0, 1:, :2]
        grid = torch.stack((xy[..., 0] / GRID_RADIUS_M, xy[..., 1] / GRID_RADIUS_M), dim=-1)
        sampled = F.grid_sample(
            inputs["terrain_risk_grid"], grid[:, None], mode="bilinear",
            padding_mode="border", align_corners=True,
        )[:, 0, 0]
        # Mild second-difference regularization prevents guidance from producing
        # zig-zag avoidance artifacts.
        smoothness = torch.diff(xy, n=2, dim=1).square().sum(dim=-1).mean(dim=1)
        return -sampled.mean(dim=1) - 0.015 * smoothness
