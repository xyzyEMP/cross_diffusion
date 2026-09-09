"""Static-map receding-horizon evaluation with kinematic execution feedback.

This is deliberately labelled kinematic closed loop: it replans from the
executed state and evaluates terrain geometry, but it is not contact dynamics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tartan.config import ROBOT_LIMITS
from tartan.data.features import build_model_features, stack_features
from tartan.data.pose_utils import discover_trajectories, load_poses, poses_to_se2, to_local_se2, wrap_angle
from tartan.data.terrain import build_elevation_map, terrain_feasibility_metrics
from tartan.planning.adaptation import adapt_trajectory
from tartan.planning.model_runner import DiffusionPlannerRunner
from tartan.planning.terrain_adapter import candidate_cost, terrain_aware_adapt


PRIMARY = {"ModernCityDowntown", "OldTownFall"}


def _global_from_local(local: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    c, s = np.cos(anchor[2]), np.sin(anchor[2])
    rotation = np.array([[c, -s], [s, c]])
    out = np.empty_like(local, dtype=np.float64)
    out[:, :2] = local[:, :2] @ rotation.T + anchor[:2]
    out[:, 2] = wrap_angle(local[:, 2] + anchor[2])
    return out


def _pad_route(route: np.ndarray, count: int) -> np.ndarray:
    if len(route) >= count:
        return route[:count]
    return np.vstack((route, np.repeat(route[-1:], count - len(route), axis=0)))


def _distance_to_polyline(points: np.ndarray, route: np.ndarray) -> np.ndarray:
    starts, ends = route[:-1, :2], route[1:, :2]
    segment = ends - starts
    denom = np.maximum(np.sum(segment * segment, axis=1), 1e-9)
    delta = points[:, None, :2] - starts[None]
    t = np.clip(np.sum(delta * segment[None], axis=-1) / denom[None], 0.0, 1.0)
    projection = starts[None] + t[..., None] * segment[None]
    return np.linalg.norm(points[:, None, :2] - projection, axis=-1).min(axis=1)


def _execute_prefix(plan: np.ndarray, anchor: np.ndarray, steps: int, rng: np.random.Generator, robot: str) -> np.ndarray:
    """Execute a plan through a noisy first-order kinematic tracking model."""
    target = _global_from_local(plan[:steps], anchor)
    output = np.empty_like(target)
    state = anchor.astype(np.float64).copy()
    tracking_gain = 0.88 if robot == "anymal" else 0.93
    noise_scale = 0.008 if robot == "anymal" else 0.005
    for index, desired in enumerate(target):
        delta = desired[:2] - state[:2]
        state[:2] += tracking_gain * delta + rng.normal(0.0, noise_scale, 2)
        state[2] = wrap_angle(state[2] + tracking_gain * wrap_angle(desired[2] - state[2]) + rng.normal(0.0, 0.002))
        output[index] = state
    return output


def _quality(local_gt: np.ndarray, anchor: np.ndarray, terrain, robot: str) -> dict[str, float]:
    metrics, _ = terrain_feasibility_metrics(local_gt, anchor, terrain, robot)
    return {
        "route_terrain_coverage": metrics["terrain_coverage_rate"],
        "route_support_coverage": metrics["support_coverage_mean"],
        "route_gt_contact_score": metrics["geometric_contact_feasibility_score"],
    }


def _constant_velocity_metrics(gt_global: np.ndarray, anchor_index: int, terrain, robot: str, offset_m: float, yaw_offset_rad: float) -> dict[str, float]:
    dt = 0.1
    anchor = gt_global[anchor_index].copy()
    anchor[:2] += offset_m * np.array([-np.sin(anchor[2]), np.cos(anchor[2])])
    anchor[2] = wrap_angle(anchor[2] + yaw_offset_rad)
    velocity = (gt_global[anchor_index, :2] - gt_global[anchor_index - 5, :2]) / (5 * dt)
    yaw_rate = wrap_angle(gt_global[anchor_index, 2] - gt_global[anchor_index - 5, 2]) / (5 * dt)
    times = np.arange(1, 81) * dt
    path = np.column_stack((anchor[:2] + times[:, None] * velocity, wrap_angle(anchor[2] + times * yaw_rate)))
    gt = gt_global[anchor_index + 1 : anchor_index + 81]
    local_path = to_local_se2(path, anchor)
    terrain_metrics, _ = terrain_feasibility_metrics(local_path, anchor, terrain, robot)
    error = np.linalg.norm(path[:, :2] - gt[:, :2], axis=1)
    route_deviation = _distance_to_polyline(path, np.vstack((gt_global[anchor_index], gt)))
    progress_total = max(np.linalg.norm(gt[-1, :2] - gt_global[anchor_index, :2]), 1e-6)
    progress = np.dot(path[-1, :2] - gt_global[anchor_index, :2], gt[-1, :2] - gt_global[anchor_index, :2]) / progress_total**2
    hard_failure = terrain_metrics["terrain_coverage_rate"] < 0.80 or terrain_metrics["support_failure_rate"] > 0.20 or terrain_metrics["step_violation_rate"] > 0.10 or terrain_metrics["slope_violation_rate"] > 0.10
    return {
        "cv_ade_m": float(error.mean()), "cv_fde_m": float(error[-1]),
        "cv_route_deviation_mean_m": float(route_deviation.mean()),
        "cv_goal_progress_ratio": float(progress),
        "cv_success": float(error[-1] < 2.0 and progress > 0.8 and not hard_failure),
        "cv_geometric_failure": float(hard_failure),
        "cv_contact_score": terrain_metrics["geometric_contact_feasibility_score"],
    }


def _rollout(runner, terrain, gt_global, anchor_index, robot, repeats, replan_steps, rng, offset_m, yaw_offset_rad):
    dt = 0.1
    history = gt_global[anchor_index - 20 : anchor_index + 1].copy()
    initial = history[-1].copy()
    initial[:2] += offset_m * np.array([-np.sin(initial[2]), np.cos(initial[2])])
    initial[2] = wrap_angle(initial[2] + yaw_offset_rad)
    history[-1] = initial
    executed = []
    inference_seconds = 0.0
    stops = 0
    replans = 0
    for elapsed in range(0, 80, replan_steps):
        current = history[-1]
        remaining = _pad_route(gt_global[anchor_index + elapsed + 1 : anchor_index + 81], 80)
        route_local = to_local_se2(remaining, current)
        history_local = to_local_se2(history[-21:], current)
        features = build_model_features(runner.config, route_local, robot)
        inputs = stack_features([features], runner.device)
        start = time.perf_counter()
        raw = np.stack([runner.predict(inputs, int(rng.integers(0, 2**31 - 1)))[0] for _ in range(repeats)])
        inference_seconds += time.perf_counter() - start
        basic = np.stack([adapt_trajectory(item, history_local, robot, dt) for item in raw])
        costs = np.asarray([candidate_cost(item, route_local, current, terrain, robot, dt) for item in basic])
        plan, diagnostics = terrain_aware_adapt(basic[int(np.argmin(costs))], history_local, current, terrain, robot, dt)
        stops += int(diagnostics["safety_stop_applied"] > 0)
        count = min(replan_steps, 80 - elapsed)
        segment = _execute_prefix(plan, current, count, rng, robot)
        executed.extend(segment)
        history = np.vstack((history, segment))
        replans += 1
    executed = np.asarray(executed)
    gt = gt_global[anchor_index + 1 : anchor_index + 81]
    local_executed = to_local_se2(executed, initial)
    terrain_metrics, _ = terrain_feasibility_metrics(local_executed, initial, terrain, robot)
    error = np.linalg.norm(executed[:, :2] - gt[:, :2], axis=1)
    route_deviation = _distance_to_polyline(executed, np.vstack((gt_global[anchor_index], gt)))
    progress_total = max(np.linalg.norm(gt[-1, :2] - gt_global[anchor_index, :2]), 1e-6)
    progress = np.dot(executed[-1, :2] - gt_global[anchor_index, :2], gt[-1, :2] - gt_global[anchor_index, :2]) / progress_total**2
    hard_failure = (
        terrain_metrics["terrain_coverage_rate"] < 0.80
        or terrain_metrics["support_failure_rate"] > 0.20
        or terrain_metrics["step_violation_rate"] > 0.10
        or terrain_metrics["slope_violation_rate"] > 0.10
    )
    return {
        "closed_loop_ade_m": float(error.mean()),
        "closed_loop_fde_m": float(error[-1]),
        "route_deviation_mean_m": float(route_deviation.mean()),
        "route_deviation_max_m": float(route_deviation.max()),
        "goal_progress_ratio": float(progress),
        "success": float(error[-1] < 2.0 and progress > 0.8 and not hard_failure),
        "geometric_failure": float(hard_failure),
        "terrain_coverage_rate": terrain_metrics["terrain_coverage_rate"],
        "support_failure_rate": terrain_metrics["support_failure_rate"],
        "slope_violation_rate": terrain_metrics["slope_violation_rate"],
        "step_violation_rate": terrain_metrics["step_violation_rate"],
        "contact_score": terrain_metrics["geometric_contact_feasibility_score"],
        "safety_stop_replans": float(stops),
        "replans": float(replans),
        "inference_seconds": inference_seconds,
        "executed": executed,
        "gt": gt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TartanGround kinematic closed-loop evaluation")
    parser.add_argument("--data-root", default="/media/yzy/1E585F67585F3CA9/data_yzy/TartanGround")
    parser.add_argument("--output-dir", default="tartan/outputs/town_focused_v2/closed_loop")
    parser.add_argument("--checkpoint", default="checkpoints/model.pth")
    parser.add_argument("--args-file", default="checkpoints/args.json")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--episodes-per-trajectory", type=int, default=3)
    parser.add_argument("--diffusion-repeats", type=int, default=2)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--min-route-coverage", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    output = (PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction_dir = output / "rollouts"
    prediction_dir.mkdir(exist_ok=True)
    data_root = Path(args.data_root)
    records = [record for record in discover_trajectories(data_root) if record.environment in PRIMARY]
    runner = DiffusionPlannerRunner(Path(args.args_file), Path(args.checkpoint), args.device)
    maps = {}
    for environment in sorted(PRIMARY):
        env_records = [record for record in records if record.environment == environment]
        maps[environment] = build_elevation_map(data_root / environment, env_records, PROJECT_ROOT / "tartan/cache/elevation")

    rows = []
    rejected = []
    perturbations = ((0.0, 0.0, "nominal"), (0.30, math.radians(5), "left_perturbed"), (-0.30, math.radians(-5), "right_perturbed"))
    start_time = time.perf_counter()
    for record_index, record in enumerate(records):
        se2 = poses_to_se2(load_poses(record.pose_path))
        first, last = 20, len(se2) - 81
        if last < first:
            continue
        anchors = np.unique(np.round(np.linspace(first, last, args.episodes_per_trajectory)).astype(int))
        for episode_index, anchor_index in enumerate(anchors):
            anchor = se2[anchor_index]
            local_gt = to_local_se2(se2[anchor_index + 1 : anchor_index + 81], anchor)
            quality = _quality(local_gt, anchor, maps[record.environment], record.robot_type)
            if quality["route_terrain_coverage"] < args.min_route_coverage or quality["route_support_coverage"] < args.min_route_coverage:
                rejected.append({"environment": record.environment, "robot_type": record.robot_type, "trajectory_id": record.trajectory_id, "anchor_index": int(anchor_index), **quality})
                continue
            for perturb_index, (offset, yaw_offset, condition) in enumerate(perturbations):
                rng = np.random.default_rng(args.seed + record_index * 1009 + episode_index * 17 + perturb_index)
                result = _rollout(runner, maps[record.environment], se2, anchor_index, record.robot_type, args.diffusion_repeats, args.replan_steps, rng, offset, yaw_offset)
                arrays = {key: result.pop(key) for key in ("executed", "gt")}
                baseline = _constant_velocity_metrics(se2, anchor_index, maps[record.environment], record.robot_type, offset, yaw_offset)
                sample_id = f"{record.key}_frame_{anchor_index:06d}_{condition}"
                np.savez_compressed(prediction_dir / f"{sample_id}.npz", **arrays, anchor=anchor)
                rows.append({"sample_id": sample_id, "environment": record.environment, "robot_type": record.robot_type, "trajectory_id": record.trajectory_id, "anchor_index": int(anchor_index), "condition": condition, "initial_lateral_offset_m": offset, "initial_yaw_offset_rad": yaw_offset, **quality, **result, **baseline})
                print(f"closed-loop {len(rows):03d}: {sample_id} success={result['success']:.0f}", flush=True)

    frame = pd.DataFrame(rows)
    rejected_frame = pd.DataFrame(rejected)
    frame.to_csv(output / "per_episode_metrics.csv", index=False)
    if not rejected_frame.empty:
        rejected_frame.to_csv(output / "rejected_low_coverage.csv", index=False)
    metrics = [column for column in frame.select_dtypes(include=[np.number]).columns if column not in {"anchor_index"}]
    summary = {
        "protocol": "static-map receding-horizon kinematic closed loop",
        "claim_boundary": "No rigid-body/contact dynamics or live sensor simulation; oracle route retained.",
        "primary_environments": sorted(PRIMARY),
        "accepted_episodes": len(frame), "rejected_low_coverage": len(rejected_frame),
        "trajectory_count": len(records), "elapsed_seconds": time.perf_counter() - start_time,
        "aggregate_mean": {key: float(frame[key].mean()) for key in metrics},
        "aggregate_std": {key: float(frame[key].std()) for key in metrics},
        "created_at": datetime.now().astimezone().isoformat(),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "platform": platform.platform(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for columns, name in [(["environment"], "summary_by_environment.csv"), (["robot_type"], "summary_by_robot.csv"), (["environment", "robot_type"], "summary_by_environment_robot.csv"), (["condition"], "summary_by_perturbation.csv")]:
        frame.groupby(columns)[metrics].agg(["mean", "std", "count"]).to_csv(output / name)
    report = [
        "# TartanGround Kinematic Closed-loop Evaluation", "", "## Scope", "",
        "This evaluation repeatedly replans from the noisy executed state on the static semantic elevation map. It is stronger than one-shot open loop, but is not rigid-body or quadruped contact-dynamics simulation.", "",
        f"- Accepted episodes: {len(frame)}", f"- Rejected for route-map coverage: {len(rejected_frame)}", f"- Source trajectories: {len(records)}", f"- Replan interval: {args.replan_steps / 10:.1f} s", f"- Minimum GT route/footprint coverage: {args.min_route_coverage:.0%}", "",
        "## Aggregate", "", "| Metric | Mean | Std |", "|---|---:|---:|",
    ]
    for key in ("success", "cv_success", "geometric_failure", "cv_geometric_failure", "closed_loop_ade_m", "cv_ade_m", "closed_loop_fde_m", "cv_fde_m", "goal_progress_ratio", "cv_goal_progress_ratio", "route_deviation_mean_m", "cv_route_deviation_mean_m", "contact_score", "cv_contact_score", "inference_seconds"):
        report.append(f"| {key} | {frame[key].mean():.4f} | {frame[key].std():.4f} |")
    report += ["", "## Interpretation boundary", "", "The route is derived from the logged future, dynamic agents are unavailable, and terrain contact is a geometric proxy. Results support zero-shot static-map trajectory-planning claims only."]
    (output / "evaluation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Closed-loop evaluation complete: {output}")


if __name__ == "__main__":
    main()
