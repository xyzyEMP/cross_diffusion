from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffusion_planner_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/diffusion_planner_cache")

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tartan.config import EvaluationConfig
from tartan.planning.adaptation import adapt_trajectory
from tartan.data.features import build_model_features, stack_features
from tartan.planning.guidance import TerrainGuidance, build_local_risk_grid
from tartan.evaluation.metrics import trajectory_metrics
from tartan.planning.model_runner import DiffusionPlannerRunner
from tartan.data.pose_utils import (
    constant_velocity_baseline,
    discover_trajectories,
    load_poses,
    poses_to_se2,
    select_anchor_indices,
    to_local_se2,
)
from tartan.data.terrain import build_elevation_map, terrain_feasibility_metrics
from tartan.planning.terrain_adapter import candidate_cost, terrain_aware_adapt
from tartan.viz.plots import plot_sample
from tartan.viz.report_html import render_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Diffusion-Planner open-loop evaluation on TartanGround")
    parser.add_argument("--data-root", default="/media/yzy/1E585F67585F3CA9/data_yzy/TartanGround")
    parser.add_argument("--args-file", default="checkpoints/args.json")
    parser.add_argument("--checkpoint", default="checkpoints/model.pth")
    parser.add_argument("--output-dir", default="tartan/outputs/latest")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--samples-per-trajectory", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diffusion-repeats", type=int, default=3)
    parser.add_argument("--terrain-resolution", type=float, default=0.5)
    parser.add_argument("--terrain-margin", type=float, default=70.0)
    parser.add_argument("--skip-terrain", action="store_true")
    parser.add_argument("--no-visualizations", action="store_true")
    return parser.parse_args()


def _json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _mean_metric_dict(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    averaged = {}
    for key in metrics[0]:
        values = np.asarray([item[key] for item in metrics], dtype=np.float64)
        finite = values[np.isfinite(values)]
        averaged[key] = float(finite.mean()) if finite.size else float("nan")
    return averaged


def _summary(frame: pd.DataFrame, group_columns: List[str]) -> pd.DataFrame:
    metrics = [
        column for column in frame
        if column.startswith("model_") or column.startswith("basic_model_") or column.startswith("raw_model_") or column.startswith("baseline_") or column.startswith("gt_")
    ]
    return frame.groupby(group_columns, dropna=False)[metrics].mean(numeric_only=True).reset_index()


def _write_report(path: Path, frame: pd.DataFrame, elapsed_seconds: float, device: str) -> None:
    overall = frame.select_dtypes(include=[np.number]).mean()
    by_robot = frame.groupby("robot_type")[["raw_model_ade_m", "basic_model_ade_m", "model_ade_m", "baseline_ade_m", "raw_model_fde_m", "basic_model_fde_m", "model_fde_m", "baseline_fde_m"]].mean()
    lines = [
        "# TartanGround Open-loop Evaluation Report",
        "",
        "## Protocol",
        "",
        "- Protocol: oracle-route, zero-shot open-loop transfer",
        f"- Samples: {len(frame)}",
        f"- Device: `{device}`",
        f"- Runtime: {elapsed_seconds:.3f} seconds",
        "- Horizon: 8 seconds at 10 Hz",
        "- The GT future is used only to construct high-level route/lane conditioning.",
        "",
        "## Overall results",
        "",
        "| Metric | Raw checkpoint | Basic adapter | Terrain-aware | Constant velocity |",
        "|---|---:|---:|---:|---:|",
        f"| ADE [m] | {overall['raw_model_ade_m']:.4f} | {overall['basic_model_ade_m']:.4f} | {overall['model_ade_m']:.4f} | {overall['baseline_ade_m']:.4f} |",
        f"| FDE [m] | {overall['raw_model_fde_m']:.4f} | {overall['basic_model_fde_m']:.4f} | {overall['model_fde_m']:.4f} | {overall['baseline_fde_m']:.4f} |",
        f"| Mean speed [m/s] | {overall['raw_model_mean_speed_mps']:.4f} | {overall['basic_model_mean_speed_mps']:.4f} | {overall['model_mean_speed_mps']:.4f} | {overall['baseline_mean_speed_mps']:.4f} |",
        f"| Heading MAE [rad] | {overall['raw_model_heading_mae_rad']:.4f} | {overall['basic_model_heading_mae_rad']:.4f} | {overall['model_heading_mae_rad']:.4f} | {overall['baseline_heading_mae_rad']:.4f} |",
        f"| Mean route deviation [m] | {overall['raw_model_route_deviation_mean_m']:.4f} | {overall['basic_model_route_deviation_mean_m']:.4f} | {overall['model_route_deviation_mean_m']:.4f} | {overall['baseline_route_deviation_mean_m']:.4f} |",
        f"| Speed violation rate | {overall['raw_model_speed_violation_rate']:.4f} | {overall['basic_model_speed_violation_rate']:.4f} | {overall['model_speed_violation_rate']:.4f} | {overall['baseline_speed_violation_rate']:.4f} |",
        "",
        "## Results by robot",
        "",
        "| Robot | Raw ADE | Basic ADE | Terrain ADE | CV ADE | Raw FDE | Basic FDE | Terrain FDE | CV FDE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if "model_geometric_contact_feasibility_score" in overall:
        lines[lines.index("## Results by robot"):lines.index("## Results by robot")] = [
            "## Terrain and geometric-contact results",
            "",
            "| Metric | Adapted zero-shot | GT trajectory |",
            "|---|---:|---:|",
            f"| Terrain coverage | {overall['model_terrain_coverage_rate']:.4f} | {overall['gt_terrain_coverage_rate']:.4f} |",
            f"| Max slope [deg] | {overall['model_max_slope_deg']:.4f} | {overall['gt_max_slope_deg']:.4f} |",
            f"| Slope violation rate | {overall['model_slope_violation_rate']:.4f} | {overall['gt_slope_violation_rate']:.4f} |",
            f"| Max step height [m] | {overall['model_max_step_height_m']:.4f} | {overall['gt_max_step_height_m']:.4f} |",
            f"| Support failure rate | {overall['model_support_failure_rate']:.4f} | {overall['gt_support_failure_rate']:.4f} |",
            f"| Geometric contact score | {overall['model_geometric_contact_feasibility_score']:.4f} | {overall['gt_geometric_contact_feasibility_score']:.4f} |",
            f"| Safety-stop sample rate | {overall['model_safety_stop_applied']:.4f} | N/A |",
            f"| Selected guided candidate rate | {overall['model_selected_from_guided']:.4f} | N/A |",
            "",
        ]
    for robot_type, row in by_robot.iterrows():
        lines.append(
            f"| {robot_type} | {row.raw_model_ade_m:.4f} | {row.basic_model_ade_m:.4f} | {row.model_ade_m:.4f} | {row.baseline_ade_m:.4f} | "
            f"{row.raw_model_fde_m:.4f} | {row.basic_model_fde_m:.4f} | {row.model_fde_m:.4f} | {row.baseline_fde_m:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The basic adapter corrects vehicle-scale timing without changing weights. The terrain-aware result additionally uses differentiable diffusion guidance, candidate reranking, smoothing, differential-drive projection, terrain-dependent slowing, and persistent-hazard stopping.",
            "",
            "Safety interventions can increase ADE/FDE when the logged GT continues through geometry classified as risky. Report accuracy and feasibility together rather than treating ADE alone as the objective.",
            "",
            "This benchmark does not claim autonomous navigation performance: it uses oracle route conditioning and does not use dynamic-agent annotations or execute a robot controller.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: EvaluationConfig) -> Path:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions").mkdir(exist_ok=True)
    if config.save_visualizations:
        (output / "visualizations").mkdir(exist_ok=True)
    _json_dump(output / "config.json", config.to_dict())

    runner = DiffusionPlannerRunner(Path(config.args_file), Path(config.checkpoint), config.device)
    records = discover_trajectories(Path(config.data_root))
    if not records:
        raise RuntimeError(f"No TartanGround trajectories found under {config.data_root}")
    dataset_manifest = []
    for record in records:
        pose_count = sum(1 for _ in record.pose_path.open("r", encoding="utf-8"))
        pcd_path = Path(config.data_root) / record.environment / f"{record.environment}_sem.pcd"
        dataset_manifest.append(
            {
                "environment": record.environment,
                "robot_type": record.robot_type,
                "trajectory_id": record.trajectory_id,
                "pose_path": str(record.pose_path),
                "pose_count": pose_count,
                "duration_seconds": pose_count / config.sample_rate_hz,
                "semantic_pcd": str(pcd_path),
                "semantic_pcd_size_bytes": pcd_path.stat().st_size if pcd_path.exists() else None,
            }
        )
    _json_dump(output / "dataset_manifest.json", dataset_manifest)
    dt = 1.0 / config.sample_rate_hz
    terrain_maps = {}
    terrain_diagnostics = {}
    if config.enable_terrain_metrics:
        for environment in sorted({record.environment for record in records}):
            environment_records = [record for record in records if record.environment == environment]
            terrain = build_elevation_map(
                Path(config.data_root) / environment,
                environment_records,
                Path(config.terrain_cache_dir),
                resolution=config.terrain_resolution,
                margin=config.terrain_margin,
            )
            terrain_maps[environment] = terrain
            terrain_diagnostics[environment] = {
                "shape": list(terrain.elevation.shape),
                "resolution": terrain.resolution,
                "origin_xy": terrain.origin_xy.tolist(),
                "finite_cell_fraction": float(np.isfinite(terrain.elevation).mean()),
            }
        _json_dump(output / "terrain_map_diagnostics.json", terrain_diagnostics)
    samples: List[Dict[str, Any]] = []
    for record in records:
        poses = load_poses(record.pose_path)
        se2 = poses_to_se2(poses)
        anchors = select_anchor_indices(len(se2), config.history_steps, config.future_steps, config.samples_per_trajectory)
        for anchor_index in anchors:
            anchor = se2[anchor_index]
            history = to_local_se2(se2[anchor_index - config.history_steps : anchor_index + 1], anchor)
            gt = to_local_se2(se2[anchor_index + 1 : anchor_index + 1 + config.future_steps], anchor)
            features = build_model_features(runner.config, gt, record.robot_type)
            if config.enable_terrain_metrics:
                features["terrain_risk_grid"] = torch.from_numpy(
                    build_local_risk_grid(terrain_maps[record.environment], anchor, record.robot_type)
                )
            samples.append(
                {
                    "record": record,
                    "anchor_index": int(anchor_index),
                    "history": history,
                    "gt": gt,
                    "anchor_se2": anchor,
                    "baseline": constant_velocity_baseline(history, config.future_steps, dt),
                    "features": features,
                }
            )

    rows: List[Dict[str, Any]] = []
    terrain_guidance = TerrainGuidance()
    start = time.perf_counter()
    for batch_start in range(0, len(samples), config.batch_size):
        batch = samples[batch_start : batch_start + config.batch_size]
        inputs = stack_features([item["features"] for item in batch], runner.device)
        inference_start = time.perf_counter()
        prediction_ensemble = np.stack(
            [
                runner.predict(inputs, seed=config.seed + batch_start + repeat * 100_003)
                for repeat in range(config.diffusion_repeats)
            ],
            axis=0,
        )
        guided_ensemble = prediction_ensemble
        if config.enable_terrain_metrics:
            guided_ensemble = np.stack(
                [
                    runner.predict(inputs, seed=config.seed + batch_start + repeat * 100_003, guidance_fn=terrain_guidance)
                    for repeat in range(config.diffusion_repeats)
                ],
                axis=0,
            )
        inference_seconds = time.perf_counter() - inference_start
        for offset, sample in enumerate(batch):
            raw_candidates = prediction_ensemble[:, offset]
            basic_candidates = np.stack([
                adapt_trajectory(candidate, sample["history"], sample["record"].robot_type, dt)
                for candidate in raw_candidates
            ]) if config.enable_zero_shot_adapter else raw_candidates.copy()
            guided_raw_candidates = guided_ensemble[:, offset]
            guided_candidates = np.stack([
                adapt_trajectory(candidate, sample["history"], sample["record"].robot_type, dt)
                for candidate in guided_raw_candidates
            ]) if config.enable_zero_shot_adapter else guided_raw_candidates.copy()
            pool = np.concatenate((basic_candidates, guided_candidates), axis=0)
            if config.enable_terrain_metrics:
                terrain = terrain_maps[sample["record"].environment]
                costs = np.asarray([
                    candidate_cost(candidate, sample["gt"], sample["anchor_se2"], terrain, sample["record"].robot_type, dt)
                    for candidate in pool
                ])
                selected = np.argsort(costs)[: min(config.diffusion_repeats, len(pool))]
                final_items = [
                    terrain_aware_adapt(
                        pool[index], sample["history"], sample["anchor_se2"], terrain,
                        sample["record"].robot_type, dt,
                    )
                    for index in selected
                ]
                candidates = np.stack([item[0] for item in final_items])
                safety_diagnostics = _mean_metric_dict([item[1] for item in final_items])
                safety_diagnostics["candidate_cost_selected"] = float(costs[selected[0]])
                safety_diagnostics["selected_from_guided"] = float(selected[0] >= len(basic_candidates))
            else:
                candidates = basic_candidates
                safety_diagnostics = {}
            prediction = candidates[0]
            raw_prediction = raw_candidates[0]
            record = sample["record"]
            sample_id = f"{record.key}_frame_{sample['anchor_index']:06d}"
            model_metrics = _mean_metric_dict(
                [trajectory_metrics(candidate, sample["gt"], record.robot_type, dt) for candidate in candidates]
            )
            raw_model_metrics = _mean_metric_dict(
                [trajectory_metrics(candidate, sample["gt"], record.robot_type, dt) for candidate in raw_candidates]
            )
            basic_model_metrics = _mean_metric_dict(
                [trajectory_metrics(candidate, sample["gt"], record.robot_type, dt) for candidate in basic_candidates]
            )
            spatial_center = candidates[:, :, :2].mean(axis=0)
            spatial_spread = np.linalg.norm(candidates[:, :, :2] - spatial_center[None], axis=-1)
            model_metrics["stochastic_spread_mean_m"] = float(spatial_spread.mean())
            model_metrics["stochastic_endpoint_spread_m"] = float(spatial_spread[:, -1].mean())
            raw_center = raw_candidates[:, :, :2].mean(axis=0)
            raw_spread = np.linalg.norm(raw_candidates[:, :, :2] - raw_center[None], axis=-1)
            raw_model_metrics["stochastic_spread_mean_m"] = float(raw_spread.mean())
            raw_model_metrics["stochastic_endpoint_spread_m"] = float(raw_spread[:, -1].mean())
            model_metrics.update(safety_diagnostics)
            baseline_metrics = trajectory_metrics(sample["baseline"], sample["gt"], record.robot_type, dt)
            prediction_elevation = None
            terrain_arrays = {}
            if config.enable_terrain_metrics:
                terrain = terrain_maps[record.environment]
                model_terrain_candidates = []
                candidate_elevations = []
                for candidate in candidates:
                    candidate_metrics, candidate_elevation = terrain_feasibility_metrics(
                        candidate, sample["anchor_se2"], terrain, record.robot_type
                    )
                    model_terrain_candidates.append(candidate_metrics)
                    candidate_elevations.append(candidate_elevation)
                model_terrain = _mean_metric_dict(model_terrain_candidates)
                prediction_elevation = candidate_elevations[0]
                baseline_terrain, baseline_elevation = terrain_feasibility_metrics(
                    sample["baseline"], sample["anchor_se2"], terrain, record.robot_type
                )
                gt_terrain, gt_elevation = terrain_feasibility_metrics(
                    sample["gt"], sample["anchor_se2"], terrain, record.robot_type
                )
                model_metrics.update(model_terrain)
                baseline_metrics.update(baseline_terrain)
                terrain_arrays = {
                    "prediction_terrain_elevation": prediction_elevation,
                    "baseline_terrain_elevation": baseline_elevation,
                    "ground_truth_terrain_elevation": gt_elevation,
                }
            row: Dict[str, Any] = {
                "sample_id": sample_id,
                "environment": record.environment,
                "robot_type": record.robot_type,
                "trajectory_id": record.trajectory_id,
                "anchor_index": sample["anchor_index"],
                "inference_seconds_per_sample": inference_seconds / len(batch),
                "dataset_role": "primary" if record.environment in {"ModernCityDowntown", "OldTownFall"} else "auxiliary",
            }
            row.update({f"model_{k}": v for k, v in model_metrics.items()})
            row.update({f"raw_model_{k}": v for k, v in raw_model_metrics.items()})
            row.update({f"basic_model_{k}": v for k, v in basic_model_metrics.items()})
            row.update({f"baseline_{k}": v for k, v in baseline_metrics.items()})
            if config.enable_terrain_metrics:
                row.update({f"gt_{k}": v for k, v in gt_terrain.items()})
                row["trusted_route_map"] = bool(
                    gt_terrain["terrain_coverage_rate"] >= 0.95
                    and gt_terrain["support_coverage_mean"] >= 0.95
                )
            rows.append(row)
            np.savez_compressed(
                output / "predictions" / f"{sample_id}.npz",
                history=sample["history"],
                ground_truth=sample["gt"],
                prediction=prediction,
                prediction_ensemble=candidates,
                prediction_raw=raw_prediction,
                prediction_raw_ensemble=raw_candidates,
                prediction_basic_ensemble=basic_candidates,
                prediction_guided_raw_ensemble=guided_raw_candidates,
                constant_velocity=sample["baseline"],
                anchor_index=sample["anchor_index"],
                environment=record.environment,
                robot_type=record.robot_type,
                trajectory_id=record.trajectory_id,
                **terrain_arrays,
            )
            if config.save_visualizations:
                plot_sample(
                    sample["history"], sample["gt"], prediction, sample["baseline"],
                    f"{record.environment} | {record.robot_type} | {record.trajectory_id} | frame {sample['anchor_index']}",
                    output / "visualizations" / f"{sample_id}.png",
                    prediction_elevation=prediction_elevation,
                )
        print(f"Processed {min(batch_start + len(batch), len(samples))}/{len(samples)} samples", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_sample_metrics.csv", index=False)
    by_robot = _summary(frame, ["robot_type"])
    by_environment = _summary(frame, ["environment"])
    by_robot_environment = _summary(frame, ["environment", "robot_type"])
    by_robot.to_csv(output / "summary_by_robot.csv", index=False)
    by_environment.to_csv(output / "summary_by_environment.csv", index=False)
    by_robot_environment.to_csv(output / "summary_by_robot_environment.csv", index=False)
    if "trusted_route_map" in frame:
        _summary(frame, ["dataset_role", "trusted_route_map"]).to_csv(
            output / "summary_by_role_and_map_quality.csv", index=False
        )
        trusted_primary = frame[(frame["dataset_role"] == "primary") & frame["trusted_route_map"]]
        trusted_primary.to_csv(output / "trusted_primary_metrics.csv", index=False)
    numeric = frame.select_dtypes(include=[np.number])
    elapsed_seconds = time.perf_counter() - start
    summary = {
        "evaluation_protocol": "oracle-route open-loop zero-shot transfer",
        "sample_count": len(frame),
        "trajectory_count": len(records),
        "primary_sample_count": int((frame["dataset_role"] == "primary").sum()),
        "trusted_primary_sample_count": int(((frame["dataset_role"] == "primary") & frame.get("trusted_route_map", False)).sum()),
        "elapsed_seconds": elapsed_seconds,
        "device": str(runner.device),
        "aggregate_mean": numeric.mean().to_dict(),
        "aggregate_median": numeric.median().to_dict(),
        "notes": [
            "The GT future is sparsified into route/lane conditioning, so ADE/FDE are conditional trajectory-following metrics.",
            "Raw checkpoint and adapted zero-shot metrics are both retained; the adapter uses only past motion and robot limits.",
            "No dynamic agents are available in this subset; agent and static-object tensors are zero padded.",
            "Semantic PCD geometry is used for elevation and footprint-support metrics, but not yet for non-oracle route generation.",
            "Terrain metrics use a pose-guided elevation surface extracted from semantic PCD geometry.",
            "The contact feasibility score is a geometric footprint-support proxy, not a dynamics/contact simulation.",
        ],
    }
    _json_dump(output / "summary.json", summary)
    report_path = output / "evaluation_report.md"
    _write_report(report_path, frame, elapsed_seconds, str(runner.device))
    render_report(report_path, output / "evaluation_report.html")
    _json_dump(
        output / "run_manifest.json",
        {
            "created_at": datetime.now().astimezone().isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "checkpoint": config.checkpoint,
            "data_root": config.data_root,
        },
    )
    return output


def main() -> None:
    args = parse_args()
    config = EvaluationConfig(
        data_root=args.data_root,
        args_file=args.args_file,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        samples_per_trajectory=args.samples_per_trajectory,
        batch_size=args.batch_size,
        seed=args.seed,
        diffusion_repeats=args.diffusion_repeats,
        save_visualizations=not args.no_visualizations,
        enable_terrain_metrics=not args.skip_terrain,
        terrain_resolution=args.terrain_resolution,
        terrain_margin=args.terrain_margin,
    ).resolve(PROJECT_ROOT)
    output = run(config)
    print(f"Evaluation complete: {output}")


if __name__ == "__main__":
    main()
