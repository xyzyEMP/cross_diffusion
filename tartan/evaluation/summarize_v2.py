"""Create the compact statistical summary for the town-focused benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tartan.viz.report_html import render_report


def _cluster_bootstrap(frame: pd.DataFrame, value: str, cluster: str, seed: int = 20260822, repeats: int = 5000) -> dict:
    grouped = frame.groupby(cluster)[value].mean()
    values = grouped.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(repeats, len(values)))].mean(axis=1)
    return {"estimate": float(values.mean()), "ci95_low": float(np.quantile(draws, 0.025)), "ci95_high": float(np.quantile(draws, 0.975)), "cluster_count": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="tartan/outputs/town_focused_v2")
    parser.add_argument("--data-root", default="/media/yzy/1E585F67585F3CA9/data_yzy/TartanGround")
    args = parser.parse_args()
    root = Path(args.output_root)
    open_frame = pd.read_csv(root / "open_loop/per_sample_metrics.csv")
    closed = pd.read_csv(root / "closed_loop/per_episode_metrics.csv")
    trusted = open_frame[(open_frame.dataset_role == "primary") & open_frame.trusted_route_map].copy()
    trusted["trajectory_key"] = trusted.environment + "/" + trusted.robot_type + "/" + trusted.trajectory_id
    closed["trajectory_key"] = closed.environment + "/" + closed.robot_type + "/" + closed.trajectory_id

    comparisons = {}
    for name, expression in {
        "open_ade_final_minus_cv_m": trusted.model_ade_m - trusted.baseline_ade_m,
        "open_fde_final_minus_cv_m": trusted.model_fde_m - trusted.baseline_fde_m,
        "open_ade_final_minus_basic_m": trusted.model_ade_m - trusted.basic_model_ade_m,
        "closed_success_minus_cv": closed.success - closed.cv_success,
        "closed_ade_minus_cv_m": closed.closed_loop_ade_m - closed.cv_ade_m,
        "closed_fde_minus_cv_m": closed.closed_loop_fde_m - closed.cv_fde_m,
        "closed_geometric_failure_minus_cv": closed.geometric_failure - closed.cv_geometric_failure,
    }.items():
        source = trusted if name.startswith("open_") else closed
        temporary = source[["trajectory_key"]].copy()
        temporary["difference"] = expression.to_numpy()
        comparisons[name] = _cluster_bootstrap(temporary, "difference", "trajectory_key")

    data_root = Path(args.data_root)
    dataset_size = sum(path.stat().st_size for path in data_root.rglob("*") if path.is_file())
    summary = {
        "dataset_size_bytes": dataset_size,
        "dataset_size_gib": dataset_size / 1024**3,
        "open_loop_samples": len(open_frame),
        "trusted_primary_samples": len(trusted),
        "primary_trajectories": int(trusted.trajectory_key.nunique()),
        "closed_loop_episodes": len(closed),
        "closed_loop_trajectories": int(closed.trajectory_key.nunique()),
        "cluster_bootstrap_95ci": comparisons,
    }
    (root / "statistical_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    om = trusted.mean(numeric_only=True)
    cm = closed.mean(numeric_only=True)
    ci_success = comparisons["closed_success_minus_cv"]
    ci_ade = comparisons["closed_ade_minus_cv_m"]
    ci_geo = comparisons["closed_geometric_failure_minus_cv"]
    lines = [
        "# Town-focused TartanGround Zero-shot Evaluation", "",
        "## What changed", "",
        f"- Dataset: {summary['dataset_size_gib']:.2f} GiB, with ModernCityDowntown and OldTownFall as primary environments.",
        f"- Open loop: {len(open_frame)} samples; the core result uses {len(trusted)} primary samples with at least 95% route and footprint map coverage.",
        f"- Kinematic closed loop: {len(closed)} episodes over {closed.trajectory_key.nunique()} trajectories, replanning every second after noisy execution feedback.",
        "- Every closed-loop anchor is evaluated nominally and with matched ±0.3 m / ±5° initial perturbations.",
        "- All confidence intervals below use trajectory-cluster bootstrap resampling (5,000 draws).", "",
        "## Trusted primary open-loop result", "",
        "| Metric | Raw checkpoint | Basic adapter | Terrain-aware | Constant velocity |", "|---|---:|---:|---:|---:|",
        f"| ADE [m] | {om.raw_model_ade_m:.3f} | {om.basic_model_ade_m:.3f} | {om.model_ade_m:.3f} | {om.baseline_ade_m:.3f} |",
        f"| FDE [m] | {om.raw_model_fde_m:.3f} | {om.basic_model_fde_m:.3f} | {om.model_fde_m:.3f} | {om.baseline_fde_m:.3f} |",
        f"| Route deviation [m] | {om.raw_model_route_deviation_mean_m:.3f} | {om.basic_model_route_deviation_mean_m:.3f} | {om.model_route_deviation_mean_m:.3f} | {om.baseline_route_deviation_mean_m:.3f} |",
        f"| Geometric contact score | N/A | N/A | {om.model_geometric_contact_feasibility_score:.3f} | {om.baseline_geometric_contact_feasibility_score:.3f} |", "",
        "## Static-map kinematic closed loop", "",
        "| Metric | Receding-horizon model | Constant velocity |", "|---|---:|---:|",
        f"| Strict success rate | {cm.success:.1%} | {cm.cv_success:.1%} |",
        f"| ADE [m] | {cm.closed_loop_ade_m:.3f} | {cm.cv_ade_m:.3f} |",
        f"| FDE [m] | {cm.closed_loop_fde_m:.3f} | {cm.cv_fde_m:.3f} |",
        f"| Route deviation [m] | {cm.route_deviation_mean_m:.3f} | {cm.cv_route_deviation_mean_m:.3f} |",
        f"| Geometric failure rate | {cm.geometric_failure:.2%} | {cm.cv_geometric_failure:.2%} |",
        f"| Contact score | {cm.contact_score:.3f} | {cm.cv_contact_score:.3f} |", "",
        "## Paired statistical differences", "",
        "Negative error/failure differences and positive success differences favor the model.", "",
        "| Comparison | Estimate | Trajectory-cluster 95% CI |", "|---|---:|---:|",
        f"| Closed-loop success − CV | {ci_success['estimate']:+.3f} | [{ci_success['ci95_low']:+.3f}, {ci_success['ci95_high']:+.3f}] |",
        f"| Closed-loop ADE − CV [m] | {ci_ade['estimate']:+.3f} | [{ci_ade['ci95_low']:+.3f}, {ci_ade['ci95_high']:+.3f}] |",
        f"| Geometric failure − CV | {ci_geo['estimate']:+.3f} | [{ci_geo['ci95_low']:+.3f}, {ci_geo['ci95_high']:+.3f}] |", "",
        "## Honest interpretation", "",
        "The adapted planner improves several aggregate metrics relative to constant velocity, but the strict closed-loop success rate remains low, especially for wheel platforms. The result should therefore be interpreted as a runnable zero-shot migration baseline, not as reliable cross-domain or cross-platform transfer.", "",
        "The test is stronger than one-shot open loop because actions change the next planning state and recovery perturbations are included. It still uses the logged future as an oracle route, a static map, geometric terrain proxies, and simplified kinematic execution; it is not rigid-body/contact-dynamics simulation.", "",
        "The next stage will introduce an explicit terrain-aware high-level intent representation between environment perception and platform-specific trajectory decoding, then use few-shot consistency alignment to improve transfer while keeping the task output at the trajectory level.", "",
        "## Files", "",
        "- `open_loop/evaluation_report.html`: detailed open-loop report.",
        "- `open_loop/trusted_primary_metrics.csv`: core high-coverage sample table.",
        "- `closed_loop/evaluation_report.md`: closed-loop protocol and aggregate.",
        "- `closed_loop/per_episode_metrics.csv`: all accepted closed-loop episodes.",
        "- `statistical_summary.json`: machine-readable cluster-bootstrap results.",
    ]
    report = root / "evaluation_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    render_report(report, root / "evaluation_report.html")


if __name__ == "__main__":
    main()
