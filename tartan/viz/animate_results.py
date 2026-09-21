"""Create progressive trajectory animations from saved evaluation NPZ files."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import shutil
import math

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
import numpy as np

from tartan.data.pose_utils import load_poses, poses_to_se2, to_local_se2
from tartan.data.semantics import GROUP_COLORS, GROUP_NAMES, UNKNOWN, build_semantic_surface
from tartan.data.terrain import ElevationMap


COLORS = {"gt": "#22c55e", "model": "#ef4444", "cv": "#3b82f6", "history": "#64748b"}
SEMANTIC_CMAP = ListedColormap(GROUP_COLORS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("tartan/outputs/final/predictions"))
    parser.add_argument("--output-dir", type=Path, default=Path("tartan/outputs/final/animations"))
    parser.add_argument("--data-root", type=Path, default=Path("/media/yzy/1E585F67585F3CA9/data_yzy/TartanGround"))
    parser.add_argument("--terrain-cache", type=Path, default=Path("tartan/cache/elevation"))
    parser.add_argument("--environments", nargs="+", default=["OldTownFall", "ModernCityDowntown"])
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--batch-index", type=int, default=0, help="zero-based case batch; changes deterministic per-group selection")
    return parser.parse_args()


def load_samples(args: argparse.Namespace) -> list[dict]:
    candidates = [
        path for path in args.input_dir.glob("*.npz")
        if any(path.name.startswith(environment + "_") for environment in args.environments)
    ]
    if len(candidates) < args.count:
        raise RuntimeError(f"Only {len(candidates)} matching samples found, requested {args.count}")
    rng = random.Random(args.seed)
    grouped: dict[tuple[str, str], list[Path]] = {}
    for path in sorted(candidates):
        parts = path.name.split("_")
        environment = next((name for name in args.environments if path.name.startswith(name + "_")), "unknown")
        robot = next((name for name in ("anymal", "diff", "omni") if f"_{name}_" in path.name), "unknown")
        grouped.setdefault((environment, robot), []).append(path)
    selected = []
    for key in sorted(grouped):
        choices = grouped[key]
        selected.append(choices[(args.batch_index * 7 + rng.randrange(len(choices))) % len(choices)])
        if len(selected) == args.count:
            break
    remaining = [path for path in candidates if path not in selected]
    if len(selected) < args.count:
        selected.extend(rng.sample(sorted(remaining), args.count - len(selected)))
    samples = []
    for path in selected:
        with np.load(path) as data:
            sample = {key: data[key].copy() for key in data.files}
        sample["path"] = path
        samples.append(sample)
    return samples


def axis_limits(sample: dict) -> tuple[float, float, float, float]:
    xy = np.concatenate(
        [sample["history"][:, :2], sample["ground_truth"][:, :2], sample["prediction"][:, :2], sample["constant_velocity"][:, :2]],
        axis=0,
    )
    minimum, maximum = np.nanmin(xy, axis=0), np.nanmax(xy, axis=0)
    center = (minimum + maximum) / 2
    radius = max(float(np.max(maximum - minimum)) * 0.58, 4.0)
    return center[0] - radius, center[0] + radius, center[1] - radius, center[1] + radius


def environment_points(sample: dict, args: argparse.Namespace, limits: tuple[float, float, float, float]):
    environment = sample["environment"].item()
    robot = sample["robot_type"].item()
    trajectory = sample["trajectory_id"].item()
    pose_path = args.data_root / environment / f"Data_{robot}" / trajectory / "pose_lcam_front.txt"
    anchor = poses_to_se2(load_poses(pose_path))[[int(sample["anchor_index"])]]
    cache_candidates = list(args.terrain_cache.glob(f"{environment}_surface_v3_*.npz"))
    if not cache_candidates:
        cache_candidates = list(args.terrain_cache.glob(f"{environment}_surface_v2_*.npz"))
    cache_path = max(cache_candidates, key=lambda path: path.stat().st_mtime)
    terrain = ElevationMap.load(cache_path)
    elevation, origin, resolution = terrain.elevation, terrain.origin_xy, terrain.resolution
    semantic = build_semantic_surface(args.data_root, terrain, args.terrain_cache)
    yy, xx = np.indices(elevation.shape)
    global_xy = np.column_stack((origin[0] + xx.ravel() * resolution, origin[1] + yy.ravel() * resolution))
    local_xy = to_local_se2(
        np.column_stack((global_xy, np.full(len(global_xy), anchor[0, 2]))), anchor[0]
    )[:, :2]
    z = elevation.ravel()
    labels = semantic.ravel()
    x0, x1, y0, y1 = limits
    visible = (
        np.isfinite(z) & (labels != UNKNOWN) & (local_xy[:, 0] >= x0) & (local_xy[:, 0] <= x1)
        & (local_xy[:, 1] >= y0) & (local_xy[:, 1] <= y1)
    )
    # A separate red overlay marks steep local elevation gradients.
    filled = np.where(np.isfinite(elevation), elevation, 0.0)
    gy, gx = np.gradient(filled, resolution)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).ravel()
    risk = visible & (slope > 25.0)
    return local_xy[visible], labels[visible], local_xy[risk]


def main() -> None:
    args = parse_args()
    samples = load_samples(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = max(1, int(math.floor(math.sqrt(len(samples)))))
    columns = int(np.ceil(len(samples) / rows))
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.5 * rows), squeeze=False)
    axes_flat = axes.ravel()
    artists = []

    for axis, sample in zip(axes_flat, samples):
        history = sample["history"]
        gt = sample["ground_truth"]
        pred = sample["prediction"]
        cv = sample["constant_velocity"]
        limits = axis_limits(sample)
        terrain_xy, terrain_labels, risk_xy = environment_points(sample, args, limits)
        axis.scatter(
            terrain_xy[:, 0], terrain_xy[:, 1], c=terrain_labels, cmap=SEMANTIC_CMAP,
            vmin=-0.5, vmax=len(GROUP_NAMES) - 0.5, s=18,
            marker="s", linewidths=0, alpha=0.72, zorder=0, label="semantic surface",
        )
        if len(risk_xy):
            axis.scatter(risk_xy[:, 0], risk_xy[:, 1], color="#ef4444", s=10, marker="s", alpha=0.22, linewidths=0, zorder=1, label="steep-risk overlay")
        axis.plot(history[:, 0], history[:, 1], color=COLORS["history"], lw=2.5, zorder=3, label="2 s history")
        axis.scatter(history[-1, 0], history[-1, 1], marker="*", s=130, color="#111827", zorder=5, label="current pose")
        gt_line, = axis.plot([], [], color=COLORS["gt"], lw=3, label="GT future")
        raw_line, = axis.plot([], [], color="#a855f7", lw=1.5, ls=":", alpha=0.8, zorder=7, label="raw checkpoint")
        basic_line, = axis.plot([], [], color="#f59e0b", lw=2, ls="--", alpha=0.9, zorder=8, label="basic adapter")
        model_line, = axis.plot([], [], color=COLORS["model"], lw=1.5, zorder=10, label="terrain-aware")
        cv_line, = axis.plot([], [], color=COLORS["cv"], lw=2, ls="--", label="constant velocity")
        gt_head = axis.scatter([], [], s=55, color=COLORS["gt"], zorder=6)
        model_head = axis.scatter([], [], s=55, color=COLORS["model"], linewidth=0, zorder=12)
        status = axis.text(
            0.02, 0.98, "", transform=axis.transAxes, va="top", fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#d1d5db"},
        )
        axis.set(xlabel="local x [m]", ylabel="local y [m]", aspect="equal")
        axis.set_xlim(*limits[:2])
        axis.set_ylim(*limits[2:])
        axis.grid(alpha=0.25)
        axis.set_title(f'{sample["environment"].item()} | {sample["robot_type"].item()} | frame {sample["anchor_index"].item()}')
        axis.legend(loc="lower left", fontsize=8)
        artists.append((gt_line, raw_line, basic_line, model_line, cv_line, gt_head, model_head, status))

    for axis in axes_flat[len(samples):]:
        axis.set_visible(False)

    key = "ground gray | sidewalk beige | vegetation green | building brown | steps orange | vehicle blue | obstacle red"
    fig.suptitle(f"TartanGround zero-shot trajectory rollout (8 s)\nSemantic colors: {key}", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    def update(frame: int):
        rendered = []
        end = min(frame + 1, 80)
        for sample, group in zip(samples, artists):
            gt_line, raw_line, basic_line, model_line, cv_line, gt_head, model_head, status = group
            gt, pred, cv = sample["ground_truth"], sample["prediction"], sample["constant_velocity"]
            raw = sample.get("prediction_raw", pred)
            basic = sample.get("prediction_basic_ensemble", pred[None])[0]
            def with_origin(values):
                return np.vstack((np.zeros((1, values.shape[1]), dtype=values.dtype), values[:end]))
            gt_draw, raw_draw, basic_draw, pred_draw, cv_draw = map(with_origin, (gt, raw, basic, pred, cv))
            gt_line.set_data(gt_draw[:, 0], gt_draw[:, 1])
            raw_line.set_data(raw_draw[:, 0], raw_draw[:, 1])
            basic_line.set_data(basic_draw[:, 0], basic_draw[:, 1])
            model_line.set_data(pred_draw[:, 0], pred_draw[:, 1])
            cv_line.set_data(cv_draw[:, 0], cv_draw[:, 1])
            gt_head.set_offsets(gt[end - 1, :2][None])
            model_head.set_offsets(pred[end - 1, :2][None])
            error = np.linalg.norm(pred[end - 1, :2] - gt[end - 1, :2])
            elevation = sample["prediction_terrain_elevation"][end - 1]
            terrain = f"{elevation:.2f} m" if np.isfinite(elevation) else "unobserved"
            status.set_text(f"t = {end / 10:.1f} s\ncurrent error = {error:.2f} m\nmodel terrain z = {terrain}")
            rendered.extend(group)
        return rendered

    animation = FuncAnimation(fig, update, frames=80, interval=100, blit=False)
    stem = f"open_loop_batch_{args.batch_index + 1:02d}"
    mp4_path = args.output_dir / f"{stem}.mp4"
    gif_path = args.output_dir / f"{stem}.gif"
    animation.save(gif_path, writer=PillowWriter(fps=args.fps))
    if shutil.which("ffmpeg"):
        animation.save(mp4_path, writer=FFMpegWriter(fps=args.fps, bitrate=3200))
    else:
        mp4_path = None
    update(79)
    fig.savefig(args.output_dir / f"{stem}_static.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    manifest = args.output_dir / f"{stem}_samples.txt"
    manifest.write_text("\n".join(str(sample["path"]) for sample in samples) + "\n", encoding="utf-8")
    if mp4_path is not None:
        print(mp4_path.resolve())
    else:
        print("MP4 skipped: ffmpeg is not installed")
    print(gif_path.resolve())


if __name__ == "__main__":
    main()
