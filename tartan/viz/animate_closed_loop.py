"""Visualize saved kinematic closed-loop executions on semantic terrain maps."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from tartan.data.pose_utils import to_local_se2
from tartan.data.semantics import GROUP_COLORS, GROUP_NAMES, UNKNOWN, build_semantic_surface
from tartan.data.terrain import ElevationMap


SEMANTIC_CMAP = ListedColormap(GROUP_COLORS)


def _load_terrain(environment: str, cache: Path) -> ElevationMap:
    candidates = list(cache.glob(f"{environment}_surface_v3_*.npz")) or list(cache.glob(f"{environment}_surface_v2_*.npz"))
    return ElevationMap.load(max(candidates, key=lambda path: path.stat().st_mtime))


def _background(environment: str, anchor: np.ndarray, limits, data_root: Path, cache: Path):
    terrain = _load_terrain(environment, cache)
    semantic = build_semantic_surface(data_root, terrain, cache)
    yy, xx = np.indices(terrain.elevation.shape)
    global_xy = np.column_stack((terrain.origin_xy[0] + xx.ravel() * terrain.resolution, terrain.origin_xy[1] + yy.ravel() * terrain.resolution))
    local = to_local_se2(np.column_stack((global_xy, np.full(len(global_xy), anchor[2]))), anchor)[:, :2]
    labels = semantic.ravel()
    z = terrain.elevation.ravel()
    x0, x1, y0, y1 = limits
    visible = np.isfinite(z) & (labels != UNKNOWN) & (local[:, 0] >= x0) & (local[:, 0] <= x1) & (local[:, 1] >= y0) & (local[:, 1] <= y1)
    return local[visible], labels[visible]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=Path("tartan/outputs/town_focused_v2/closed_loop"))
    parser.add_argument("--output-dir", type=Path, default=Path("tartan/outputs/town_focused_v2/visualizations/closed_loop"))
    parser.add_argument("--data-root", type=Path, default=Path("/media/yzy/1E585F67585F3CA9/data_yzy/TartanGround"))
    parser.add_argument("--terrain-cache", type=Path, default=Path("tartan/cache/elevation"))
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--batch-index", type=int, default=0, help="zero-based difficulty batch")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.result_root / "per_episode_metrics.csv")
    selected = []
    for (_, _), group in frame.groupby(["environment", "robot_type"], sort=True):
        # Five batches span easy to hard cases with unique ranks where possible.
        ordered = group.sort_values("closed_loop_fde_m")
        rank = round((args.batch_index % 5) * (len(ordered) - 1) / 4)
        selected.append(ordered.iloc[rank])

    samples = []
    for row in selected:
        data = np.load(args.result_root / "rollouts" / f"{row.sample_id}.npz")
        anchor, executed, gt = data["anchor"], data["executed"], data["gt"]
        samples.append({"row": row, "anchor": anchor, "executed": to_local_se2(executed, anchor), "gt": to_local_se2(gt, anchor)})

    rows = 2
    columns = math.ceil(len(samples) / rows)
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.5 * rows), squeeze=False)
    artists = []
    for axis, sample in zip(axes.ravel(), samples):
        xy = np.vstack((sample["executed"][:, :2], sample["gt"][:, :2], np.zeros((1, 2))))
        minimum, maximum = xy.min(axis=0), xy.max(axis=0)
        center = (minimum + maximum) / 2
        radius = max(float(np.max(maximum - minimum)) * .62, 4.0)
        limits = (center[0] - radius, center[0] + radius, center[1] - radius, center[1] + radius)
        terrain_xy, labels = _background(sample["row"].environment, sample["anchor"], limits, args.data_root, args.terrain_cache)
        axis.scatter(terrain_xy[:, 0], terrain_xy[:, 1], c=labels, cmap=SEMANTIC_CMAP, vmin=-.5, vmax=len(GROUP_NAMES)-.5, s=16, marker="s", linewidths=0, alpha=.7)
        gt_line, = axis.plot([], [], color="#22c55e", lw=2.5, label="logged route")
        run_line, = axis.plot([], [], color="#ef4444", lw=1.6, label="executed closed loop", zorder=8)
        gt_head = axis.scatter([], [], s=50, color="#22c55e", zorder=9)
        run_head = axis.scatter([], [], s=50, color="#ef4444", linewidth=0, zorder=10)
        status = axis.text(.02, .98, "", transform=axis.transAxes, va="top", bbox={"facecolor":"white", "alpha":.86, "edgecolor":"#d1d5db"})
        axis.scatter([0], [0], marker="*", s=120, color="#111827", zorder=11, label="start")
        axis.set(xlim=limits[:2], ylim=limits[2:], xlabel="local x [m]", ylabel="local y [m]", aspect="equal")
        axis.set_title(f"{sample['row'].environment} | {sample['row'].robot_type} | {sample['row'].condition}")
        axis.grid(alpha=.2)
        axis.legend(loc="lower left", fontsize=8)
        artists.append((gt_line, run_line, gt_head, run_head, status))
    for axis in axes.ravel()[len(samples):]: axis.set_visible(False)
    fig.suptitle("TartanGround kinematic closed-loop replay (8 s)", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, .96))

    def update(index: int):
        output = []
        end = index + 1
        for sample, group in zip(samples, artists):
            gt_line, run_line, gt_head, run_head, status = group
            gt = np.vstack((np.zeros((1, 3)), sample["gt"][:end]))
            run = np.vstack((np.zeros((1, 3)), sample["executed"][:end]))
            gt_line.set_data(gt[:, 0], gt[:, 1]); run_line.set_data(run[:, 0], run[:, 1])
            gt_head.set_offsets(gt[-1:, :2]); run_head.set_offsets(run[-1:, :2])
            error = np.linalg.norm(run[-1, :2] - gt[-1, :2])
            status.set_text(f"t = {end/10:.1f} s\nerror = {error:.2f} m")
            output.extend(group)
        return output

    animation = FuncAnimation(fig, update, frames=80, interval=100, blit=False)
    stem = f"closed_loop_batch_{args.batch_index + 1:02d}"
    gif = args.output_dir / f"{stem}.gif"
    animation.save(gif, writer=PillowWriter(fps=args.fps))
    if shutil.which("ffmpeg"):
        animation.save(args.output_dir / f"{stem}.mp4", writer=FFMpegWriter(fps=args.fps, bitrate=3200))
    update(79)
    fig.savefig(args.output_dir / f"{stem}_static.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    (args.output_dir / f"{stem}_samples.txt").write_text("\n".join(row.sample_id for row in selected) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
