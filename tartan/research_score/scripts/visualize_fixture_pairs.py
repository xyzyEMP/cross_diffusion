from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def draw_pair(ax, row):
    diff = np.asarray(row["trajectory_diff"])[:, :2]
    anymal = np.asarray(row["trajectory_anymal"])[:, :2]
    ax.plot(diff[:, 0], diff[:, 1], color="#d62728", lw=1.7, label="Diff")
    ax.plot(anymal[:, 0], anymal[:, 1], color="#1f77b4", lw=1.7, label="ANYmal")
    ax.scatter([0], [0], s=10, color="black", zorder=3)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.2)
    ax.set_title("%s\nmean=%.3fm max=%.3fm" % (
        row["pair_id"], row["pair_mean_distance_m"], row["pair_max_distance_m"]), fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = load_rows(args.manifest)
    metrics = []
    for index, row in enumerate(rows):
        occupancy = np.load(row["common_map_reference"])
        resolution = float(row["map_resolution_m"])
        full_diff = np.asarray(row["full_route_diff_world"])
        full_anymal = np.asarray(row["full_route_anymal_world"])
        diff = np.asarray(row["trajectory_diff"])[:, :2]
        anymal = np.asarray(row["trajectory_anymal"])[:, :2]
        distance = np.linalg.norm(diff - anymal, axis=1)
        metric = {"pair_id": row["pair_id"], "family": row["family"],
                  "mean_distance_m": float(distance.mean()), "max_distance_m": float(distance.max())}
        metrics.append(metric)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].imshow(occupancy, origin="lower", cmap="Greys", extent=[0, occupancy.shape[1] * resolution, 0, occupancy.shape[0] * resolution])
        axes[0].plot(full_diff[:, 0], full_diff[:, 1], color="#d62728", lw=1.7, label="Diff")
        axes[0].plot(full_anymal[:, 0], full_anymal[:, 1], color="#1f77b4", lw=1.7, label="ANYmal")
        axes[0].scatter(*row["common_start_world"], s=20, color="black", label="start")
        axes[0].scatter(*row["common_goal_world"], s=25, color="#2ca02c", marker="*", label="goal")
        axes[0].set_aspect("equal", adjustable="box"); axes[0].set_title("full route + common map"); axes[0].legend(fontsize=7)
        draw_pair(axes[1], row); axes[1].set_title("local first 8m\nmean=%.3fm max=%.3fm" % (metric["mean_distance_m"], metric["max_distance_m"])); axes[1].legend(fontsize=7)
        fig.suptitle(row["pair_id"]); fig.tight_layout()
        fig.savefig(output / ("%02d_%s.png" % (index, row["pair_id"].replace(":", "_"))), dpi=150)
        plt.close(fig)
    columns = 5
    rows_count = int(np.ceil(len(rows) / columns))
    fig, axes = plt.subplots(rows_count, columns, figsize=(columns * 3.2, rows_count * 3.0), squeeze=False)
    for ax, row in zip(axes.flat, rows):
        draw_pair(ax, row)
    for ax in axes.flat[len(rows):]: ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("All accepted synthetic pairs: local first 8m", y=.995)
    fig.tight_layout(rect=[0, 0, 1, .98])
    fig.savefig(output / "all_pairs_local8m.png", dpi=160)
    plt.close(fig)
    with (output / "pair_difference_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    (output / "pair_difference_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    families = defaultdict(list)
    for metric in metrics: families[metric["family"]].append(metric["mean_distance_m"])
    summary = {"pairs": len(metrics), "minimum_mean_distance_m": min(m["mean_distance_m"] for m in metrics),
               "overall_mean_distance_m": float(np.mean([m["mean_distance_m"] for m in metrics])),
               "maximum_mean_distance_m": max(m["mean_distance_m"] for m in metrics),
               "families": {name: {"pairs": len(values), "min_m": min(values), "mean_m": float(np.mean(values)), "max_m": max(values)} for name, values in sorted(families.items())}}
    (output / "pair_difference_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
