from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pair.matching import CandidatePair


def visualize_pair(pair: CandidatePair, output_path: Path) -> None:
    segment_a, segment_b = pair.segment_a, pair.segment_b
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.plot(segment_a.points[:, 0], segment_a.points[:, 1], label=segment_a.embodiment)
    axis.plot(segment_b.points[:, 0], segment_b.points[:, 1], label=segment_b.embodiment)
    axis.scatter(*segment_a.entry_position[:2], marker="o", s=70, label="A entry")
    axis.scatter(*segment_b.entry_position[:2], marker="o", s=70, label="B entry")
    axis.scatter(*segment_a.exit_position[:2], marker="X", s=80, label="A exit")
    axis.scatter(*segment_b.exit_position[:2], marker="X", s=80, label="B exit")
    axis.set_title(
        f"{segment_a.map_id}: {segment_a.embodiment} / {segment_b.embodiment}\n"
        f"entry={pair.entry_distance:.2f} m, exit={pair.exit_distance:.2f} m, "
        f"mean={pair.mean_path_distance:.2f} m, chamfer={pair.chamfer_distance:.2f} m"
    )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def visualize_reconstructed_pair(record, paths, output_path):
    """At most six deterministic plots, all in the observed Diff body frame."""
    output_path = Path(output_path)
    if output_path.exists():
        return
    figure, axis = plt.subplots(figsize=(8, 7))
    for side, path in zip(("Diff", "Omni"), paths):
        axis.plot(path[:, 0], path[:, 1], label=side+" rebuilt 8m / 80 valid")
        axis.scatter(*path[0, :2], marker="o")
        axis.scatter(*path[-1, :2], marker="X", label=side+" own goal")
    metrics = record["metrics_8m"]
    axis.set_title(record["pair_id"]+"\nDiff observed-anchor common frame; common_goal_id=null\n"+
                   ", ".join(f"{key}={metrics[key]:.3f}m" for key in ("entry_distance", "exit_distance", "mean_path_distance", "max_path_distance"))+"\n"+
                   ("accepted "+record["split"] if record["pair_valid"] else "rejected: "+",".join(record["rejection_reasons"])), fontsize=8)
    axis.set(xlabel="Diff local x [m]", ylabel="Diff local y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=.3)
    axis.legend(fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)
