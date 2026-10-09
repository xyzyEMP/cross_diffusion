from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.pair.matching import CandidatePair


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
