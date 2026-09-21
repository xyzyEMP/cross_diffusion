from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_sample(
    history: np.ndarray,
    gt: np.ndarray,
    prediction: np.ndarray,
    baseline: np.ndarray,
    title: str,
    path: Path,
    prediction_elevation: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=140)
    ax.plot(history[:, 0], history[:, 1], color="0.55", linewidth=2, label="history")
    ax.plot(gt[:, 0], gt[:, 1], color="#2ca02c", linewidth=2.5, label="GT / oracle route")
    prediction_line, = ax.plot(
        prediction[:, 0], prediction[:, 1], color="#d62728", linewidth=1.2,
        zorder=8, label="terrain-aware prediction",
    )
    if prediction_elevation is not None and np.isfinite(prediction_elevation).any():
        valid = np.isfinite(prediction_elevation)
        scatter = ax.scatter(
            prediction[valid, 0], prediction[valid, 1], c=prediction_elevation[valid],
            cmap="Greys", s=12, zorder=2, alpha=0.68, label="predicted terrain elevation",
        )
        fig.colorbar(scatter, ax=ax, shrink=0.72, label="terrain elevation [m]")
    ax.plot(baseline[:, 0], baseline[:, 1], color="#1f77b4", linestyle="--", linewidth=1.8, label="constant velocity")
    ax.scatter([0], [0], c="black", s=40, marker="o", zorder=5, label="current pose")
    ax.scatter([gt[-1, 0]], [gt[-1, 1]], c="#2ca02c", s=55, marker="*", zorder=5, label="goal")
    ax.set_title(title)
    ax.set_xlabel("local x [m]")
    ax.set_ylabel("local y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
