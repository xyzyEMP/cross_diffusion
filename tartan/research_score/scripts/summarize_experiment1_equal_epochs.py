from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = ("pretrain_finetune", "pretrain_adapter", "joint_train", "emb_cond_diffusion")
BUDGETS = (1, 10, 20, 50, 100)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.transfer_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for method in METHODS:
        for budget in BUDGETS:
            if method == "pretrain_finetune":
                validation = json.loads(
                    (
                        root
                        / "finetune_equal100epochs_seed11_epochscan_fine_selection"
                        / "validation"
                        / f"{budget}pct"
                        / "validation_ranked.json"
                    ).read_text()
                )[0]
                summary_path = (
                    root
                    / "finetune_equal100epochs_seed11_epochscan_fine_selection"
                    / "evaluation"
                    / f"pretrain_finetune_{budget}pct_seed11"
                    / "summary.json"
                )
                epoch = int(validation["epoch"])
                val_loss = float(validation["val_loss"])
            else:
                train_dir = root / "comparators_equal100epochs_fine_select_seed11"
                metrics = json.loads(
                    (train_dir / f"{method}_{budget}pct_seed11" / "metrics.json").read_text()
                )
                epoch = int(metrics["best_target_update"] // metrics["selection_every_updates"])
                val_loss = float(metrics["best_validation_loss"])
                summary_path = (
                    root
                    / "comparators_equal100epochs_fine_select_seed11_eval_nonoverlap8m"
                    / f"{method}_{budget}pct_seed11"
                    / "summary.json"
                )
            summary = json.loads(summary_path.read_text())
            rows.append(
                {
                    "method": method,
                    "budget_pct": budget,
                    "best_epoch": epoch,
                    "val_loss": val_loss,
                    "sr": float(summary["sr"]),
                    "cr": float(summary["cr"]),
                    "spl": float(summary["spl"]),
                    "macro_sr": float(summary["episode_macro"]["sr"]),
                    "goal_progress": float(summary["goal_progress"]),
                    "route_failure_rate": float(summary["route_failure_rate"]),
                }
            )

    with (output / "experiment1_main_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / "experiment1_main_table.json").write_text(json.dumps(rows, indent=2) + "\n")

    lines = [
        "# 实验一：等100 epochs、逐epochvalidation选点（seed 11）",
        "",
        "| 方法 | 预算 | Best epoch | Val loss | SR | CR | SPL | Macro SR | Goal progress |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['budget_pct']}% | {row['best_epoch']} | "
            f"{row['val_loss']:.6f} | {row['sr']:.4f} | {row['cr']:.4f} | "
            f"{row['spl']:.4f} | {row['macro_sr']:.4f} | {row['goal_progress']:.4f} |"
        )
    (output / "experiment1_main_table.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
