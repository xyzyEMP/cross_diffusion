from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from tartan.research_score.scripts.train_method_formal import validate
from tartan.research_score.training.cached_target_dataset import CachedTargetDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--args", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    config = Config(args.args, None)
    config.device = "cuda"
    model = Diffusion_Planner(config).cuda()
    val_dataset = CachedTargetDataset(args.val_cache, config, None, args.seed)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    rows = []
    for path in sorted(Path(args.snapshots).glob("epoch_*.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        loss = validate(model, config, val_loader, False, args.amp)
        row = {
            "epoch": int(checkpoint["epoch"]),
            "target_updates": int(checkpoint["target_updates"]),
            "val_loss": float(loss),
            "checkpoint": str(path),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    rows.sort(key=lambda row: row["epoch"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "validation_by_epoch.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    ranked = sorted(rows, key=lambda row: row["val_loss"])
    (output / "validation_ranked.json").write_text(
        json.dumps(ranked, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
