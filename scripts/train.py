from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.diffusion.planner import Diffusion_Planner
from utils.config import Config
from evaluation.metrics_navigation import aggregate
from scripts.evaluate import (
    episode_macro,
    load_segments,
    rollout,
)
from engine.training_utils import (
    load_ckpt,
    loss_step,
    publish,
    validate,
)
from datasets.cached_target_dataset import CachedTargetDataset


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def navigation_validate(model, config, records, seeds):
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()
    was_training = model.training
    model.eval()
    rows = []
    try:
        with torch.no_grad():
            for seed in seeds:
                for index, record in enumerate(records):
                    row = rollout(
                        record,
                        config,
                        model,
                        index,
                        inference_seed=seed,
                    )
                    row["inference_seed"] = seed
                    rows.append(row)
    finally:
        model.train(was_training)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
    valid = [row for row in rows if row["included_in_denominator"]]
    micro = aggregate(valid)
    macro = episode_macro(valid)
    return rows, micro, macro


def selection_key(row):
    return (
        float(row["macro_spl"]),
        float(row["macro_sr"]),
        -float(row["macro_cr"]),
        float(row["macro_goal_progress"]),
        -int(row["epoch"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--val-navigation-manifest", required=True)
    parser.add_argument("--args", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", type=int, choices=[1, 10, 20, 50, 100], required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--navigation-every-epochs", type=int, default=5)
    parser.add_argument("--scheduler-every-epochs", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--inference-seeds", type=int, nargs="+", default=[11, 23, 47])
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    config = Config(args.args, None)
    config.device = "cuda"
    model = Diffusion_Planner(config)
    load_ckpt(model, args.checkpoint)
    model = model.cuda()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        threshold=1e-4,
        threshold_mode="rel",
        min_lr=5e-6,
    )

    train_dataset = CachedTargetDataset(args.train_cache, config, args.budget, args.seed)
    val_dataset = CachedTargetDataset(args.val_cache, config, None, args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(args.batch_size, len(train_dataset)),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    navigation_records = load_segments(args.val_navigation_manifest)

    best_row = None
    best_state = None
    stale_navigation_checks = 0
    target_updates = 0
    loss_history = []
    navigation_history = []
    started = time.time()
    stop_reason = "max_epochs"

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_losses = []
        for inputs, trajectory, valid_mask, _ in train_loader:
            loss = loss_step(
                model,
                config,
                inputs,
                trajectory,
                valid_mask,
                args.amp,
            )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            scaler.step(optimizer)
            scaler.update()
            target_updates += 1
            train_losses.append(float(loss))

        val_loss = validate(model, config, val_loader, args.amp)
        loss_row = {
            "epoch": epoch,
            "target_updates": target_updates,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        loss_history.append(loss_row)

        if epoch % args.scheduler_every_epochs == 0:
            scheduler.step(val_loss)

        if epoch % args.navigation_every_epochs != 0:
            continue

        per_rollout, micro, macro = navigation_validate(
            model, config, navigation_records, args.inference_seeds
        )
        nav_row = {
            "epoch": epoch,
            "target_updates": target_updates,
            "macro_spl": float(macro["spl"]),
            "macro_sr": float(macro["sr"]),
            "macro_cr": float(macro["cr"]),
            "macro_goal_progress": float(macro["goal_progress"]),
            "macro_route_failure_rate": float(macro["route_failure_rate"]),
            "micro_spl": float(micro["spl"]),
            "micro_sr": float(micro["sr"]),
            "micro_cr": float(micro["cr"]),
            "micro_goal_progress": float(micro["goal_progress"]),
            "inference_seeds": list(args.inference_seeds),
            "rollouts": len(per_rollout),
            "seconds": time.time() - started,
        }
        navigation_history.append(nav_row)
        print(json.dumps(nav_row), flush=True)

        if best_row is None or selection_key(nav_row) > selection_key(best_row):
            best_row = nav_row
            best_state = clone_state(model)
            stale_navigation_checks = 0
        else:
            stale_navigation_checks += 1

        if epoch >= args.min_epochs and stale_navigation_checks >= args.early_stop_patience:
            stop_reason = "navigation_patience"
            break

    completed_epoch = loss_history[-1]["epoch"]
    publish(
        {
            "model": best_state,
            "method": "pretrain_finetune",
            "budget": args.budget,
            "seed": args.seed,
            "selection": "macro_spl_macro_sr_low_macro_cr_macro_goal_progress_early_epoch",
            "navigation_metrics": best_row,
        },
        output / "navigation_best.pt",
    )
    publish(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": completed_epoch,
            "target_updates": target_updates,
        },
        output / "last.pt",
    )
    result = {
        "status": "complete",
        "protocol": "finetune_navigation_selection_earlystop_v1",
        "budget": args.budget,
        "seed": args.seed,
        "target_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "validation_navigation_tasks": len(navigation_records),
        "inference_seeds": list(args.inference_seeds),
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "min_epochs": args.min_epochs,
        "completed_epoch": completed_epoch,
        "stop_reason": stop_reason,
        "early_stop_patience": args.early_stop_patience,
        "navigation_every_epochs": args.navigation_every_epochs,
        "scheduler_every_epochs": args.scheduler_every_epochs,
        "target_updates": target_updates,
        "best_navigation": best_row,
        "minimum_validation_loss": min(row["val_loss"] for row in loss_history),
        "minimum_validation_loss_epoch": min(loss_history, key=lambda row: row["val_loss"])["epoch"],
        "loss_history": loss_history,
        "navigation_history": navigation_history,
        "seconds": time.time() - started,
    }
    publish(result, output / "metrics.json", is_json=True)


if __name__ == "__main__":
    main()
