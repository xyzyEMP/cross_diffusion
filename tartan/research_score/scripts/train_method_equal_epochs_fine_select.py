from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from tartan.research_score.model.score_decomposition import ScoreDecompositionPlanner
from tartan.research_score.scripts.train_method_formal import (
    ABILITIES,
    METHODS,
    load_ckpt,
    loss_step,
    publish,
    source_batch,
    validate,
)
from tartan.research_score.training.cached_target_dataset import CachedTargetDataset


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--args", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", type=int, choices=[1, 10, 20, 50, 100], required=True)
    parser.add_argument("--max-target-updates", type=int, required=True)
    parser.add_argument("--selection-every-updates", type=int, required=True)
    parser.add_argument("--scheduler-every-updates", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    config = Config(args.args, None)
    config.device = "cuda"
    backbone = Diffusion_Planner(config)
    load_ckpt(backbone, args.checkpoint)
    wrapped = args.method in {"pretrain_adapter", "emb_cond_diffusion"}
    model = ScoreDecompositionPlanner(backbone).cuda() if wrapped else backbone.cuda()
    if args.method == "pretrain_adapter":
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
        model.backbone.eval()

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    learning_rate = 2e-4 if args.method == "pretrain_adapter" else 1e-4
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=5e-6
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
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    source = (
        torch.load(args.source_cache, map_location="cpu", weights_only=False)
        if args.method in {"joint_train", "emb_cond_diffusion"}
        else None
    )

    selection_best = float("inf")
    selection_best_update = 0
    selection_best_state = None
    coarse_best = float("inf")
    coarse_best_update = 0
    coarse_best_state = None
    selection_history = []
    scheduler_history = []
    target_updates = 0
    source_updates = 0
    global_step = 0
    epoch = 0
    next_selection = args.selection_every_updates
    next_scheduler = args.scheduler_every_updates
    train_target = []
    train_source = []
    started = time.time()

    while target_updates < args.max_target_updates:
        epoch += 1
        model.train()
        if args.method == "pretrain_adapter":
            model.backbone.eval()
        for inputs, trajectory, valid_mask, _ in train_loader:
            if target_updates >= args.max_target_updates:
                break
            target_loss = loss_step(
                model, config, inputs, trajectory, valid_mask, wrapped, 1, args.amp
            )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(target_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            scaler.step(optimizer)
            scaler.update()
            target_updates += 1
            global_step += 1
            train_target.append(float(target_loss))

            if source is not None:
                source_inputs, source_trajectory, source_mask = source_batch(
                    source, len(trajectory)
                )
                source_loss = loss_step(
                    model,
                    config,
                    source_inputs,
                    source_trajectory,
                    source_mask,
                    wrapped,
                    0,
                    args.amp,
                )
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(source_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                scaler.step(optimizer)
                scaler.update()
                source_updates += 1
                global_step += 1
                train_source.append(float(source_loss))

            do_selection = target_updates >= next_selection or target_updates >= args.max_target_updates
            do_scheduler = target_updates >= next_scheduler or target_updates >= args.max_target_updates
            if not (do_selection or do_scheduler):
                continue

            val_loss = validate(model, config, val_loader, wrapped, args.amp)
            if args.method == "pretrain_adapter":
                model.backbone.eval()

            if do_selection:
                row = {
                    "epoch": epoch,
                    "target_updates": target_updates,
                    "val_loss": val_loss,
                }
                selection_history.append(row)
                if val_loss < selection_best - 1e-5:
                    selection_best = val_loss
                    selection_best_update = target_updates
                    selection_best_state = clone_state(model)
                next_selection += args.selection_every_updates

            if do_scheduler:
                scheduler.step(val_loss)
                row = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "target_updates": target_updates,
                    "source_updates": source_updates,
                    "target_loss": float(np.mean(train_target)),
                    "source_loss": float(np.mean(train_source)) if train_source else None,
                    "val_loss": val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "seconds": time.time() - started,
                }
                scheduler_history.append(row)
                print(json.dumps(row), flush=True)
                train_target = []
                train_source = []
                if val_loss < coarse_best - 1e-5:
                    coarse_best = val_loss
                    coarse_best_update = target_updates
                    coarse_best_state = clone_state(model)
                next_scheduler += args.scheduler_every_updates

    publish(
        {
            "model": selection_best_state,
            "target_updates": selection_best_update,
            "method": args.method,
            "budget": args.budget,
            "seed": args.seed,
            "val_loss": selection_best,
            "selection": "every_epoch_validation_loss",
        },
        output / "best.pt",
    )
    publish(
        {
            "model": coarse_best_state,
            "target_updates": coarse_best_update,
            "method": args.method,
            "budget": args.budget,
            "seed": args.seed,
            "val_loss": coarse_best,
            "selection": "every_10_epochs_validation_loss",
        },
        output / "best_coarse.pt",
    )
    publish(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        output / "last.pt",
    )
    result = {
        "status": "complete",
        "protocol": "equal_100_epochs_fine_selection_coarse_scheduler_v1",
        "method": args.method,
        "budget": args.budget,
        "seed": args.seed,
        "target_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "source_cache_samples": len(source["trajectory"]) if source is not None else 0,
        "batch_size": args.batch_size,
        "completed_epochs": epoch,
        "target_updates": target_updates,
        "source_updates": source_updates,
        "selection_every_updates": args.selection_every_updates,
        "scheduler_every_updates": args.scheduler_every_updates,
        "best_target_update": selection_best_update,
        "best_validation_loss": selection_best,
        "coarse_best_target_update": coarse_best_update,
        "coarse_best_validation_loss": coarse_best,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "seconds": time.time() - started,
        "selection_history": selection_history,
        "scheduler_history": scheduler_history,
    }
    publish(result, output / "metrics.json", is_json=True)


if __name__ == "__main__":
    main()
