from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from tartan.research_score.model.score_decomposition import ScoreDecompositionPlanner
from tartan.research_score.scripts.build_proxy_cpu_inputs import publish_text
from tartan.research_score.training.cached_target_dataset import CachedTargetDataset
from tartan.research_score.training.losses import proxy_objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--args", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(11)
    config = Config(args.args, None)
    config.device = "cpu"
    model = ScoreDecompositionPlanner(Diffusion_Planner(config))
    payload = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"], strict=True)
    model.embodiment_encoder.reinitialize_id(2, 11)
    model.eval()

    dataset = CachedTargetDataset(args.cache, config, None, 11)
    inputs, trajectory, mask, sample_ids = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    inputs = config.observation_normalizer(inputs)
    raw = torch.zeros(1, 11, 80, 4)
    raw[:, 0] = trajectory
    target = config.state_normalizer(raw)
    current = torch.zeros(1, 11, 1, 4)
    current[:, 0, 0] = inputs["ego_current_state"][:, :4]
    time = torch.tensor([0.5])
    mean, std = model.backbone.sde.marginal_prob(target, time)
    generator = torch.Generator(device="cpu").manual_seed(11)
    query = mean + std.view(1, 1, 1, 1) * torch.randn(target.shape, generator=generator)
    denoise_inputs = dict(inputs)
    denoise_inputs.update({"sampled_trajectories": torch.cat([current, query], 2), "diffusion_time": time})
    ability = torch.zeros(1, 10)
    ability_mask = torch.zeros(1, 10)
    with torch.no_grad():
        _, anymal = model(denoise_inputs, torch.tensor([1]), ability, ability_mask)
        _, diff = model(denoise_inputs, torch.tensor([2]), ability, ability_mask)
    pred_anymal = anymal["score"][:, 0, 1:]
    pred_diff = diff["score"][:, 0, 1:]
    losses = proxy_objective("tartan_recorded_unpaired", pred_anymal, target[:, 0], mask,
                             pred_diff, target[:, 0], mask, weights={})
    shared_equal = torch.equal(anymal["decomposition"].x0_shared, diff["decomposition"].x0_shared)
    result = {"status": "PASS_CPU_DENOISING_FORWARD", "sample_id": sample_ids[0],
              "score_shape": list(pred_anymal.shape), "finite_anymal": bool(torch.isfinite(pred_anymal).all()),
              "finite_diff": bool(torch.isfinite(pred_diff).all()), "shared_equal_when_context_equal": shared_equal,
              "conditioned_mean_abs_difference": float((pred_anymal - pred_diff).abs().mean()),
              "real_unpaired_loss": float(losses["total"]),
              "note": "same target is used only for numerical interface smoke; no Diff-ANYmal pair claim"}
    if not (result["finite_anymal"] and result["finite_diff"] and shared_equal):
        raise RuntimeError(json.dumps(result, sort_keys=True))
    publish_text(Path(args.output), json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
