from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from tartan.research_score.model.score_decomposition import ScoreDecompositionPlanner
from tartan.research_score.scripts.build_proxy_cpu_inputs import publish_text
from tartan.research_score.training.losses import proxy_objective


def context(cache, config, index):
    zero = lambda *shape: torch.zeros(shape, dtype=torch.float32)
    result = {"ego_current_state": cache["ego_current_state"][index:index + 1],
              "neighbor_agents_past": zero(1, config.agent_num, config.time_len, config.agent_state_dim),
              "static_objects": zero(1, config.static_objects_num, config.static_objects_state_dim),
              "lanes": cache["lanes"][index:index + 1],
              "lanes_speed_limit": zero(1, config.lane_num, 1),
              "lanes_has_speed_limit": torch.zeros(1, config.lane_num, 1, dtype=torch.bool),
              "route_lanes": cache["route_lanes"][index:index + 1],
              "route_lanes_speed_limit": zero(1, config.route_num, 1),
              "route_lanes_has_speed_limit": torch.zeros(1, config.route_num, 1, dtype=torch.bool)}
    return config.observation_normalizer(result)


def normalized_target(config, trajectory):
    raw = torch.zeros(1, 11, 80, 4)
    raw[:, 0] = trajectory
    return config.state_normalizer(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--args", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(11)
    config = Config(args.args, None); config.device = "cpu"
    model = ScoreDecompositionPlanner(Diffusion_Planner(config))
    payload = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"], strict=True)
    model.embodiment_encoder.reinitialize_id(2, 11)
    model.eval()
    cache = torch.load(args.cache, map_location="cpu")
    index = next(i for i, pair_id in enumerate(cache["pair_ids"]) if cache["pair_valid"][i] and "straight" not in pair_id)
    base_context = context(cache, config, index)
    target_diff = normalized_target(config, cache["trajectory_diff"][index:index + 1])
    target_anymal = normalized_target(config, cache["trajectory_anymal"][index:index + 1])
    time = torch.tensor([0.5])
    generator = torch.Generator(device="cpu").manual_seed(11)
    epsilon = torch.randn(target_diff.shape, generator=generator)
    mean_diff, std = model.backbone.sde.marginal_prob(target_diff, time)
    mean_anymal, _ = model.backbone.sde.marginal_prob(target_anymal, time)
    query_diff = mean_diff + std.view(1, 1, 1, 1) * epsilon
    query_anymal = mean_anymal + std.view(1, 1, 1, 1) * epsilon
    current = torch.zeros(1, 11, 1, 4)
    current[:, 0, 0] = base_context["ego_current_state"][:, :4]
    ability = torch.zeros(1, 10); ability_mask = torch.zeros(1, 10)

    def run(query, platform_id):
        inputs = dict(base_context)
        inputs.update({"sampled_trajectories": torch.cat([current, query], 2), "diffusion_time": time})
        return model(inputs, torch.tensor([platform_id]), ability, ability_mask)[1]["decomposition"]

    diff_self = run(query_diff, 2)
    anymal_self = run(query_anymal, 1)
    shared_other_view = run(query_diff, 1)
    mask_diff = cache["valid_mask_diff"][index:index + 1]
    mask_anymal = cache["valid_mask_anymal"][index:index + 1]
    losses = proxy_objective("synthetic_fixture", diff_self.x0_total[:, 0, 1:], target_diff[:, 0], mask_diff,
                             anymal_self.x0_total[:, 0, 1:], target_anymal[:, 0], mask_anymal,
                             pair_valid=torch.ones(1, dtype=torch.bool),
                             shared_a=diff_self.x0_shared[:, 0, 1:], shared_b=shared_other_view.x0_shared[:, 0, 1:],
                             swap_ab=anymal_self.x0_total[:, 0, 1:], swap_ba=diff_self.x0_total[:, 0, 1:],
                             delta_a=diff_self.x0_embodiment_delta[:, 0, 1:],
                             delta_b=anymal_self.x0_embodiment_delta[:, 0, 1:],
                             weights={"invariant": .2, "swap": .5, "separation": .1})
    losses["total"].backward()
    backbone_grad = sum(float(parameter.grad.abs().sum()) for parameter in model.backbone.parameters() if parameter.grad is not None)
    residual_grad = sum(float(parameter.grad.abs().sum()) for parameter in model.residual.parameters() if parameter.grad is not None)
    id_grad = model.embodiment_encoder.id.weight.grad
    result = {"status": "PASS_CPU_FIXTURE_FORWARD_BACKWARD", "pair_id": cache["pair_ids"][index],
              "losses": {key: float(value.detach()) for key, value in losses.items()},
              "backbone_grad_l1": backbone_grad, "residual_grad_l1": residual_grad,
              "diff_id_grad_l1": float(id_grad[2].abs().sum()), "anymal_id_grad_l1": float(id_grad[1].abs().sum()),
              "l_inv_interpretation": "structural identity only because both contexts are the same common map-goal representation"}
    if not all(torch.isfinite(value) for value in losses.values()) or min(backbone_grad, residual_grad, result["diff_id_grad_l1"], result["anymal_id_grad_l1"]) <= 0:
        raise RuntimeError(json.dumps(result, sort_keys=True))
    publish_text(Path(args.output), json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
