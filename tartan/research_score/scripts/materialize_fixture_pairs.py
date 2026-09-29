from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from diffusion_planner.utils.config import Config
from tartan.data.features import _polyline_features, _route_segments
from tartan.research_score.scripts.build_proxy_cpu_inputs import plan_grid


def route_features(config, route):
    lanes = np.zeros((config.lane_num, config.lane_len, config.lane_state_dim), np.float32)
    routes = np.zeros((config.route_num, config.route_len, config.route_state_dim), np.float32)
    count = 0
    for segment in _polyline_features(_route_segments(route, 4, config.lane_len), .8):
        if count < config.route_num:
            routes[count] = segment[:, :config.route_state_dim]
        if count < config.lane_num:
            lanes[count] = segment[:, :config.lane_state_dim]
        count += 1
    return torch.from_numpy(lanes), torch.from_numpy(routes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--args", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    config = Config(args.args, None)
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    ego, lanes, routes, diff, anymal, mask_diff, mask_anymal = [], [], [], [], [], [], []
    common_profile = {"footprint_radius_m": 0.0, "max_heading_change_bins": 3, "turn_penalty": 0.0}
    for row in rows:
        occupancy = np.load(row["common_map_reference"]).astype(bool)
        start = np.asarray(row["common_start_world"], np.float32)
        goal = np.asarray(row["common_goal_world"], np.float32)
        common_route = plan_grid(occupancy, start, goal, common_profile, float(row["map_resolution_m"])) - start
        lane, route = route_features(config, common_route)
        ego.append(torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32))
        lanes.append(lane); routes.append(route)
        diff.append(torch.tensor(row["trajectory_diff"], dtype=torch.float32))
        anymal.append(torch.tensor(row["trajectory_anymal"], dtype=torch.float32))
        mask_diff.append(torch.tensor(row["valid_mask_diff"], dtype=torch.bool))
        mask_anymal.append(torch.tensor(row["valid_mask_anymal"], dtype=torch.bool))
    payload = {"ego_current_state": torch.stack(ego), "lanes": torch.stack(lanes),
               "route_lanes": torch.stack(routes), "trajectory_diff": torch.stack(diff),
               "trajectory_anymal": torch.stack(anymal), "valid_mask_diff": torch.stack(mask_diff),
               "valid_mask_anymal": torch.stack(mask_anymal), "pair_valid": torch.tensor([row["pair_valid"] for row in rows]),
               "pair_ids": [row["pair_id"] for row in rows], "platform_id_diff": 2, "platform_id_anymal": 1,
               "context_semantics": "common_raw_map_goal_route; identical context makes L_inv structural-only"}
    fd, temporary = tempfile.mkstemp(prefix="fixture_pairs_", suffix=".pt")
    os.close(fd)
    try:
        torch.save(payload, temporary)
        subprocess.run(["dd", "if=" + temporary, "of=" + str(output), "conv=fsync", "status=none"], check=True)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    print(json.dumps({"status": "complete", "pairs": len(rows), "valid": int(payload["pair_valid"].sum()),
                      "output": str(output), "bytes": output.stat().st_size}, sort_keys=True))


if __name__ == "__main__":
    main()
