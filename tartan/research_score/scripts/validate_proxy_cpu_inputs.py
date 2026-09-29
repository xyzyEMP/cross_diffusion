from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tartan.research_score.scripts.build_proxy_cpu_inputs import inflate, publish_text


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def arc_length(values):
    xy = np.asarray(values, np.float64)[:, :2]
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def collision_free(row, platform):
    occupancy = np.load(row["common_map_reference"]).astype(bool)
    resolution = float(row["map_resolution_m"])
    profile = row["platform_constraints"][platform]
    blocked = inflate(occupancy, int(np.ceil(float(profile["footprint_radius_m"]) / resolution)))
    route = np.asarray(row["full_route_" + platform + "_world"], np.float64)
    cells = np.rint(route[:, [1, 0]] / resolution).astype(int)
    return not blocked[cells[:, 0], cells[:, 1]].any()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--d029-summary", required=True)
    parser.add_argument("--d029-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    inputs = Path(args.inputs)
    train = read_jsonl(inputs / "real_unpaired_train.jsonl")
    val = read_jsonl(inputs / "real_unpaired_val.jsonl")
    fixture = read_jsonl(inputs / "synthetic_paired_fixture.jsonl")
    registry = json.loads((inputs / "platform_registry.json").read_text())
    d029 = json.loads(Path(args.d029_summary).read_text())
    cache = torch.load(args.d029_cache, map_location="cpu")

    checks = {}
    checks["real_counts"] = len(train) == 64 and len(val) == 16
    checks["real_unpaired_only"] = all(not row["pair_valid"] and row["allowed_losses"] == ["L_diff"] for row in train + val)
    checks["platform_balance"] = all(sum(row["platform_name"] == platform for row in train) == 32 for platform in ("diff", "anymal")) and all(sum(row["platform_name"] == platform for row in val) == 8 for platform in ("diff", "anymal"))
    train_episodes = {(row["platform_name"], row["episode_id"]) for row in train}
    val_episodes = {(row["platform_name"], row["episode_id"]) for row in val}
    checks["episode_isolation"] = train_episodes.isdisjoint(val_episodes)
    checks["no_anymal_test"] = not any(any(token in row["episode_id"] for token in ("P2001", "P2003", "P2012", "P2015", "P2018")) for row in train + val)
    checks["fixture_contract"] = len(fixture) >= 18 and all(row["pair_valid"] for row in fixture) and all(len(row["trajectory_diff"]) == 80 and len(row["trajectory_anymal"]) == 80 for row in fixture)
    valid_fixture = [row for row in fixture if row["pair_valid"]]
    invalid_fixture = [row for row in fixture if not row["pair_valid"]]
    checks["fixture_shared_task"] = all(np.allclose(row["goal_diff"], row["common_goal"]) and np.allclose(row["goal_anymal"], row["common_goal"]) and row["context_diff"]["map_reference"] == row["context_anymal"]["map_reference"] == row["common_map_reference"] for row in valid_fixture)
    checks["fixture_oracle_goal"] = all(np.linalg.norm(np.asarray(row["full_route_diff"])[-1] - np.asarray(row["goal_diff"])) <= .3 and np.linalg.norm(np.asarray(row["full_route_anymal"])[-1] - np.asarray(row["goal_anymal"])) <= .3 for row in fixture)
    checks["fixture_local_frame"] = all(np.allclose(np.asarray(row["trajectory_diff"])[0, :2], [0.0, 0.0]) and np.allclose(np.asarray(row["trajectory_anymal"])[0, :2], [0.0, 0.0]) and np.allclose(row["common_start"], [0.0, 0.0]) for row in fixture)
    checks["fixture_fixed_arc"] = all(abs(arc_length(row["trajectory_diff"]) - 8.0) <= .04 and abs(arc_length(row["trajectory_anymal"]) - 8.0) <= .04 for row in fixture)
    checks["fixture_collision_free"] = all(collision_free(row, platform) for row in fixture for platform in ("diff", "anymal"))
    checks["fixture_constraint_effect"] = all(np.linalg.norm(np.asarray(row["trajectory_diff"])[:, :2] - np.asarray(row["trajectory_anymal"])[:, :2], axis=1).mean() > .05 for row in valid_fixture)
    checks["fixture_no_identity_pairs"] = all(not np.allclose(row["trajectory_diff"], row["trajectory_anymal"]) for row in valid_fixture)
    checks["fixture_no_invalid_controls"] = len(invalid_fixture) == 0
    checks["fixture_inv_scope"] = all(row["l_inv_scope"] == "structural_identity_only_common_context" for row in fixture)
    checks["platform_registry"] = {name: value["id"] for name, value in registry["platforms"].items()} == {"car": 0, "anymal": 1, "diff": 2}
    checks["d029_counts"] = d029["targets"] == {"1": 20, "10": 205, "100": 2048}
    checks["d029_cache"] = len(cache["sample_ids"]) == 2048 and sum("1" in value["11"] for value in cache["budget_membership"]) == 20 and sum("10" in value["11"] for value in cache["budget_membership"]) == 205
    if not all(checks.values()):
        raise RuntimeError("CPU preflight failed: " + json.dumps(checks, sort_keys=True))
    report = {"status": "PASS_CPU_GPU_PENDING", "checks": checks,
              "gpu_tasks_not_run": ["proxy_real_smoke", "proxy_fixture_smoke", "four_way_swap_sampling", "experiment1_d029_12_runs", "closed_loop_evaluation"]}
    publish_text(Path(args.output), json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
