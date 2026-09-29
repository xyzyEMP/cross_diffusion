from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from tartan.data.pose_utils import load_poses, poses_to_se2, to_local_se2
from tartan.research_score.data.core import resample_fixed_arc


def publish_bytes(path, source):
    if path.exists():
        raise FileExistsError(path)
    subprocess.run(["dd", "if=" + str(source), "of=" + str(path), "conv=fsync", "status=none"], check=True)


def publish_text(path, text):
    fd, temporary = tempfile.mkstemp(prefix="proxy_cpu_", suffix=".txt")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        publish_bytes(path, Path(temporary))
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def rank(seed, value):
    return hashlib.sha256((str(seed) + "|" + str(value)).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def balanced_take(rows, count, seed):
    queues = defaultdict(list)
    for row in rows:
        queues[str(row["episode_id"])].append(row)
    for episode in queues:
        queues[episode].sort(key=lambda row: (rank(seed, row["sample_id"]), row["sample_id"]))
    chosen = []
    episodes = sorted(queues, key=lambda value: (rank(seed, value), value))
    while len(chosen) < min(count, len(rows)):
        changed = False
        for episode in episodes:
            if queues[episode] and len(chosen) < count:
                chosen.append(queues[episode].pop(0))
                changed = True
        if not changed:
            break
    return chosen


def proxy_row(row, platform, platform_id):
    result = dict(row)
    result.update({"evidence_kind": "tartan_recorded_unpaired", "platform_name": platform,
                   "platform_id": platform_id, "pair_id": None, "pair_valid": False,
                   "allowed_losses": ["L_diff"], "ability": [0.0] * 10,
                   "ability_mask": [0.0] * 10})
    result.setdefault("provenance", {})["proxy_protocol"] = "D028/v1.3.1-proxy"
    return result


def diff_rows(root, episode, split, seed):
    folder = root / "Data_diff" / episode
    se2 = poses_to_se2(load_poses(folder / "pose_lcam_front.txt"))
    rows = []
    for anchor in range(20, len(se2) - 2, 10):
        occ = folder / "coarse_occ" / ("occupancy_coarse5_%06d_sparse.npy" % anchor)
        if not occ.exists():
            continue
        future = se2[anchor + 1:min(len(se2), anchor + 81)]
        if len(future) < 2:
            continue
        local = to_local_se2(future, se2[anchor])
        trajectory, valid = resample_fixed_arc(local, 8.0, 80)
        valid_indices = np.flatnonzero(valid)
        goal = trajectory[valid_indices[-1], :2] if len(valid_indices) else np.zeros(2, np.float32)
        sid = "tartan:diff:%s:%06d" % (episode, anchor)
        rows.append({"sample_id": sid, "episode_id": "diff:" + episode,
                     "map_id": "ModularNeighborhood", "split": split,
                     "branch": "moving_planning" if int(valid.sum()) == 80 else "stop_or_short",
                     "anchor_index": anchor, "domain": "tartanground", "embodiment": "diff",
                     "fixed_goal": {"xy_local": goal.tolist(), "requested_distance_m": 8.0,
                                    "rule": "future_arc_clamped_then_frozen_for_proxy_input"},
                     "route_set": {"source": "current_coarse_occupancy_plus_fixed_goal",
                                   "map_reference": str(occ), "future_gt_dependency": False},
                     "trajectory": {"fixed_arc_length_80": trajectory.tolist(),
                                    "valid_mask": valid.astype(int).tolist(),
                                    "raw_reference": str(folder / "pose_lcam_front.txt")},
                     "provenance": {"selection": "complete_episode_split_then_hash_window_sample",
                                    "seed": seed}})
    return rows


def xy_to_state(xy):
    delta = np.gradient(xy, axis=0)
    yaw = np.arctan2(delta[:, 1], delta[:, 0])
    return np.column_stack([xy, np.cos(yaw), np.sin(yaw)]).astype(np.float32)


def inflate(occupancy, radius_cells):
    result = occupancy.copy()
    ys, xs = np.nonzero(occupancy)
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy > radius_cells * radius_cells:
                continue
            yy, xx = ys + dy, xs + dx
            valid = (yy >= 0) & (yy < occupancy.shape[0]) & (xx >= 0) & (xx < occupancy.shape[1])
            result[yy[valid], xx[valid]] = True
    return result


DIRECTIONS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))


def heading_distance(a, b):
    distance = abs(a - b) % 8
    return min(distance, 8 - distance)


def plan_grid(occupancy, start_xy, goal_xy, profile, resolution):
    blocked = inflate(occupancy, int(math.ceil(profile["footprint_radius_m"] / resolution)))
    to_cell = lambda xy: (int(round(xy[1] / resolution)), int(round(xy[0] / resolution)))
    start, goal = to_cell(start_xy), to_cell(goal_xy)
    if blocked[start] or blocked[goal]:
        raise RuntimeError("start or goal blocked after platform inflation")
    initial = (start[0], start[1], 0)
    queue = [(0.0, 0.0, initial)]
    costs, parents, final = {initial: 0.0}, {}, None
    while queue:
        _, value, state = heapq.heappop(queue)
        y, x, old_heading = state
        if (y, x) == goal:
            final = state
            break
        if value != costs.get(state):
            continue
        for heading, (dx, dy) in enumerate(DIRECTIONS):
            turn = heading_distance(old_heading, heading)
            if turn > profile["max_heading_change_bins"]:
                continue
            ny, nx = y + dy, x + dx
            if ny < 0 or nx < 0 or ny >= blocked.shape[0] or nx >= blocked.shape[1] or blocked[ny, nx]:
                continue
            if dx and dy and (blocked[y, nx] or blocked[ny, x]):
                continue
            new_state = (ny, nx, heading)
            new_cost = value + math.hypot(dx, dy) * resolution + profile["turn_penalty"] * turn
            if new_cost >= costs.get(new_state, float("inf")):
                continue
            costs[new_state], parents[new_state] = new_cost, state
            heuristic = math.hypot(goal[1] - nx, goal[0] - ny) * resolution
            heapq.heappush(queue, (new_cost + heuristic, new_cost, new_state))
    if final is None:
        raise RuntimeError("platform route is unreachable")
    cells = []
    while True:
        cells.append((final[1], final[0]))
        if final == initial:
            break
        final = parents[final]
    cells.reverse()
    return np.asarray(cells, np.float32) * resolution


def path_length(xy):
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def resample_prefix(xy, length_m=8.0, count=80):
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if cumulative[-1] + 1e-6 < length_m:
        raise RuntimeError("full route shorter than fixed horizon")
    query = np.linspace(0.0, length_m, count)
    sampled = np.column_stack([np.interp(query, cumulative, xy[:, axis]) for axis in (0, 1)])
    return xy_to_state(sampled)


def add_rect(occupancy, resolution, x0, x1, y0, y1):
    ix0, ix1 = int(math.floor(x0 / resolution)), int(math.ceil(x1 / resolution))
    iy0, iy1 = int(math.floor(y0 / resolution)), int(math.ceil(y1 / resolution))
    occupancy[max(0, iy0):min(occupancy.shape[0], iy1 + 1), max(0, ix0):min(occupancy.shape[1], ix1 + 1)] = True


def make_scene(family, variant, resolution=0.2):
    occupancy = np.zeros((71, 71), dtype=bool)
    shift = (variant - 4.5) * 0.16
    if family == "narrow_gate":
        add_rect(occupancy, resolution, 6.8, 7.2, 0.0, 6.55 + shift)
        add_rect(occupancy, resolution, 6.8, 7.2, 7.45 + shift, 10.0)
        add_rect(occupancy, resolution, 6.8, 7.2, 12.1, 14.0)
    elif family == "sharp_turn":
        add_rect(occupancy, resolution, 4.0 + shift, 10.5, 5.4, 6.4)
        add_rect(occupancy, resolution, 9.5, 10.5, 5.4, 11.2 - shift)
    elif family == "multi_route":
        add_rect(occupancy, resolution, 5.2, 8.8, 4.2 + shift, 9.8 + shift)
        add_rect(occupancy, resolution, 8.8, 10.2, 8.8 + shift, 9.8 + shift)
    elif family == "offset_gate":
        gap = 6.45 + variant * 0.20
        add_rect(occupancy, resolution, 5.8, 6.4, 0.0, gap)
        add_rect(occupancy, resolution, 5.8, 6.4, gap + 0.90, 14.0)
        add_rect(occupancy, resolution, 8.8, 9.4, 0.0, 5.0 + shift)
        add_rect(occupancy, resolution, 8.8, 9.4, 10.2 + shift, 14.0)
    elif family == "slalom":
        add_rect(occupancy, resolution, 4.0, 5.0, 0.0, 7.8 + shift)
        add_rect(occupancy, resolution, 7.0, 8.0, 6.2 - shift, 14.0)
        add_rect(occupancy, resolution, 10.0, 11.0, 0.0, 7.8 + shift)
    elif family == "island_chain":
        add_rect(occupancy, resolution, 4.0, 5.7, 6.0 + shift, 8.4 + shift)
        add_rect(occupancy, resolution, 7.0, 8.7, 4.2 - shift, 6.6 - shift)
        add_rect(occupancy, resolution, 9.5, 11.0, 7.2 + shift, 9.6 + shift)
    return occupancy


def fixtures(output):
    rows = []
    maps = output / "fixture_maps"
    maps.mkdir()
    resolution = 0.2
    profiles = {"diff": {"footprint_radius_m": 0.55, "max_heading_change_bins": 1, "turn_penalty": 0.30},
                "anymal": {"footprint_radius_m": 0.25, "max_heading_change_bins": 3, "turn_penalty": 0.05}}
    minimum_pair_mean_distance_m = 0.05
    families = ("narrow_gate", "sharp_turn", "multi_route", "offset_gate", "slalom", "island_chain")
    accepted_trajectory_pairs = []
    for family in families:
        for variant in range(10):
            occupancy = make_scene(family, variant, resolution)
            try:
                start = np.asarray([1.0, 7.0], np.float32)
                common_goal = np.asarray([13.0, 7.0 + (variant - 4.5) * 0.12], np.float32)
                full_diff = plan_grid(occupancy, start, common_goal, profiles["diff"], resolution)
                full_anymal = plan_grid(occupancy, start, common_goal, profiles["anymal"], resolution)
            except RuntimeError:
                continue
            trajectory_diff, trajectory_anymal = resample_prefix(full_diff), resample_prefix(full_anymal)
            pair_distance = np.linalg.norm(trajectory_diff[:, :2] - trajectory_anymal[:, :2], axis=1)
            pair_mean_distance_m = float(pair_distance.mean())
            if pair_mean_distance_m <= minimum_pair_mean_distance_m:
                continue
            trajectory_pair = np.concatenate([trajectory_diff[:, :2], trajectory_anymal[:, :2]], axis=1)
            if any(np.array_equal(trajectory_pair, accepted) for accepted in accepted_trajectory_pairs):
                continue
            accepted_trajectory_pairs.append(trajectory_pair.copy())
            map_path = maps / ("%s_%02d.npy" % (family, variant))
            fd, temporary = tempfile.mkstemp(prefix="fixture_map_", suffix=".npy")
            os.close(fd)
            try:
                np.save(temporary, occupancy.astype(np.uint8))
                publish_bytes(map_path, Path(temporary))
            finally:
                if Path(temporary).exists():
                    Path(temporary).unlink()
            trajectory_diff[:, :2] -= start
            trajectory_anymal[:, :2] -= start
            full_diff_local, full_anymal_local = full_diff - start, full_anymal - start
            pair_id = "fixture:%s:%02d" % (family, variant)
            rows.append({"pair_id": pair_id, "task_id": pair_id, "family": family,
                         "evidence_kind": "synthetic_fixture", "pair_valid": True,
                         "invalid_reason": None,
                         "common_start": [0.0, 0.0], "common_goal": (common_goal - start).tolist(),
                         "goal_diff": (common_goal - start).tolist(), "goal_anymal": (common_goal - start).tolist(),
                         "common_start_world": start.tolist(), "common_goal_world": common_goal.tolist(),
                         "goal_diff_world": common_goal.tolist(), "goal_anymal_world": common_goal.tolist(),
                         "common_map_reference": str(map_path), "map_resolution_m": resolution,
                         "context_diff": {"view": "common_occupancy_identity", "map_reference": str(map_path)},
                         "context_anymal": {"view": "common_occupancy_identity", "map_reference": str(map_path)},
                         "l_inv_scope": "structural_identity_only_common_context",
                         "platform_constraints": profiles, "oracle_planner": "shared_heading_lattice_astar_v1",
                         "full_route_diff": full_diff_local.tolist(), "full_route_anymal": full_anymal_local.tolist(),
                         "full_route_diff_world": full_diff.tolist(), "full_route_anymal_world": full_anymal.tolist(),
                         "full_route_length_diff_m": path_length(full_diff),
                         "full_route_length_anymal_m": path_length(full_anymal),
                         "trajectory_diff": trajectory_diff.tolist(), "trajectory_anymal": trajectory_anymal.tolist(),
                         "pair_mean_distance_m": pair_mean_distance_m,
                         "pair_max_distance_m": float(pair_distance.max()),
                         "pair_selection_rule": "mean_aligned_prefix_distance_gt_%.2fm" % minimum_pair_mean_distance_m,
                         "valid_mask_diff": [1] * 80, "valid_mask_anymal": [1] * 80,
                         "allowed_losses": ["L_diff", "L_inv", "L_swap", "L_sep"],
                         "supports_claims": {"pipeline_validation": True, "real_cross_embodiment_claim": False}})
    if not rows or len({row["family"] for row in rows}) < 4:
        raise RuntimeError("too few distinct non-identity fixture families")
    return rows


def write_jsonl(path, rows):
    publish_text(path, "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tartan-root", required=True)
    parser.add_argument("--anymal-train", required=True)
    parser.add_argument("--anymal-val", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)

    train = balanced_take(read_jsonl(args.anymal_train), 32, args.seed)
    val = balanced_take(read_jsonl(args.anymal_val), 8, args.seed)
    episodes = sorted([p.name for p in (Path(args.tartan_root) / "Data_diff").iterdir() if p.is_dir()],
                      key=lambda value: (rank(args.seed, value), value))
    if len(episodes) < 5:
        raise RuntimeError("expected five independent Diff episodes")
    split = {episode: ("train" if i < 3 else "val" if i == 3 else "heldout")
             for i, episode in enumerate(episodes)}
    diff_train_pool = sum((diff_rows(Path(args.tartan_root), ep, "train", args.seed)
                           for ep in episodes if split[ep] == "train"), [])
    diff_val_pool = sum((diff_rows(Path(args.tartan_root), ep, "val", args.seed)
                         for ep in episodes if split[ep] == "val"), [])
    diff_train = balanced_take(diff_train_pool, 32, args.seed)
    diff_val = balanced_take(diff_val_pool, 8, args.seed)
    real_train = [proxy_row(row, "anymal", 1) for row in train] + [proxy_row(row, "diff", 2) for row in diff_train]
    real_val = [proxy_row(row, "anymal", 1) for row in val] + [proxy_row(row, "diff", 2) for row in diff_val]
    write_jsonl(out / "real_unpaired_train.jsonl", real_train)
    write_jsonl(out / "real_unpaired_val.jsonl", real_val)
    fixture_rows = fixtures(out)
    write_jsonl(out / "synthetic_paired_fixture.jsonl", fixture_rows)
    registry = {"protocol": "D028/D029-v1.3.1-proxy", "checkpoint": args.checkpoint,
                "platforms": {"car": {"id": 0, "checkpoint_semantics": "existing"},
                              "anymal": {"id": 1, "checkpoint_semantics": "existing"},
                              "diff": {"id": 2, "checkpoint_semantics": "seed11_reinitialize_row_only"}},
                "unused_embedding_rows": [3], "ability_policy": "unknown_zero_with_zero_validity_mask"}
    publish_text(out / "platform_registry.json", json.dumps(registry, indent=2, sort_keys=True) + "\n")
    summary = {"status": "CPU_INPUTS_COMPLETE_GPU_PENDING", "diff_episode_split": split,
               "counts": {"real_train": len(real_train), "real_val": len(real_val), "fixture": len(fixture_rows)},
               "real_pair_valid_count": sum(bool(row["pair_valid"]) for row in real_train + real_val),
               "anymal_test_used": False, "checkpoint_exists": Path(args.checkpoint).exists()}
    publish_text(out / "input_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
