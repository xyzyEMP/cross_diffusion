from __future__ import annotations

import argparse
import csv
import json
import random
import os
import tempfile
from collections import Counter

import numpy as np
from pathlib import Path
from typing import Dict, List

import yaml

from pair.matching import CandidatePair, MiningConfig, mine_candidate_pairs
from pair.trajectory import LocalSegment, extract_local_segments, load_trajectory
from pair.visualization import visualize_pair


RESULT_FIELDS = [
    "pair_id",
    "map_id",
    "trajectory_a",
    "trajectory_b",
    "segment_a",
    "segment_b",
    "embodiment_a",
    "embodiment_b",
    "entry_a",
    "entry_b",
    "exit_a",
    "exit_b",
    "center_distance",
    "spatial_overlap",
    "entry_distance",
    "exit_distance",
    "mean_path_distance",
    "max_path_distance",
    "chamfer_distance",
    "path_length_a",
    "path_length_b",
    "path_length_ratio",
]


def _load_segments(
    data_root: Path, embodiment: str, config: MiningConfig
) -> List[LocalSegment]:
    sequences = sorted(
        path.parent
        for path in (data_root / f"Data_{embodiment}").glob("*/pose_lcam_front.txt")
    )
    if not sequences:
        raise FileNotFoundError(f"No {embodiment} trajectories found under {data_root}")
    segments = []
    for sequence in sequences:
        trajectory = load_trajectory(sequence, embodiment)
        segments.extend(
            extract_local_segments(
                trajectory, config.segment_length, config.segment_stride
            )
        )
    return segments


def _write_results(candidates: List[CandidatePair], output_dir: Path) -> None:
    records = [candidate.to_record() for candidate in candidates]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidates.json").write_text(json.dumps(records, indent=2))
    with (output_dir / "candidates.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, list) else value
                    for key, value in record.items()
                }
            )


def _write_visualizations(
    candidates: List[CandidatePair], output_dir: Path, config: MiningConfig
) -> None:
    count = min(config.visualization_count, len(candidates))
    selected = random.Random(config.random_seed).sample(candidates, count)
    for index, candidate in enumerate(selected):
        visualize_pair(candidate, output_dir / "visualizations" / f"{index:03d}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mine", "reconstruct"], default="mine")
    parser.add_argument("--trajectory-manifest", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--candidate-config", type=Path)
    parser.add_argument("--base-window-dir", type=Path, help="Approved native8m supplement from frozen base windows")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--embodiment-a", choices=["diff"], default="diff")
    parser.add_argument("--embodiment-b", choices=["omni"], default="omni")
    args = parser.parse_args()

    if args.mode == "reconstruct":
        required = ("trajectory_manifest", "candidate_manifest", "candidate_config", "output_root", "run_id")
        if any(getattr(args, key) is None for key in required):
            parser.error("reconstruct requires trajectory/candidate manifests, candidate config, output-root and run-id")
        reconstruct_pairs(args)
        return
    if args.data_root is None or args.output_dir is None:
        parser.error("mine requires data-root and output-dir")
    from tartan.research_score.artifacts import new_run_id
    output = args.output_dir / new_run_id("pair_mining")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(args.config.read_text())
    (output / "run.json").write_text(json.dumps({"data_root": str(args.data_root), "config": str(args.config), "embodiment_a": args.embodiment_a, "embodiment_b": args.embodiment_b, "output": str(output)}))

    values: Dict[str, object] = yaml.safe_load(args.config.read_text())
    config = MiningConfig.from_mapping({k: v for k, v in values.items() if k != "proxy_gate"})
    segments_a = _load_segments(args.data_root, args.embodiment_a, config)
    segments_b = _load_segments(args.data_root, args.embodiment_b, config)
    candidates = mine_candidate_pairs(segments_a, segments_b, config)
    _write_results(candidates, output)
    _write_visualizations(candidates, output, config)
    print(
        f"segments: {args.embodiment_a}={len(segments_a)}, "
        f"{args.embodiment_b}={len(segments_b)}; candidates={len(candidates)}"
    )
    print(f"results: {output}")



def _publish_text(text, path):
    """Resume only an identical complete artifact; never overwrite frozen inputs."""
    path = Path(path)
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"Existing artifact conflicts with reconstruction: {path}")
        return
    from tartan.research_score.artifacts import publish_text
    publish_text(text, path)


def _json_lines(rows):
    return "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)


def _pose_matrix(anchor):
    x, y, yaw = anchor
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0., 0., 1.]])


def _common_path(values, transform):
    values = np.asarray(values, float)
    return np.column_stack((values[:, :2] @ transform[:2, :2].T + transform[:2, 2],
                            values[:, 2:4] @ transform[:2, :2].T))


def reconstruct_pairs(args):
    from tartan.data.pose_utils import poses_to_se2,read_proxy_trajectories,load_proxy_se2
    from tartan.research_score.artifacts import validate_output_name
    from tartan.research_score.data.core import proxy_window
    from pair.matching import proxy_pair_metrics, proxy_gate_reasons

    validate_output_name(args.run_id)
    gate = yaml.safe_load(args.config.read_text()).get("proxy_gate")
    if not gate:
        raise ValueError("Explicit frozen proxy_gate configuration is required")
    expected_gate = {"length_m": 8., "num_points": 80, "arc_metric": "xy", "center_distance_max_m": 3., "entry_distance_lt_m": 1., "exit_distance_lt_m": 1., "mean_distance_lt_m": 1., "max_distance_lt_m": 2., "common_frame": "diff_observed_anchor", "full_valid_required": True, "same_split_required": True, "allow_icp": False, "allow_threshold_relaxation": False}
    if any(gate.get(key) != value for key, value in expected_gate.items()):
        raise ValueError("Proxy gate differs from approved frozen thresholds/frame policy")
    if gate.get("candidate_policy") == "legacy10m_plus_native8m" and args.base_window_dir is None:
        raise ValueError("Approved candidate policy requires --base-window-dir")
    mining_values = yaml.safe_load(args.candidate_config.read_text())
    mining_config = MiningConfig.from_mapping({k: v for k, v in mining_values.items() if k != "proxy_gate"})
    if mining_config.segment_length != 10 or mining_config.segment_stride != 2.5 or mining_config.resample_points != 64:
        raise ValueError("Candidate replay requires original 10m/2.5m/64 configuration")
    rows = read_proxy_trajectories(args.trajectory_manifest)
    index = {}
    for row in rows:
        key = (row["map_id"], row["embodiment"], row["trajectory_id"])
        if key in index:
            raise ValueError(f"Duplicate trajectory identity: {key}")
        index[key] = row
    candidates = json.loads(args.candidate_manifest.read_text())
    accepted = {"train": [], "val": []}
    windows = {"train": {}, "val": {}}
    rejected, seen, replay = [], set(), {}
    plot_rows = []
    for candidate in sorted(candidates, key=lambda row: row["pair_id"]):
        record = {"pair_id": candidate["pair_id"], "candidate": candidate,
                  "candidate_manifest": str(args.candidate_manifest), "candidate_config": str(args.candidate_config),
                  "trajectory_manifest": str(args.trajectory_manifest), "pair_level": "observational_matched",
                  "intervention": False, "counterfactual": False, "common_goal_id": None,
                  "pair_valid": False, "rejection_reasons": []}
        try:
            sides = []
            for side, expected in (("a", "diff"), ("b", "omni")):
                if candidate["embodiment_"+side] != expected:
                    raise ValueError("wrong_pair_platform")
                key = (candidate["map_id"], expected, candidate["trajectory_"+side])
                row = index.get(key)
                if row is None:
                    raise ValueError("trajectory_not_in_frozen_manifest")
                sides.append(row)
            record["source_splits"] = [row["split"] for row in sides]
            record["source_trajectory_keys"] = [row["trajectory_key"] for row in sides]
            if sides[0]["split"] != sides[1]["split"]:
                raise ValueError("cross_split")
            split = sides[0]["split"]
            if split not in ("train", "val"):
                raise ValueError("pair_split_not_train_or_val")
            record["split"] = split
            record["map_id"] = candidate["map_id"]
            rebuilt, anchors, old_segments = [], [], []
            for side, row in zip(("a", "b"), sides):
                pose_path = Path(row["pose_path"])
                key = str(pose_path)
                if key not in replay:
                    trajectory = load_trajectory(pose_path.parent, row["embodiment"])
                    se2 = load_proxy_se2(row)
                    segments = extract_local_segments(trajectory, mining_config.segment_length, mining_config.segment_stride)
                    replay[key] = (se2, {segment.segment_id: segment for segment in segments})
                se2, segments = replay[key]
                old = segments.get(candidate["segment_"+side])
                if old is None:
                    raise ValueError("candidate_segment_not_replayable")
                if not np.allclose(old.entry_position, candidate["entry_"+side], atol=1e-5, rtol=0):
                    raise ValueError("candidate_entry_replay_mismatch")
                anchor = old.start_frame
                gates = row.get("gates", {})
                if not gates.get("reference_heading",gates.get("body_heading",False)) or not gates.get("occupancy_frame"):
                    record["source_gate_evidence_"+side] = {"gates": gates, "reasons": row.get("gate_reasons", []), "frame_convention": row.get("frame_convention"), "body_heading_source": row.get("body_heading_source"), "pose_path": row.get("pose_path"), "metadata_path": row.get("metadata_path"), "frame_evidence": row.get("frame_evidence")}
                    raise ValueError("unverified_reference_or_occupancy_frame")
                window = proxy_window(row, se2, anchor, args.trajectory_manifest)
                if window is None:
                    raise ValueError("anchor_history_or_frame_gate")
                rebuilt.append(window)
                anchors.append(se2[anchor])
                old_segments.append(old)
                record[side] = {"trajectory_key": row["trajectory_key"], "sample_id": window["sample_id"],
                                "segment_id": old.segment_id, "segment_index": int(old.segment_id.rsplit(":", 1)[1]),
                                "anchor_frame": anchor, "old_raw_frame_range": [old.start_frame, old.end_frame],
                                "old_xyz_arc_start_m": int(old.segment_id.rsplit(":", 1)[1])*mining_config.segment_stride,
                                "old_entry_to_observed_anchor_m": float(np.linalg.norm(old.entry_position[:2]*np.array([1.,-1.] if row.get("world_frame_policy")=="ned_to_nwu" else [1.,1.])-se2[anchor, :2])),
                                "raw_reference": str(pose_path), "rebuilt_raw_frame_range": [window["trajectory"]["source_start_frame"], window["trajectory"]["source_end_frame"]],
                                "rebuilt_measured_arc_m": window["trajectory"]["measured_arc_m"],
                                "fixed_goal": window["fixed_goal"], "time_source": row.get("time_source"),
                                "body_heading_source": row.get("body_heading_source"), "reference_pose_policy":row.get("reference_pose_policy"), "reference_heading_source":row.get("reference_heading_source"), "frame_convention": row.get("frame_convention")}
            transforms = [np.eye(3), np.linalg.inv(_pose_matrix(anchors[0])) @ _pose_matrix(anchors[1])]
            paths = [_common_path(window["trajectory"]["fixed_arc_length_80"], transform) for window, transform in zip(rebuilt, transforms)]
            for window, transform, path in zip(rebuilt, transforms, paths):
                original = np.asarray(window["trajectory"]["fixed_arc_length_80"])
                if not np.allclose(_common_path(path, np.linalg.inv(transform)), original, atol=1e-4, rtol=0):
                    raise ValueError("se2_roundtrip_error")
            for window in rebuilt:
                if np.asarray(window["trajectory"]["valid_mask"]).shape != (80,) or not np.asarray(window["trajectory"]["valid_mask"]).all():
                    raise ValueError("incomplete_8m_or_mask")
            metrics = proxy_pair_metrics(*paths)
            record.update(T_a=transforms[0].tolist(), T_b=transforms[1].tolist(), metrics_8m=metrics,
                          common_origin_world=anchors[0][:2].tolist(), common_yaw_world=float(anchors[0][2]),
                          common_frame_source="Diff observed reference anchor from frozen pose/frame evidence",
                          goals_common=[path[-1, :2].tolist() for path in paths], representation={"length_m": 8, "points": 80, "arc": "XY"})
            reasons = proxy_gate_reasons(metrics)
            identity = tuple(window["sample_id"] for window in rebuilt)
            if identity in seen:
                reasons.append("duplicate_pair")
            seen.add(identity)
            record["rejection_reasons"] = reasons
            if reasons:
                rejected.append(record)
                plot_rows.append((record, paths))
                continue
            record["pair_valid"] = True
            accepted[split].append(record)
            for window in rebuilt:
                windows[split][window["sample_id"]] = window
            plot_rows.append((record, paths))
        except (ValueError, KeyError, OSError) as error:
            record["rejection_reasons"].append(str(error))
            rejected.append(record)
    supplement = {}
    if args.base_window_dir is not None:
        supplement = supplement_native_pairs(args, index, accepted, windows, gate)
    output = args.output_root / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    previous=output/'gate_summary.json'
    if previous.exists() and json.loads(previous.read_text())['status']=='BLOCKED_NO_TRAIN_PAIRS' and (accepted['train'] or (args.trajectory_manifest.parent/'trajectory_evidence.json').exists()):
        import shutil
        archive=args.output_root.parent/'runs'/args.run_id/'blocked_pair_evidence'
        archive.mkdir(parents=True,exist_ok=True)
        for name in ('train.jsonl','val.jsonl','train_windows.jsonl','val_windows.jsonl','rejected.jsonl','gate_summary.json'):
            source=output/name
            if source.exists():
                destination=archive/name
                if destination.exists():raise FileExistsError(destination)
                shutil.move(str(source),str(destination))
    for split in ("train", "val"):
        _publish_text(_json_lines(accepted[split]), output / f"{split}.jsonl")
        _publish_text(_json_lines(sorted(windows[split].values(), key=lambda row: row["sample_id"])), output / f"{split}_windows.jsonl")
    _publish_text(_json_lines(rejected), output / "rejected.jsonl")
    summary = {"status": "PAIR_READY" if accepted["train"] else "BLOCKED_NO_TRAIN_PAIRS",
               "input_count": len(candidates)+sum(supplement.get("added", {}).values()),
               "legacy_input_count": len(candidates), "accepted": {s: len(v) for s, v in accepted.items()},
               "rejected_count": len(rejected), "rejection_reasons": dict(Counter(reason for row in rejected for reason in row["rejection_reasons"])),
               "unique_windows": {s: len(v) for s, v in windows.items()},
               "window_reuse_count": {s: 2*len(accepted[s])-len(windows[s]) for s in accepted},
               "input_split_combinations": dict(Counter("/".join(row.get("source_splits", ["unresolved"])) for row in rejected+accepted["train"]+accepted["val"])),
               "trajectory_pairs": {s: dict(Counter(row["a"]["trajectory_key"]+"__"+row["b"]["trajectory_key"] for row in values)) for s, values in accepted.items()},
               "unique_trajectories": {s: len({row[side]["trajectory_key"] for row in values for side in ("a", "b")}) for s, values in accepted.items()},
               "gate": gate, "candidate_config": mining_values, "run_id": args.run_id,
               "sources": {"candidate_manifest": str(args.candidate_manifest), "trajectory_manifest": str(args.trajectory_manifest)},
               "native_supplement": supplement}
    _publish_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False)+"\n", output / "gate_summary.json")
    from pair.visualization import visualize_reconstructed_pair
    selected = []
    for split in ("train", "val"):
        selected.extend([item for item in plot_rows if item[0].get("split") == split and item[0]["pair_valid"]][:2])
    near = sorted([item for item in plot_rows if not item[0]["pair_valid"]], key=lambda item: abs(item[0]["metrics_8m"]["entry_distance"]-1))[:2]
    for position, (record, paths) in enumerate((selected+near)[:6]):
        visualize_reconstructed_pair(record, paths, output / "selected_plots" / f"pair_{position:02d}.png")
    print(json.dumps(summary, indent=2))


def supplement_native_pairs(args, index, accepted, windows, gate):
    """Union approved real base-window pairs with rebuilt legacy pairs by side IDs."""
    from scipy.spatial import cKDTree
    from pair.matching import proxy_pair_metrics, proxy_gate_reasons
    approval = json.loads(Path(gate["supplement_approval_record"]).read_text())
    if approval.get("status") != "APPROVED" or gate.get("candidate_policy") != "legacy10m_plus_native8m":
        raise ValueError("Native supplement requires explicit approval")
    if args.base_window_dir.resolve() != args.trajectory_manifest.parent.resolve():
        raise ValueError("Native windows must use the same frozen trajectory manifest")
    added = {}
    for split in ("train", "val"):
        source = args.base_window_dir / f"base_{split}.jsonl"
        base = [json.loads(line) for line in source.read_text().splitlines() if line]
        for row in base:
            trajectory = index[(row["map_id"], row["embodiment"], row["trajectory_id"])]
            if trajectory["split"] != split or row["split"] != split or row["embodiment"] not in ("diff", "omni"):
                raise ValueError("Native supplement split/platform mismatch")
            if row["anchor_index"] % 10 or row["anchor_index"] < 20:
                raise ValueError("Native supplement must use frozen10-frame grid")
        a = [w for w in base if w["embodiment"] == "diff" and all(w["trajectory"]["valid_mask"])]
        b = [w for w in base if w["embodiment"] == "omni" and all(w["trajectory"]["valid_mask"])]
        tree = cKDTree([w["current_state"]["anchor_world_se2"][:2] for w in b])
        seen = {(r["a"]["sample_id"], r["b"]["sample_id"]) for r in accepted[split]}
        added[split] = 0
        for wa in a:
            inv = np.linalg.inv(_pose_matrix(wa["current_state"]["anchor_world_se2"]))
            for j in sorted(tree.query_ball_point(wa["current_state"]["anchor_world_se2"][:2], 1.)):
                wb = b[j]
                if wa["map_id"] != wb["map_id"] or (wa["sample_id"], wb["sample_id"]) in seen:
                    continue
                transform = inv @ _pose_matrix(wb["current_state"]["anchor_world_se2"])
                paths = [np.asarray(wa["trajectory"]["fixed_arc_length_80"]), _common_path(wb["trajectory"]["fixed_arc_length_80"], transform)]
                metrics = proxy_pair_metrics(*paths)
                if proxy_gate_reasons(metrics):
                    continue
                record = {"pair_id": "native8m:"+wa["sample_id"]+"__"+wb["sample_id"],
                          "split": split, "map_id": wa["map_id"], "pair_valid": True,
                          "candidate_kind": "native8m_frozen_base_window", "candidate_manifest": str(source),
                          "trajectory_manifest": str(args.trajectory_manifest), "pair_level": "observational_matched",
                          "intervention": False, "counterfactual": False, "common_goal_id": None,
                          "source_splits": [split, split], "source_trajectory_keys": [wa["trajectory_key"], wb["trajectory_key"]],
                          "rejection_reasons": [], "T_a": np.eye(3).tolist(), "T_b": transform.tolist(),
                          "metrics_8m": metrics, "common_origin_world": wa["current_state"]["anchor_world_se2"][:2],
                          "common_yaw_world": wa["current_state"]["anchor_world_se2"][2],
                          "common_frame_source": "Diff observed reference anchor from frozen pose/frame evidence",
                          "representation": {"length_m": 8, "points": 80, "arc": "XY"},
                          "goals_common": [p[-1, :2].tolist() for p in paths]}
                for side, window in (("a", wa), ("b", wb)):
                    record[side] = {k: window[k] for k in ("sample_id", "trajectory_key", "time_source", "body_heading_source", "reference_pose_policy", "reference_heading_source", "frame_convention")}
                    record[side].update(anchor_frame=window["anchor_index"], rebuilt_raw_frame_range=[window["trajectory"]["source_start_frame"],window["trajectory"]["source_end_frame"]], raw_reference=window["trajectory"]["raw_reference"], fixed_goal=window["fixed_goal"])
                    windows[split][window["sample_id"]] = window
                accepted[split].append(record)
                seen.add((wa["sample_id"], wb["sample_id"]))
                added[split] += 1
        accepted[split].sort(key=lambda r: r["pair_id"])
    if {s: len(v) for s, v in accepted.items()} != approval["approved_pairs"]:
        raise ValueError("Supplement differs from approved218/20 diagnostic")
    return {"policy": "legacy10m_plus_native8m", "added": added, "approval_record": gate["supplement_approval_record"], "anchor_stride_frames": 10}


if __name__ == "__main__":
    main()
