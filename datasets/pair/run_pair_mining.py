from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import yaml

from datasets.pair.matching import CandidatePair, MiningConfig, mine_candidate_pairs
from datasets.pair.trajectory import LocalSegment, extract_local_segments, load_trajectory
from utils.visualization.pair import visualize_pair


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
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embodiment-a", default="anymal")
    parser.add_argument("--embodiment-b", default="diff")
    args = parser.parse_args()

    values: Dict[str, object] = yaml.safe_load(args.config.read_text())
    config = MiningConfig.from_mapping(values)
    segments_a = _load_segments(args.data_root, args.embodiment_a, config)
    segments_b = _load_segments(args.data_root, args.embodiment_b, config)
    candidates = mine_candidate_pairs(segments_a, segments_b, config)
    _write_results(candidates, args.output_dir)
    _write_visualizations(candidates, args.output_dir, config)
    print(
        f"segments: {args.embodiment_a}={len(segments_a)}, "
        f"{args.embodiment_b}={len(segments_b)}; candidates={len(candidates)}"
    )
    print(f"results: {args.output_dir}")


if __name__ == "__main__":
    main()
