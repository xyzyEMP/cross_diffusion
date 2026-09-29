from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from tartan.research_score.data.budget_sampler import nested_window_budgets


def _publish_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(prefix="d029_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary_path = Path(temporary)
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    source = Path(args.input)
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    result = nested_window_budgets(rows, seed=args.seed)
    for row in rows:
        old = row.get("budget_membership")
        if old:
            row["legacy_budget_membership_v124"] = old
        row["budget_membership"] = {str(args.seed): result["memberships"][str(row["sample_id"])]}
        row.setdefault("provenance", {})["budget_protocol"] = "D029/v1.3.1-proxy"

    manifest_text = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    summary = {key: value for key, value in result.items() if key not in {"budgets", "memberships"}}
    summary.update({"status": "CPU_COMPLETE_GPU_TRAINING_PENDING", "input_manifest": str(source),
                    "output_manifest": str(Path(args.output)),
                    "split_rule": "complete_episode_16_3_5_then_train_window_budget",
                    "validation_and_test_unchanged": True})
    _publish_text(Path(args.output), manifest_text)
    _publish_text(Path(args.summary), json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
