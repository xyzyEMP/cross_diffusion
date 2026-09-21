"""Validate Stage-03 v1.2.1 artifacts against the frozen scientific contract."""
import argparse, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path


REV = "score-decomp-transfer-v1.2.1"


def load_json(path):
    return json.loads(path.read_text())


def load_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run, output = Path(args.run), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    checks, failures = [], []

    def check(name, condition, detail):
        row = {"name": name, "passed": bool(condition), "detail": detail}
        checks.append(row)
        if not condition:
            failures.append(row)

    manifest = load_json(run / "run_manifest.json")
    selected = load_json(run / "selected_length.json")
    profile = load_json(run / "transfer_primary/profile_manifest.json")
    proxy = load_json(run / "proxy_pair_auxiliary/profile_manifest.json")
    strict = load_json(run / "strict_pair/profile_manifest.json")
    split_summary = load_json(run / "split_summary.json")
    rows = []
    by_split = {}
    for split in ("train", "val", "test"):
        by_split[split] = load_jsonl(run / f"transfer_primary/target_{split}_windows.jsonl")
        rows.extend(by_split[split])

    check("protocol_revision", all(x.get("protocol_revision") == REV for x in (manifest, selected, profile, proxy, strict)), REV)
    check("selected_hash", profile["selected_length_sha256"] == sha256(run / "selected_length.json") == manifest["selected_length_sha256"], profile["selected_length_sha256"])
    check("selection_formal_branches", selected["formal_branches"] == ["source_car", "target_anymal"] and selected["proxy_excluded"] is True, selected["formal_branches"])
    eligible = [float(k) for k, v in selected["coverage"].items() if min(v.values()) >= selected["min_train_coverage"]]
    check("largest_eligible_length", bool(eligible) and selected["selected_length_m"] == max(eligible), {"eligible": eligible, "selected": selected["selected_length_m"]})
    check("coverage_threshold", min(selected["coverage"][str(int(selected["selected_length_m"]))].values()) >= .90, selected["coverage"])
    check("claim_scope", profile["claim_scope"] == "same-map unseen-episode transfer" and profile["unseen_map_claim"] is False, profile["claim_scope"])
    check("proxy_diagnostic_only", proxy["included_in_formal_training"] is False and proxy["included_in_length_selection"] is False and proxy["included_in_main_table"] is False, proxy["status"])
    check("strict_blocked", strict["status"] == "BLOCKED_WAITING_DATA", strict["status"])

    episode_sets = {s: {r["episode_id"] for r in rs} for s, rs in by_split.items()}
    overlap = {f"{a}_{b}": sorted(episode_sets[a] & episode_sets[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    check("episode_disjoint", not any(overlap.values()), overlap)
    sample_ids = [r["sample_id"] for r in rows]
    check("sample_unique", len(sample_ids) == len(set(sample_ids)), {"rows": len(rows), "unique": len(set(sample_ids))})
    check("split_fields_match_files", all(r["split"] == split for split, rs in by_split.items() for r in rs), {s: len(v) for s, v in by_split.items()})
    check("all_rows_retained_by_branch", all(r["branch"] in {"moving_planning", "stop_or_short"} for r in rows), Counter(r["branch"] for r in rows))
    check("fixed_goal_is_explicit", all(len(r["fixed_goal"]["xy_local"]) == 2 and "frozen" in r["fixed_goal"]["rule"] for r in rows), len(rows))
    check("route_has_no_future_dependency", all(r["route_set"]["future_gt_dependency"] is False for r in rows), len(rows))
    required = {"sample_id", "episode_id", "map_id", "split", "budget_membership", "anchor_index", "current_state", "fixed_goal", "route_set", "trajectory", "provenance"}
    missing = Counter(field for r in rows for field in required - set(r))
    check("canonical_schema", not missing, dict(missing))

    budget_failures = []
    for seed, groups in split_summary["budgets"].items():
        one, ten, full = map(set, (groups["1"], groups["10"], groups["100"]))
        if not one <= ten <= full:
            budget_failures.append(seed)
    check("nested_episode_budgets", not budget_failures, budget_failures)
    for row in rows:
        check_key = row["trajectory"]
        if not {"values_ref", "mask_ref", "row", "representation", "raw_reference", "raw_points", "timestamps_s"} <= set(check_key):
            failures.append({"name": "trajectory_schema", "passed": False, "detail": row["sample_id"]})
            break

    result = {"protocol_revision": REV, "status": "PASS" if not failures else "FAIL", "checks_passed": sum(x["passed"] for x in checks), "checks_failed": len(failures), "target_rows": len(rows), "split_rows": {s: len(v) for s, v in by_split.items()}, "branch_rows": dict(Counter(r["branch"] for r in rows)), "checks": checks, "failures": failures}
    (output / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "checks_passed", "checks_failed", "target_rows", "split_rows", "branch_rows")}, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
