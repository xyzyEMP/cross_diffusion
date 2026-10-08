import pytest

from tartan.research_score.data.budget_sampler import nested_window_budgets
from tartan.research_score.data.core import assert_no_split_leak




def test_d029_exact_nested_episode_coverage_and_order_invariance():
    rows = [
        {"sample_id": f"e{episode}:{i:03d}", "episode_id": f"e{episode}",
         "branch": "moving_planning" if i % 3 else "stop_or_short", "split": "train"}
        for episode in range(16) for i in range(128)
    ]
    result = nested_window_budgets(rows, seed=11)
    assert result["targets"] == {"1": 20, "10": 205, "100": 2048}
    one, ten, full = (set(result["budgets"][tag]) for tag in ("1", "10", "100"))
    assert one < ten < full
    assert len({sid.split(":")[0] for sid in one}) == 16
    assert result["budgets"] == nested_window_budgets(list(reversed(rows)), seed=11)["budgets"]


def test_d029_rejects_cross_split_rows():
    rows = [
        {"sample_id": "a", "episode_id": "e1", "branch": "moving_planning", "split": "train"},
        {"sample_id": "b", "episode_id": "e2", "branch": "moving_planning", "split": "val"},
    ]
    with pytest.raises(ValueError, match="train split"):
        nested_window_budgets(rows)


def test_no_leak():
    assert_no_split_leak([{"map_id": "a", "trajectory_id": "t", "split": "train"}])


def test_budget_builder_publishes_matching_cache_and_manifest(tmp_path, monkeypatch):
    import json, sys, torch
    from tartan.research_score.scripts.build_training_budgets import main
    rows = [{"sample_id": f"e{e}:{i:03d}", "episode_id": f"e{e}", "branch": "moving_planning", "split": "train"} for e in range(16) for i in range(128)]
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("".join(json.dumps(r)+"\n" for r in rows))
    cache = tmp_path / "features.pt"
    torch.save({"sample_ids": [r["sample_id"] for r in rows], "trajectory": torch.zeros(2048, 1, 4)}, cache)
    output = tmp_path / "budgets"
    monkeypatch.setattr(sys, "argv", ["build_training_budgets", "--manifest", str(manifest), "--cache", str(cache), "--output", str(output)])
    main()
    published = torch.load(output / "target_train_features_d029_extended.pt", weights_only=False)
    written = [json.loads(line) for line in (output / "target_train_windows_d029_extended.jsonl").read_text().splitlines()]
    assert published["sample_ids"] == [r["sample_id"] for r in written]
    assert published["budget_membership"] == [r["budget_membership"] for r in written]
    assert {tag: sum(tag in m["11"] for m in published["budget_membership"]) for tag in ["1", "10", "20", "50", "100"]} == {"1": 20, "10": 205, "20": 410, "50": 1024, "100": 2048}


def test_proxy_manifest_validator_blocks_missing_evidence_and_split_tampering():
    from tartan.research_score.data.core import proxy_split
    from tartan.research_score.scripts.validate_transfer_manifests import proxy_checks
    identities = [{'trajectory_key':f'm/{platform}/{i}', 'map_id':'m', 'embodiment':platform,
                   'trajectory_id':str(i), 'pose_count':40, 'pose_path':'real_pose_reference',
                   'metadata_path':'real_metadata_reference', 'occupancy_dir':'real_map_reference',
                   'gates':{'time':False,'body_heading':False,'occupancy_frame':False},
                   'gate_reasons':['unverified'], 'sample_rate_hz':None}
                  for platform, count in [('diff',2),('omni',2),('anymal',1)] for i in range(count)]
    frozen = proxy_split(identities)
    empty = {name:[] for name in ['base_train','base_val','anymal_test']}
    checks = proxy_checks(frozen, empty)
    assert checks['status'] == 'BLOCKED_FRAME'
    assert len(checks['uncovered_trajectories']) == 5
    assert next(c for c in checks['checks'] if c['name']=='frozen_full_trajectory_split')['passed']
    anymal = next(r for r in frozen if r['embodiment']=='anymal')
    anymal['split'] = 'train'
    checks = proxy_checks(frozen, empty)
    assert not next(c for c in checks['checks'] if c['name']=='frozen_full_trajectory_split')['passed']
