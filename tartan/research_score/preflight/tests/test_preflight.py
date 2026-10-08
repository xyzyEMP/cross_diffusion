from pathlib import Path

from tartan.research_score.preflight.registry import run_checks
from unittest.mock import patch
import shutil


def config(tmp_path):
    ckpt = tmp_path / "m"; ckpt.write_bytes(b"x")
    keys = ("project_root", "tartan_root", "source_args", "source_checkpoint", "output_root", "train_cache", "val_cache", "source_cache", "val_navigation_manifest", "test_navigation_manifest")
    return {"paths": {key: str(tmp_path if key.endswith("root") else ckpt) for key in keys}, "trajectory": {"length_m": 8, "points": 80}, "fairness": {"future_gt_as_input": False}}



def result_map(items): return {x.id: x for x in items}


def test_missing_path_fails(tmp_path):
    cfg = config(tmp_path); cfg["paths"]["tartan_root"] = str(tmp_path / "missing")
    assert result_map(run_checks(cfg, "transfer_primary", tmp_path / "new"))["path.tartan_root"].status == "FAIL"


def test_bad_protocol_fails(tmp_path):
    cfg = config(tmp_path); cfg["trajectory"]["points"] = 79
    assert result_map(run_checks(cfg, "transfer_primary", tmp_path / "new"))["protocol.frozen_fields"].status == "FAIL"


def test_unimplemented_profile_fails(tmp_path):
    cfg = config(tmp_path)
    assert result_map(run_checks(cfg, "proposed_b", tmp_path / "new"))["profile"].status == "FAIL"


def test_existing_output_fails(tmp_path):
    cfg = config(tmp_path); report = tmp_path / "used"; report.mkdir(); (report / "x").write_text("x")
    assert result_map(run_checks(cfg, "transfer_primary", report))["output.no_overwrite"].status == "FAIL"


def test_future_input_rejected(tmp_path):
    cfg = config(tmp_path); cfg["fairness"]["future_gt_as_input"] = True
    assert result_map(run_checks(cfg, "transfer_primary", tmp_path / "new"))["protocol.no_future_input"].status == "FAIL"


def test_sentinel_disk_capacity_is_unknown_warn(tmp_path):
    cfg = config(tmp_path)
    sentinel = shutil._ntuple_diskusage(2**63, 0, 2**63)
    with patch("tartan.research_score.preflight.registry.shutil.disk_usage", return_value=sentinel):
        result = result_map(run_checks(cfg, "transfer_primary", tmp_path / "new"))["disk.free"]
    assert result.status == "WARN"
    assert result.evidence["capacity_known"] is False
    assert result.evidence["free_bytes"] is None
