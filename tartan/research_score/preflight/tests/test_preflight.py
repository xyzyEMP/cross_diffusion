from pathlib import Path

from tartan.research_score.preflight.registry import run_checks
from unittest.mock import patch
import shutil


def config(tmp_path):
    ckpt = tmp_path / "m"; ckpt.write_bytes(b"x")
    return {"paths": {"project_root": str(tmp_path), "tartan_root": str(tmp_path), "nuplan_root": str(tmp_path), "source_args": str(ckpt), "source_checkpoint": str(ckpt), "output_root": str(tmp_path), "cf_pair_root": ""}, "protocol": {"condition_mode": "route_set", "max_candidates": 6, "num_points": 80, "length_selection_stage": "03"}, "claims": {"car_dog_scientific": False}}


def result_map(items): return {x.id: x for x in items}


def test_missing_path_fails(tmp_path):
    cfg = config(tmp_path); cfg["paths"]["tartan_root"] = str(tmp_path / "missing")
    assert result_map(run_checks(cfg, "transfer_primary", tmp_path / "new"))["path.tartan_root"].status == "FAIL"


def test_bad_protocol_fails(tmp_path):
    cfg = config(tmp_path); cfg["protocol"]["max_candidates"] = 7
    assert result_map(run_checks(cfg, "transfer_primary", tmp_path / "new"))["protocol.frozen_fields"].status == "FAIL"


def test_proxy_claim_fails(tmp_path):
    cfg = config(tmp_path); cfg["claims"]["car_dog_scientific"] = True
    assert result_map(run_checks(cfg, "proxy_pair_auxiliary", tmp_path / "new"))["claims.proxy_scope"].status == "FAIL"


def test_strict_without_pairs_fails(tmp_path):
    cfg = config(tmp_path)
    assert result_map(run_checks(cfg, "strict_pair", tmp_path / "new"))["strict_pair.data"].status == "FAIL"


def test_existing_output_fails(tmp_path):
    cfg = config(tmp_path); report = tmp_path / "used"; report.mkdir(); (report / "x").write_text("x")
    assert result_map(run_checks(cfg, "transfer_primary", report))["output.no_overwrite"].status == "FAIL"


def test_length_cannot_be_handfilled(tmp_path):
    cfg = config(tmp_path); cfg["protocol"]["length_m"] = 12
    assert result_map(run_checks(cfg, "transfer_primary", tmp_path / "new"))["protocol.length_pending_stage03"].status == "FAIL"


def test_sentinel_disk_capacity_is_unknown_warn(tmp_path):
    cfg = config(tmp_path)
    sentinel = shutil._ntuple_diskusage(2**63, 0, 2**63)
    with patch("tartan.research_score.preflight.registry.shutil.disk_usage", return_value=sentinel):
        result = result_map(run_checks(cfg, "transfer_primary", tmp_path / "new"))["disk.free"]
    assert result.status == "WARN"
    assert result.evidence["capacity_known"] is False
    assert result.evidence["free_bytes"] is None
