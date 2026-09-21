from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Dict, List

import torch

from .result import CheckResult


PROFILES = {"transfer_primary", "proxy_pair_auxiliary", "strict_pair"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_checks(config: Dict, profile: str, report_dir: Path) -> List[CheckResult]:
    results: List[CheckResult] = []
    results.append(CheckResult("profile", "P0", "PASS" if profile in PROFILES else "FAIL", {"profile": profile}, "Use one frozen real profile."))
    paths = config.get("paths", {})
    for key in ("project_root", "tartan_root", "nuplan_root", "source_args", "source_checkpoint", "output_root"):
        value = paths.get(key, "")
        path = Path(value) if value else None
        ok = bool(path and path.exists())
        results.append(CheckResult(f"path.{key}", "P0", "PASS" if ok else "FAIL", {"path": value}, "Set the path in local.env."))
    output_root = Path(paths.get("output_root", "/nonexistent"))
    writable = output_root.exists() and os.access(output_root, os.W_OK)
    results.append(CheckResult("output.writable", "P0", "PASS" if writable else "FAIL", {"path": str(output_root)}, "Create a writable output root."))
    usage = shutil.disk_usage(output_root) if output_root.exists() else None
    free = usage.free if usage else 0
    sentinel = bool(usage and (usage.total >= 2**62 or usage.free >= 2**62))
    disk_status = "WARN" if sentinel else ("PASS" if free >= 5 * 1024**3 else "WARN")
    evidence = {"capacity_known": not sentinel, "free_bytes": None if sentinel else free, "raw_free_bytes": free}
    results.append(CheckResult("disk.free", "P1", disk_status, evidence, "Shared filesystem reports a sentinel capacity; verify quota with the storage service." if sentinel else "Keep at least 5 GiB free for Stage 02."))
    cuda = torch.cuda.is_available()
    results.append(CheckResult("cuda.available", "P0", "PASS" if cuda else "FAIL", {"torch": torch.__version__, "cuda_available": cuda, "device_count": torch.cuda.device_count()}, "Attach at least one CUDA GPU; CPU substitution is forbidden."))
    protocol = config.get("protocol", {})
    frozen = protocol.get("condition_mode") == "route_set" and protocol.get("max_candidates") == 6 and protocol.get("num_points") == 80
    results.append(CheckResult("protocol.frozen_fields", "P0", "PASS" if frozen else "FAIL", protocol, "Keep route_set/6 candidates/80 points."))
    no_length = "length_m" not in protocol and protocol.get("length_selection_stage") == "03"
    results.append(CheckResult("protocol.length_pending_stage03", "P0", "PASS" if no_length else "FAIL", {"length_m_present": "length_m" in protocol, "selection_stage": protocol.get("length_selection_stage")}, "Do not select physical length before Stage 03."))
    if str(config.get("stage")) == "03":
        trajectory = config.get("trajectory", {})
        route = config.get("route", {})
        length_rule_ok = (trajectory.get("length_candidates_m") == [8, 10, 12, 15, 20]
                          and float(trajectory.get("min_train_coverage", 0)) == 0.90
                          and trajectory.get("length_m") == "derived_from_train_audit")
        results.append(CheckResult("stage03.length_rule", "P0", "PASS" if length_rule_ok else "FAIL", trajectory, "Use train-only frozen length-selection rule."))
        route_ok = route.get("source") == "map_goal" and route.get("max_candidates") == 6
        results.append(CheckResult("stage03.route_no_future", "P0", "PASS" if route_ok else "FAIL", route, "Non-oracle route must be map_goal with <=6 candidates."))
        budgets = config.get("budgets", {})
        budget_ok = budgets.get("percent") == [1, 10, 100] and budgets.get("nested") is True and len(budgets.get("joint_seeds", [])) == 5
        results.append(CheckResult("stage03.budgets", "P0", "PASS" if budget_ok else "FAIL", budgets, "Freeze nested 1/10/100 budgets and 5 joint seeds."))
    claim_ok = profile != "proxy_pair_auxiliary" or config.get("claims", {}).get("car_dog_scientific", False) is False
    results.append(CheckResult("claims.proxy_scope", "P0", "PASS" if claim_ok else "FAIL", config.get("claims", {}), "Proxy cannot support Car-Dog scientific claims."))
    if profile == "strict_pair":
        root = paths.get("cf_pair_root", "")
        results.append(CheckResult("strict_pair.data", "P0", "PASS" if root and Path(root).exists() else "FAIL", {"cf_pair_root": root}, "Provide audited strict pairs."))
    ckpt = Path(paths.get("source_checkpoint", "/nonexistent"))
    if ckpt.is_file():
        results.append(CheckResult("checkpoint.sha256", "P0", "PASS", {"sha256": file_sha256(ckpt), "size_bytes": ckpt.stat().st_size}))
    report_target = report_dir.resolve()
    collision = report_target.exists() and any(report_target.iterdir())
    results.append(CheckResult("output.no_overwrite", "P0", "FAIL" if collision else "PASS", {"report": str(report_target), "already_nonempty": collision}, "Use a new immutable run_id."))
    return results
