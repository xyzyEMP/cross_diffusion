from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from .registry import run_checks


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _resolve(value):
    if isinstance(value, dict):
        return {k: _resolve(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v) for v in value]
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        name = value[2:-1]
        if name not in os.environ:
            raise RuntimeError(f"Missing environment variable: {name}")
        return os.environ[name]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = _resolve(yaml.safe_load(config_path.read_text()))
    config["stage"] = str(args.stage)
    config["profile"] = args.profile
    config["run_id"] = args.run_id
    report = Path(args.report)
    checks = run_checks(config, args.profile, report)
    report.mkdir(parents=True, exist_ok=False)
    resolved = yaml.safe_dump(config, sort_keys=True)
    _write_text_atomic(report / "config.resolved.yaml", resolved)
    digest = hashlib.sha256(resolved.encode()).hexdigest()
    _write_text_atomic(report / "config.sha256", digest + "\n")
    env = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda_device_count": torch.cuda.device_count()}
    _write_text_atomic(report / "environment.json", json.dumps(env, indent=2))
    usage = __import__("shutil").disk_usage(config["paths"]["output_root"])
    sentinel = usage.total >= 2**62 or usage.free >= 2**62
    _write_text_atomic(report / "resource_estimate.json", json.dumps({"stage": str(args.stage), "minimum_free_bytes": 5 * 1024**3, "capacity_known": not sentinel, "actual_free_bytes": None if sentinel else usage.free, "raw_reported_free_bytes": usage.free, "gpu_required_for_completion": str(args.stage) not in {"03"}}, indent=2))
    failed = [c for c in checks if c.status == "FAIL"]
    payload = {"created_at": datetime.now(timezone.utc).isoformat(), "run_id": args.run_id, "overall": "FAIL" if any(c.severity == "P0" for c in failed) else "PASS", "checks": [c.to_dict() for c in checks]}
    _write_text_atomic(report / "report.json", json.dumps(payload, indent=2))
    p0 = [c.id for c in failed if c.severity == "P0"]
    p1 = [c.id for c in failed if c.severity == "P1"]
    md = [f"# Preflight Stage {args.stage}", "", f"- Overall: **{payload['overall']}**", f"- Profile: `{args.profile}`", f"- Run ID: `{args.run_id}`", f"- P0: `{p0}`", f"- P1: `{p1}`", f"- Paths: `{config['paths']}`", f"- May continue: **{'no' if p0 else 'yes'}**", "", "## Checks", "", "| ID | Severity | Status | Remediation |", "|---|---|---|---|"]
    md += [f"| {c.id} | {c.severity} | {c.status} | {c.remediation} |" for c in checks]
    _write_text_atomic(report / "report.md", "\n".join(md) + "\n")
    raise SystemExit(2 if p0 else 0)
