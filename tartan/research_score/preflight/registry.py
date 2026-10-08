"""Read-only checks for the current transfer training inputs."""
import os
import shutil
from pathlib import Path
import torch
from .result import CheckResult


def run_checks(config, profile, report_dir):
    if profile == "proxy_ab":return proxy_checks(config, report_dir)
    results = []
    def check(name, ok, evidence, remedy):
        results.append(CheckResult(name, "P0", "PASS" if ok else "FAIL", evidence, remedy))
    check("profile", profile == "transfer_primary", {"profile": profile}, "Only the implemented transfer pipeline can run.")
    paths = config.get("paths", {})
    for key in ("project_root", "tartan_root", "source_args", "source_checkpoint", "output_root", "train_cache", "val_cache", "source_cache", "val_navigation_manifest", "test_navigation_manifest"):
        value = paths.get(key, "")
        path = Path(value) if value else None
        check("path." + key, bool(path and path.exists()), {"path": value}, "Set the actual frozen input path.")
    root = Path(paths.get("output_root", "/nonexistent"))
    check("output.writable", root.exists() and os.access(root, os.W_OK), {"path": str(root)}, "Use the shared output volume.")
    usage = shutil.disk_usage(root) if root.exists() else None
    sentinel = bool(usage and (usage.total >= 2**62 or usage.free >= 2**62))
    free = usage.free if usage else 0
    results.append(CheckResult("disk.free", "P1", "WARN" if sentinel or free < 5 * 1024**3 else "PASS", {"capacity_known": not sentinel, "free_bytes": None if sentinel else free}, "Check actual available quota before training."))
    check("cuda.available", torch.cuda.is_available(), {"torch": torch.__version__, "device_count": torch.cuda.device_count()}, "Attach CUDA for formal training.")
    trajectory = config.get("trajectory", {})
    check("protocol.frozen_fields", trajectory.get("length_m") == 8 and trajectory.get("points") == 80, trajectory, "Keep the frozen 8m/80-point representation.")
    fairness = config.get("fairness", {})
    check("protocol.no_future_input", fairness.get("future_gt_as_input") is False, fairness, "Use observed map and fixed goal.")
    check("output.no_overwrite", not report_dir.exists(), {"report": str(report_dir)}, "Use a new timestamped run directory.")
    return results


def proxy_checks(config, report_dir):
    import json
    results=[]
    def check(name,ok,evidence,remedy):results.append(CheckResult(name,"P0","PASS" if ok else "FAIL",evidence,remedy))
    root=Path(config["paths"]["output_root"]);data_id=config["paths"]["data_id"]
    data=root/"data"/data_id;cache=root/"cache"/data_id;pairs=root/"pairs"/data_id
    init=config["initialization"]
    check("source.provenance",init.get("provenance_verified") is True and bool(init.get("provenance_record")) and Path(init.get("provenance_record") or "/nonexistent").is_file(),init,"Provide evidence of original nuPlan pretraining; no filename-only approval.")
    for name,path in {"args":Path(init["args"]),"checkpoint":Path(init["checkpoint"]),"trajectories":data/"trajectories.jsonl","base_train":cache/"base_train.pt","base_val":cache/"base_val.pt","navigation_val":data/"navigation_val.jsonl","pair_train":cache/"pair_train.pt","pair_val":cache/"pair_val.pt","pair_gate":pairs/"gate_summary.json","trajectory_summary":data/"trajectory_summary.json"}.items():
        check("path."+name,path.is_file(),{"path":str(path)},"Complete the corresponding frozen CPU stage.")
    if (data/"trajectories.jsonl").is_file():
        from tartan.data.pose_utils import read_proxy_trajectories
        trajectories=read_proxy_trajectories(data/"trajectories.jsonl")
        gaps=[r["trajectory_key"] for r in trajectories if not r.get("gates") or not all(r["gates"].values())]
        check("data.time_reference_evidence",not gaps,{"unverified_trajectories":gaps},"Verify time and the approved reference frame for all three platforms.")
    if (pairs/"gate_summary.json").is_file():
        gate=json.loads((pairs/"gate_summary.json").read_text());check("pairs.train_nonempty",gate.get("accepted_train",gate.get("train_pairs",gate.get("accepted",{}).get("train",0)))>0,gate,"No qualified train pair blocks B; do not relax frozen thresholds.")
    check("protocol.frozen",config["seed"]==11 and config["trajectory"]["length_m"]==8 and config["trajectory"]["points"]==80,{"seed":config["seed"],"trajectory":config["trajectory"]},"Keep the approved Proxy settings.")
    check("output.writable",root.exists() and os.access(root,os.W_OK),{"path":str(root)},"Use the formal output disk.")
    usage=shutil.disk_usage(root) if root.exists() else None
    sentinel=bool(usage and (usage.total>=2**62 or usage.free>=2**62))
    if sentinel:results.append(CheckResult("disk.free","P1","WARN",{"capacity_known":False,"reported_free_bytes":usage.free},"Check actual quota; filesystem reports an unbounded sentinel."))
    else:check("disk.free",bool(usage and usage.free>=10*1024**3),{"capacity_known":True,"free_bytes":usage.free if usage else None},"Provide at least 10 GiB output disk space.")
    cuda=torch.cuda.is_available();phase=config["phase"]
    results.append(CheckResult("cuda.available","P0" if phase=="gpu" else "P1","PASS" if cuda else ("FAIL" if phase=="gpu" else "GPU_PENDING"),{"phase":phase,"device_count":torch.cuda.device_count()},"Attach CUDA for GPU smoke and formal training."))
    check("output.no_overwrite",not report_dir.exists(),{"path":str(report_dir)},"Keep the existing report; use an explicit distinct report path.")
    return results
