from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

import numpy as np
import torch

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from tartan.research_score.data.schema import source_input_schema
from tartan.research_score.artifacts import validate_output_name


def _write_json_atomic(path: Path, payload) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--args", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-cache", help="Real nuPlan feature cache required for CUDA forward")
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA audit requested but CUDA is unavailable; CPU fallback is forbidden")
    out = Path(args.output)
    validate_output_name(out.name)
    out.mkdir(parents=True, exist_ok=False)
    cfg = Config(args.args, guidance_fn=None)
    cfg.device = args.device
    model = Diffusion_Planner(cfg)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("ema_state_dict", payload.get("model", payload))
    state = {k.removeprefix("module."): v for k, v in state.items()}
    incompatible = model.load_state_dict(state, strict=True)
    audit = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "strict_load": True,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "diffusion_model_type": cfg.diffusion_model_type,
        "sde": {
            "type": type(model.sde).__name__,
            "T": float(model.sde.T),
            "beta_min": float(model.sde._beta_min),
            "beta_max": float(model.sde._beta_max),
        },
        "future_len": cfg.future_len,
        "time_len": cfg.time_len,
        "sample_rate_hz": 10,
        "source_horizon_seconds": cfg.future_len / 10,
        "output_semantics": "body-level SE(2): x, y, cos(yaw), sin(yaw); no foot/joint/motor command",
        "source_normalizer_policy": "frozen from args.json",
        "autonomous_navigation_claim": False,
        "device_requested": args.device,
    }
    normalizer_path = out / "source_normalizers.json"
    _write_json_atomic(normalizer_path, {
        "policy": "frozen_from_checkpoint_args",
        "observation": cfg.observation_normalizer.to_dict(),
        "state": cfg.state_normalizer.to_dict(),
    })
    normalizer_reference = {
        "path": normalizer_path.name,
    }
    audit["source_normalizers"] = normalizer_reference
    _write_json_atomic(out / "source_input_schema.json", source_input_schema(cfg, normalizer_reference))
    if args.device == "cpu":
        _write_json_atomic(out / "source_model_audit.json", audit)
        print(json.dumps({"status": "STATIC_AUDIT_COMPLETE", "cuda_signature": "NOT_RUN"}))
        return
    model.eval().cuda()
    if not args.source_cache:
        raise ValueError("CUDA forward requires --source-cache with real nuPlan observations")
    cache = torch.load(args.source_cache, map_location="cpu", weights_only=False)
    keys = source_input_schema(cfg)["fields"]
    inputs = {key: cache[key][:1].cuda() for key in keys}
    normalized = cfg.observation_normalizer(inputs)
    outputs = []
    for _ in range(2):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        with torch.no_grad():
            _, result = model(normalized)
        pred = result["prediction"].detach().cpu().numpy()
        if not np.isfinite(pred).all():
            raise RuntimeError("CUDA output contains NaN/Inf")
        outputs.append(pred)
    if not np.array_equal(outputs[0], outputs[1]):
        raise RuntimeError("Fixed-seed CUDA outputs are not bitwise identical")
    np.savez_compressed(out / "source_output_signature.npz", prediction=outputs[0], seed=args.seed, mean=outputs[0].mean(), std=outputs[0].std())
    audit.update({"cuda_forward": True, "deterministic_repeat": True, "output_mean": float(outputs[0].mean()), "output_std": float(outputs[0].std()), "output_shape": list(outputs[0].shape)})
    _write_json_atomic(out / "source_model_audit.json", audit)
    print(json.dumps({"status": "CUDA_AUDIT_COMPLETE"}))


if __name__ == "__main__":
    main()
