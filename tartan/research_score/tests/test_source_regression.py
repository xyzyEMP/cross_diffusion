from pathlib import Path

import numpy as np
import pytest
import torch

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from tartan.research_score.data.schema import CanonicalObservation, CanonicalTrajectory, RunIdentity, source_input_schema


ROOT = Path(__file__).resolve().parents[3]


def test_source_schema_observed_inputs():
    cfg = Config(str(ROOT / "checkpoints/args.json"), guidance_fn=None)
    schema = source_input_schema(cfg)
    assert schema["future_gt_as_input"] is False
    assert schema["fields"]["route_lanes"]["shape"] == ["B", 25, 20, 12]
    assert schema["normalization"]["policy"] == "frozen_from_checkpoint_args"
    assert schema["normalization"]["observation"]
    assert schema["normalization"]["state"]["mean"]
    assert cfg.diffusion_model_type == "x_start"
    assert cfg.future_len == 80 and cfg.time_len == 21


def test_canonical_dataclasses_construct():
    CanonicalObservation("s", [1.0, 2.0], [[0.0, 0.0, 0.0]], [], [], "anymal")
    CanonicalTrajectory([[0.0, 0.0]], [1.0], [0.0], [True], [0.1])
    RunIdentity("r", "transfer_primary", "d", "train", "m", "100", 1)




@pytest.mark.skipif(not torch.cuda.is_available(), reason="Source checkpoint regression requires an attached GPU")
def test_checkpoint_strict_cuda_load():
    cfg = Config(str(ROOT / "checkpoints/args.json"), guidance_fn=None); cfg.device = "cuda"
    model = Diffusion_Planner(cfg)
    payload = torch.load(ROOT / "checkpoints/model.pth", map_location="cpu", weights_only=False)
    state = payload.get("ema_state_dict", payload.get("model", payload))
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    model.cuda()
