from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from contextlib import nullcontext

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config


class DiffusionPlannerRunner:
    def __init__(self, args_file: Path, checkpoint: Path, device: str = "auto") -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        self.device = torch.device(device)
        self.config = Config(str(args_file), guidance_fn=None)
        # config.device is used only while constructing a few modules.
        self.config.device = str(self.device)
        self.model = Diffusion_Planner(self.config)
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint_data.get("ema_state_dict", checkpoint_data.get("model", checkpoint_data))
        state = {key.removeprefix("module."): value for key, value in state.items()}
        incompatible = self.model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
        self.model.eval().to(self.device)

    def predict(self, inputs: Dict[str, torch.Tensor], seed: int, guidance_fn=None) -> np.ndarray:
        torch.manual_seed(seed)
        normalized = self.config.observation_normalizer(inputs)
        self.model.decoder.decoder._guidance_fn = guidance_fn
        context = torch.no_grad() if guidance_fn is None else nullcontext()
        with context:
            _, outputs = self.model(normalized)
        prediction = outputs["prediction"][:, 0].detach().cpu().numpy()
        yaw = np.arctan2(prediction[..., 3], prediction[..., 2])
        return np.concatenate((prediction[..., :2], yaw[..., None]), axis=-1).astype(np.float32)
