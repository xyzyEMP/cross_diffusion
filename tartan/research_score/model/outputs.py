from dataclasses import dataclass
from typing import Optional
import torch
@dataclass
class DecompositionOutput:
    x0_shared: torch.Tensor
    x0_embodiment_delta: torch.Tensor
    x0_total: torch.Tensor
    eps_shared: Optional[torch.Tensor]=None
    eps_embodiment: Optional[torch.Tensor]=None
    eps_total: Optional[torch.Tensor]=None
    z_shared: Optional[torch.Tensor]=None
    z_embodiment: Optional[torch.Tensor]=None
