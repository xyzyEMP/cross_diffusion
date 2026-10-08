from contextlib import nullcontext
import torch
from torch import nn
from .embodiment import EmbodimentEncoder
from .adapters import ZeroResidualAdapter
from .outputs import DecompositionOutput

class ScoreDecompositionPlanner(nn.Module):
    def __init__(self,backbone,ability_dim=10,proxy=False,seed=11):
        super().__init__()
        self.backbone = backbone
        self.proxy = proxy
        # B-specific construction cannot consume the base noise/dropout RNG stream.
        with torch.random.fork_rng(devices=[]) if proxy else nullcontext():
            if proxy:
                torch.random.default_generator.manual_seed(seed)
            self.embodiment_encoder = EmbodimentEncoder(ability_dim, proxy=proxy)
            self.residual = ZeroResidualAdapter()
            if proxy:
                self.classifier = nn.Sequential(nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 2))
        self.to(next(backbone.parameters()).device)

    def _delta(self, shared, condition):
        delta = self.residual(shared, condition)
        if delta.ndim >= 4:
            ego = torch.zeros_like(delta)
            if self.proxy:
                ego[:, 0, 1:] = delta[:, 0, 1:]
            else:
                ego[:, 0] = delta[:, 0]
            delta = ego
        return delta
    def forward(self,inputs,embodiment_id=None,ability=None,ability_mask=None,source_regression_mode=False,sigma_clamp=1e-4,id_mask=None):
        if self.proxy and not source_regression_mode:
            if "proxy_context" in inputs:
                raise ValueError("proxy_context is internal and cannot be supplied from cache")
            inputs = dict(inputs)
            ability = self.backbone.encoder.encoder.encode_proxy_history(inputs)
            inputs["proxy_context"] = ability
        # A supplied noised trajectory denotes the denoising-loss path even
        # while the module is in eval mode.  Sampling is the only path that
        # installs the per-step correction callback.
        if not self.training and not source_regression_mode and "sampled_trajectories" not in inputs:
            if embodiment_id is None:raise ValueError("research_mode requires embodiment_id")
            z=self.embodiment_encoder(embodiment_id,ability,ability_mask,id_mask)
            def correction(flat):
                shared=flat.reshape(*flat.shape[:-1],-1,4);delta=self._delta(shared,z)
                return (shared+delta).reshape_as(flat)
            conditioned=dict(inputs);conditioned["score_correction_fn"]=correction
            return self.backbone(conditioned)
        enc,out=self.backbone(inputs);shared=out["score"]
        if source_regression_mode:return enc,out
        if embodiment_id is None:raise ValueError("research_mode requires embodiment_id")
        z=self.embodiment_encoder(embodiment_id,ability,ability_mask,id_mask);delta=self._delta(shared,z)
        total=shared+delta
        xt=inputs.get("sampled_trajectories");t=inputs.get("diffusion_time")
        es=ee=et=None
        if xt is not None and t is not None:
            sde=getattr(self.backbone,"sde",None)
            if sde is None:
                mean_coeff=torch.exp(-.5*t).view(-1,*([1]*(xt.ndim-1)))
                sigma=(1-mean_coeff.square()).sqrt().clamp_min(sigma_clamp)
            else:
                mean_coeff,sigma=sde.marginal_prob(torch.ones_like(xt),t)
                sigma=sigma.clamp_min(sigma_clamp)
            es=(xt-mean_coeff*shared)/sigma;ee=-mean_coeff*delta/sigma;et=(xt-mean_coeff*total)/sigma
        return enc,{**out,"score":total,"decomposition":DecompositionOutput(shared,delta,total,es,ee,et,enc.get("encoding"),z)}
