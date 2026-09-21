import torch
from torch import nn
from .embodiment import EmbodimentEncoder
from .adapters import ZeroResidualAdapter
from .outputs import DecompositionOutput

class ScoreDecompositionPlanner(nn.Module):
    def __init__(self,backbone,ability_dim=10):
        super().__init__();self.backbone=backbone;self.embodiment_encoder=EmbodimentEncoder(ability_dim);self.residual=ZeroResidualAdapter()
    def forward(self,inputs,embodiment_id=None,ability=None,ability_mask=None,source_regression_mode=False,sigma_clamp=1e-4):
        # A supplied noised trajectory denotes the denoising-loss path even
        # while the module is in eval mode.  Sampling is the only path that
        # installs the per-step correction callback.
        if not self.training and not source_regression_mode and "sampled_trajectories" not in inputs:
            if embodiment_id is None:raise ValueError("research_mode requires embodiment_id")
            z=self.embodiment_encoder(embodiment_id,ability,ability_mask)
            def correction(flat):
                shared=flat.reshape(*flat.shape[:-1],-1,4);delta=self.residual(shared,z)
                if delta.shape[1]>1:
                    ego=torch.zeros_like(delta);ego[:,0]=delta[:,0];delta=ego
                return (shared+delta).reshape_as(flat)
            conditioned=dict(inputs);conditioned["score_correction_fn"]=correction
            return self.backbone(conditioned)
        enc,out=self.backbone(inputs);shared=out["score"]
        if source_regression_mode:return enc,out
        if embodiment_id is None:raise ValueError("research_mode requires embodiment_id")
        z=self.embodiment_encoder(embodiment_id,ability,ability_mask);delta=self.residual(shared,z)
        if delta.ndim>=4 and delta.shape[1]>1:
            ego_only=torch.zeros_like(delta);ego_only[:,0]=delta[:,0];delta=ego_only
        total=shared+delta
        xt=inputs.get("sampled_trajectories");t=inputs.get("diffusion_time")
        es=ee=et=None
        if xt is not None and t is not None:
            mean_coeff=torch.exp(-.5*t).view(-1,*([1]*(xt.ndim-1)));sigma=(1-mean_coeff.square()).sqrt().clamp_min(sigma_clamp)
            es=(xt-mean_coeff*shared)/sigma;ee=-mean_coeff*delta/sigma;et=(xt-mean_coeff*total)/sigma
        return enc,{**out,"score":total,"decomposition":DecompositionOutput(shared,delta,total,es,ee,et,enc.get("encoding"),z)}
