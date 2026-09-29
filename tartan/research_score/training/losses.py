import torch
from torch.autograd import Function
def masked_mse(pred,target,mask):
    e=(pred-target).square().sum(-1);return (e*mask).sum()/mask.sum().clamp_min(1)
def diffusion_loss(total,target,mask):return masked_mse(total,target,mask)
def invariant_loss(a,b,pair_valid):return masked_mse(a,b,pair_valid[:,None].expand(a.shape[:-1]))
def swap_loss(shared_a,res_b,target_b,shared_b,res_a,target_a,mask_a,mask_b):return masked_mse(shared_a+res_b,target_b,mask_b)+masked_mse(shared_b+res_a,target_a,mask_a)
def residual_loss(delta,mask):return masked_mse(delta,torch.zeros_like(delta),mask)
class _GRL(Function):
    @staticmethod
    def forward(ctx,x,l):ctx.l=l;return x.view_as(x)
    @staticmethod
    def backward(ctx,g):return -ctx.l*g,None
def grad_reverse(x,l=1.):return _GRL.apply(x,l)

def proxy_objective(evidence_kind,pred_a,target_a,mask_a,pred_b,target_b,mask_b,
                    pair_valid=None,shared_a=None,shared_b=None,swap_ab=None,swap_ba=None,
                    delta_a=None,delta_b=None,weights=None,adversarial_loss=None):
    """Apply D028 supervision boundaries to one balanced two-platform batch."""
    weights=weights or {};ldiff=.5*(diffusion_loss(pred_a,target_a,mask_a)+diffusion_loss(pred_b,target_b,mask_b))
    aux_requested=any(float(weights.get(k,0.)) for k in ("invariant","swap","separation"))
    if evidence_kind=="tartan_recorded_unpaired":
        if aux_requested:raise ValueError("real_unpaired permits L_diff only")
        return {"total":ldiff,"diffusion":ldiff}
    if evidence_kind!="synthetic_fixture":raise ValueError("unknown evidence_kind")
    if pair_valid is None:raise ValueError("synthetic_fixture requires pair_valid")
    pair_mask_a=mask_a & pair_valid[:,None];pair_mask_b=mask_b & pair_valid[:,None]
    linv=masked_mse(shared_a,shared_b,pair_mask_a & pair_mask_b)
    lswap=.5*(masked_mse(swap_ab,target_b,pair_mask_b)+masked_mse(swap_ba,target_a,pair_mask_a))
    lres=.5*(residual_loss(delta_a,pair_mask_a)+residual_loss(delta_b,pair_mask_b))
    ladv=ldiff.new_zeros(()) if adversarial_loss is None else adversarial_loss
    lsep=ladv+.01*lres
    total=ldiff+float(weights.get("invariant",0.))*linv+float(weights.get("swap",0.))*lswap+float(weights.get("separation",0.))*lsep
    return {"total":total,"diffusion":ldiff,"invariant":linv,"swap":lswap,"separation":lsep,"residual":lres,"adversarial":ladv}
