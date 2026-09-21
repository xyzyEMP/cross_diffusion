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
