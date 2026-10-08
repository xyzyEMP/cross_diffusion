"""Transfer loss primitives and approved Proxy common-frame objectives."""
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


def proxy_masked_mse(pred, target, mask, denominator=None):
    """Four-coordinate SSE per valid point, never hide an empty batch."""
    mask = mask.bool()
    count = mask.sum() if denominator is None else denominator
    if torch.as_tensor(count).item() <= 0:
        raise ValueError("Proxy loss has no valid points")
    return ((pred - target).square().sum(-1) * mask).sum() / count


def proxy_common_frame(value, transform, normalizer, delta=False):
    """Inverse ego normalizer then SE2; deltas receive no mean or translation."""
    std = normalizer.std.to(device=value.device, dtype=value.dtype)[0, 0]
    mean = normalizer.mean.to(device=value.device, dtype=value.dtype)[0, 0]
    physical = value * std if delta else value * std + mean
    from tartan.research_score.data.core import torch_transform_trajectory
    return torch_transform_trajectory(physical, transform, delta=delta) / value.new_tensor([20., 20., 1., 1.])


def proxy_pair_losses(decomp_a, decomp_b, target_a, target_b, mask_a, mask_b,
                      pair_valid, T_a, T_b, normalizer, classifier,
                      labels_a=None, labels_b=None, denominators=None):
    """Q2 objectives. a/b retain their own inputs; only common-frame delta swaps."""
    denominators = {} if denominators is None else denominators
    mask_a, mask_b = mask_a.bool(), mask_b.bool()
    valid = pair_valid.bool()
    common_mask = valid[:, None] & mask_a & mask_b
    if not valid.all() or not common_mask.any(dim=1).all():
        raise ValueError("Proxy pair batch contains a rejected or empty pair")
    def shared(d, transform):
        return proxy_common_frame(d.x0_shared[:, 0, 1:], transform, normalizer)
    def delta(d, transform):
        return proxy_common_frame(d.x0_embodiment_delta[:, 0, 1:], transform, normalizer, delta=True)
    sa, sb = shared(decomp_a, T_a), shared(decomp_b, T_b)
    da, db = delta(decomp_a, T_a), delta(decomp_b, T_b)
    ya = proxy_common_frame(target_a, T_a, normalizer)
    yb = proxy_common_frame(target_b, T_b, normalizer)
    pair_diff = .5 * (proxy_masked_mse(decomp_a.x0_total[:, 0, 1:], target_a, mask_a, denominators.get("a"))
                      + proxy_masked_mse(decomp_b.x0_total[:, 0, 1:], target_b, mask_b, denominators.get("b")))
    inv = proxy_masked_mse(sa, sb, common_mask, denominators.get("common"))
    swap = .5 * (proxy_masked_mse(sa + db, yb, common_mask, denominators.get("common"))
                 + proxy_masked_mse(sb + da, ya, common_mask, denominators.get("common")))
    res = .5 * (proxy_masked_mse(da, torch.zeros_like(da), common_mask, denominators.get("common"))
                + proxy_masked_mse(db, torch.zeros_like(db), common_mask, denominators.get("common")))
    weights = common_mask[..., None].to(sa.dtype)
    pool = torch.cat([(sa * weights).sum(1) / weights.sum(1),
                      (sb * weights).sum(1) / weights.sum(1)], 0)
    if labels_a is None:
        labels_a = torch.zeros(len(sa), dtype=torch.long, device=sa.device)
    if labels_b is None:
        labels_b = torch.ones(len(sb), dtype=torch.long, device=sb.device)
    labels = torch.cat([labels_a, labels_b]).to(device=sa.device, dtype=torch.long)
    if not (labels.eq(0).sum() == labels.eq(1).sum() == len(sa)):
        raise ValueError("Pair classifier requires balanced Diff=0 and Omni=1 labels")
    ce = torch.nn.functional.cross_entropy(classifier(grad_reverse(pool)), labels, reduction="sum")
    ce = ce / denominators.get("classifier", len(labels))
    return {"pair_diff": pair_diff, "inv": inv, "swap": swap,
            "res": res, "ce": ce, "sep": ce + res,
            "pair_points": common_mask.sum()}
