import torch
from tartan.research_score.training.losses import diffusion_loss,invariant_loss,swap_loss,grad_reverse,proxy_objective
def test_losses_targets_and_invalid_pair():
 z=torch.zeros(2,3,4,requires_grad=True);o=torch.ones_like(z);m=torch.ones(2,3,dtype=torch.bool);assert diffusion_loss(z,z,m)==0
 v=torch.zeros(2,dtype=torch.bool);q=invariant_loss(z,o,v);q.backward();assert z.grad.abs().sum()==0
 assert swap_loss(z,o,o,z,o,o,m,m)==0
def test_grl_direction():
 x=torch.tensor([1.],requires_grad=True);grad_reverse(x).sum().backward();assert x.grad.item()==-1

def test_proxy_real_rejects_pair_losses():
 x=torch.zeros(2,3,4);m=torch.ones(2,3,dtype=torch.bool)
 try:proxy_objective("tartan_recorded_unpaired",x,x,m,x,x,m,weights={"swap":.5})
 except ValueError as e:assert "L_diff only" in str(e)
 else:raise AssertionError("real data must reject pair supervision")

def test_proxy_fixture_invalid_pair_has_no_auxiliary_gradient():
 a=torch.zeros(2,3,4,requires_grad=True);b=torch.ones(2,3,4,requires_grad=True);m=torch.ones(2,3,dtype=torch.bool);valid=torch.zeros(2,dtype=torch.bool)
 q=proxy_objective("synthetic_fixture",a,a.detach(),m,b,b.detach(),m,pair_valid=valid,shared_a=a,shared_b=b,swap_ab=a,swap_ba=b,delta_a=a,delta_b=b,weights={"invariant":.2,"swap":.5,"separation":.1})
 assert q["invariant"]==0 and q["swap"]==0 and q["residual"]==0
