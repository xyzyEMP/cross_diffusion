import torch
from tartan.research_score.training.losses import diffusion_loss,invariant_loss,swap_loss,grad_reverse
def test_losses_targets_and_invalid_pair():
 z=torch.zeros(2,3,4,requires_grad=True);o=torch.ones_like(z);m=torch.ones(2,3,dtype=torch.bool);assert diffusion_loss(z,z,m)==0
 v=torch.zeros(2,dtype=torch.bool);q=invariant_loss(z,o,v);q.backward();assert z.grad.abs().sum()==0
 assert swap_loss(z,o,o,z,o,o,m,m)==0
def test_grl_direction():
 x=torch.tensor([1.],requires_grad=True);grad_reverse(x).sum().backward();assert x.grad.item()==-1
