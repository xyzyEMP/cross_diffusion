import torch
from tartan.research_score.training.losses import diffusion_loss,invariant_loss,swap_loss,grad_reverse
def test_losses_targets_and_invalid_pair():
 z=torch.zeros(2,3,4,requires_grad=True);o=torch.ones_like(z);m=torch.ones(2,3,dtype=torch.bool);assert diffusion_loss(z,z,m)==0
 v=torch.zeros(2,dtype=torch.bool);q=invariant_loss(z,o,v);q.backward();assert z.grad.abs().sum()==0
 assert swap_loss(z,o,o,z,o,o,m,m)==0
def test_grl_direction():
 x=torch.tensor([1.],requires_grad=True);grad_reverse(x).sum().backward();assert x.grad.item()==-1


def test_proxy_common_frame_affine_and_delta():
 from types import SimpleNamespace
 from tartan.research_score.training.losses import proxy_common_frame, proxy_masked_mse
 normalizer=SimpleNamespace(std=torch.tensor([[[20.,20.,1.,1.]]]),mean=torch.tensor([[[1.,2.,0.,0.]]]))
 transform=torch.tensor([[[0.,-1.,3.],[1.,0.,4.],[0.,0.,1.]]])
 x=torch.tensor([[[.1,.2,1.,0.]]])
 shared=proxy_common_frame(x,transform,normalizer)
 delta=proxy_common_frame(x,transform,normalizer,delta=True)
 assert torch.allclose(shared,torch.tensor([[[-.15,.35,0.,1.]]]))
 assert torch.allclose(delta,torch.tensor([[[-.2,.1,0.,1.]]]))
 try:proxy_masked_mse(x,x,torch.zeros(1,1,dtype=torch.bool))
 except ValueError:pass
 else:raise AssertionError("empty mask accepted")


def test_proxy_pair_half_sum_and_classifier_gradients():
 from types import SimpleNamespace
 from tartan.research_score.training.losses import proxy_pair_losses
 from torch import nn
 x=torch.zeros(2,1,4,4,requires_grad=True)
 delta=torch.zeros_like(x,requires_grad=True)
 d=SimpleNamespace(x0_shared=x,x0_embodiment_delta=delta,x0_total=x+delta)
 target=torch.ones(2,3,4)
 masks=torch.ones(2,3,dtype=torch.bool)
 T=torch.eye(3).expand(2,3,3)
 n=SimpleNamespace(std=torch.tensor([[[20.,20.,1.,1.]]]),mean=torch.zeros(1,1,4))
 classifier=nn.Sequential(nn.Linear(4,64),nn.SiLU(),nn.Linear(64,2))
 q=proxy_pair_losses(d,d,target,target,masks,masks,torch.ones(2,dtype=torch.bool),T,T,n,classifier)
 assert q["pair_diff"]==4 and q["swap"]==4 and q["inv"]==0
 (q["pair_diff"]+.1*q["swap"]+.01*q["sep"]).backward()
 assert torch.isfinite(x.grad).all() and classifier[-1].weight.grad is not None


def test_proxy_forward_keeps_sde_noise_rank_four():
 from types import SimpleNamespace
 from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
 from tartan.research_score.scripts.train_transfer import proxy_forward
 class Probe:
  sde=VPSDE_linear()
  def __call__(self,x):
   assert x['sampled_trajectories'].shape==(2,11,81,4)
   return None,{'score':x['sampled_trajectories']}
 c=SimpleNamespace(observation_normalizer=lambda x:x,state_normalizer=lambda x:x)
 batch=({'ego_current_state':torch.zeros(2,10)},torch.zeros(2,80,4),torch.ones(2,80,dtype=torch.bool),{'split':['train','train'],'platform_id':torch.tensor([2,3])})
 loss,*_=proxy_forward(Probe(),c,batch,torch.Generator().manual_seed(11),None,False,'cpu')
 assert torch.isfinite(loss)
