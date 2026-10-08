import numpy as np,torch
from torch import nn
from tartan.research_score.model.score_decomposition import ScoreDecompositionPlanner
from tartan.research_score.data.source_representation import TrajectoryRepresentationBridge
class B(nn.Module):
 def __init__(self):super().__init__();self.w=nn.Parameter(torch.ones(1))
 def forward(self,x):
  y=x["sampled_trajectories"]*self.w;return {"encoding":torch.ones(len(y),2,3,device=y.device)},{"score":y}
def inp(b=2):return {"sampled_trajectories":torch.randn(b,3,81,4),"diffusion_time":torch.ones(b)*.5}
def test_zero_residual_and_source_regression():
 m=ScoreDecompositionPlanner(B());x=inp();base=m.backbone(x)[1]["score"];_,o=m(x,torch.zeros(2,dtype=torch.long),torch.zeros(2,10));assert torch.equal(base,o["score"]);assert torch.equal(m(x,source_regression_mode=True)[1]["score"],base)
def test_parameterization_and_condition_isolation():
 m=ScoreDecompositionPlanner(B());x=inp();_,a=m(x,torch.zeros(2,dtype=torch.long),torch.zeros(2,10));d=a["decomposition"];assert torch.allclose(d.eps_total,d.eps_shared+d.eps_embodiment,atol=1e-5);s=d.x0_shared.clone();m.residual.net[-1].weight.data.fill_(.1);_,b=m(x,torch.ones(2,dtype=torch.long),torch.ones(2,10));assert torch.equal(s,b["decomposition"].x0_shared);assert not torch.equal(d.x0_total,b["decomposition"].x0_total);assert torch.count_nonzero(b["decomposition"].x0_embodiment_delta[:,1:])==0
def test_bridge_line():
 x=np.c_[np.linspace(0,10,101),np.zeros(101),np.zeros(101)];y,m=TrajectoryRepresentationBridge()(x);assert y.shape==(80,4) and abs(y[-1,0]-8)<1e-5 and m.all()

def test_reinitialize_only_new_diff_row():
 m=ScoreDecompositionPlanner(B());before=m.embodiment_encoder.id.weight.detach().clone();m.embodiment_encoder.reinitialize_id(2,11);after=m.embodiment_encoder.id.weight.detach();assert torch.equal(before[:2],after[:2]);assert not torch.equal(before[2],after[2]);assert torch.equal(before[3],after[3])

def test_epsilon_diagnostic_uses_backbone_sde():
 class S:
  def marginal_prob(self,x,t):return x*.25,torch.ones_like(t).view(-1,1,1,1)*.75
 b=B();b.sde=S();m=ScoreDecompositionPlanner(b);x=inp();_,o=m(x,torch.zeros(2,dtype=torch.long),torch.zeros(2,10));d=o["decomposition"];expected=(x["sampled_trajectories"]-.25*d.x0_shared)/.75;assert torch.allclose(d.eps_shared,expected)


def test_proxy_history_initialization_masks_and_second_step_gradient():
 from diffusion_planner.model.module.encoder import Encoder
 from diffusion_planner.utils.config import Config
 from pathlib import Path
 config=Config(str(Path(__file__).resolve().parents[3]/'checkpoints/args.json'),None)
 config.device='cpu'
 a,b=Encoder(config),Encoder(config)
 state=torch.random.get_rng_state().clone()
 a.enable_proxy_history();b.enable_proxy_history()
 assert torch.equal(state,torch.random.get_rng_state())
 for x,y in zip(a.proxy_history.parameters(),b.proxy_history.parameters()):assert torch.equal(x,y)
 assert a.proxy_projection.weight.count_nonzero()==0
 inputs={'ego_history':torch.ones(2,20,4),'history_mask':torch.ones(2,20,dtype=torch.bool),
         'history_dt':torch.ones(2,19)*.1,'history_dt_mask':torch.ones(2,19,dtype=torch.bool),
         'motion_rms':torch.ones(2,3),'motion_mask':torch.ones(2,3,dtype=torch.bool)}
 inputs['history_mask'][:,0]=False
 first=a.encode_proxy_history(inputs)
 inputs['ego_history'][:,0]=float('nan')
 assert torch.equal(first,a.encode_proxy_history(inputs))
 optimizer=torch.optim.SGD(list(a.proxy_history.parameters())+list(a.proxy_projection.parameters()),lr=.1)
 for step in range(2):
  optimizer.zero_grad();context=a.encode_proxy_history(inputs)
  (a.proxy_projection(context)-1).square().mean().backward()
  grad=a.proxy_history.net[0].weight.grad.abs().sum()
  assert grad==0 if step==0 else grad>0
  optimizer.step()
 assert context.shape==(2,22)


def test_proxy_condition_dropout_and_future_only_delta():
 from tartan.research_score.model.embodiment import EmbodimentEncoder
 e=EmbodimentEncoder(proxy=True)
 context=torch.ones(2,22)
 out=e(torch.tensor([0,1]),context,id_mask=torch.zeros(2))
 assert torch.equal(out[0],out[1])
 m=ScoreDecompositionPlanner(B(),proxy=True)
 m.residual.net[-1].bias.data.fill_(1)
 delta=m._delta(torch.zeros(2,3,81,4),torch.zeros(2,64))
 assert delta[:,0,1:].eq(1).all()
 assert delta[:,:,0].count_nonzero()==0 and delta[:,1:].count_nonzero()==0


def test_proxy_sampling_reuses_context_and_corrects_each_step():
 from types import SimpleNamespace
 class Sampling(B):
  def __init__(self):
   super().__init__();self.calls=0
   self.encoder=SimpleNamespace(encoder=self)
  def encode_proxy_history(self,inputs):
   self.calls+=1
   return torch.zeros(2,22)
  def forward(self,inputs):
   # Mathematical denoising callback check; no generated training records.
   flat=torch.zeros(2,3,81*4)
   correction=inputs['score_correction_fn']
   first=correction(flat);second=correction(first)
   return {},{'prediction':second.reshape(2,3,81,4)}
 backbone=Sampling();model=ScoreDecompositionPlanner(backbone,proxy=True).eval()
 model.residual.net[-1].bias.data.fill_(1)
 _,out=model({},embodiment_id=torch.tensor([1,2]))
 y=out['prediction']
 assert backbone.calls==1
 assert y[:,0,1:].eq(2).all() and y[:,:,0].count_nonzero()==0
 assert y[:,1:].count_nonzero()==0
