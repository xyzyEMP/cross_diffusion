import numpy as np,torch
from torch import nn
from tartan.research_score.model.score_decomposition import ScoreDecompositionPlanner
from tartan.research_score.model.representation_bridge import TrajectoryRepresentationBridge
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
