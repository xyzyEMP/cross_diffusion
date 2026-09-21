import torch
from torch import nn

class ZeroResidualAdapter(nn.Module):
    def __init__(self, trajectory_dim=4, context_dim=64, hidden_dim=64):
        super().__init__();self.net=nn.Sequential(nn.Linear(trajectory_dim+context_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,trajectory_dim));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,context):
        c=context
        while c.ndim<x.ndim:c=c.unsqueeze(1)
        c=c.expand(*x.shape[:-1],c.shape[-1]);return self.net(torch.cat([x,c],-1))
