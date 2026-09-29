import torch
from torch import nn

class EmbodimentEncoder(nn.Module):
    def __init__(self, ability_dim=10, hidden_dim=64, embodiments=4):
        super().__init__();self.id=nn.Embedding(embodiments,hidden_dim);self.net=nn.Sequential(nn.Linear(ability_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim))
    def forward(self, embodiment_id, ability, ability_mask=None):
        if ability_mask is not None: ability=ability*ability_mask
        return self.id(embodiment_id)+self.net(ability)

    def reinitialize_id(self, embodiment_id, seed):
        """Deterministically initialize one new platform row, preserving all others."""
        if embodiment_id < 0 or embodiment_id >= self.id.num_embeddings:
            raise IndexError("embodiment_id outside embedding registry")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        value = torch.randn(self.id.embedding_dim, generator=generator, dtype=self.id.weight.dtype)
        with torch.no_grad():
            self.id.weight[embodiment_id].copy_(value.to(self.id.weight.device))
