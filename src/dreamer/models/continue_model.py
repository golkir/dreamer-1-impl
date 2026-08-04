import torch
from torch import nn
from dreamer.models.mlp import MLP
from dreamer.config import DreamerConfig


class ContinueModel(nn.Module):
    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config = config
        self.mlp = MLP(self.config.rssm.h_dim + self.config.rssm.z_dim, 1)

    def forward(self, posterior, h):
        x = torch.cat([posterior, h], dim=-1)
        x = self.mlp(x)
        dist = torch.distributions.Bernoulli(logits=x)
        return dist
