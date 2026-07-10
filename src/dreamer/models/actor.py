import torch
from torch import nn
from torch.distributions import TanhTransform
from einops import rearrange

from dreamer.models.mlp import MLP
from dreamer.config import DreamerConfig
from dreamer.utils import create_normal_dist_from_output

class Actor(nn.Module):
    """
    q(a_r | s_r)

    the output should be:
    actions are vector valued, each action in the vector is Gaussian output by dense neural net passed through
    the tanh function.

    Input: sr_dim - let's say B, L, H
    We want B, L, A, 2, we can use rearrange to achieve this

    Shape:
    """

    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.z_dim = config.rssm.z_dim
        self.h_dim = config.rssm.h_dim
        self.mlp = MLP(self.z_dim + self.h_dim, config.action_dim * 2)

    def actor_loss(self, lambda_values: torch.Tensor):
        actor_loss = -lambda_values.mean()
        return actor_loss

    def forward(
        self, h_r: torch.Tensor, z_r: torch.Tensor
    ) -> tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        parameters are component of the RSSM imaginary state (h_r and z_r)
        """
        x = torch.cat([h_r, z_r], -1)
        x = self.mlp(x)

        dist = create_normal_dist_from_output(x, activation=torch.tanh)

        dist = torch.distributions.TransformedDistribution(dist, TanhTransform())
        action = torch.distributions.Independent(dist, 1).rsample()
        return dist, action
