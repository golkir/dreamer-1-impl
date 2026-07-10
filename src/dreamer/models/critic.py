import torch
from torch import nn
import torch.functional as F
from dreamer.models.mlp import MLP
from dreamer.config import DreamerConfig

from dreamer.utils import create_normal_dist_from_output
"""
Loss: min E[Sum(v_model(s_r) - V(s_r)^2)]
"""

class Critic(nn.Module):
    """
    v(s_tao) = computes value of imagined trajectory
    """
    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.h_dim = config.rssm.h_dim
        self.z_dim = config.rssm.z_dim
        self.mlp = MLP(self.h_dim + self.z_dim, 1)


    def compute_lambda_return(self, rewards, values, gamma = 0.99, lam = 0.95):
        
        H = rewards.shape[1]

        returns = torch.empty_like(rewards)

        ret = values[:, -1]

        for t in reversed(range(H)):
            ret = rewards[:, t] + gamma * (
                (1 - lam) * values[:, t + 1]
                + lam * ret
            )
            returns[:, t] = ret

        return returns
        
    def forward(self, sr, rewards):
        x = self.mlp(sr)
        dist = create_normal_dist_from_output(x, event_shape=1)
        return dist