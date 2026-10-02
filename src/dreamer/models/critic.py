import torch
from torch import nn
import torch.functional as F
from dreamer.models.mlp import MLP
from dreamer.config import DreamerConfig

from dreamer.utils import create_normal_dist_from_params

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

    def compute_lambda_return(self, rewards, values, discounts, lam=0.95):
        """
        TD(lambda) returns along an imagined trajectory (time on dim 1):

            V_t = r_t + discount_t * ((1 - lambda) * v_{t+1} + lambda * V_{t+1})

        bootstrapped with V_{H-1} = v_{H-1}. ``discounts`` is gamma times the
        predicted continue probability. The last entry is only the bootstrap.
        """

        H = rewards.shape[1] - 1

        returns = torch.empty_like(rewards)

        ret = values[:, -1]
        returns[:, -1] = ret

        for t in reversed(range(H)):
            ret = rewards[:, t] + discounts[:, t] * ((1.0 - lam) * values[:, t + 1] + lam * ret)
            returns[:, t] = ret

        return returns

    def forward(self, hr, sr):
        x = torch.cat([hr, sr], dim=-1)
        x = self.mlp(x)

        dist = create_normal_dist_from_params(x, std=1, event_shape=1)
        return dist
