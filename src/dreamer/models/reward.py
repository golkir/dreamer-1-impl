import torch
from torch import nn
from dreamer.models.mlp import MLP
from dreamer.utils import create_normal_dist_from_params


class RewardModel(nn.Module):
    """
    q(r_tao | s_tao)
    """

    def __init__(self, st_dim):
        super().__init__()
        self.mlp = MLP(st_dim, 1)

    def reward_loss(dist, reward_true):
        # reward_true shape: (B, L, 1)
        log_prob = dist.log_prob(reward_true)
        return -log_prob.mean()

    def forward(self, ht: torch.Tensor, st: torch.Tensor):
        x = torch.cat([ht, st], dim=-1)
        x = self.mlp(x)
        dist = create_normal_dist_from_params(x, std=1.0, event_shape=1)
        return dist
