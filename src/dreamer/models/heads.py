"""Dense prediction heads on top of the latent features."""

import math

import torch
import torch.nn.functional as F
from torch import distributions as D
from torch import nn

from dreamer.models.mlp import MLP
from dreamer.utils import OneHotDist, TanhNormal


class NormalHead(nn.Module):
    """Unit-variance Gaussian over a scalar; used for reward and value."""

    def __init__(self, feat_dim: int, units: int, layers: int, activation: str):
        super().__init__()
        self.mlp = MLP(feat_dim, 1, units, layers, activation)

    def forward(self, feat: torch.Tensor) -> D.Normal:
        return D.Normal(self.mlp(feat).squeeze(-1).float(), 1.0)


class ContinueHead(nn.Module):
    """Bernoulli probability that the episode continues after this state."""

    def __init__(self, feat_dim: int, units: int, layers: int, activation: str):
        super().__init__()
        self.mlp = MLP(feat_dim, 1, units, layers, activation)

    def forward(self, feat: torch.Tensor) -> D.Bernoulli:
        return D.Bernoulli(logits=self.mlp(feat).squeeze(-1).float())


class Actor(nn.Module):
    """Policy q(a | s): tanh-Gaussian for continuous actions, one-hot categorical
    (straight-through gradients) for discrete actions."""

    def __init__(
        self,
        feat_dim: int,
        action_dim: int,
        discrete: bool,
        units: int,
        layers: int,
        activation: str,
        init_std: float = 5.0,
        min_std: float = 1e-4,
        mean_scale: float = 5.0,
    ):
        super().__init__()
        self.discrete = discrete
        out_dim = action_dim if discrete else 2 * action_dim
        self.mlp = MLP(feat_dim, out_dim, units, layers, activation)
        # softplus(raw_init_std) == init_std, so the initial std is init_std
        self.raw_init_std = math.log(math.exp(init_std) - 1)
        self.min_std = min_std
        self.mean_scale = mean_scale

    def forward(self, feat: torch.Tensor) -> TanhNormal | OneHotDist:
        x = self.mlp(feat).float()
        if self.discrete:
            return OneHotDist(x)
        mean, std = x.chunk(2, -1)
        mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
        std = F.softplus(std + self.raw_init_std) + self.min_std
        return TanhNormal(mean, std)
