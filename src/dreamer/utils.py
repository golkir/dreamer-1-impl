from __future__ import annotations

import contextlib
import random
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as D
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


@contextlib.contextmanager
def freeze(modules: Iterable[nn.Module]):
    """Disable gradients for the modules' parameters while still letting gradients
    flow through their computations back to the inputs (e.g. to the actor)."""
    params = [p for m in modules for p in m.parameters()]
    old = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in zip(params, old):
            p.requires_grad_(flag)


def lambda_return(
    reward: torch.Tensor,
    value: torch.Tensor,
    discount: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    """TD(lambda) returns for an imagined trajectory of states s_0 .. s_H.

    All inputs have shape (H + 1, ...), indexed by state. ``reward[t]`` and
    ``discount[t]`` belong to arriving at s_t, so the first entries are unused:

        R_H = v(s_H)
        R_t = r_{t+1} + discount_{t+1} * ((1 - lambda) * v(s_{t+1}) + lambda * R_{t+1})

    Returns R_0 .. R_{H-1}, shape (H, ...).
    """
    horizon = reward.shape[0] - 1
    returns = []
    last = value[-1]
    for t in reversed(range(horizon)):
        last = reward[t + 1] + discount[t + 1] * ((1 - lambda_) * value[t + 1] + lambda_ * last)
        returns.append(last)
    return torch.stack(returns[::-1])


class TanhNormal:
    """Diagonal Gaussian squashed by tanh, as used by the Dreamer actor."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.normal = D.Normal(mean, std)
        self.dist = D.Independent(
            D.TransformedDistribution(self.normal, D.TanhTransform(cache_size=1)), 1
        )

    def rsample(self) -> torch.Tensor:
        return self.dist.rsample()

    def mode(self) -> torch.Tensor:
        return torch.tanh(self.normal.mean)

    def entropy(self) -> torch.Tensor:
        # No closed form after the tanh; use the pre-squash Gaussian entropy.
        return self.normal.entropy().sum(-1)


class OneHotDist:
    """Categorical over one-hot actions with straight-through gradients."""

    def __init__(self, logits: torch.Tensor):
        self.dist = D.OneHotCategorical(logits=logits)

    def rsample(self) -> torch.Tensor:
        sample = self.dist.sample()
        probs = self.dist.probs
        return sample + probs - probs.detach()

    def mode(self) -> torch.Tensor:
        return F.one_hot(self.dist.probs.argmax(-1), self.dist.probs.shape[-1]).float()

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy()
