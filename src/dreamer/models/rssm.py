"""
Recurrent State-Space Model.

The latent state has a deterministic part h_t (GRU) and a stochastic part z_t:

    h_t = f(h_{t-1}, z_{t-1}, a_{t-1})          recurrent model
    z_t ~ p(z_t | h_t)                          prior / transition model
    z_t ~ q(z_t | h_t, e_t)                     posterior / representation model

where e_t is the encoded observation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from torch import distributions as D
from torch import nn

from dreamer.config import ModelConfig


@dataclass
class State:
    mean: torch.Tensor
    std: torch.Tensor
    stoch: torch.Tensor
    deter: torch.Tensor

    @property
    def feat(self) -> torch.Tensor:
        return torch.cat([self.stoch, self.deter], -1)

    @property
    def dist(self) -> D.Distribution:
        return D.Independent(D.Normal(self.mean, self.std), 1)

    def map(self, fn) -> "State":
        return State(*(fn(getattr(self, f.name)) for f in fields(self)))

    def detach(self) -> "State":
        return self.map(torch.Tensor.detach)

    @staticmethod
    def stack(states: list["State"], dim: int) -> "State":
        return State(
            *(torch.stack([getattr(s, f.name) for s in states], dim) for f in fields(State))
        )


class RSSM(nn.Module):
    def __init__(self, action_dim: int, embed_dim: int, config: ModelConfig):
        super().__init__()
        self.stoch_size = config.stoch_size
        self.deter_size = config.deter_size
        self.min_std = config.min_std
        act = getattr(nn, config.dense_act)
        hidden = config.hidden_size

        self.img_in = nn.Sequential(nn.Linear(config.stoch_size + action_dim, hidden), act())
        self.gru = nn.GRUCell(hidden, config.deter_size)
        self.prior_net = nn.Sequential(
            nn.Linear(config.deter_size, hidden), act(), nn.Linear(hidden, 2 * config.stoch_size)
        )
        self.post_net = nn.Sequential(
            nn.Linear(config.deter_size + embed_dim, hidden),
            act(),
            nn.Linear(hidden, 2 * config.stoch_size),
        )

    @property
    def feat_dim(self) -> int:
        return self.stoch_size + self.deter_size

    def initial(self, batch_size: int, device) -> State:
        zeros = lambda n: torch.zeros(batch_size, n, device=device)  # noqa: E731
        return State(
            zeros(self.stoch_size),
            zeros(self.stoch_size),
            zeros(self.stoch_size),
            zeros(self.deter_size),
        )

    def _stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = x.float().chunk(2, -1)
        return mean, F.softplus(std) + self.min_std

    def img_step(self, prev: State, action: torch.Tensor) -> State:
        """Prior step: advance the state with an action, without an observation."""
        x = self.img_in(torch.cat([prev.stoch, action], -1))
        deter = self.gru(x, prev.deter).float()
        mean, std = self._stats(self.prior_net(deter))
        stoch = mean + std * torch.randn_like(std)
        return State(mean, std, stoch, deter)

    def obs_step(
        self, prev: State, action: torch.Tensor, embed: torch.Tensor
    ) -> tuple[State, State]:
        """Posterior step: returns (posterior, prior) for the next time step."""
        prior = self.img_step(prev, action)
        mean, std = self._stats(self.post_net(torch.cat([prior.deter, embed], -1)))
        stoch = mean + std * torch.randn_like(std)
        return State(mean, std, stoch, prior.deter), prior

    def observe(
        self, embed: torch.Tensor, action: torch.Tensor, is_first: torch.Tensor
    ) -> tuple[State, State]:
        """Filter a batch of sequences (B, T, ...). ``action[:, t]`` is the action
        that led to step t; the state is reset wherever ``is_first`` is set."""
        batch, length = embed.shape[:2]
        state = self.initial(batch, embed.device)
        posts, priors = [], []
        for t in range(length):
            keep = (1.0 - is_first[:, t]).unsqueeze(-1)
            state = state.map(lambda x: x * keep)
            state, prior = self.obs_step(state, action[:, t] * keep, embed[:, t])
            posts.append(state)
            priors.append(prior)
        return State.stack(posts, 1), State.stack(priors, 1)

    def kl_loss(
        self, post: State, prior: State, free_nats: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """KL(posterior || prior), averaged, clipped below at ``free_nats``."""
        kl = D.kl_divergence(post.dist, prior.dist).mean()
        return torch.clamp(kl, min=free_nats), kl
