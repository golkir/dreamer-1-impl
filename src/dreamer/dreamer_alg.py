from __future__ import annotations

import contextlib
import itertools

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from dreamer.config import Config
from dreamer.models.decoder import Decoder
from dreamer.models.encoder import ObservationEncoder
from dreamer.models.heads import Actor, ContinueHead, NormalHead
from dreamer.models.rssm import RSSM, State
from dreamer.replay_buffer import Batch
from dreamer.utils import freeze, lambda_return

PolicyState = tuple[State, torch.Tensor]  # (latent state, previous action)

torch.distributions.Distribution.set_default_validate_args(False)


class Dreamer(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        action_space: gym.Space,
        config: Config,
    ):
        super().__init__()
        self.config = config
        m, t = config.model, config.train

        self.discrete = isinstance(action_space, gym.spaces.Discrete)
        if self.discrete:
            self.action_dim = int(action_space.n)
        else:
            self.action_dim = int(np.prod(action_space.shape))

        # World model
        self.encoder = ObservationEncoder(obs_shape, m)
        self.rssm = RSSM(self.action_dim, self.encoder.embed_dim, m)
        feat_dim = self.rssm.feat_dim
        self.decoder = Decoder(feat_dim, obs_shape, m)
        self.reward = NormalHead(feat_dim, m.dense_units, m.reward_layers, m.dense_act)
        self.cont = (
            ContinueHead(feat_dim, m.dense_units, m.continue_layers, m.dense_act)
            if t.use_continue
            else None
        )
        # Behavior
        self.actor = Actor(
            feat_dim,
            self.action_dim,
            self.discrete,
            m.dense_units,
            m.actor_layers,
            m.dense_act,
            init_std=m.actor_init_std,
            min_std=m.actor_min_std,
            mean_scale=m.actor_mean_scale,
        )
        self.value = NormalHead(feat_dim, m.dense_units, m.value_layers, m.dense_act)

        self.world_model = [
            mod
            for mod in (self.encoder, self.rssm, self.decoder, self.reward, self.cont)
            if mod is not None
        ]
        world_params = list(itertools.chain(*(mod.parameters() for mod in self.world_model)))

        def adam(params, lr):
            return torch.optim.AdamW(params, lr=lr, eps=t.adam_eps, weight_decay=t.weight_decay)

        self.model_opt = adam(world_params, t.model_lr)
        self.actor_opt = adam(self.actor.parameters(), t.actor_lr)
        self.value_opt = adam(self.value.parameters(), t.value_lr)

        # Mixed precision: networks run in fp16, distributions and losses stay fp32.
        # One loss scaler per optimizer, created on first use on the model's device.
        self.amp = config.run.amp
        self._scalers: dict[str, torch.amp.GradScaler] = {}

        if config.run.compile:
            # Fuse the many small ops of the per-step recurrences, which run 50 (observe)
            # and 15 (imagine) times per update. Only functions are compiled, so
            # parameter names and checkpoints are unchanged.
            self.rssm.img_step = torch.compile(self.rssm.img_step)
            self.rssm.obs_step = torch.compile(self.rssm.obs_step)
            self.actor.mlp.compile()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


    @staticmethod
    def preprocess(obs: torch.Tensor) -> torch.Tensor:
        return obs.float() / 255.0 - 0.5

    def transform_reward(self, reward: torch.Tensor) -> torch.Tensor:
        if self.config.train.reward_transform == "tanh":
            return torch.tanh(reward)
        return reward

    def _autocast(self):
        if not self.amp:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=torch.float16)

    def _optimize(
        self, name: str, optimizer: torch.optim.Optimizer, loss: torch.Tensor, params
    ) -> torch.Tensor:
        """Backward, clip, step. Returns the gradient norm before clipping."""
        if name not in self._scalers:
            self._scalers[name] = torch.amp.GradScaler(self.device.type, enabled=self.amp)
        scaler = self._scalers[name]
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = nn.utils.clip_grad_norm_(params, self.config.train.grad_clip)
        scaler.step(optimizer)  # skipped if the fp16 gradients overflowed
        scaler.update()
        return norm

    def train_step(self, batch: Batch) -> dict[str, float]:
        post, metrics = self.train_world_model(batch)
        metrics.update(self.train_behavior(post.detach()))
        # One device -> host transfer instead of a sync per metric.
        values = torch.stack([v.float() for v in metrics.values()]).tolist()
        return dict(zip(metrics, values))

    def train_world_model(self, batch: Batch) -> tuple[State, dict[str, torch.Tensor]]:
        t = self.config.train
        obs = self.preprocess(batch.observation)
        with self._autocast():
            embed = self.encoder(obs)
            post, prior = self.rssm.observe(embed, batch.action, batch.is_first)
            feat = post.feat

            image_loss = -self.decoder(feat).log_prob(obs).mean()
            reward_loss = -self.reward(feat).log_prob(self.transform_reward(batch.reward)).mean()
            kl_loss, kl = self.rssm.kl_loss(post, prior, t.free_nats)
            loss = t.kl_scale * kl_loss + image_loss + reward_loss
            metrics = {}
            if self.cont is not None:
                cont_loss = -self.cont(feat).log_prob(1.0 - batch.is_terminal).mean()
                loss = loss + t.continue_scale * cont_loss
                metrics["continue_loss"] = cont_loss.detach()

        grad_norm = self._optimize(
            "model", self.model_opt, loss, [p for mod in self.world_model for p in mod.parameters()]
        )

        metrics.update(
            model_loss=loss.detach(),
            image_loss=image_loss.detach(),
            reward_loss=reward_loss.detach(),
            kl=kl.detach(),
            prior_entropy=prior.dist.entropy().mean().detach(),
            post_entropy=post.dist.entropy().mean().detach(),
            model_grad_norm=grad_norm.detach(),
        )
        return post, metrics

    def imagine(self, start: State, horizon: int) -> tuple[State, torch.Tensor]:
        """Roll the actor out in latent space. Returns states s_0..s_H stacked on
        dim 0 and the policy entropy at s_0..s_{H-1}."""
        state = start
        states, entropies = [start], []
        for _ in range(horizon):
            # The actor sees stopped-gradient features; gradients reach it through
            # the actions it feeds into the dynamics.
            dist = self.actor(state.feat.detach())
            state = self.rssm.img_step(state, dist.rsample())
            states.append(state)
            entropies.append(dist.entropy())
        return State.stack(states, 0), torch.stack(entropies)

    def train_behavior(self, post: State) -> dict[str, torch.Tensor]:
        t = self.config.train
        start = post.map(lambda x: x.reshape(-1, x.shape[-1]))

        # Gradients flow through the world model and value net into the actions,
        # but only the actor's parameters are updated here.
        with freeze(self.world_model + [self.value]), self._autocast():
            states, entropy = self.imagine(start, t.horizon)
            feat = states.feat  # (H + 1, N, F)
            reward = self.reward(feat).mean
            value = self.value(feat).mean
            if self.cont is not None:
                discount = t.discount * self.cont(feat).mean
            else:
                discount = torch.full_like(reward, t.discount)
            returns = lambda_return(reward, value, discount, t.lambda_)  # (H, N)
            # Weight step t by the probability of still being in the episode.
            weights = torch.cumprod(
                torch.cat([torch.ones_like(discount[:1]), discount[1:-1]]), 0
            ).detach()
            actor_loss = -(weights * returns).mean()
            if t.actor_entropy:
                actor_loss = actor_loss - t.actor_entropy * (weights * entropy).mean()

        actor_norm = self._optimize("actor", self.actor_opt, actor_loss, self.actor.parameters())

        with self._autocast():
            value_dist = self.value(feat[:-1].detach())
            value_loss = -(weights * value_dist.log_prob(returns.detach())).mean()
        value_norm = self._optimize("value", self.value_opt, value_loss, self.value.parameters())

        return {
            "actor_loss": actor_loss.detach(),
            "value_loss": value_loss.detach(),
            "actor_entropy": entropy.mean().detach(),
            "imag_reward": reward.mean().detach(),
            "imag_value": value.mean().detach(),
            "imag_return": returns.mean().detach(),
            "actor_grad_norm": actor_norm.detach(),
            "value_grad_norm": value_norm.detach(),
        }

    # ------------------------------------------------------------------ acting

    @torch.no_grad()
    def policy(
        self,
        obs: np.ndarray,
        state: PolicyState | None,
        explore: bool,
        expl_amount: float = 0.0,
    ) -> tuple[np.ndarray | int, PolicyState]:
        """Select an action for a single observation. Pass ``state=None`` at the
        start of an episode. Returns the env action and the next policy state."""
        device = self.device
        if state is None:
            state = (self.rssm.initial(1, device), torch.zeros(1, self.action_dim, device=device))
        latent, prev_action = state
        obs_t = self.preprocess(torch.as_tensor(obs, device=device).unsqueeze(0))
        latent, _ = self.rssm.obs_step(latent, prev_action, self.encoder(obs_t))
        dist = self.actor(latent.feat)
        action = dist.rsample() if explore else dist.mode()
        if explore and expl_amount > 0:
            action = self._explore(action, expl_amount)
        return self.to_env_action(action[0]), (latent, action)

    def _explore(self, action: torch.Tensor, amount: float) -> torch.Tensor:
        if self.discrete:
            if np.random.rand() < amount:
                index = torch.randint(self.action_dim, (action.shape[0],), device=action.device)
                return nn.functional.one_hot(index, self.action_dim).float()
            return action
        return torch.clamp(action + amount * torch.randn_like(action), -1.0, 1.0)

    def to_env_action(self, action: torch.Tensor) -> np.ndarray | int:
        if self.discrete:
            return int(action.argmax(-1).item())
        return action.cpu().numpy().astype(np.float32)

    def to_buffer_action(self, env_action: np.ndarray | int) -> np.ndarray:
        if self.discrete:
            return np.eye(self.action_dim, dtype=np.float32)[env_action]
        return np.asarray(env_action, dtype=np.float32)

    # ------------------------------------------------------------------ diagnostics

    @torch.no_grad()
    def video_prediction(self, batch: Batch, context: int = 5, length: int = 20) -> torch.Tensor:
        """Reconstruct ``context`` frames, then predict open loop from the real
        actions. Returns a (C, rows * 3H, length * W) image in [0, 1] whose row
        triplets show truth, model and error."""
        n = min(6, batch.observation.shape[0])
        length = min(length, batch.observation.shape[1])
        obs = self.preprocess(batch.observation[:n, :length])
        action = batch.action[:n, :length]
        embed = self.encoder(obs[:, :context])
        post, _ = self.rssm.observe(embed, action[:, :context], batch.is_first[:n, :context])
        state = post.map(lambda x: x[:, -1])
        imagined = []
        for t in range(context, length):
            state = self.rssm.img_step(state, action[:, t])
            imagined.append(state)
        feats = [post.feat]
        if imagined:
            feats.append(State.stack(imagined, 1).feat)
        model = self.decoder(torch.cat(feats, 1)).mean + 0.5
        truth = obs + 0.5
        error = (model - truth + 1) / 2
        grid = torch.cat([truth, model, error], 3)  # (n, T, C, 3H, W)
        grid = grid.permute(2, 0, 3, 1, 4)  # (C, n, 3H, T, W)
        c, rows, h, cols, w = grid.shape
        return grid.reshape(c, rows * h, cols * w).clamp(0, 1).cpu()

    # ------------------------------------------------------------------ checkpoints

    def checkpoint(self) -> dict:
        return {
            "model": self.state_dict(),
            "model_opt": self.model_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "value_opt": self.value_opt.state_dict(),
            "scalers": {name: scaler.state_dict() for name, scaler in self._scalers.items()},
        }

    def load_checkpoint(self, ckpt: dict) -> None:
        self.load_state_dict(ckpt["model"])
        self.model_opt.load_state_dict(ckpt["model_opt"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.value_opt.load_state_dict(ckpt["value_opt"])
        for name, state in ckpt.get("scalers", {}).items():
            if state:  # empty when the checkpoint was trained without amp
                self._scalers[name] = torch.amp.GradScaler(self.device.type, enabled=self.amp)
                self._scalers[name].load_state_dict(state)
