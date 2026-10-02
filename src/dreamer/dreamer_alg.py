from typing import Sequence
from einops import rearrange
import itertools
import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
from torch.optim import AdamW
import torch.optim as optim
from dreamer.models.rssm import RSSM
from dreamer.models.encoder import ObservationEncoder
from dreamer.models.decoder import Decoder
from dreamer.models.actor import Actor
from dreamer.models.reward import RewardModel
from dreamer.models.critic import Critic
from dreamer.models.mlp import MLP
from dreamer.models.continue_model import ContinueModel
from dreamer.replay_buffer import ReplayBuffer
from dataclasses import dataclass, fields, field
from dreamer.config import DreamerConfig, RSSMTrajectory, ImaginationTrajectory
from dreamer.utils import create_normal_dist_from_params, frozen


def preprocess_obs(obs, device):
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs)

    # uint8 images in [0, 255] -> floats in [-0.5, 0.5]
    obs = obs.float() / 255.0 - 0.5

    if obs.ndim == 3:
        obs = obs.unsqueeze(0).unsqueeze(0)

    return obs.to(device)


torch.distributions.Distribution.set_default_validate_args(False)


class Dreamer(nn.Module):
    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config = config
        self.device = config.device

        self.buffer = ReplayBuffer(
            config.observation_shape,
            config.action_dim,
            self.device,
            config.replay_buffer_size,
        )

        self.rssm = RSSM(config.action_dim, config)
        self.encoder = ObservationEncoder(3, config)
        self.decoder = Decoder(config=config)
        self.reward_model = RewardModel(config.rssm.z_dim + config.rssm.h_dim)
        self.actor = Actor(config)
        self.critic = Critic(config)
        self.continue_model = ContinueModel(config)
        self.dynamic_trajectory = RSSMTrajectory()
        self.imagination_trajectory = ImaginationTrajectory()

        self.world_params = params = list(
            itertools.chain(
                self.rssm.parameters(),
                self.reward_model.parameters(),
                self.decoder.parameters(),
                self.encoder.parameters(),
                self.continue_model.parameters(),
            )
        )

        device_type = torch.device(self.device).type
        fused = device_type == "cuda"
        self.world_optimizer = optim.Adam(params, lr=config.world_lr, fused=fused)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.actor_lr, fused=fused
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.critic_lr, fused=fused
        )

        # mixed precision: autocast for the forward passes, plus loss scaling for fp16
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
            config.mixed_precision
        )
        use_scaler = config.mixed_precision == "fp16"
        self.world_scaler, self.actor_scaler, self.critic_scaler = (
            torch.amp.GradScaler(device_type, enabled=use_scaler) for _ in range(3)
        )

        if config.compile:
            # the per-step modules called inside the Python loops of
            # learn_dynamics/learn_behavior; distributions stay eager.
            # Training, imagination and acting each need their own variant (shapes,
            # requires_grad, autocast), more than dynamo's default limit of 8.
            torch._dynamo.config.recompile_limit = 32
            self.rssm.recurrent = torch.compile(self.rssm.recurrent)
            self.rssm.posterior_mlp.compile()
            self.rssm.prior_mlp.compile()
            self.actor.mlp.compile()

        self.num_total_episode = 0

    def autocast(self):
        return torch.autocast(
            torch.device(self.device).type,
            dtype=self.amp_dtype,
            enabled=self.amp_dtype is not None,
        )

    def _update(self, loss, optimizer, scaler, params, norm_type=2.0):
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(params, self.config.grad_clip, norm_type=norm_type)
        scaler.step(optimizer)
        scaler.update()

    def learn_dynamics(self, data):
        """
        Learn world model via learning representation and transition model

        Args:
            data: batch that includes observation, action, and reward
        """
        data.observation = preprocess_obs(data.observation, self.device)
        with self.autocast():
            obs_embed = self.encoder(data.observation)  # B, S, Obs_emb_dim
            z, h = self.rssm.state_init(self.config.batch_size)

            for t in range(1, self.config.seq_length):
                # a sampled chunk may span two episodes: reset the state where one ended
                keep = 1.0 - data.done[:, t - 1]
                z, h = z * keep, h * keep
                h = self.rssm.recurrent(z, data.action[:, t - 1] * keep, h)
                z, posterior_dist = self.rssm.observe(h, obs_embed[:, t])
                self.dynamic_trajectory.append(
                    posterior=z,
                    posterior_mean=posterior_dist.mean,
                    posterior_std=posterior_dist.stddev,
                    h=h,
                )

            traj = self.dynamic_trajectory.stack_fields()
            # the prior depends only on h, so it runs once over the whole sequence
            # instead of once per step inside the loop
            _, prior_dist = self.rssm.transition(traj.h)
        losses = self.optimize_world_model(data, traj, prior_dist)
        return (
            traj.posterior.detach(),
            traj.h.detach(),
            losses,
        )  # why detached are returned here

    def learn_behavior(
        self, zs: torch.Tensor, hs: torch.Tensor
    ) -> ImaginationTrajectory:
        """
        Learn the behavior by imagining trajectories
        """
        # The actor loss backpropagates through the world model and the critic, but
        # only the actor is updated, so skip computing their weight gradients.
        with frozen(
            self.rssm, self.reward_model, self.continue_model, self.critic
        ), self.autocast():
            ss = rearrange(zs, "b l s -> (b l) s")  # (B*L, s_dim)
            hh = rearrange(hs, "b l h -> (b l) h")  # (B*L, h_dim)
            for _ in range(self.config.horizon):
                action_dist, action = self.actor(hh, ss)
                hh = self.rssm.recurrent(ss, action, hh)
                ss, _ = self.rssm.transition(hh)
                self.imagination_trajectory.append(prior=ss, h=hh)

            traj = self.imagination_trajectory.stack_fields()

            reward_pred = self.reward_model(traj.h, traj.prior).mean
            values = self.critic(traj.h, traj.prior).mean
            continues = self.continue_model(traj.prior, traj.h).mean

            lambda_values = self.critic.compute_lambda_return(
                reward_pred, values, self.config.discount * continues, self.config.lam
            )

            # the last entry is only the bootstrap value
            actor_loss = self.actor.actor_loss(lambda_values[:, :-1])
            self._update(
                actor_loss, self.actor_optimizer, self.actor_scaler, self.actor.parameters()
            )

        with self.autocast():
            value_dist = self.critic(
                traj.h.detach()[:, :-1],
                traj.prior.detach()[:, :-1],
            )
            critic_loss = -torch.mean(
                value_dist.log_prob(lambda_values[:, :-1].detach())
            )
        self._update(
            critic_loss, self.critic_optimizer, self.critic_scaler, self.critic.parameters()
        )

        losses = {
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "value_mean": values.detach().mean(),
            "reward_mean": reward_pred.detach().mean(),
        }
        return traj, losses

    def optimize_world_model(self, data, trajectory: RSSMTrajectory, prior_dist):
        with self.autocast():
            decoded_obs_dist = self.decoder(
                trajectory.h, trajectory.posterior
            )  # takes posterior and h
            continue_dist = self.continue_model(trajectory.posterior, trajectory.h)
            reward_dist = self.reward_model(trajectory.h, trajectory.posterior)

        reconstruction_observation_loss = decoded_obs_dist.log_prob(
            data.observation[:, 1:]
        )

        # The state at step t is inferred from observation t, which was reached by
        # action t-1, so it predicts reward[t-1] and terminated[t-1]. Where step t-1
        # ended an episode, observation t starts a new one: no reward, not terminal.
        keep = 1.0 - data.done[:, :-1]
        # binary cross-entropy, computed from the logits
        continue_loss = -continue_dist.log_prob(1 - data.terminated[:, :-1] * keep)

        reward_loss = reward_dist.log_prob(data.reward[:, :-1] * keep)

        prior_dist = create_normal_dist_from_params(
            prior_dist.mean, prior_dist.stddev, event_shape=1
        )
        posterior_dist = create_normal_dist_from_params(
            trajectory.posterior_mean, trajectory.posterior_std, event_shape=1
        )

        kl_divergence_loss = torch.mean(
            torch.distributions.kl.kl_divergence(posterior_dist, prior_dist)
        )
        kl_divergence_loss = kl_divergence_loss.clamp(min=self.config.free_nats)

        world_model_loss = (
            kl_divergence_loss
            - reconstruction_observation_loss.mean()
            - reward_loss.mean()
            + continue_loss.mean()
        )
        self._update(
            world_model_loss,
            self.world_optimizer,
            self.world_scaler,
            self.world_params,
            norm_type=self.config.grad_norm_type,
        )

        # tensors, not floats: .item() would sync with the GPU on every update
        return {
            "world_loss": world_model_loss.detach(),
            "reward_loss": -reward_loss.detach().mean(),
            "reconstruction_loss": -reconstruction_observation_loss.detach().mean(),
            "kl_loss": kl_divergence_loss.detach(),
        }

    @torch.no_grad()
    def policy(self, observation, state=None, explore=False):
        """
        One acting step: update the latent state with the new observation and pick an action.

        state carries (posterior, h, previous action) between steps; pass None at the
        start of an episode. Returns the action as a numpy array and the new state.
        """
        if state is None:
            posterior, h = self.rssm.state_init(1)
            action = torch.zeros(1, self.config.action_dim, device=self.device)
        else:
            posterior, h, action = state

        embedded_observation = self.encoder(preprocess_obs(observation, self.device))
        h = self.rssm.recurrent(posterior, action, h)
        posterior, _ = self.rssm.observe(h, embedded_observation.reshape(1, -1))

        action_dist, action = self.actor(h, posterior)
        if explore:
            # Gaussian exploration noise (Dreamer: 0.3)
            action = action + self.config.expl_noise * torch.randn_like(action)
            action = action.clamp(-1.0, 1.0)
        else:
            # the deterministic policy
            action = torch.tanh(action_dist.base_dist.mean)

        return action.cpu().numpy()[0], (posterior, h, action)

    def environment_interaction(self, env, num_interaction_episodes, train=True):
        score_lst = []
        for _epi in range(num_interaction_episodes):
            observation, _info = env.reset()
            state = None
            score = 0.0
            done = False

            while not done:
                env_action, state = self.policy(observation, state, explore=train)
                if hasattr(env, "action_space"):
                    env_action = np.clip(
                        env_action, env.action_space.low, env.action_space.high
                    )

                next_observation, reward, terminated, truncated, info = env.step(
                    env_action
                )
                done = terminated or truncated

                if train:
                    self.buffer.add(
                        observation, env_action, reward, next_observation, done, terminated
                    )
                score += reward
                observation = next_observation

            if train:
                self.num_total_episode += 1
            score_lst.append(score)

        return score_lst

    def evaluate(self, env):
        return self.environment_interaction(env, self.config.num_evaluate, train=False)
