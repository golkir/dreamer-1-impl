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
from dreamer.replay_buffer import ReplayBuffer
from dataclasses import dataclass, fields, field
from dreamer.config import DreamerConfig, RSSMTrajectory, ImaginationTrajectory
from dreamer.utils import create_normal_dist_from_params


def preprocess_obs(obs, device):
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs)

    obs = obs.float()

    if obs.ndim == 3:
        obs = obs.unsqueeze(0).unsqueeze(0)

    return obs.to(device)


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
        self.decoder = Decoder(
            latent_dim=config.rssm.h_dim + config.rssm.z_dim, config=config
        )
        self.reward_model = RewardModel(config.rssm.z_dim + config.rssm.h_dim)
        self.actor = Actor(config)
        self.critic = Critic(config)
        self.dynamic_trajectory = RSSMTrajectory()
        self.imagination_trajectory = ImaginationTrajectory()

        params = list(
            itertools.chain(
                self.rssm.parameters(),
                self.reward_model.parameters(),
                self.decoder.parameters(),
                self.encoder.parameters(),
            )
        )

        print("before adam")
        self.world_optimizer = optim.Adam(
            params,
            lr=config.world_lr,
        )

        print("after adam")
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.critic_lr
        )

        self.num_total_episode = 0

    def learn_dynamics(self, data):
        """
        Learn world model via learning representation and transition model

        Args:
            data: batch that includes observation, action, and reward
        """
        B = self.config.batch_size
        obs_embed = self.encoder(data.observation)  # B, S, Obs_emb_dim
        prior, h = self.rssm.state_init(self.config.batch_size)

        for t in range(1, self.config.seq_length):
            h = self.rssm.recurrent(prior, data.action[:, t - 1], h)
            posterior_sample, posterior_dist = self.rssm.observe(h, obs_embed[:, t])
            prior, prior_dist = self.rssm.transition(h)
            self.dynamic_trajectory.append(
                prior=prior,
                prior_mean=prior_dist.mean,
                prior_std=prior_dist.stddev,
                posterior=posterior_sample,
                posterior_mean=posterior_dist.mean,
                posterior_std=posterior_dist.stddev,
                h=h,
            )

            prior = posterior_sample

        traj = self.dynamic_trajectory.stack_fields()
        losses = self.optimize_world_model(data, traj)
        return traj.posterior.detach(), traj.h.detach(), losses

    def learn_behavior(
        self, zs: torch.Tensor, hs: torch.Tensor
    ) -> ImaginationTrajectory:
        """
        Learn the behavior by imagining trajectories
        """
        ss = rearrange(zs, "b l s -> (b l) s")  # (B*L, s_dim)
        hh = rearrange(hs, "b l h -> (b l) h")  # (B*L, h_dim)
        for _ in range(self.config.horizon):
            action_dist, action = self.actor(hh, ss)
            hh = self.rssm.recurrent(ss, action, hh)
            ss, _ = self.rssm.transition(hh)
            self.imagination_trajectory.append(prior=ss, h=hh)

        traj = self.imagination_trajectory.stack_fields()

        losses = self.optimize_behavior(traj)

        return traj, losses

    def optimize_world_model(self, data, trajectory: RSSMTrajectory):
        
        decoded_obs_dist = self.decoder(trajectory.h, trajectory.prior)
        reward_dist = self.reward_model(trajectory.h, trajectory.posterior)

        reward_loss = reward_dist.log_prob(data.reward[:, 1:])  # log likelihood

        prior_dist = create_normal_dist_from_params(
            trajectory.prior_mean, trajectory.prior_std, event_shape=1
        )
        posterior_dist = create_normal_dist_from_params(
            trajectory.posterior_mean, trajectory.posterior_std, event_shape=1
        )

        kl_divergence_loss = torch.mean(
            torch.distributions.kl.kl_divergence(prior_dist, posterior_dist)
        )
        kl_divergence_loss = torch.max(
            torch.tensor(self.config.free_nats).to(self.device), kl_divergence_loss
        )
        reconstruction_observation_loss = decoded_obs_dist.log_prob(
            data.observation[:, 1:]
        )
        world_model_loss = (
            kl_divergence_loss
            - reconstruction_observation_loss.mean()
            - reward_loss.mean()
        )
        self.world_optimizer.zero_grad()
        world_model_loss.backward()
        self.world_optimizer.step()

        return {
            "world_loss": world_model_loss.item(),
            "reward_loss": (-reward_loss.mean()).item(),
            "reconstruction_loss": (-reconstruction_observation_loss.mean()).item(),
            "kl_loss": kl_divergence_loss.item(),
        }

    def optimize_behavior(self, traj: ImaginationTrajectory):

        reward_pred = self.reward_model(traj.h, traj.prior).mean

        print(traj.h.shape, "Traj h shape")
        print(traj.prior.shape, "Prior")

        values = self.critic(traj.h, traj.prior).mean
        lambda_values = self.critic.compute_lambda_return(
            reward_pred, values
        )  # this function should be checked

        actor_loss = self.actor.actor_loss(lambda_values)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
        self.actor_optimizer.step()

        value_dist = self.critic(
            traj.prior.detach()[:, :-1],
            traj.h.detach()[:, :-1],
        )

        critic_loss = -torch.mean(value_dist.log_prob(lambda_values[:, :-1].detach()))
        self.critic_optimizer.zero_grad()

        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip)
        self.critic_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "value_mean": values.mean().item(),
            "reward_mean": reward_pred.mean().item(),
        }

    @torch.no_grad()
    def environment_interaction(self, env, num_interaction_episodes, train=True):
        score_lst = []
        for _epi in range(num_interaction_episodes):
            posterior, h = self.rssm.state_init(1)
            action = torch.zeros(1, self.config.action_dim).to(self.device)

            observation, _info = env.reset()

            embedded_observation = self.encoder(
                preprocess_obs(observation, self.device)
            )

            print(embedded_observation.shape, "emb shae")

            score = 0.0
            done = False

            while not done:
                h = self.rssm.recurrent(posterior, action, h)
                embedded_observation = embedded_observation.reshape(1, -1)
                posterior, _ = self.rssm.observe(h, embedded_observation)

                action_dist, action = self.actor(posterior, h)
                action = action.detach()

                env_action = action.cpu().numpy()[0]
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
                        observation, env_action, reward, next_observation, done
                    )
                score += reward
                embedded_observation = self.encoder(
                    preprocess_obs(next_observation, self.device)
                )
                observation = next_observation

            if train:
                self.num_total_episode += 1
            else:
                score_lst.append(score)

        if not train and score_lst:
            evaluate_score = float(np.mean(score_lst))
            print("evaluate score : ", evaluate_score)

    def evaluate(self, env):
        self.environment_interaction(env, self.config.num_evaluate, train=False)
