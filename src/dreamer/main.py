"""
Usage:
    accelerate launch main.py --env-name cartpole_balance --num-iterations 1000
    accelerate launch main.py --env-backend gym --env-name Pendulum-v1 --mixed-precision fp16
"""

import argparse
import os
import random

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from dreamer.config import DreamerConfig
from dreamer.envs import make_dmc_env
from dreamer.dreamer_alg import Dreamer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default=None)
    parser.add_argument("--env-backend", type=str, default=None, choices=["dmc", "gym"])
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--mixed-precision", type=str, default=None, choices=["no", "fp16", "bf16"]
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="path to a checkpoint to resume from"
    )
    return parser.parse_args()


def apply_overrides(config: DreamerConfig, args: argparse.Namespace) -> DreamerConfig:
    if args.env_name is not None:
        config.environment.env_name = args.env_name
    if args.env_backend is not None:
        config.env_backend = args.env_backend
    if args.num_iterations is not None:
        config.num_iterations = args.num_iterations
    if args.seed is not None:
        config.seed = args.seed
    if args.mixed_precision is not None:
        config.mixed_precision = args.mixed_precision
    return config


def save_checkpoint(dreamer: Dreamer, iteration: int, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "model_state": dreamer.state_dict(),
            "world_optimizer": dreamer.world_optimizer.state_dict(),
            "actor_optimizer": dreamer.actor_optimizer.state_dict(),
            "critic_optimizer": dreamer.critic_optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(dreamer: Dreamer, path: str, map_location="cpu") -> int:
    ckpt = torch.load(path, map_location=map_location)
    dreamer.load_state_dict(ckpt["model_state"])
    dreamer.world_optimizer.load_state_dict(ckpt["world_optimizer"])
    dreamer.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
    dreamer.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
    return ckpt["iteration"]


def main() -> None:
    args = parse_args()
    config = DreamerConfig()
    set_seed(config.seed)
    device = config.device
    env_config = config.environment

    env = make_dmc_env(
        env_config.domain_name,
        env_config.task_name,
        env_config.seed,
        env_config.visualize_reward,
        env_config.from_pixels,
        env_config.height,
        env_config.width,
        env_config.frame_skip,
        env_config.pixel_norm,
    )

    dreamer = Dreamer(config)

    start_iteration = 0
    if args.resume:
        start_iteration = load_checkpoint(dreamer, args.resume, map_location=device)

    writer = SummaryWriter(config.log_dir)

    progress = tqdm(
        range(start_iteration, config.num_iterations),
        desc="Training",
        dynamic_ncols=True,
    )

    if len(dreamer.buffer) < 1:
        dreamer.environment_interaction(env, config.seed_episodes)

    for iteration in progress:
        data = dreamer.buffer.sample(config.batch_size, config.seq_length)
        posteriors, h, world_stats = dreamer.learn_dynamics(data)
        traj, behavior_stats = dreamer.learn_behavior(posteriors, h)

        progress.set_postfix(
            world=f"{world_stats['world_loss']:.2f}",
            recon=f"{world_stats['reconstruction_loss']:.2f}",
            kl=f"{world_stats['kl_loss']:.3f}",
            actor=f"{behavior_stats['actor_loss']:.2f}",
            critic=f"{behavior_stats['critic_loss']:.2f}",
            replay=len(dreamer.buffer),
            episodes=dreamer.num_total_episode,
        )

        dreamer.environment_interaction(env, config.num_interaction_episodes)

        if writer is not None and iteration % 10 == 0:
            writer.add_scalar(
                "train/episode_count", dreamer.num_total_episode, iteration
            )

        if iteration % config.eval_every == 0:
            dreamer.evaluate(env)

        if iteration % config.checkpoint_every == 0:
            save_checkpoint(
                dreamer,
                iteration,
                os.path.join(config.checkpoint_dir, f"dreamer_{iteration}.pt"),
            )

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
