"""
Usage:
    python -m dreamer.main --env-name walker_walk
    python -m dreamer.main --env-name cartpole_balance --num-iterations 200 --resume checkpoints/dreamer/dreamer_100.pt
"""

import argparse
import os
import random
import sys
import types

import numpy as np
import torch

# If TensorFlow is installed (e.g. on Kaggle), TensorBoard imports it, which loads
# Keras/JAX and makes MuJoCo's OpenGL rendering segfault. This marker makes
# TensorBoard use its built-in TensorFlow stub instead.
sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("notf"))
from torch.utils.tensorboard import SummaryWriter  # noqa: E402
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
    parser.add_argument(
        "--env-name", type=str, default=None, help="DMC <domain>_<task>, e.g. walker_walk"
    )
    parser.add_argument("--env-backend", type=str, default=None, choices=["dmc", "gym"])
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--mixed-precision", type=str, default=None, choices=["no", "fp16", "bf16"]
    )
    parser.add_argument(
        "--compile", action="store_true", help="torch.compile the per-step RSSM/actor modules"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="path to a checkpoint to resume from"
    )
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda (default: auto)")
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    return parser.parse_args()


def apply_overrides(config: DreamerConfig, args: argparse.Namespace) -> DreamerConfig:
    if args.env_name is not None:
        domain, task = args.env_name.split("_", 1)
        if domain == "ball" and task.startswith("in_cup_"):  # ball_in_cup_catch
            domain, task = "ball_in_cup", task[len("in_cup_") :]
        config.environment.domain_name = domain
        config.environment.task_name = task
    if args.env_backend is not None:
        config.env_backend = args.env_backend
    if args.num_iterations is not None:
        config.num_iterations = args.num_iterations
    if args.seed is not None:
        config.seed = config.environment.seed = args.seed
    if args.mixed_precision is not None:
        config.mixed_precision = args.mixed_precision
    if args.compile:
        config.compile = True
    if args.log_dir is not None:
        config.log_dir = args.log_dir
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    config.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return config


def save_checkpoint(dreamer: Dreamer, iteration: int, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "num_total_episode": dreamer.num_total_episode,
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
    dreamer.num_total_episode = ckpt.get("num_total_episode", 0)
    return ckpt["iteration"]


def main() -> None:
    args = parse_args()
    config = apply_overrides(DreamerConfig(), args)
    set_seed(config.seed)
    device = config.device
    env_config = config.environment

    if torch.device(device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # input shapes are fixed

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
    config.action_dim = env.action_space.shape[0]
    # Replay never needs more steps than the run collects (DMC: 1000 env steps per episode).
    episode_steps = 1000 // env_config.frame_skip
    episodes = config.seed_episodes + config.num_iterations * config.num_interaction_episodes
    config.replay_buffer_size = min(config.replay_buffer_size, episodes * episode_steps)

    dreamer = Dreamer(config).to(device)
    print(
        f"{env_config.domain_name}_{env_config.task_name} on {device}, "
        f"replay {config.replay_buffer_size} steps"
    )

    start_iteration = 0
    if args.resume:
        start_iteration = load_checkpoint(dreamer, args.resume, map_location=device) + 1

    writer = SummaryWriter(config.log_dir)

    progress = tqdm(
        range(start_iteration, config.num_iterations),
        desc="Training",
        dynamic_ncols=True,
    )

    if len(dreamer.buffer) < 1:
        dreamer.environment_interaction(env, config.seed_episodes)

    for iteration in progress:
        for _ in range(config.train_steps):
            data = dreamer.buffer.sample(config.batch_size, config.seq_length)
            posteriors, h, world_stats = dreamer.learn_dynamics(data)
            traj, behavior_stats = dreamer.learn_behavior(posteriors, h)
        # one GPU sync per iteration, for the stats of the last update
        world_stats = {k: v.item() for k, v in world_stats.items()}
        behavior_stats = {k: v.item() for k, v in behavior_stats.items()}

        scores = dreamer.environment_interaction(env, config.num_interaction_episodes)

        progress.set_postfix(
            world=f"{world_stats['world_loss']:.2f}",
            recon=f"{world_stats['reconstruction_loss']:.2f}",
            kl=f"{world_stats['kl_loss']:.3f}",
            actor=f"{behavior_stats['actor_loss']:.2f}",
            critic=f"{behavior_stats['critic_loss']:.2f}",
            score=f"{np.mean(scores):.1f}",
            replay=len(dreamer.buffer),
            episodes=dreamer.num_total_episode,
        )

        env_steps = dreamer.num_total_episode * 1000  # DMC episodes are 1000 env steps
        for name, value in {**world_stats, **behavior_stats}.items():
            writer.add_scalar(f"train/{name}", value, env_steps)
        writer.add_scalar("train/score", np.mean(scores), env_steps)
        writer.add_scalar("train/episode_count", dreamer.num_total_episode, env_steps)

        if iteration % config.eval_every == 0:
            eval_score = float(np.mean(dreamer.evaluate(env)))
            writer.add_scalar("eval/score", eval_score, env_steps)
            tqdm.write(f"iteration {iteration}: eval score {eval_score:.1f}")

        if iteration % config.checkpoint_every == 0 or iteration == config.num_iterations - 1:
            save_checkpoint(
                dreamer,
                iteration,
                os.path.join(config.checkpoint_dir, f"dreamer_{iteration}.pt"),
            )

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
