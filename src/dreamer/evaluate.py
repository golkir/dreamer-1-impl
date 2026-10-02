"""
Evaluate a trained Dreamer agent and optionally record a video.

    python -m dreamer.evaluate checkpoints/dreamer --episodes 10 --video walker.mp4
    python -m dreamer.evaluate checkpoints/dreamer/dreamer_100.pt --env-name walker_walk

The checkpoint can be a dreamer_<iteration>.pt file or a directory (the latest one is
used). Checkpoints saved before the config was stored need --env-name.
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import torch

from dreamer.checkpoint import checkpoint_config, latest_checkpoint, load_checkpoint
from dreamer.config import DreamerConfig
from dreamer.dreamer_alg import Dreamer
from dreamer.envs import make_dmc_env
from dreamer.main import parse_env_name, set_seed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", help="dreamer_<iteration>.pt, or a directory of them")
    parser.add_argument(
        "--env-name", help="DMC <domain>_<task>; required for checkpoints without a config"
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    parser.add_argument("--video", help="write the first episode to this .mp4")
    parser.add_argument("--fps", type=int, default=None, help="default: real time")
    args = parser.parse_args(argv)

    path = args.checkpoint
    if os.path.isdir(path):
        path = latest_checkpoint(path)

    config = checkpoint_config(path)
    if config is None:
        if args.env_name is None:
            parser.error(f"{path} doesn't store its environment; pass --env-name")
        config = DreamerConfig()
    if args.env_name is not None:
        config.environment.domain_name, config.environment.task_name = parse_env_name(
            args.env_name
        )
    config.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config.compile = False
    config.mixed_precision = "no"
    config.replay_buffer_size = 1  # acting only

    set_seed(args.seed)
    env_config = config.environment
    env = make_dmc_env(
        env_config.domain_name,
        env_config.task_name,
        args.seed,
        height=env_config.height,
        width=env_config.width,
        frame_skip=env_config.frame_skip,
    )
    config.action_dim = env.action_space.shape[0]

    agent = Dreamer(config).to(config.device)
    iteration = load_checkpoint(agent, path, map_location=config.device, optimizers=False)
    agent.eval()
    print(
        f"{path} (iteration {iteration}) on "
        f"{env_config.domain_name}_{env_config.task_name}, {config.device}"
    )

    returns = []
    for episode in range(args.episodes):
        obs, _ = env.reset()
        state, done, total, frames = None, False, 0.0, [obs]
        while not done:
            action, state = agent.policy(obs, state, explore=False)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
            if args.video and episode == 0:
                frames.append(obs)
        returns.append(total)
        print(f"episode {episode}: return {total:.1f}", flush=True)
        if args.video and episode == 0:
            # one frame per agent step, i.e. per control_timestep * frame_skip seconds
            fps = args.fps or round(1 / (env.env.control_timestep() * env.action_repeat))
            write_video(args.video, frames, fps)
    env.close()
    print(
        f"mean return {np.mean(returns):.1f} +- {np.std(returns):.1f} over {len(returns)} episodes"
    )


def write_video(path: str, frames: list[np.ndarray], fps: int, scale: int = 4) -> None:
    c, h, w = frames[0].shape
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * scale, h * scale)
    )
    for frame in frames:
        image = frame.transpose(1, 2, 0)
        image = np.repeat(image, 3, -1) if c == 1 else image[..., ::-1]  # RGB -> BGR
        writer.write(
            cv2.resize(
                np.ascontiguousarray(image), (w * scale, h * scale), interpolation=cv2.INTER_NEAREST
            )
        )
    writer.release()
    print(f"saved {path}")


if __name__ == "__main__":
    main()
