"""
Evaluate a trained Dreamer agent and optionally record a video.

    python -m dreamer.evaluate runs/dmc_walker_walk_s0 --episodes 10 --video walker.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dreamer.config import Config
from dreamer.dreamer_alg import Dreamer
from dreamer.envs import make_env
from dreamer.utils import resolve_device, set_seed


def load_agent(run_dir: Path, checkpoint: str, device: torch.device) -> tuple[Dreamer, Config]:
    config = Config.from_dict(json.loads((run_dir / "config.json").read_text()))
    env = make_env(config.env, config.run.seed)
    agent = Dreamer(env.observation_space.shape, env.action_space, config).to(device)
    env.close()
    state = torch.load(run_dir / checkpoint, map_location=device, weights_only=False)
    agent.load_state_dict(state["model"] if "model" in state else state)
    agent.eval()
    return agent, config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path, help="run directory containing config.json")
    parser.add_argument("--checkpoint", default="latest.pt", help="file inside run_dir")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--video", type=Path, help="write the first episode to this .mp4")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = resolve_device(args.device)
    agent, config = load_agent(args.run_dir, args.checkpoint, device)
    env = make_env(config.env, args.seed)

    returns = []
    for episode in range(args.episodes):
        obs, _ = env.reset()
        state, done, total, frames = None, False, 0.0, [obs]
        while not done:
            action, state = agent.policy(obs, state, explore=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
            if args.video and episode == 0:
                frames.append(obs)
        returns.append(total)
        print(f"episode {episode}: return {total:.1f}", flush=True)
        if args.video and episode == 0:
            write_video(args.video, frames)
    print(
        f"mean return {np.mean(returns):.1f} +- {np.std(returns):.1f} over {len(returns)} episodes"
    )


def write_video(path: Path, frames: list[np.ndarray], fps: int = 30, scale: int = 4) -> None:
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
