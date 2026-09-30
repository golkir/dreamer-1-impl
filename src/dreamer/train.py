"""
Train Dreamer.

Examples:
    # DeepMind Control, paper settings (1M env steps)
    python -m dreamer.train --preset dmc --task walker_walk

    # Atari
    python -m dreamer.train --preset atari --task pong

    # Quick end-to-end check on CPU
    python -m dreamer.train --preset debug

    # Override any config field, resume an interrupted run
    python -m dreamer.train --task cheetah_run --set train.batch_size=32 --resume

Logs (TensorBoard + metrics.jsonl), config.json and checkpoints are written to
<logdir>/<run name>/. Watch them with ``tensorboard --logdir runs``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from dreamer.config import PRESETS, Config, EnvConfig, make_config, parse_overrides, set_field
from dreamer.dreamer_alg import Dreamer
from dreamer.envs import make_env
from dreamer.replay_buffer import ReplayBuffer
from dreamer.utils import resolve_device, set_seed

EnvFn = Callable[[EnvConfig, int], gym.Env]


class Logger:
    """Averages scalars between writes; logs to TensorBoard, JSONL and stdout."""

    def __init__(self, logdir: Path):
        self.writer = SummaryWriter(str(logdir))
        self.jsonl = open(logdir / "metrics.jsonl", "a")
        self.scalars: dict[str, list[float]] = defaultdict(list)

    def scalar(self, name: str, value: float) -> None:
        self.scalars[name].append(float(value))

    def image(self, name: str, image: torch.Tensor, step: int) -> None:
        self.writer.add_image(name, image, step)

    def write(self, step: int) -> None:
        means = {k: float(np.mean(v)) for k, v in self.scalars.items() if v}
        self.scalars.clear()
        for name, value in means.items():
            self.writer.add_scalar(name, value, step)
        self.writer.flush()
        self.jsonl.write(json.dumps({"step": step, **means}) + "\n")
        self.jsonl.flush()
        labels = {
            "episode/return": "return",
            "eval/return": "eval_return",
            "train/model_loss": "model_loss",
            "train/actor_loss": "actor_loss",
            "train/value_loss": "value_loss",
            "perf/fps": "fps",
        }
        shown = {label: means[k] for k, label in labels.items() if k in means}
        print(f"[{step:>9}] " + "  ".join(f"{k} {v:.3g}" for k, v in shown.items()), flush=True)

    def close(self) -> None:
        self.writer.close()
        self.jsonl.close()


class Collector:
    """Steps a single environment and writes every transition to replay."""

    def __init__(self, env: gym.Env, buffer: ReplayBuffer, agent: Dreamer, action_repeat: int):
        self.env = env
        self.buffer = buffer
        self.agent = agent
        self.action_repeat = action_repeat
        self.obs: np.ndarray | None = None
        self.policy_state = None
        self.episode_return = 0.0
        self.episode_length = 0

    def collect(
        self, env_steps: int, random: bool, expl_amount: float = 0.0
    ) -> tuple[list[dict], int]:
        """Collect at least ``env_steps`` environment steps. Returns the finished
        episodes and the number of env steps actually taken."""
        episodes = []
        steps = 0
        while steps < env_steps:
            if self.obs is None:
                self.obs, _ = self.env.reset()
                self.policy_state = None
                self.episode_return, self.episode_length = 0.0, 0
                self.buffer.add(self.obs, np.zeros(self.agent.action_dim), 0.0, True, False)

            if random:
                action = self.env.action_space.sample()
            else:
                action, self.policy_state = self.agent.policy(
                    self.obs, self.policy_state, explore=True, expl_amount=expl_amount
                )
            self.obs, reward, terminated, truncated, _ = self.env.step(action)
            self.buffer.add(
                self.obs, self.agent.to_buffer_action(action), reward, False, terminated
            )
            self.episode_return += float(reward)
            self.episode_length += 1
            steps += self.action_repeat
            if terminated or truncated:
                episodes.append(
                    {
                        "return": self.episode_return,
                        "length": self.episode_length * self.action_repeat,
                    }
                )
                self.obs = None
        return episodes, steps


def evaluate(agent: Dreamer, env: gym.Env, episodes: int) -> list[float]:
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        state, done, total = None, False, 0.0
        while not done:
            action, state = agent.policy(obs, state, explore=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
    return returns


def exploration_amount(config: Config, env_step: int) -> float:
    t = config.train
    amount = t.expl_amount
    if t.expl_decay:
        amount *= 0.5 ** (env_step / t.expl_decay)
    return max(t.expl_min, amount)


def crossed(before: int, after: int, every: int) -> bool:
    return every > 0 and after // every > before // every


def save_checkpoint(path: Path, agent: Dreamer, counters: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({**agent.checkpoint(), "counters": counters}, tmp)
    os.replace(tmp, path)


def train(config: Config, resume: bool = False, env_fn: EnvFn = make_env) -> dict:
    """Run a full training loop. Returns the final counters and last eval score."""
    t, r = config.train, config.run
    set_seed(r.seed)
    device = resolve_device(r.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logdir = Path(r.logdir) / config.run_name
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "config.json").write_text(config.to_json())
    print(f"Run directory: {logdir}  device: {device}", flush=True)
    if device.type == "cpu":
        print("WARNING: training on CPU; this is only practical for debugging.", flush=True)

    env = env_fn(config.env, r.seed)
    eval_env = env_fn(config.env, r.seed + 10_000)
    obs_shape = env.observation_space.shape
    agent = Dreamer(obs_shape, env.action_space, config).to(device)
    # No point reserving more replay than the run can fill (+ slack for resets).
    max_steps = t.steps // config.env.action_repeat + 10_000
    capacity = min(t.buffer_capacity, max_steps)
    buffer = ReplayBuffer(obs_shape, agent.action_dim, capacity, seed=r.seed)
    collector = Collector(env, buffer, agent, config.env.action_repeat)
    logger = Logger(logdir)
    print(
        f"Parameters: {sum(p.numel() for p in agent.parameters()) / 1e6:.2f}M  "
        f"replay: {capacity} steps (up to {capacity * np.prod(obs_shape) / 2**30:.1f} GiB)",
        flush=True,
    )

    counters = {"env_step": 0, "updates": 0, "episodes": 0}
    ckpt_path = logdir / "latest.pt"
    replay_path = logdir / "replay.npz"
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        agent.load_checkpoint(ckpt)
        counters.update(ckpt["counters"])
        if replay_path.exists():
            buffer.load(str(replay_path))
        print(
            f"Resumed from {ckpt_path} at step {counters['env_step']} "
            f"with {len(buffer)} replay steps",
            flush=True,
        )
    elif resume:
        print(f"No checkpoint at {ckpt_path}; starting from scratch", flush=True)

    def log_episodes(episodes: list[dict]) -> None:
        for ep in episodes:
            counters["episodes"] += 1
            logger.scalar("episode/return", ep["return"])
            logger.scalar("episode/length", ep["length"])

    def checkpoint() -> None:
        save_checkpoint(ckpt_path, agent, counters)
        if r.save_replay:
            buffer.save(str(replay_path))

    # Fill replay before training: random actions for a fresh run, the current
    # policy when resuming without a saved buffer.
    if len(buffer) == 0:
        need = max(t.prefill_steps, (t.batch_length + 1) * config.env.action_repeat)
        fresh = counters["updates"] == 0
        print(
            f"Prefilling replay with {need} env steps ({'random' if fresh else 'policy'} actions)",
            flush=True,
        )
        episodes, steps = collector.collect(
            need, random=fresh, expl_amount=exploration_amount(config, counters["env_step"])
        )
        log_episodes(episodes)
        counters["env_step"] += steps

    last_eval = None
    timer, timer_step = time.time(), counters["env_step"]
    try:
        while counters["env_step"] < t.steps:
            before = counters["env_step"]

            agent.train()
            for _ in range(t.train_steps):
                batch = buffer.sample(t.batch_size, t.batch_length, device)
                for name, value in agent.train_step(batch).items():
                    logger.scalar(f"train/{name}", value)
            counters["updates"] += t.train_steps

            agent.eval()
            expl = exploration_amount(config, before)
            episodes, steps = collector.collect(t.train_every, random=False, expl_amount=expl)
            log_episodes(episodes)
            counters["env_step"] += steps
            step = counters["env_step"]
            logger.scalar("train/expl_amount", expl)

            if crossed(before, step, r.eval_every):
                returns = evaluate(agent, eval_env, r.eval_episodes)
                last_eval = float(np.mean(returns))
                logger.scalar("eval/return", last_eval)
            if crossed(before, step, r.video_every):
                batch = buffer.sample(6, t.batch_length, device)
                logger.image("openl/truth_model_error", agent.video_prediction(batch), step)
            if crossed(before, step, r.log_every) or step >= t.steps:
                now = time.time()
                logger.scalar("perf/fps", (step - timer_step) / max(now - timer, 1e-6))
                logger.scalar("perf/updates", counters["updates"])
                logger.scalar("perf/replay_size", len(buffer))
                timer, timer_step = now, step
                logger.write(step)
            if crossed(before, step, r.checkpoint_every):
                checkpoint()
                torch.save(agent.state_dict(), logdir / f"weights_{step}.pt")
    except KeyboardInterrupt:
        print("Interrupted; saving checkpoint", flush=True)
        checkpoint()
        raise
    finally:
        logger.close()
        env.close()
        eval_env.close()

    checkpoint()
    torch.save(agent.state_dict(), logdir / "weights_final.pt")
    print(f"Done. Checkpoint: {ckpt_path}", flush=True)
    return {**counters, "eval_return": last_eval, "logdir": str(logdir)}


def build_config(args: argparse.Namespace) -> Config:
    if args.config:
        config = Config.from_dict(json.loads(Path(args.config).read_text()))
        for key, value in parse_overrides(args.set).items():
            set_field(config, key, value)
    else:
        config = make_config(args.preset, parse_overrides(args.set))
    shortcuts = {
        "env.suite": args.suite,
        "env.task": args.task,
        "train.steps": args.steps,
        "run.seed": args.seed,
        "run.device": args.device,
        "run.logdir": args.logdir,
        "run.name": args.name,
    }
    for key, value in shortcuts.items():
        if value is not None:
            set_field(config, key, value)
    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Dreamer (v1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", default="dmc", choices=sorted(PRESETS))
    parser.add_argument("--config", help="config.json of a previous run to start from")
    parser.add_argument("--suite", choices=["dmc", "atari", "gym"])
    parser.add_argument("--task", help="e.g. walker_walk (dmc), pong (atari), Pendulum-v1 (gym)")
    parser.add_argument("--steps", type=lambda s: int(float(s)), help="env steps, e.g. 1e6")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", help="auto | cpu | cuda | cuda:N | mps")
    parser.add_argument("--logdir")
    parser.add_argument("--name", help="run directory name (default: suite_task_sSEED)")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.FIELD=VALUE",
        help="override any config field, e.g. --set train.batch_size=16 (repeatable)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="continue from latest.pt in the run directory"
    )
    parser.add_argument("--print-config", action="store_true", help="print the config and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config(args)
    if args.print_config:
        print(config.to_json())
        return
    train(config, resume=args.resume)


if __name__ == "__main__":
    main()
