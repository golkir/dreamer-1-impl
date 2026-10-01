"""
Measure where training time goes on this machine and project the run time.

    python -m dreamer.benchmark --preset dmc --task walker_walk
    python -m dreamer.benchmark --preset dmc --set run.amp=true --set run.compile=true

Reports the time per gradient update (world model / behavior), how busy the GPU
is during an update, the cost of one environment step, and the projected hours
for the configured ``train.steps``. A low GPU-busy share means the update is
limited by Python/kernel-launch overhead (try run.compile); a high share means
it is limited by GPU compute (try run.amp).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from dreamer.config import PRESETS, make_config, parse_overrides, set_field
from dreamer.dreamer_alg import Dreamer
from dreamer.envs import make_env
from dreamer.replay_buffer import ReplayBuffer
from dreamer.utils import resolve_device, set_seed


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(fn, device: torch.device, repeats: int) -> float:
    _sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync(device)
    return (time.perf_counter() - start) / repeats


def gpu_busy_fraction(fn, device: torch.device, repeats: int = 3) -> float | None:
    """Share of wall time the GPU spends executing kernels while running ``fn``."""
    if device.type != "cuda":
        return None
    from torch.profiler import ProfilerActivity, profile

    _sync(device)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        _sync(device)
        wall = time.perf_counter() - start
    kernel_us = sum(e.self_device_time_total for e in prof.key_averages())
    return kernel_us / 1e6 / wall


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preset", default="dmc", choices=sorted(PRESETS))
    parser.add_argument("--task")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE")
    parser.add_argument("--updates", type=int, default=20, help="timed gradient updates")
    parser.add_argument("--env-steps", type=int, default=200, help="timed agent steps")
    args = parser.parse_args(argv)

    config = make_config(args.preset, parse_overrides(args.set))
    if args.task:
        set_field(config, "env.task", args.task)
    t = config.train
    set_seed(0)
    device = resolve_device(args.device)

    env = make_env(config.env, seed=0)
    obs_shape = env.observation_space.shape
    agent = Dreamer(obs_shape, env.action_space, config).to(device)
    print(
        f"device {device} | {config.env.suite}:{config.env.task} | batch "
        f"{t.batch_size}x{t.batch_length} | amp={config.run.amp} compile={config.run.compile}",
        flush=True,
    )

    # Update timing does not depend on the data, so fill replay with random frames.
    buffer = ReplayBuffer(obs_shape, agent.action_dim, 4 * t.batch_length, seed=0)
    rng = np.random.default_rng(0)
    for i in range(4 * t.batch_length):
        buffer.add(
            rng.integers(0, 256, obs_shape, dtype=np.uint8),
            rng.uniform(-1, 1, agent.action_dim),
            float(rng.normal()),
            i % t.batch_length == 0,
            False,
        )
    batch = buffer.sample(t.batch_size, t.batch_length, device)

    warmup = 5 if config.run.compile else 2
    print(
        f"warming up ({warmup} updates{', compiling' if config.run.compile else ''})...", flush=True
    )
    for _ in range(warmup):
        agent.train_step(batch)

    posts = []

    def world_model():
        post, _ = agent.train_world_model(batch)
        posts.append(post.detach())

    t_world = _timed(world_model, device, args.updates)
    t_behavior = _timed(lambda: agent.train_behavior(posts[-1]), device, args.updates)
    t_sample = _timed(lambda: buffer.sample(t.batch_size, t.batch_length, device), device, 10)
    busy = gpu_busy_fraction(lambda: agent.train_step(batch), device)
    t_update = t_world + t_behavior + t_sample

    # Environment: simulation + rendering vs. the policy's forward pass.
    obs, _ = env.reset()
    state, t_env, t_policy = None, 0.0, 0.0
    for _ in range(args.env_steps):
        start = time.perf_counter()
        action, state = agent.policy(obs, state, explore=True, expl_amount=t.expl_amount)
        t_policy += time.perf_counter() - start
        start = time.perf_counter()
        obs, _, terminated, truncated, _ = env.step(action)
        t_env += time.perf_counter() - start
        if terminated or truncated:
            obs, _ = env.reset()
            state = None
    env.close()
    t_env /= args.env_steps
    t_policy /= args.env_steps
    t_step = t_env + t_policy

    ms = 1e3
    print(f"\nper gradient update: {t_update * ms:7.0f} ms")
    print(f"  world model        {t_world * ms:7.0f} ms")
    print(f"  behavior           {t_behavior * ms:7.0f} ms")
    print(f"  replay sampling    {t_sample * ms:7.0f} ms")
    if busy is not None:
        verdict = "overhead-bound: try run.compile" if busy < 0.6 else "compute-bound: try run.amp"
        print(f"  GPU busy           {busy:7.0%}  ({verdict})")
    print(f"per agent step:      {t_step * ms:7.1f} ms")
    print(f"  env (sim + render) {t_env * ms:7.1f} ms")
    print(f"  policy             {t_policy * ms:7.1f} ms")

    # One training phase = train_steps updates + train_every env steps.
    agent_steps = t.train_every / config.env.action_repeat
    phase = t.train_steps * t_update + agent_steps * t_step
    phases = (t.steps - t.prefill_steps) / t.train_every
    evals = t.steps / config.run.eval_every if config.run.eval_every else 0
    # Assumes full-length eval episodes (1000 env steps on DMC unless time_limit is set).
    episode = (config.env.time_limit or 1000) / config.env.action_repeat
    eval_time = evals * config.run.eval_episodes * episode * t_step
    total = phases * phase + eval_time
    print(
        f"\nprojected for {t.steps:,} env steps: {total / 3600:.1f} h "
        f"(training {phases * t.train_steps * t_update / 3600:.1f} h, "
        f"collection {phases * agent_steps * t_step / 3600:.1f} h, eval {eval_time / 3600:.1f} h); "
        f"~{t.train_every / phase:.1f} env steps/s"
    )


if __name__ == "__main__":
    main()
