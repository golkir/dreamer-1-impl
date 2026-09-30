import json
import os
from pathlib import Path

import pytest

from dreamer.train import build_config, parse_args, train


def test_cli_builds_config():
    args = parse_args(
        [
            "--preset",
            "atari",
            "--task",
            "breakout",
            "--steps",
            "1e5",
            "--seed",
            "3",
            "--set",
            "train.batch_size=8",
        ]
    )
    config = build_config(args)
    assert (config.env.suite, config.env.task) == ("atari", "breakout")
    assert config.train.steps == 100_000
    assert config.run.seed == 3 and config.train.batch_size == 8
    assert config.run_name == "atari_breakout_s3"


@pytest.mark.parametrize("discrete", [False, True])
def test_training_loop_and_resume(debug_config, dummy_env_fn, discrete):
    config = debug_config
    config.train.steps = 1_200
    config.train.use_continue = discrete
    config.run.save_replay = True
    env_fn = lambda c, s: dummy_env_fn(c, s, discrete=discrete, terminate=discrete)  # noqa: E731

    result = train(config, env_fn=env_fn)
    logdir = Path(result["logdir"])
    assert result["env_step"] >= 1_200
    assert result["updates"] == 2 * config.train.train_steps  # 500 prefill + 2 x 500
    assert result["eval_return"] is not None
    for name in ("latest.pt", "replay.npz", "config.json", "metrics.jsonl", "weights_final.pt"):
        assert (logdir / name).exists(), name
    rows = [json.loads(line) for line in (logdir / "metrics.jsonl").read_text().splitlines()]
    assert any("train/model_loss" in row for row in rows)
    assert any("episode/return" in row for row in rows)

    config.train.steps = 2_000
    resumed = train(config, resume=True, env_fn=env_fn)
    assert resumed["env_step"] >= 2_000
    assert resumed["updates"] > result["updates"]


@pytest.mark.slow
@pytest.mark.parametrize(
    "suite,task,discrete",
    [("dmc", "cartpole_balance", False), ("atari", "pong", True), ("gym", "Pendulum-v1", False)],
)
def test_real_environments(suite, task, discrete):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import gymnasium as gym

    from dreamer.config import make_config
    from dreamer.envs import make_env

    config = make_config("debug", {"env.suite": suite, "env.task": task})
    env = make_env(config.env, seed=0)
    obs, _ = env.reset()
    assert obs.shape == (3, 64, 64) and obs.dtype == "uint8"
    assert isinstance(env.action_space, gym.spaces.Discrete) == discrete
    if not discrete:
        assert (env.action_space.low == -1).all() and (env.action_space.high == 1).all()
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    assert obs.shape == (3, 64, 64)
    env.close()
