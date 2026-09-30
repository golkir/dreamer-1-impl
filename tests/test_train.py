import json
import os
import subprocess
import sys
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
    config.run.tensorboard = not discrete  # cover both logging paths
    env_fn = lambda c, s: dummy_env_fn(c, s, discrete=discrete, terminate=discrete)  # noqa: E731

    result = train(config, env_fn=env_fn)
    logdir = Path(result["logdir"])
    assert result["env_step"] >= 1_200
    assert result["updates"] == 2 * config.train.train_steps  # 500 prefill + 2 x 500
    assert result["eval_return"] is not None
    for name in ("latest.pt", "replay.npz", "config.json", "metrics.jsonl", "weights_final.pt"):
        assert (logdir / name).exists(), name
    assert list((logdir / "openl").rglob("*.png"))
    assert bool(list(logdir.glob("events.out.tfevents.*"))) == config.run.tensorboard
    rows = [json.loads(line) for line in (logdir / "metrics.jsonl").read_text().splitlines()]
    assert any("train/model_loss" in row for row in rows)
    assert any("episode/return" in row for row in rows)

    config.train.steps = 2_000
    resumed = train(config, resume=True, env_fn=env_fn)
    assert resumed["env_step"] >= 2_000
    assert resumed["updates"] > result["updates"]


def _run_isolated(code: str, extra_path: str | None = None) -> str:
    """Run ``code`` in a fresh interpreter: native libraries loaded by other tests
    (TensorBoard, Triton, ...) can break OpenGL initialization in this process."""
    env = dict(os.environ)
    if extra_path:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [extra_path, env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300, env=env
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return result.stdout


def test_importing_train_does_not_load_tensorboard():
    out = _run_isolated(
        "import sys, dreamer.train; print('torch.utils.tensorboard' in sys.modules)"
    )
    assert out.strip() == "False"


def test_tensorboard_logging_never_imports_tensorflow(tmp_path):
    # With TensorFlow installed (e.g. on Kaggle), TensorBoard would import it, pulling
    # in Keras/JAX whose LLVM crashes MuJoCo's OpenGL. Simulate an installed TF that
    # fails loudly if anything imports it.
    fake = tmp_path / "site" / "tensorflow"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("raise RuntimeError('tensorflow was imported')")
    out = _run_isolated(
        f"""
import torch
from pathlib import Path
from dreamer.train import Logger

logger = Logger(Path({str(tmp_path)!r}), tensorboard=True)
logger.scalar("a/b", 1.0)
logger.image("openl/x", torch.rand(3, 8, 8), 1)
logger.write(1)
logger.close()
print("ok")
""",
        extra_path=str(tmp_path / "site"),
    )
    assert out.strip().endswith("ok")
    assert list(tmp_path.glob("events.out.tfevents.*"))


@pytest.mark.slow
@pytest.mark.parametrize(
    "suite,task,discrete",
    [("dmc", "cartpole_balance", False), ("atari", "pong", True), ("gym", "Pendulum-v1", False)],
)
def test_real_environments(suite, task, discrete):
    out = _run_isolated(
        f"""
import gymnasium as gym
from dreamer.config import make_config
from dreamer.envs import make_env

env = make_env(make_config("debug", {{"env.suite": "{suite}", "env.task": "{task}"}}).env, seed=0)
obs, _ = env.reset()
assert obs.shape == (3, 64, 64) and obs.dtype == "uint8", obs.shape
assert isinstance(env.action_space, gym.spaces.Discrete) == {discrete}
if not {discrete}:
    assert (env.action_space.low == -1).all() and (env.action_space.high == 1).all()
obs, *_ = env.step(env.action_space.sample())
assert obs.shape == (3, 64, 64)
env.close()
print("ok")
"""
    )
    assert out.strip().endswith("ok")
