"""
Configuration for Dreamer (v1, Hafner et al. 2020, "Dream to Control").

Defaults follow the paper's DeepMind Control setup. Named presets adjust them for
other suites (Atari) or for fast CPU smoke tests ("debug"). Any field can be
overridden from the command line with ``--set section.field=value``.
"""

from __future__ import annotations

import ast
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnvConfig:
    suite: str = "dmc"  # "dmc" | "atari" | "gym"
    # dmc: "<domain>_<task>" (e.g. "walker_walk", "ball_in_cup_catch")
    # atari: game name (e.g. "pong", "breakout") or a full id like "ALE/Pong-v5"
    # gym: any Gymnasium id that supports render_mode="rgb_array" (e.g. "Pendulum-v1")
    task: str = "walker_walk"
    action_repeat: int = 2
    size: tuple[int, int] = (64, 64)
    grayscale: bool = False
    # Max episode length in *environment* steps (before action repeat).
    # 0 keeps the suite's own limit (1000 for DMC, 108k frames for Atari).
    time_limit: int = 0
    camera: int = -1  # dmc camera id; -1 picks a sensible default per domain
    # Atari only
    sticky_actions: bool = True
    noop_max: int = 30
    terminal_on_life_loss: bool = False

    @property
    def channels(self) -> int:
        return 1 if self.grayscale else 3


@dataclass
class ModelConfig:
    # RSSM
    deter_size: int = 200
    stoch_size: int = 30
    hidden_size: int = 200
    min_std: float = 0.1
    # Conv encoder / decoder (64x64 images)
    cnn_depth: int = 32
    encoder_kernels: tuple[int, ...] = (4, 4, 4, 4)
    decoder_kernels: tuple[int, ...] = (5, 5, 6, 6)
    cnn_act: str = "ReLU"
    # Dense heads
    dense_units: int = 400
    dense_act: str = "ELU"
    reward_layers: int = 2
    value_layers: int = 3
    actor_layers: int = 4
    continue_layers: int = 3
    # Actor distribution (continuous actions)
    actor_init_std: float = 5.0
    actor_min_std: float = 1e-4
    actor_mean_scale: float = 5.0


@dataclass
class TrainConfig:
    # Budget, counted in environment steps (i.e. including action repeat)
    steps: int = 1_000_000
    prefill_steps: int = 5_000  # random-policy steps before training (5 DMC episodes)
    train_every: int = 1_000  # env steps collected between training phases
    train_steps: int = 100  # gradient updates per training phase
    # Replay
    buffer_capacity: int = 1_000_000  # agent steps; 64x64x3 uint8 ~ 12 KB per step
    batch_size: int = 50
    batch_length: int = 50
    # World model
    model_lr: float = 6e-4
    free_nats: float = 3.0
    kl_scale: float = 1.0
    # Learn a continuation (not-terminal) head. Off for DMC: its episodes only end
    # by time limit, never by termination.
    use_continue: bool = False
    continue_scale: float = 10.0
    reward_transform: str = "none"  # "none" | "tanh" (squash rewards, used on Atari)
    # Behavior
    horizon: int = 15
    discount: float = 0.99
    lambda_: float = 0.95
    actor_lr: float = 8e-5
    value_lr: float = 8e-5
    actor_entropy: float = 0.0
    # Optimization
    grad_clip: float = 100.0
    adam_eps: float = 1e-5
    weight_decay: float = 0.0
    # Exploration: Gaussian noise std for continuous actions, epsilon-greedy
    # probability for discrete ones. Halves every expl_decay env steps (0 = constant)
    # and never drops below expl_min.
    expl_amount: float = 0.3
    expl_decay: int = 0
    expl_min: float = 0.0


@dataclass
class RunConfig:
    name: str = ""  # defaults to "<suite>_<task>_s<seed>"
    logdir: str = "runs"
    seed: int = 0
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "cuda:1" | "mps"
    log_every: int = 1_000  # env steps
    eval_every: int = 10_000
    eval_episodes: int = 5
    video_every: int = 20_000  # open-loop prediction images; 0 disables
    tensorboard: bool = True  # metrics.jsonl and PNG images are always written
    checkpoint_every: int = 50_000
    save_replay: bool = False  # store the replay buffer next to latest.pt (large!)


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    run: RunConfig = field(default_factory=RunConfig)

    @property
    def run_name(self) -> str:
        if self.run.name:
            return self.run.name
        task = self.env.task.replace("/", "-")
        return f"{self.env.suite}_{task}_s{self.run.seed}"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        config = cls()
        for section, values in data.items():
            for key, value in values.items():
                set_field(config, f"{section}.{key}", value)
        return config


# --------------------------------------------------------------------------- presets

PRESETS: dict[str, dict[str, Any]] = {
    # Paper defaults for DeepMind Control from pixels.
    "dmc": {},
    # Atari: discrete actions, 4-frame action repeat, terminations and reward squashing.
    "atari": {
        "env.suite": "atari",
        "env.task": "pong",
        "env.action_repeat": 4,
        "env.time_limit": 108_000,
        "train.steps": 10_000_000,
        "train.prefill_steps": 50_000,
        "train.use_continue": True,
        "train.reward_transform": "tanh",
        "train.discount": 0.995,
        "train.actor_entropy": 1e-3,
        "train.expl_amount": 0.4,
        "train.expl_decay": 200_000,
        "train.expl_min": 0.1,
        "run.eval_every": 100_000,
        "run.eval_episodes": 3,
        "run.video_every": 100_000,
        "run.checkpoint_every": 250_000,
    },
    # Tiny networks and budgets to check the whole pipeline end to end on a CPU.
    "debug": {
        "env.task": "cartpole_balance",
        "model.deter_size": 32,
        "model.stoch_size": 8,
        "model.hidden_size": 32,
        "model.cnn_depth": 4,
        "model.dense_units": 32,
        "train.steps": 2_000,
        "train.prefill_steps": 500,
        "train.train_every": 500,
        "train.train_steps": 5,
        "train.batch_size": 4,
        "train.batch_length": 10,
        "train.horizon": 5,
        "train.buffer_capacity": 10_000,
        "run.log_every": 500,
        "run.eval_every": 1_000,
        "run.eval_episodes": 1,
        "run.video_every": 1_000,
        "run.checkpoint_every": 1_000,
    },
}


def make_config(preset: str = "dmc", overrides: dict[str, Any] | None = None) -> Config:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    config = Config()
    for key, value in {**PRESETS[preset], **(overrides or {})}.items():
        set_field(config, key, value)
    return config


# ------------------------------------------------------------------ override parsing


def set_field(config: Config, dotted_key: str, value: Any) -> None:
    """Set ``section.field`` on ``config``, converting strings to the field's type."""
    try:
        section_name, field_name = dotted_key.split(".")
        section = getattr(config, section_name)
        current = getattr(section, field_name)
    except (ValueError, AttributeError) as e:
        raise KeyError(f"Unknown config key {dotted_key!r}") from e
    if isinstance(value, str) and not isinstance(current, str):
        value = _parse_value(value)
    if isinstance(current, tuple):
        value = tuple(value)
    elif isinstance(current, bool):
        value = bool(value)
    elif isinstance(current, float) and isinstance(value, int):
        value = float(value)
    elif isinstance(current, int) and isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, type(current)):
        raise TypeError(f"{dotted_key} expects {type(current).__name__}, got {value!r}")
    setattr(section, field_name, value)


def _parse_value(text: str) -> Any:
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def parse_overrides(items: list[str]) -> dict[str, Any]:
    overrides = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override {item!r} must look like section.field=value")
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides
