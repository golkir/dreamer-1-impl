"""
Environment factory for Dreamer.

Every environment follows the Gymnasium API and yields observations as
``uint8`` arrays of shape ``(C, H, W)``; normalization happens inside the agent.

Suites:
    dmc    DeepMind Control Suite from pixels, continuous actions
    atari  Arcade Learning Environment via Gymnasium (ale-py), discrete actions
    gym    any Gymnasium env that can render rgb_array frames
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np

from dreamer.config import EnvConfig
from dreamer.wrappers import ActionRepeat, ChannelsFirst, PixelObservation, TimeLimit

# Headless rendering for MuJoCo; must be set before dm_control is imported.
# Override with MUJOCO_GL=osmesa if EGL is unavailable.
os.environ.setdefault("MUJOCO_GL", "egl")


class DMCEnv(gym.Env):
    """DeepMind Control task rendered from pixels, with action repeat."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        domain: str,
        task: str,
        size: tuple[int, int] = (64, 64),
        action_repeat: int = 2,
        camera: int = -1,
        grayscale: bool = False,
        seed: int | None = None,
    ):
        from dm_control import suite

        self._env = suite.load(domain, task, task_kwargs={"random": seed})
        self.size = size
        self.action_repeat = action_repeat
        self.grayscale = grayscale
        self.camera = camera if camera >= 0 else (2 if domain == "quadruped" else 0)

        spec = self._env.action_spec()
        self.action_space = gym.spaces.Box(
            spec.minimum.astype(np.float32),
            spec.maximum.astype(np.float32),
            dtype=np.float32,
        )
        channels = 1 if grayscale else 3
        self.observation_space = gym.spaces.Box(0, 255, (channels, *size), dtype=np.uint8)

    def render(self) -> np.ndarray:
        return self._env.physics.render(*self.size, camera_id=self.camera)

    def _observation(self) -> np.ndarray:
        frame = self.render()
        if self.grayscale:
            frame = (frame @ np.array([0.299, 0.587, 0.114]))[..., None]
            frame = frame.astype(np.uint8)
        return np.ascontiguousarray(frame.transpose(2, 0, 1))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._env.reset()
        return self._observation(), {}

    def step(self, action):
        reward = 0.0
        for _ in range(self.action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0.0
            if time_step.last():
                break
        # DMC episodes end by time limit (discount stays 1); a zero discount marks
        # a true termination.
        last = time_step.last()
        terminated = last and time_step.discount == 0.0
        return self._observation(), reward, terminated, last and not terminated, {}


def _make_dmc(config: EnvConfig, seed: int) -> gym.Env:
    domain, task = config.task.split("_", 1)
    if domain == "ball" and task.startswith("in_cup"):  # "ball_in_cup_catch"
        domain, task = "ball_in_cup", task[len("in_cup_") :]
    env: gym.Env = DMCEnv(
        domain,
        task,
        size=config.size,
        action_repeat=config.action_repeat,
        camera=config.camera,
        grayscale=config.grayscale,
        seed=seed,
    )
    if config.time_limit:
        env = TimeLimit(env, config.time_limit // config.action_repeat)
    return env


def _make_atari(config: EnvConfig, seed: int) -> gym.Env:
    import ale_py

    gym.register_envs(ale_py)
    if "/" in config.task:
        env_id = config.task
    else:
        name = "".join(part.capitalize() for part in config.task.split("_"))
        env_id = f"ALE/{name}-v5"
    env = gym.make(
        env_id,
        frameskip=1,  # frame skipping is done by AtariPreprocessing
        repeat_action_probability=0.25 if config.sticky_actions else 0.0,
        full_action_space=False,
        max_num_frames_per_episode=config.time_limit or None,
    )
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=config.noop_max,
        frame_skip=config.action_repeat,
        screen_size=config.size,
        terminal_on_life_loss=config.terminal_on_life_loss,
        grayscale_obs=config.grayscale,
        grayscale_newaxis=True,
        scale_obs=False,
    )
    env = ChannelsFirst(env)
    env.action_space.seed(seed)
    return env


def _make_gym(config: EnvConfig, seed: int) -> gym.Env:
    env = gym.make(config.task, render_mode="rgb_array")
    env = ActionRepeat(env, config.action_repeat)
    env = PixelObservation(env, config.size, config.grayscale)
    env = ChannelsFirst(env)
    if config.time_limit:
        env = TimeLimit(env, config.time_limit // config.action_repeat)
    env.action_space.seed(seed)
    return env


_SUITES = {"dmc": _make_dmc, "atari": _make_atari, "gym": _make_gym}


def make_env(config: EnvConfig, seed: int = 0) -> gym.Env:
    if config.suite not in _SUITES:
        raise ValueError(f"Unknown suite {config.suite!r}; choose from {list(_SUITES)}")
    env = _SUITES[config.suite](config, seed)
    space = env.action_space
    if isinstance(space, gym.spaces.Box) and not (
        np.allclose(space.low, -1) and np.allclose(space.high, 1)
    ):
        # The actor outputs actions in [-1, 1].
        env = gym.wrappers.RescaleAction(env, -1.0, 1.0)
    # Seed the suite's RNG on the first reset; later resets continue the stream.
    env.reset(seed=seed)
    return env
