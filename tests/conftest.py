import gymnasium as gym
import numpy as np
import pytest

from dreamer.config import make_config


class DummyPixelEnv(gym.Env):
    """Fast stand-in for a pixel environment: random images, fixed-length episodes,
    optional terminations."""

    def __init__(self, discrete=False, episode_length=12, terminate=False, channels=3):
        self.observation_space = gym.spaces.Box(0, 255, (channels, 64, 64), dtype=np.uint8)
        if discrete:
            self.action_space = gym.spaces.Discrete(4)
        else:
            self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.episode_length = episode_length
        self.terminate = terminate
        self._t = 0

    def _obs(self):
        return self.np_random.integers(0, 256, self.observation_space.shape, dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        reward = float(np.sum(action)) if not np.isscalar(action) else float(action == 1)
        done = self._t >= self.episode_length
        return self._obs(), reward, done and self.terminate, done and not self.terminate, {}


@pytest.fixture
def debug_config(tmp_path):
    config = make_config("debug")
    config.run.logdir = str(tmp_path / "runs")
    config.run.device = "cpu"
    return config


@pytest.fixture
def dummy_env_fn():
    def make(env_config, seed, **kwargs):
        env = DummyPixelEnv(channels=env_config.channels, **kwargs)
        env.reset(seed=seed)
        return env

    return make
