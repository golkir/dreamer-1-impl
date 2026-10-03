"""Gymnasium wrappers that turn an environment into Dreamer's pixel interface."""

from __future__ import annotations

import cv2
import gymnasium as gym
import numpy as np


class ActionRepeat(gym.Wrapper):
    """Repeat each action ``repeat`` times and sum the rewards."""

    def __init__(self, env: gym.Env, repeat: int):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.repeat):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return observation, total_reward, terminated, truncated, info


class TimeLimit(gym.Wrapper):
    """Truncate episodes after ``max_steps`` wrapper steps."""

    def __init__(self, env: gym.Env, max_steps: int):
        super().__init__(env)
        self.max_steps = max_steps
        self._step = 0

    def reset(self, **kwargs):
        self._step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._step += 1
        truncated = truncated or self._step >= self.max_steps
        return observation, reward, terminated, truncated, info


class PixelObservation(gym.ObservationWrapper):
    """Replace the native observation with the rendered image, resized to ``size``."""

    def __init__(self, env: gym.Env, size: tuple[int, int], grayscale: bool = False):
        super().__init__(env)
        self.size = size
        self.grayscale = grayscale
        channels = 1 if grayscale else 3
        self.observation_space = gym.spaces.Box(0, 255, (*size, channels), dtype=np.uint8)

    def observation(self, observation):
        frame = self.env.render()
        frame = cv2.resize(frame, self.size[::-1], interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)[..., None]
        return frame


class ChannelsFirst(gym.ObservationWrapper):
    """Convert HWC (or HW) uint8 images to CHW."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        shape = env.observation_space.shape
        h, w, c = shape if len(shape) == 3 else (*shape, 1)
        self.observation_space = gym.spaces.Box(0, 255, (c, h, w), dtype=np.uint8)

    def observation(self, observation):
        if observation.ndim == 2:
            observation = observation[..., None]
        return np.ascontiguousarray(np.transpose(observation, (2, 0, 1)))
