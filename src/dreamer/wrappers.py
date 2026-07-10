

from __future__ import annotations

import cv2
import gymnasium as gym
import numpy as np


class ActionRepeat(gym.Wrapper):
    """
    Repeat each action for `repeat` environment steps and
    accumulate rewards.
    """

    def __init__(self, env: gym.Env, repeat: int):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0

        for _ in range(self.repeat):
            observation, reward, terminated, truncated, info = self.env.step(action)

            total_reward += reward

            if terminated or truncated:
                break

        return observation, total_reward, terminated, truncated, info


class PixelObservation(gym.ObservationWrapper):
    """
    Ignore the native observation and always return
    the rendered RGB image.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        frame = env.render()

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=frame.shape,
            dtype=np.uint8,
        )

    def observation(self, observation):
        return self.env.render()


class ResizeObservation(gym.ObservationWrapper):
    """
    Resize HWC images.
    """

    def __init__(self, env: gym.Env, size: int):
        super().__init__(env)

        self.size = size

        channels = env.observation_space.shape[-1]

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(size, size, channels),
            dtype=np.uint8,
        )

    def observation(self, observation):
        return cv2.resize(
            observation,
            (self.size, self.size),
            interpolation=cv2.INTER_AREA,
        )


class NormalizeImageObs(gym.ObservationWrapper):
    """
    Convert uint8 -> float32 in [-0.5, 0.5].
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.observation_space = gym.spaces.Box(
            low=-0.5,
            high=0.5,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        return observation.astype(np.float32) / 255.0 - 0.5


class ChannelsFirst(gym.ObservationWrapper):
    """
    Convert HWC -> CHW.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        h, w, c = env.observation_space.shape

        self.observation_space = gym.spaces.Box(
            low=-0.5,
            high=0.5,
            shape=(c, h, w),
            dtype=np.float32,
        )

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))