"""
Environment factory for Dreamer.

Supports:
    - DeepMind Control Suite (dm_control)
    - Gymnasium environments

Produces observations in the format expected by Dreamer:

    shape: (3, H, W)
    dtype: float32
    range: [-0.5, 0.5]
"""


import os 

os.environ["MUJOCO_GL"] = "egl"
import gymnasium as gym
import numpy as np

from dm_control import suite


import gymnasium as gym
import numpy as np


class DMCEnv:
    def __init__(
        self,
        domain_name: str,
        task_name: str,
        height: int = 64,
        width: int = 64,
        camera_id: int | None = None,
        action_repeat: int = 2,
    ):
        self.env = suite.load(domain_name, task_name)

        self.height = height
        self.width = width
        self.action_repeat = action_repeat

        if camera_id is None:
            camera_id = 2 if domain_name == "quadruped" else 0
        self.camera_id = camera_id

        spec = self.env.action_spec()

        self.action_space = type("ActionSpace", (), {})()
        self.action_space.low = spec.minimum.astype(np.float32)
        self.action_space.high = spec.maximum.astype(np.float32)
        self.action_space.shape = spec.shape

        self.observation_shape = (3, height, width)

    def _render(self):
        image = self.env.physics.render(
            height=self.height,
            width=self.width,
            camera_id=self.camera_id,
        )

        image = image.astype(np.float32) / 255.0 - 0.5
        image = np.transpose(image, (2, 0, 1))

        return image

    def reset(self):
        self.env.reset()
        obs = self._render()
        return obs, {}

    def step(self, action):
        reward = 0.0

        for _ in range(self.action_repeat):
            ts = self.env.step(action)
            reward += ts.reward or 0.0
            if ts.last():
                break

        obs = self._render()

        return (
            obs,
            reward,
            ts.last(),
            False,
            {},
        )

    def close(self):
        pass


def make_dmc_env(
    domain_name,
    task_name,
    seed,
    visualize_reward=False,
    from_pixels=True,
    height=64,
    width=64,
    frame_skip=2,
    pixel_norm=True,
):
    del seed
    del visualize_reward
    del from_pixels
    del pixel_norm

    return DMCEnv(
        domain_name=domain_name,
        task_name=task_name,
        height=height,
        width=width,
        action_repeat=frame_skip,
    )
