"""
Sequence replay buffer.

Each row stores one time step using the convention

    observation[t]  image observed at step t
    action[t]       action that led to observation[t] (zeros on the first step)
    reward[t]       reward received when arriving at observation[t]
    is_first[t]     observation[t] starts a new episode
    is_terminal[t]  observation[t] is terminal (not set for time-limit truncation)

so the latent state inferred from observation[t] predicts reward[t] directly.
Episodes are written back to back; sampled chunks may span an episode boundary,
which the world model handles by resetting its state wherever ``is_first`` is set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(slots=True)
class Batch:
    observation: torch.Tensor  # (B, T, C, H, W) uint8
    action: torch.Tensor  # (B, T, A) float32
    reward: torch.Tensor  # (B, T) float32
    is_first: torch.Tensor  # (B, T) float32
    is_terminal: torch.Tensor  # (B, T) float32


class ReplayBuffer:
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_dim: int,
        capacity: int,
        seed: int | None = None,
    ):
        self.capacity = int(capacity)
        self.observation = np.zeros((self.capacity, *observation_shape), np.uint8)
        self.action = np.zeros((self.capacity, action_dim), np.float32)
        self.reward = np.zeros(self.capacity, np.float32)
        self.is_first = np.zeros(self.capacity, bool)
        self.is_terminal = np.zeros(self.capacity, bool)
        self.index = 0
        self.full = False
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.capacity if self.full else self.index

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        is_first: bool,
        is_terminal: bool,
    ) -> None:
        i = self.index
        self.observation[i] = observation
        self.action[i] = action
        self.reward[i] = reward
        self.is_first[i] = is_first
        self.is_terminal[i] = is_terminal
        self.index = (i + 1) % self.capacity
        self.full = self.full or self.index == 0

    def sample(self, batch_size: int, length: int, device="cpu") -> Batch:
        """Sample ``batch_size`` contiguous chunks of ``length`` steps."""
        size = len(self)
        if size < length:
            raise ValueError(f"Need at least {length} steps in replay, have {size}")
        # Start offsets relative to the oldest step, so chunks never run across the
        # write pointer from the newest data into the oldest.
        oldest = self.index if self.full else 0
        starts = oldest + self.rng.integers(0, size - length + 1, size=batch_size)
        idx = (starts[:, None] + np.arange(length)[None]) % self.capacity

        is_first = self.is_first[idx].copy()
        # The first step of a chunk starts from a fresh latent state.
        is_first[:, 0] = True

        def to_tensor(x: np.ndarray, dtype=None) -> torch.Tensor:
            return torch.as_tensor(x, dtype=dtype).to(device, non_blocking=True)

        return Batch(
            observation=to_tensor(self.observation[idx]),
            action=to_tensor(self.action[idx]),
            reward=to_tensor(self.reward[idx]),
            is_first=to_tensor(is_first, torch.float32),
            is_terminal=to_tensor(self.is_terminal[idx], torch.float32),
        )

    # ------------------------------------------------------------------ persistence

    def state_dict(self) -> dict:
        n = len(self)
        # Store in chronological order so reloading into any capacity is simple.
        order = (np.arange(n) + (self.index if self.full else 0)) % self.capacity
        return {
            "observation": self.observation[order],
            "action": self.action[order],
            "reward": self.reward[order],
            "is_first": self.is_first[order],
            "is_terminal": self.is_terminal[order],
        }

    def load_state_dict(self, state: dict) -> None:
        n = min(len(state["reward"]), self.capacity)
        for key in ("observation", "action", "reward", "is_first", "is_terminal"):
            getattr(self, key)[:n] = state[key][-n:]
        self.index = n % self.capacity
        self.full = n == self.capacity

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            np.savez(f, **self.state_dict())

    def load(self, path: str) -> None:
        with np.load(path) as data:
            self.load_state_dict(dict(data))
