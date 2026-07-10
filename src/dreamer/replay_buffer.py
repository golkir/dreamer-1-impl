from dataclasses import dataclass
import numpy as np
import torch


@dataclass(slots=True)
class ReplayBatch:
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_observation: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_size: int,
        device: torch.device,
        capacity: int,
    ):
        self.device = device
        self.capacity = int(capacity)

        # uint8 for images, float32 otherwise
        state_dtype = (
            np.uint8 if len(observation_shape) == 3 else np.float32
        )

        self.observation = np.empty(
            (self.capacity, *observation_shape),
            dtype=state_dtype,
        )
        self.next_observation = np.empty_like(self.observation)

        self.action = np.empty(
            (self.capacity, action_size),
            dtype=np.float32,
        )

        self.reward = np.empty(
            (self.capacity, 1),
            dtype=np.float32,
        )

        self.done = np.empty(
            (self.capacity, 1),
            dtype=np.float32,
        )

        self.buffer_index = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.buffer_index

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ):
        idx = self.buffer_index

        self.observation[idx] = observation
        self.action[idx] = action
        self.reward[idx] = reward
        self.next_observation[idx] = next_observation
        self.done[idx] = done

        self.buffer_index = (idx + 1) % self.capacity
        self.full |= self.buffer_index == 0

    def sample(
        self,
        batch_size: int,
        chunk_size: int,
    ) -> ReplayBatch:
        """
        Returns tensors with shape:

            observation      (B, T, ...)
            action           (B, T, A)
            reward           (B, T, 1)
            next_observation (B, T, ...)
            done             (B, T, 1)

        where
            B = batch_size
            T = chunk_size
        """

        last_start = self.buffer_index - chunk_size + 1

        if not self.full:
            assert (
                last_start > 0
            ), "Not enough samples in replay buffer."

        max_start = self.capacity if self.full else last_start

        starts = np.random.randint(
            0,
            max_start,
            size=(batch_size, 1),
        )

        offsets = np.arange(chunk_size).reshape(1, -1)

        indices = (starts + offsets) % self.capacity

        return ReplayBatch(
            observation=torch.as_tensor(
                self.observation[indices],
                device=self.device,
                dtype=torch.float32,
            ),
            action=torch.as_tensor(
                self.action[indices],
                device=self.device,
            ),
            reward=torch.as_tensor(
                self.reward[indices],
                device=self.device,
            ),
            next_observation=torch.as_tensor(
                self.next_observation[indices],
                device=self.device,
                dtype=torch.float32,
            ),
            done=torch.as_tensor(
                self.done[indices],
                device=self.device,
            ),
        )