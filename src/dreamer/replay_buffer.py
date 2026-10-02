from dataclasses import dataclass
import numpy as np
import torch


@dataclass(slots=True)
class ReplayBatch:
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor  # episode ended after this step (termination or time limit)
    terminated: torch.Tensor  # episode ended by a true termination


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
        self.terminated = np.empty_like(self.done)

        self.buffer_index = 0
        self.full = False
        self.pin_memory = torch.device(device).type == "cuda"

    def __len__(self):
        return self.capacity if self.full else self.buffer_index

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        terminated: bool = False,
    ):
        idx = self.buffer_index

        self.observation[idx] = observation
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = done
        self.terminated[idx] = terminated

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

        if self.full:
            # count from the oldest step so a chunk never runs from the newest
            # data into the oldest across the write position
            starts = self.buffer_index + np.random.randint(
                0, self.capacity - chunk_size + 1, size=(batch_size, 1)
            )
        else:
            starts = np.random.randint(0, last_start, size=(batch_size, 1))

        offsets = np.arange(chunk_size).reshape(1, -1)

        indices = (starts + offsets) % self.capacity

        return ReplayBatch(
            observation=self._to_device(self.observation[indices]),
            action=self._to_device(self.action[indices]),
            reward=self._to_device(self.reward[indices]),
            done=self._to_device(self.done[indices]),
            terminated=self._to_device(self.terminated[indices]),
        )

    def _to_device(self, array: np.ndarray) -> torch.Tensor:
        # images stay uint8 until they reach the device: 4x less to copy,
        # and the float conversion runs there (see preprocess_obs)
        tensor = torch.from_numpy(array)
        if self.pin_memory:
            tensor = tensor.pin_memory()
        return tensor.to(self.device, non_blocking=self.pin_memory)
