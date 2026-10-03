import numpy as np
import pytest

from dreamer.replay_buffer import ReplayBuffer


def fill(buffer, steps, episode_length=7):
    for i in range(steps):
        first = i % episode_length == 0
        buffer.add(np.full((1, 4, 4), i % 256, np.uint8), np.full(2, i), float(i), first, False)


def test_sample_shapes_and_first_flag():
    buffer = ReplayBuffer((1, 4, 4), 2, capacity=100, seed=0)
    fill(buffer, 30)
    batch = buffer.sample(8, 10)
    assert batch.observation.shape == (8, 10, 1, 4, 4)
    assert batch.action.shape == (8, 10, 2)
    assert batch.reward.shape == (8, 10)
    assert (batch.is_first[:, 0] == 1).all()


def test_chunks_are_contiguous_after_wraparound():
    buffer = ReplayBuffer((1, 4, 4), 2, capacity=50, seed=0)
    fill(buffer, 137)  # wraps several times; newest step is 136
    assert len(buffer) == 50
    for _ in range(20):
        reward = buffer.sample(16, 10).reward.numpy()
        # Consecutive steps only: never jump from the newest data to the oldest.
        np.testing.assert_array_equal(np.diff(reward, axis=1), 1.0)
        assert reward.min() >= 137 - 50


def test_not_enough_data():
    buffer = ReplayBuffer((1, 4, 4), 2, capacity=50)
    fill(buffer, 5)
    with pytest.raises(ValueError):
        buffer.sample(2, 10)


def test_save_load_roundtrip(tmp_path):
    buffer = ReplayBuffer((1, 4, 4), 2, capacity=50, seed=0)
    fill(buffer, 80)
    buffer.save(str(tmp_path / "replay.npz"))
    restored = ReplayBuffer((1, 4, 4), 2, capacity=50, seed=0)
    restored.load(str(tmp_path / "replay.npz"))
    assert len(restored) == 50
    reward = restored.sample(16, 10).reward.numpy()
    np.testing.assert_array_equal(np.diff(reward, axis=1), 1.0)
    assert reward.min() >= 30
