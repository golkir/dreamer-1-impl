import copy

import gymnasium as gym
import numpy as np
import pytest
import torch

from dreamer.dreamer_alg import Dreamer
from dreamer.replay_buffer import ReplayBuffer

OBS_SHAPE = (3, 64, 64)


def make_agent(config, discrete):
    space = gym.spaces.Discrete(4) if discrete else gym.spaces.Box(-1, 1, (2,), np.float32)
    config.train.use_continue = discrete
    return Dreamer(OBS_SHAPE, space, config)


def make_batch(agent, config):
    buffer = ReplayBuffer(OBS_SHAPE, agent.action_dim, 200, seed=0)
    rng = np.random.default_rng(0)
    for i in range(60):
        first = i % 20 == 0
        action = (
            np.zeros(agent.action_dim, np.float32)
            if first
            else rng.uniform(-1, 1, agent.action_dim)
        )
        buffer.add(
            rng.integers(0, 256, OBS_SHAPE, dtype=np.uint8),
            action,
            rng.normal(),
            first,
            i % 20 == 19,
        )
    return buffer.sample(config.train.batch_size, config.train.batch_length)


def params(module):
    return [p.detach().clone() for p in module.parameters()]


def changed(before, module):
    return all(not torch.equal(b, a) for b, a in zip(before, module.parameters()))


@pytest.mark.parametrize("discrete", [False, True])
def test_train_step_updates_every_component(debug_config, discrete):
    torch.manual_seed(0)
    # With free nats the prior gets no gradient while the KL is small; disable
    # them so every parameter must receive a gradient.
    debug_config.train.free_nats = 0.0
    agent = make_agent(debug_config, discrete)
    batch = make_batch(agent, debug_config)
    before = {
        name: params(getattr(agent, name))
        for name in ("encoder", "rssm", "decoder", "reward", "actor", "value")
    }
    metrics = agent.train_step(batch)
    assert all(np.isfinite(v) for v in metrics.values()), metrics
    for name, snapshot in before.items():
        assert changed(snapshot, getattr(agent, name)), f"{name} was not updated"
    if discrete:
        assert "continue_loss" in metrics


def test_behavior_step_leaves_world_model_untouched(debug_config):
    torch.manual_seed(0)
    agent = make_agent(debug_config, discrete=False)
    post, _ = agent.train_world_model(make_batch(agent, debug_config))
    world = copy.deepcopy(agent.world_model)
    agent.train_behavior(post.detach())
    for old, new in zip(world, agent.world_model):
        for a, b in zip(old.parameters(), new.parameters()):
            assert torch.equal(a, b)
        assert all(p.requires_grad for p in new.parameters())


@pytest.mark.parametrize("discrete", [False, True])
def test_policy_actions(debug_config, discrete):
    agent = make_agent(debug_config, discrete)
    obs = np.zeros(OBS_SHAPE, np.uint8)
    state = None
    for explore in (True, False, True):
        action, state = agent.policy(obs, state, explore=explore, expl_amount=0.5)
        if discrete:
            assert isinstance(action, int) and 0 <= action < 4
            assert agent.to_buffer_action(action).sum() == 1
        else:
            assert action.shape == (2,) and np.abs(action).max() <= 1


def test_checkpoint_roundtrip(debug_config):
    agent = make_agent(debug_config, discrete=False)
    agent.train_step(make_batch(agent, debug_config))
    clone = make_agent(debug_config, discrete=False)
    clone.load_checkpoint(agent.checkpoint())
    for a, b in zip(agent.parameters(), clone.parameters()):
        assert torch.equal(a, b)
    assert clone.model_opt.state_dict()["state"]


def test_video_prediction_image(debug_config):
    agent = make_agent(debug_config, discrete=False)
    image = agent.video_prediction(make_batch(agent, debug_config), context=3, length=8)
    rows = min(6, debug_config.train.batch_size)
    assert image.shape == (3, rows * 3 * 64, 8 * 64)
    assert 0 <= image.min() and image.max() <= 1


def test_amp_train_step_and_checkpoint(debug_config):
    torch.manual_seed(0)
    debug_config.run.amp = True
    agent = make_agent(debug_config, discrete=False)
    batch = make_batch(agent, debug_config)
    for _ in range(12):  # the loss scaler may skip a few overflowing steps at first
        metrics = agent.train_step(batch)
    losses = {k: v for k, v in metrics.items() if not k.endswith("grad_norm")}
    assert all(np.isfinite(v) for v in losses.values()), losses
    ckpt = agent.checkpoint()
    assert ckpt["scalers"]["model"]["scale"] < 65536  # warm-up lowered the scale
    clone = make_agent(debug_config, discrete=False)
    clone.load_checkpoint(ckpt)
    assert clone._scalers["model"].get_scale() == agent._scalers["model"].get_scale()


def test_loads_checkpoint_without_scaler_state(debug_config):
    agent = make_agent(debug_config, discrete=False)
    ckpt = agent.checkpoint()
    del ckpt["scalers"]  # checkpoints written before mixed precision existed
    debug_config.run.amp = True
    clone = make_agent(debug_config, discrete=False)
    clone.load_checkpoint(ckpt)
    assert np.isfinite(list(clone.train_step(make_batch(clone, debug_config)).values())[0])


@pytest.mark.slow
def test_compiled_train_step(debug_config):
    torch.manual_seed(0)
    debug_config.run.compile = True
    agent = make_agent(debug_config, discrete=True)
    metrics = agent.train_step(make_batch(agent, debug_config))
    assert all(np.isfinite(v) for v in metrics.values()), metrics
    action, _ = agent.policy(np.zeros(OBS_SHAPE, np.uint8), None, explore=False)
    assert 0 <= action < 4
