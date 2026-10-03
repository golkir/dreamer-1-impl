import numpy as np
import pytest
import torch
from torch import nn

from dreamer.utils import OneHotDist, TanhNormal, freeze, lambda_return


def brute_force_lambda_return(reward, value, discount, lam):
    """Weighted average of n-step returns, straight from the definition."""
    horizon = len(reward) - 1
    out = []
    for t in range(horizon):

        def n_step(n):
            ret, scale = 0.0, 1.0
            for k in range(1, n + 1):
                ret += scale * reward[t + k]
                scale *= discount[t + k]
            return ret + scale * value[t + n]

        steps = horizon - t
        ret = sum((1 - lam) * lam ** (n - 1) * n_step(n) for n in range(1, steps))
        out.append(ret + lam ** (steps - 1) * n_step(steps))
    return np.array(out)


@pytest.mark.parametrize("lam", [0.0, 0.5, 0.95, 1.0])
def test_lambda_return_matches_definition(lam):
    rng = np.random.default_rng(0)
    reward, value = rng.normal(size=9), rng.normal(size=9)
    discount = rng.uniform(0.5, 1.0, size=9)
    expected = brute_force_lambda_return(reward, value, discount, lam)
    actual = lambda_return(*(torch.tensor(x)[:, None] for x in (reward, value, discount)), lam)
    np.testing.assert_allclose(actual[:, 0].numpy(), expected, rtol=1e-6)


def test_lambda_return_td0():
    reward, value = torch.randn(5, 3), torch.randn(5, 3)
    discount = torch.full((5, 3), 0.9)
    ret = lambda_return(reward, value, discount, 0.0)
    torch.testing.assert_close(ret, reward[1:] + 0.9 * value[1:])


def test_freeze_blocks_param_grads_but_not_input_grads():
    net = nn.Linear(3, 1)
    x = torch.randn(4, 3, requires_grad=True)
    with freeze([net]):
        net(x).sum().backward()
    assert net.weight.grad is None
    assert x.grad is not None
    assert net.weight.requires_grad


def test_action_distributions():
    dist = TanhNormal(torch.zeros(5, 2), torch.ones(5, 2))
    sample = dist.rsample()
    assert sample.shape == (5, 2) and sample.abs().max() <= 1
    assert dist.entropy().shape == (5,)

    logits = torch.randn(5, 4, requires_grad=True)
    onehot = OneHotDist(logits)
    sample = onehot.rsample()
    torch.testing.assert_close(sample.detach().sum(-1), torch.ones(5))
    (sample * torch.arange(4.0)).sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0  # straight-through
    assert onehot.mode().argmax(-1).eq(logits.argmax(-1)).all()
