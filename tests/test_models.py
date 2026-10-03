import pytest
import torch

from dreamer.config import ModelConfig
from dreamer.models.decoder import Decoder
from dreamer.models.encoder import ObservationEncoder
from dreamer.models.rssm import RSSM


@pytest.mark.parametrize("channels", [1, 3])
def test_encoder_decoder_shapes(channels):
    config = ModelConfig()
    obs_shape = (channels, 64, 64)
    encoder = ObservationEncoder(obs_shape, config)
    assert encoder.embed_dim == 1024
    embed = encoder(torch.zeros(2, 5, *obs_shape))
    assert embed.shape == (2, 5, 1024)
    decoder = Decoder(230, obs_shape, config)
    dist = decoder(torch.zeros(2, 5, 230))
    assert dist.mean.shape == (2, 5, *obs_shape)
    assert dist.log_prob(torch.zeros(2, 5, *obs_shape)).shape == (2, 5)


def test_decoder_rejects_mismatched_size():
    with pytest.raises(ValueError):
        Decoder(230, (3, 84, 84), ModelConfig())


def test_rssm_observe_and_reset():
    config = ModelConfig(deter_size=16, stoch_size=4, hidden_size=16)
    rssm = RSSM(action_dim=2, embed_dim=8, config=config)
    batch, length = 3, 6
    embed, action = torch.randn(batch, length, 8), torch.randn(batch, length, 2)
    is_first = torch.zeros(batch, length)
    is_first[:, 0] = 1
    is_first[1, 3] = 1  # episode boundary inside sequence 1
    post, prior = rssm.observe(embed, action, is_first)
    assert post.stoch.shape == (batch, length, 4)
    assert prior.deter.shape == (batch, length, 16)
    torch.testing.assert_close(post.deter, prior.deter)
    # After a reset the deterministic state only depends on the (zero) initial
    # state, so it equals the deterministic state at t = 0.
    torch.testing.assert_close(post.deter[1, 3], post.deter[1, 0])
    assert not torch.allclose(post.deter[0, 3], post.deter[0, 0])
    assert (post.std >= config.min_std).all()
