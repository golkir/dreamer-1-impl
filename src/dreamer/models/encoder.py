import torch
from torch import nn

from dreamer.config import ModelConfig


class ObservationEncoder(nn.Module):
    """Conv encoder: (..., C, 64, 64) images in [-0.5, 0.5] -> (..., embed_dim)."""

    def __init__(self, obs_shape: tuple[int, int, int], config: ModelConfig):
        super().__init__()
        act = getattr(nn, config.cnn_act)
        layers: list[nn.Module] = []
        in_channels = obs_shape[0]
        for i, kernel in enumerate(config.encoder_kernels):
            out_channels = config.cnn_depth * 2**i
            layers += [nn.Conv2d(in_channels, out_channels, kernel, stride=2), act()]
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers, nn.Flatten())
        with torch.no_grad():
            self.embed_dim = self.cnn(torch.zeros(1, *obs_shape)).shape[-1]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_shape = obs.shape[:-3]
        x = self.cnn(obs.reshape(-1, *obs.shape[-3:]))
        return x.reshape(*batch_shape, -1)
