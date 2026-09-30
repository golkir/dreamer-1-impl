import torch
from torch import distributions as D
from torch import nn

from dreamer.config import ModelConfig


class Decoder(nn.Module):
    """Transposed-conv decoder: latent features -> Normal(image, 1) over (C, H, W)."""

    def __init__(self, feat_dim: int, obs_shape: tuple[int, int, int], config: ModelConfig):
        super().__init__()
        act = getattr(nn, config.cnn_act)
        depth = config.cnn_depth
        self.obs_shape = obs_shape
        self.fc = nn.Linear(feat_dim, 32 * depth)
        channels = [32 * depth, 4 * depth, 2 * depth, depth, obs_shape[0]]
        layers: list[nn.Module] = []
        for i, kernel in enumerate(config.decoder_kernels):
            layers.append(nn.ConvTranspose2d(channels[i], channels[i + 1], kernel, stride=2))
            if i < len(config.decoder_kernels) - 1:
                layers.append(act())
        self.deconv = nn.Sequential(*layers)
        with torch.no_grad():
            out = self.deconv(torch.zeros(1, 32 * depth, 1, 1)).shape[1:]
        if tuple(out) != tuple(obs_shape):
            raise ValueError(
                f"Decoder produces {tuple(out)} but observations are {obs_shape}; "
                "adjust model.decoder_kernels or env.size"
            )

    def forward(self, feat: torch.Tensor) -> D.Distribution:
        batch_shape = feat.shape[:-1]
        x = self.fc(feat).reshape(-1, self.fc.out_features, 1, 1)
        mean = self.deconv(x).reshape(*batch_shape, *self.obs_shape)
        return D.Independent(D.Normal(mean, 1.0), len(self.obs_shape))
