import torch
from torch import nn

from dreamer.config import DreamerConfig
from dreamer.utils import create_normal_dist_from_params

class Decoder(nn.Module):
    """
    Decode RSSM latent state (h, z) back into an RGB observation.
    """

    def __init__(self, latent_dim: int, config: DreamerConfig):
        super().__init__()

        self.config = config
        decoder_cfg = config.decoder

        self.depth = decoder_cfg.depth
        activation = getattr(nn, decoder_cfg.activation)

        self.obs_shape = (
            decoder_cfg.obs_channels,
            decoder_cfg.obs_height,
            decoder_cfg.obs_width,
        )

        self.init_channels = self.depth * 32

        self.fc = nn.Linear(latent_dim, self.init_channels)

        channels = [
            self.init_channels,
            self.depth * 4,
            self.depth * 2,
            self.depth,
            decoder_cfg.obs_channels,
        ]

        layers = []

        for i in range(4):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=decoder_cfg.kernel_sizes[i],
                    stride=decoder_cfg.strides[i],
                )
            )

            if i != 3:
                layers.append(activation())

        self.decoder = nn.Sequential(*layers)

        # Guarantees the requested output size
        self.output_resize = nn.AdaptiveAvgPool2d(
            (decoder_cfg.obs_height, decoder_cfg.obs_width)
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        h: (..., h_dim)
        z: (..., z_dim)

        returns:
            (..., C, H, W)
        """

        leading_shape = h.shape[:-1]
        x = torch.cat([h, z], dim=-1)
        x = x.reshape(-1, x.shape[-1])
        x = self.fc(x)
        x = x.view(-1, self.init_channels, 1, 1)
        x = self.decoder(x)
        x = self.output_resize(x)
        print(x.shape, "Before view")
        x = x.view(*leading_shape, *self.obs_shape)
        print(x.shape, "after view")
        dist = create_normal_dist_from_params(x, std=1, event_shape=len(self.config.observation_shape))
        return dist 


if __name__ == "__main__":
    from dataclasses import dataclass
    
    @dataclass
    class Config:
        input_shape: tuple
        depth: int
        activation: str
        kernel_size_1: int = 5
        kernel_size_2: int = 5
        kernel_size_3: int = 6
        kernel_size_4: int = 6

    config = Config(input_shape=(3, 100, 100), depth=32, activation="ELU")

    decoder = Decoder(latent_dim=230, config=config)

    h_mock = torch.randn(2, 5, 130)
    z_mock = torch.randn(2, 5, 100)

    res = decoder(h_mock, z_mock)
    print("Output Tensor Shape:", res.shape)
