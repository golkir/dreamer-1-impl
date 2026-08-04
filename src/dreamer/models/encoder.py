import torch
from torch import nn
from einops import rearrange
from dreamer.config import DreamerConfig


class ObservationEncoder(nn.Module):
    def __init__(self, input_channels: int, config: DreamerConfig):
        super().__init__()
        self.config = config
        self.depth = self.config.encoder.depth # default: 32
        activation = getattr(nn, self.config.encoder.activation)

        layers = []
        curr_in_channels = input_channels
        curr_out = self.depth

        for multiplier in range(1, 5):
            
            layers.append(
                nn.Conv2d(
                    curr_in_channels,
                    curr_out,
                    kernel_size=self.config.encoder.kernel_sizes[multiplier - 1],
                    stride=self.config.encoder.strides[multiplier - 1],
                )
            )
            layers.append(activation())
            curr_in_channels = curr_out
            curr_out *= 2


        self.cnn = nn.Sequential(*layers)


    def forward(self, x):
        # (B, S, 3, H, W)
        B, S = x.shape[:2]
        x = rearrange(x, "b s c h w -> (b s) c h w")
        x = self.cnn(x)
        x = rearrange(x, "(b s) c h w -> b s (c h w)", b=B, s=S)
        return x


if __name__ == "__main__":

    from dataclasses import dataclass

    im = torch.randn((5, 3, 100, 100))

    @dataclass
    class Config:
        input_shape: tuple
        kernel_size: tuple
        stride: int
        depth: int
        activation: str
        feature_dim: int

    config = Config(
        input_shape=(3, 100, 100),
        kernel_size=(3, 3),
        stride=2,
        depth=32,
        activation="ELU",
        feature_dim=1024,
    )

    encoder = ObservationEncoder(3, config)

    res = encoder(im)

    print(res.shape, "shape after cnn")
