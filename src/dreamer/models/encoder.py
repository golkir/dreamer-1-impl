import torch
from torch import nn
from dreamer.config import DreamerConfig


class ObservationEncoder(nn.Module):
    def __init__(self, input_channels: int, config: DreamerConfig):
        super().__init__()
        self.config = config
        self.depth = self.config.encoder.depth
        activation = getattr(nn, self.config.encoder.activation)

        layers = []
        curr_in_channels = input_channels

        for multiplier in range(1, 5):
            out_channels = self.depth * multiplier
            layers.append(
                nn.Conv2d(
                    curr_in_channels,
                    out_channels,
                    kernel_size=self.config.encoder.kernel_sizes[multiplier-1],
                    stride=self.config.encoder.strides[multiplier-1],
                )
            )
            layers.append(activation())
            curr_in_channels = out_channels

        self.cnn = nn.Sequential(*layers)

        input_shape = getattr(self.config, "observation_shape", (input_channels, 64, 64))

        fc_input_dim = self._get_conv_output_dim(input_shape)

        self.fc = nn.Linear(fc_input_dim, self.config.encoder.feature_dim)

    def _get_conv_output_dim(self, input_shape):
        """Performs a safe dry-run to determine the exact flattening dimension."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.cnn(dummy_input)
            return int(torch.numel(dummy_output))

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.cnn(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
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
