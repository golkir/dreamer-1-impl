import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, dim, dim_out, num_layers=2, hidden_size=256):
        super().__init__()

        layers = []
        dims = (dim, *((hidden_size,) * (num_layers - 1)))

        for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            is_last = ind == (len(dims) - 1)

            layers.extend(
                [
                    nn.Linear(layer_dim_in, layer_dim_out),
                    nn.GELU() if not is_last else nn.Identity(),
                ]
            )

        self.net = nn.Sequential(*layers, nn.Linear(hidden_size, dim_out))

    def forward(self, x):
        return self.net(x)