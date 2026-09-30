from torch import nn


class MLP(nn.Module):
    """``layers`` hidden layers of ``units`` units followed by a linear output."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        units: int = 400,
        layers: int = 2,
        activation: str = "ELU",
    ):
        super().__init__()
        act = getattr(nn, activation)
        modules: list[nn.Module] = []
        dim = in_dim
        for _ in range(layers):
            modules += [nn.Linear(dim, units), act()]
            dim = units
        modules.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)
