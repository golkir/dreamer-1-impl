from contextlib import contextmanager

import torch
import torch.nn.functional as F


@contextmanager
def frozen(*modules):
    """Don't compute weight gradients for these modules; gradients still flow through them."""
    params = [p for m in modules for p in m.parameters() if p.requires_grad]
    for p in params:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p in params:
            p.requires_grad_(True)


def create_normal_dist_from_output(
    x,
    mean_scale=1,
    init_std=0,
    min_std=0.1,
    activation=None,
    event_shape=None,
):
    # distribution math stays in fp32 under autocast
    mean, std_logits = torch.chunk(x.float(), 2, dim=-1)

    mean = mean / mean_scale
    if activation is not None:
        mean = activation(mean)
    mean = mean * mean_scale

    std = F.softplus(std_logits + init_std) + min_std

    return create_normal_dist_from_params(mean, std, event_shape)


def create_normal_dist_from_params(
    mean,
    std,
    event_shape=None,
):

    # distribution math stays in fp32 under autocast
    if torch.is_tensor(mean):
        mean = mean.float()
    if torch.is_tensor(std):
        std = std.float()
    dist = torch.distributions.Normal(mean, std)

    if event_shape is not None:
        dist = torch.distributions.Independent(dist, event_shape)

    return dist


# def create_normal_dist(
#     x,
#     std=None,
#     mean_scale=1,
#     init_std=0,
#     min_std=0.1,
#     activation=None,
#     event_shape=None,
# ):
#     if std == None:
#         mean, std = torch.chunk(x, 2, -1)
#         mean = mean / mean_scale
#         if activation:
#             mean = activation(mean)
#         mean = mean_scale * mean
#         std = F.softplus(std + init_std) + min_std
#     else:
#         mean = x
#     dist = torch.distributions.Normal(mean, std)
#     if event_shape:
#         dist = torch.distributions.Independent(dist, event_shape)
#     return dist