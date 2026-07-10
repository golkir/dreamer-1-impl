import torch
import torch.nn.functional as F

def create_normal_dist_from_output(
    x,
    mean_scale=1,
    init_std=0,
    min_std=0.1,
    activation=None,
    event_shape=None,
):
    mean, std_logits = torch.chunk(x, 2, dim=-1)

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

    dist = torch.distributions.Normal(mean, std)

    if event_shape is not None:
        dist = torch.distributions.Independent(dist, event_shape)

    return dist