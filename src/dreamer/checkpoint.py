"""
Saving and loading Dreamer checkpoints (dreamer_<iteration>.pt).

Checkpoints written before the config was stored load the same way;
checkpoint_config() returns None for them.
"""

import dataclasses
import os
import re

import torch

from dreamer.config import DreamerConfig, config_from_dict
from dreamer.dreamer_alg import Dreamer


def save_checkpoint(dreamer: Dreamer, iteration: int, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "num_total_episode": dreamer.num_total_episode,
            "model_state": dreamer.state_dict(),
            "world_optimizer": dreamer.world_optimizer.state_dict(),
            "actor_optimizer": dreamer.actor_optimizer.state_dict(),
            "critic_optimizer": dreamer.critic_optimizer.state_dict(),
            "config": dataclasses.asdict(dreamer.config),
        },
        path,
    )


def load_checkpoint(
    dreamer: Dreamer, path: str, map_location="cpu", optimizers: bool = True
) -> int:
    ckpt = torch.load(path, map_location=map_location)
    dreamer.load_state_dict(ckpt["model_state"])
    if optimizers:
        dreamer.world_optimizer.load_state_dict(ckpt["world_optimizer"])
        dreamer.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        dreamer.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
    dreamer.num_total_episode = ckpt.get("num_total_episode", 0)
    return ckpt["iteration"]


def checkpoint_config(path: str) -> DreamerConfig | None:
    """The config the checkpoint was trained with, or None for older checkpoints."""
    values = torch.load(path, map_location="cpu").get("config")
    return None if values is None else config_from_dict(DreamerConfig, values)


def latest_checkpoint(directory: str) -> str:
    """The dreamer_<iteration>.pt with the highest iteration in a directory."""
    found = {}
    for name in os.listdir(directory):
        match = re.fullmatch(r"dreamer_(\d+)\.pt", name)
        if match:
            found[int(match.group(1))] = name
    if not found:
        raise FileNotFoundError(f"no dreamer_<iteration>.pt checkpoints in {directory}")
    return os.path.join(directory, found[max(found)])
