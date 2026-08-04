from dataclasses import dataclass, field, fields
import torch

class Trajectory:
    def append(self, **kwargs) -> None:
        for f in fields(self):
            getattr(self, f.name).append(kwargs[f.name])

    def stack_fields(self, dim: int = 1):
        stacked = {}

        for f in fields(self):
            values = getattr(self, f.name)

            if len(values) > 0:
                stacked[f.name] = torch.stack(values, dim=dim)
            else:
                stacked[f.name] = []
            setattr(self, f.name, [])
        return type(self)(**stacked)

@dataclass
class RSSMTrajectory(Trajectory):
    prior: list[torch.Tensor] = field(default_factory=list)
    prior_mean: list[torch.Tensor] = field(default_factory=list)
    prior_std: list[torch.Tensor] = field(default_factory=list)
    posterior: list[torch.Tensor] = field(default_factory=list)
    posterior_mean: list[torch.Tensor] = field(default_factory=list)
    posterior_std: list[torch.Tensor] = field(default_factory=list)
    h: list[torch.Tensor] = field(default_factory=list)


@dataclass
class ImaginationTrajectory(Trajectory):
    prior: list[torch.Tensor] = field(default_factory=list)
    h: list[torch.Tensor] = field(default_factory=list)


@dataclass
class EnvironmentConfig:
    benchmark: str = "dmc"

    # DeepMind Control Suite
    domain_name: str = "quadruped"
    task_name: str = "run"
    seed: int = 0
    visualize_reward: bool = False

    # Observation settings
    from_pixels: bool = True
    height: int = 64
    width: int = 64
    frame_skip: int = 2
    pixel_norm: bool = False

    # Gym backend (only used when benchmark == "gym")
    gym_id: str = "CartPole-v1"

@dataclass
class EncoderConfig:
    obs_channels: int = 3
    obs_height: int = 64
    obs_width: int = 64
    depth: int = 32
    kernel_sizes: tuple = (4, 4, 4, 4)
    feature_dim: int = 1024
    strides: tuple = (2, 2, 2, 2)
    activation: str = "ReLU"



@dataclass
class DecoderConfig:
    obs_channels: int = 3
    obs_height: int = 64
    obs_width: int = 64
    depth: int = 32
    kernel_sizes: tuple = (5, 5, 6, 6)
    strides: tuple = (2, 2, 2, 2)
    activation: str = "ReLU"


@dataclass
class RSSMConfig:
    z_dim: int = 30
    h_dim: int = 200
    obs_emb_dim: int = 1024
    hidden_dim: int = 200
    min_std: float = 0.1


@dataclass
class DreamerConfig:
    # environment / data
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    env_backend: str = "dmc"  # "dmc" | "gym"
    action_repeat: int = 2
    observation_dim: int = 1024
    observation_shape: tuple = (3, 64, 64)
    action_dim: int = 12
    seq_length: int = 4
    image_size: int = 64

    # model
    rssm: RSSMConfig = field(default_factory=RSSMConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    horizon: int = 15

    # optimization
    replay_buffer_size: int = 100
    world_lr: float = 6e-4
    actor_lr: float = 8e-5
    critic_lr: float = 8e-5
    grad_clip: float = 100.0
    grad_norm_type: float = 2.0
    free_nats: float = 3.0  # KL free bits, prevents posterior collapse
    discount: float = 0.99
    lam: float = 0.95  # lambda for the lambda-return

    # training loop
    seed_episodes: int = 1
    num_iterations: int = 1000
    batch_size: int = 5
    num_interaction_episodes: int = 1
    num_evaluate: int = 5
    eval_every: int = 10
    checkpoint_every: int = 50

    # misc
    seed: int = 0
    device: str = "cpu"
    mixed_precision: str = "no"  # "no" | "fp16" | "bf16" -> passed to Accelerator
    log_dir: str = "runs/dreamer"
    checkpoint_dir: str = "checkpoints/dreamer"
