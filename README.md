# Dreamer (v1) in PyTorch

A compact implementation of **Dreamer** from Hafner et al., *Dream to Control: Learning
Behaviors by Latent Imagination* (ICLR 2020), with training support for
DeepMind Control (from pixels), Atari and pixel-rendered Gymnasium environments.

## Quick start

```bash
uv sync                                            # install (Python 3.12)
uv run pytest                                      # tests, ~15 s on CPU
uv run dreamer-train --preset debug                # tiny end-to-end run on CPU, ~1 min
```

### Training on a GPU machine

```bash
git clone <this repo> && cd kg_project
scripts/train.sh dmc walker_walk          # installs uv + deps, then trains (1M env steps)
scripts/train.sh atari pong               # Atari (10M env steps by default)
tensorboard --logdir runs                 # watch progress
```

`scripts/train.sh` always passes `--resume`, so re-running the same command continues
from the last checkpoint (useful on preemptible instances). Press Ctrl-C and it saves a
checkpoint before exiting.

Equivalent direct invocation:

```bash
uv run dreamer-train --preset dmc --task cheetah_run --seed 1
uv run dreamer-train --preset atari --task breakout --steps 5e6
uv run dreamer-train --suite gym --task Pendulum-v1 --steps 2e5
```

Evaluate a run and record a video:

```bash
uv run dreamer-eval runs/dmc_walker_walk_s0 --episodes 10 --video walker.mp4
```

**GPU notes**

- The locked `torch` wheel is built for CUDA 13 and needs a recent NVIDIA driver. If
  `torch.cuda.is_available()` prints `False` on a GPU box, install a wheel that matches the
  driver, e.g. `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`.
  Note that `uv sync` will switch back to the locked version.
- DMC renders headless with `MUJOCO_GL=egl` (the default here). If EGL is missing, use
  `MUJOCO_GL=osmesa` with `PYOPENGL_PLATFORM=osmesa` (needs `libosmesa6`).
- The device is picked automatically (`cuda` > `mps` > `cpu`); force it with `--device cuda:1`.
- Environments with TensorFlow installed (e.g. Kaggle): importing TensorFlow loads Keras/JAX,
  whose bundled LLVM makes MuJoCo's first render segfault. The logger therefore forces
  TensorBoard's TensorFlow-free mode. If a DMC render still crashes, check what else imports
  TensorFlow, or pass `--set run.tensorboard=false`; `metrics.jsonl` and the PNG images are
  written either way.

## Configuration

All settings live in dataclasses in [`src/dreamer/config.py`](src/dreamer/config.py),
grouped into `env`, `model`, `train` and `run`. Choose a preset, then override any field:

```bash
uv run dreamer-train --preset dmc --task hopper_hop \
    --set train.batch_size=32 --set model.dense_units=300 --set run.eval_every=20000
uv run dreamer-train --preset atari --print-config         # show the resolved config
uv run dreamer-train --config runs/<run>/config.json       # start from a saved config
```

| preset  | what it is |
|---------|------------|
| `dmc`   | Paper defaults: 64×64 RGB, action repeat 2, batch 50×50, horizon 15, 1M env steps |
| `atari` | Discrete actions, action repeat 4, sticky actions, continuation head, `tanh` reward squashing, ε-greedy exploration, 10M env steps |
| `debug` | Tiny networks and budgets for checking the pipeline on a CPU |

Tasks: DMC uses `domain_task` (`walker_walk`, `cheetah_run`, `cartpole_swingup`,
`ball_in_cup_catch`, `finger_spin`, `reacher_easy`, `quadruped_run`, ...). Atari uses the game
name (`pong`, `breakout`, `space_invaders`, ...) or a full id such as `ALE/Pong-v5`.

Step counts (`train.steps`, `run.eval_every`, etc.) are **environment steps**, including action
repeat, as in the paper.

## Outputs

Each run writes to `runs/<suite>_<task>_s<seed>/`:

| file | contents |
|------|----------|
| `events.out.tfevents.*` | TensorBoard: losses, KL, entropies, imagined returns, episode/eval returns, FPS, and `openl` images (rows: truth / model / error; the first 5 frames are reconstructions, the rest are open-loop predictions) |
| `metrics.jsonl` | the same scalars, one JSON line per log step |
| `openl/truth_model_error/*.png` | the open-loop prediction images as PNG files (always written) |
| `config.json` | the resolved config |
| `latest.pt` | full checkpoint (weights + optimizers + counters) used by `--resume` |
| `weights_<step>.pt` | periodic weight snapshots |
| `replay.npz` | replay buffer, only with `--set run.save_replay=true` (can be several GB) |

When resuming without a saved replay buffer, the buffer is refilled with the current policy.

## How it works

```
src/dreamer/
  config.py        dataclass configs, presets, --set parsing
  envs.py          DMC / Atari / Gym factories -> uint8 (C, 64, 64) observations
  wrappers.py      Gymnasium wrappers (action repeat, pixels, channels-first, time limit)
  replay_buffer.py sequence replay (obs, action, reward, is_first, is_terminal)
  models/          encoder, decoder, RSSM, dense heads (reward/value/continue/actor)
  dreamer_alg.py   the agent: world-model and behavior learning, acting, diagnostics
  train.py         training loop, logging, checkpointing, CLI
  evaluate.py      evaluation and video recording
```

Each training phase takes `train.train_steps` (100) gradient steps and then collects
`train.train_every` (1000) env steps with the exploration policy:

1. **World model**: a conv encoder and the RSSM filter replayed 50-step sequences. The loss
   is image NLL + reward NLL + max(free_nats, KL(posterior ‖ prior)), plus the continuation
   NLL when enabled. The latent state is reset at episode boundaries (`is_first`).
2. **Behavior**: from every posterior state, the actor imagines 15 steps. Its loss is the
   negative λ-return (λ = 0.95, γ = 0.99), with gradients backpropagated through the frozen
   dynamics, reward and value models. The value net regresses those returns.
3. **Acting**: the actor gets the filtered posterior and samples an action. Exploration adds
   Gaussian noise (0.3) for continuous actions, or ε-greedy for discrete ones. Evaluation
   uses the mode of the actor distribution.

For discrete actions (Atari), the actor is a one-hot categorical with straight-through
gradients and a small entropy bonus. Everything else is shared with the continuous case.
