# Dreamer (v1) in PyTorch

A compact implementation of **Dreamer** from Hafner et al., *Dream to Control: Learning
Behaviors by Latent Imagination* (ICLR 2020), with training support for
DeepMind Control (from pixels), Atari and pixel-rendered Gymnasium environments.

## Quick start

```bash
uv sync                                            # install (Python 3.12)
uv run pytest                                      # tests
uv run dreamer-train --preset debug                # tiny end-to-end run on CPU, ~1 min
```

Train (the device is picked automatically: `cuda` > `mps` > `cpu`; override with `--device`):

```bash
uv run dreamer-train --preset dmc --task cheetah_run --seed 1
uv run dreamer-train --preset atari --task breakout --steps 5e6
uv run dreamer-train --suite gym --task Pendulum-v1 --steps 2e5
tensorboard --logdir runs
```

Add `--resume` to continue from the last checkpoint. Ctrl-C saves a checkpoint before exiting.
`scripts/train.sh dmc walker_walk` installs dependencies and trains with `--resume`.

Evaluate a run and record a video:

```bash
uv run dreamer-eval runs/dmc_walker_walk_s0 --episodes 10 --video walker.mp4
```

## Differences from other implementations

Compared with minimal Dreamer v1 ports this implementation has:

- **Optional speed-ups.** fp16 mixed precision (`run.amp`) and `torch.compile` of the
  recurrent rollouts (`run.compile`), both checkpoint-compatible, plus a profiling tool
  (`dreamer.benchmark`).
- **Diagnostics.** Open-loop prediction images (truth / model / error), TensorBoard and
  `metrics.jsonl` logging, and periodic evaluation.

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

Tasks: DMC uses `domain_task` (`walker_walk`, `cheetah_run`, `cartpole_swingup`, ...). Atari uses
the game name (`pong`, `breakout`, ...) or a full id such as `ALE/Pong-v5`.

## Speed

```bash
uv run python -m dreamer.benchmark --preset dmc --task walker_walk
uv run python -m dreamer.benchmark --preset dmc --task walker_walk --set run.amp=true --set run.compile=true
```

Reports time per gradient update, GPU utilization, environment step cost and projected total
training time. `run.amp=true` helps when the GPU is the bottleneck; `run.compile=true` helps
when it is waiting on Python.

## Outputs

Each run writes to `runs/<suite>_<task>_s<seed>/`:

| file | contents |
|------|----------|
| `events.out.tfevents.*` | TensorBoard scalars and `openl` images (rows: truth / model / error; the first 5 frames are reconstructions, the rest are open-loop predictions) |
| `metrics.jsonl` | the same scalars, one JSON line per log step |
| `openl/truth_model_error/*.png` | the open-loop prediction images |
| `config.json` | the resolved config |
| `latest.pt` | full checkpoint used by `--resume` |
| `weights_<step>.pt` | periodic weight snapshots |
| `replay.npz` | replay buffer, only with `--set run.save_replay=true` |

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

