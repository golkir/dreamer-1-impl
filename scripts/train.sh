#!/usr/bin/env bash
# One-command training on a fresh GPU machine. Re-running the same command resumes
# from the last checkpoint, so it is safe to use on preemptible instances.
#
#   scripts/train.sh dmc walker_walk            # DeepMind Control, seed 0
#   scripts/train.sh atari pong 1               # Atari Pong, seed 1
#   scripts/train.sh dmc cheetah_run 0 --set train.batch_size=32
#
# Output goes to runs/<suite>_<task>_s<seed>/ (TensorBoard logs, checkpoints) and
# the console log to runs/<suite>_<task>_s<seed>.log.
set -euo pipefail
cd "$(dirname "$0")/.."

preset=${1:-dmc}
task=${2:-walker_walk}
seed=${3:-0}
shift $(($# < 3 ? $# : 3))

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --quiet

uv run python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

suite=$preset
[[ $preset == debug ]] && suite=dmc
mkdir -p runs
uv run dreamer-train --preset "$preset" --task "$task" --seed "$seed" --resume "$@" \
  2>&1 | grep --line-buffered -v "libEGL warning" | tee -a "runs/${suite}_${task}_s${seed}.log"
