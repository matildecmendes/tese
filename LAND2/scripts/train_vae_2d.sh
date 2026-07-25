#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/train_vae_2d.sh DATASET_PATH [RUN_NAME] [extra Python arguments]"
  echo "Example smoke test: bash scripts/train_vae_2d.sh /data/LIDC vaeLAND2D --max_samples 100 --epochs 1 --batch_size 2"
  exit 1
fi

DATASET_PATH="$1"
RUN_NAME="${2:-vaeLAND2D}"
if [[ $# -ge 2 ]]; then shift 2; else shift 1; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$(dirname "$REPO_ROOT"):${PYTHONPATH:-}"

python -B "$REPO_ROOT/src/land2d/train_vae_2d.py" \
  --dataset_path "$DATASET_PATH" \
  --model_config_file "$REPO_ROOT/src/land2d/configs/config_vae_2d.json" \
  --train_config_file "$REPO_ROOT/src/land2d/configs/config_vae_train_2d.json" \
  --model_dir "$REPO_ROOT/checkpoints/vae_2d" \
  --log_dir "$REPO_ROOT/logs/vae_2d" \
  --run_name "$RUN_NAME" \
  "$@"
