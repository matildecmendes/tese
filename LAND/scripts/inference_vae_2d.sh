#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/inference_vae_2d.sh DATASET_PATH VAE_PATH OUTPUT_DIR [NUM_SAMPLES] [extra Python arguments]"
  exit 1
fi

DATASET_PATH="$1"
VAE_PATH="$2"
OUTPUT_DIR="$3"
NUM_SAMPLES="${4:-16}"
if [[ $# -ge 4 ]]; then shift 4; else shift 3; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$(dirname "$REPO_ROOT"):${PYTHONPATH:-}"

python -B "$REPO_ROOT/src/land2d/inference_vae_2d.py" \
  --dataset_path "$DATASET_PATH" \
  --vae_path "$VAE_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --num_samples "$NUM_SAMPLES" \
  "$@"
