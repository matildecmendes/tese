#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/check_dataset_2d.sh DATASET_PATH [MAX_FILES]"
  exit 1
fi

DATASET_PATH="$1"
MAX_FILES="${2:-1000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$(dirname "$REPO_ROOT"):${PYTHONPATH:-}"

python -B "$REPO_ROOT/src/land2d/check_dataset_2d.py" \
  --dataset_path "$DATASET_PATH" \
  --max_files "$MAX_FILES" \
  --resolution 256
