#!/usr/bin/env bash
# Local (WSL / single-GPU) environment profile.
# Edit paths below if you keep the data elsewhere, then from code/ run:
#     source scripts/env_local.sh
set -a

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "$CODE_DIR/.." && pwd)"

# raw BP4D (fixed raw layout: 2D+3D/, Thermal/, Physiology/)
export RAW_DATA_PATH="${RAW_DATA_PATH:-$REPO_DIR/data/raw/BP4D}"

# canonical per-session layout (output of data/prepare_bp4d.py)
export DATA_PATH="${DATA_PATH:-$REPO_DIR/data/processed/bp4d_canonical}"

export DATA_SET="${DATA_SET:-bp4d+}"
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/output}"
export NUM_WORKERS="${NUM_WORKERS:-4}"

set +a

echo "[env_local] RAW_DATA_PATH = $RAW_DATA_PATH"
echo "[env_local] DATA_PATH     = $DATA_PATH"
echo "[env_local] OUTPUT_DIR    = $OUTPUT_DIR"
