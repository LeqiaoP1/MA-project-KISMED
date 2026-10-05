#!/usr/bin/env bash
# Lichtenberg HPC -- LOGIN-NODE ONLY: pre-fetch the Stage-1 ("initial") ViT
# encoder weights into the shared cache.
#
# WHY: the compute nodes have NO internet. runners/*.py resolve a
# `--pretrained_encoder base` / `--finetune large` spec by DOWNLOADING on first
# use, which works on a login node but hangs/fails inside a batch job. Download
# once here, then every job reuses the cache.
#
# Run from code/ (do NOT sbatch this):
#   bash scripts/hpc/download_weights.sh            # VideoMAE ViT-B (~1.3 GB)
#   bash scripts/hpc/download_weights.sh --all      # base + large (~4 GB)
#   bash scripts/hpc/download_weights.sh --list     # show the spec table
#   bash scripts/hpc/download_weights.sh mae:base   # 2-D MAE control
#
# Cache location: $WEIGHTS_DIR, else $PROJ_DIR/models/initial (in-repo, matches
# scripts/env_local.sh). NOTE env_hpc.sh itself resolves INITIAL_MODELS_DIR to
# "$(dirname "$PROJ_DIR")/models/initial" == $HOME/models/initial, i.e. OUTSIDE
# the repo; this script and submit_pretrain.sbatch deliberately override that so
# the two never disagree. Export WEIGHTS_DIR to relocate both at once.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SH="$SCRIPT_DIR/../env_hpc.sh"
if [ ! -f "$ENV_SH" ]; then
    echo "ERROR: env_hpc.sh not found at $ENV_SH" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_SH"
activate_project_env

cd "$CODE_DIR"

export INITIAL_MODELS_DIR="${WEIGHTS_DIR:-$PROJ_DIR/models/initial}"
mkdir -p "$INITIAL_MODELS_DIR"

echo "[download_weights] cache : $INITIAL_MODELS_DIR"
echo "[download_weights] python: $(command -v python)"

if [ "$#" -gt 0 ]; then
    python -u runners/run_download_weights.py "$@"
else
    python -u runners/run_download_weights.py base
fi

echo
echo "[download_weights] cached under $INITIAL_MODELS_DIR:"
ls -lh "$INITIAL_MODELS_DIR" 2>/dev/null || true
echo "[download_weights] done -- submit_pretrain.sbatch / smoke_pretrain.sbatch"
echo "                   will now reuse this cache."
