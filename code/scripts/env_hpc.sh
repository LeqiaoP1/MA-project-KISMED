#!/usr/bin/env bash
# Lichtenberg HPC environment profile (EDIT the values marked EDIT below).
# This file is sourced by scripts/hpc/submit_*.sbatch jobs.
set -a

# --- Python environment (Lichtenberg supports venv) ------------------------ #
export VENV="${VENV:-$HOME/MA-project-KISMED/.env}"            # EDIT: e.g. $HOME/.venvs/kismed
# optional alternative: use a conda env instead
export CONDA_ENV="${CONDA_ENV:-}"                    # EDIT (leave empty if using VENV)

# --- paths on the cluster -------------------------------------------------- #
export RAW_DATA_PATH="${WORK_PROJ}/BP4D+"          # EDIT
export DATA_PATH="${HOME}/MA-project-KISMED/data/processed/bp4d_canonical"  # EDIT
export PROJ_DIR="${HOME}/MA-project-KISMED"
export OUTPUT_DIR="${PROJ_DIR}/output"                       # EDIT
export CODE_DIR="${PROJ_DIR}/code"

# Stage-1 initial (downloaded) ViT encoder weights. Must be on a path every
# compute node can read: pre-fetch with runners/run_download_weights.py on a
# LOGIN node (compute nodes have no internet), then reuse the cache.
export INITIAL_MODELS_DIR="${INITIAL_MODELS_DIR:-$(dirname "$PROJ_DIR")/models/initial}"
export DATA_SET="${DATA_SET:-bp4d+}"
export NUM_WORKERS="${NUM_WORKERS:-4}"

# --- SLURM resource template (overridable per job / on the command line) --- #
export PARTITION="${PARTITION:-gpu}"        # EDIT: partition to submit to
export ACCOUNT="${ACCOUNT:-}"               # EDIT (optional)
export GPU_TYPE="${GPU_TYPE:-a100:4gb?}"    # EDIT: gres suffix if required, e.g. a100:40gb
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"  # EDIT: GPUs requested per node

set +a

activate_project_env() {
    if [ -n "$VENV" ] && [ -f "$VENV/bin/activate" ]; then
        # shellcheck disable=SC1090
        source "$VENV/bin/activate"
        echo "[env_hpc] activated venv: $VENV"
    elif [ -n "$CONDA_ENV" ]; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
        echo "[env_hpc] activated conda env: $CONDA_ENV"
    else
        echo "[env_hpc] WARNING: neither VENV nor CONDA_ENV set; relying on loaded python."
    fi
}
