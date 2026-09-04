#!/usr/bin/env bash
# Lichtenberg HPC environment profile (EDIT the values marked EDIT below).
# This file is sourced by scripts/hpc/submit_*.sbatch jobs.
set -a

# --- Python environment (Lichtenberg supports venv) ------------------------ #
export VENV="${VENV:-/path/to/your/venv}"            # EDIT: e.g. $HOME/.venvs/kismed
# optional alternative: use a conda env instead
export CONDA_ENV="${CONDA_ENV:-}"                    # EDIT (leave empty if using VENV)

# --- paths on the cluster -------------------------------------------------- #
export RAW_DATA_PATH="${RAW_DATA_PATH:-/path/on/hpc/data/raw/BP4D}"          # EDIT
export DATA_PATH="${DATA_PATH:-/path/on/hpc/data/processed/bp4d_canonical}"  # EDIT
export OUTPUT_DIR="${OUTPUT_DIR:-/path/on/hpc/output}"                       # EDIT
export CODE_DIR="${CODE_DIR:-$SLURM_SUBMIT_DIR/code}"
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
