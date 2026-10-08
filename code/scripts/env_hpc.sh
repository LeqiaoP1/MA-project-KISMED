#!/usr/bin/env bash
# Lichtenberg HPC environment profile (EDIT the values marked EDIT below).
# This file is sourced by scripts/hpc/submit_*.sbatch jobs.
set -a

# --- Python environment (Lichtenberg supports venv) ------------------------ #
export VENV="${VENV:-$HOME/MA-project-KISMED/.env}"            # EDIT: e.g. $HOME/.venvs/kismed
# optional alternative: use a conda env instead
export CONDA_ENV="${CONDA_ENV:-}"                    # EDIT (leave empty if using VENV)

# --- paths on the cluster -------------------------------------------------- #
export RAW_DATA_PATH="${WORK_PROJ}/test"          # EDIT
export DATA_PATH="${HOME}/MA-project-KISMED/data/processed/bp4d_canonical"  # EDIT
export PROJ_DIR="${HOME}/MA-project-KISMED"
# Checkpoints are ~1.2 GB EACH (30 of them at save_ckpt_freq 5 ~= 36 GB), and
# $PROJ_DIR/output sits on $HOME's small quota -- a full Stage-2/3 run fills it
# and dies mid-save. Prefer the scratch file system when the site provides one.
# NOTE $WORK_SCRATCH is a SITE variable, not defined by this profile, so the
# fallback keeps the old behaviour on a cluster that has no scratch.
export OUTPUT_DIR="${OUTPUT_DIR:-${WORK_SCRATCH:-$PROJ_DIR}/output}"  # EDIT
export CODE_DIR="${PROJ_DIR}/code"

# Stage-1 initial (downloaded) ViT encoder weights. Must be on a path every
# compute node can read: pre-fetch with runners/run_download_weights.py on a
# LOGIN node (compute nodes have no internet), then reuse the cache.
export INITIAL_MODELS_DIR="${INITIAL_MODELS_DIR:-$(dirname "$PROJ_DIR")/models/initial}"
export DATA_SET="${DATA_SET:-bp4d+}"
# MEASURED (2026-10-08): the Stage-2 loop is CPU-work-bound, not GPU-bound -- a
# single Blackwell GPU does the step in 0.20 s while the iteration takes 2.45 s,
# and forcing OMP_NUM_THREADS to 1 only SHIFTS time between loader and compute
# (data 1.27->0.65 s, step 1.15->1.72 s, total unchanged). Batch 16 x N workers
# means ~3200 .wmv frames decoded+resized per iteration PER RANK, so the worker
# count is limited by the CORES the job actually owns: --cpus-per-task must
# cover num_workers + the training process. This default matches every tir_roi
# config's `num_workers: 8`; the old 4 silently overrode them.
export NUM_WORKERS="${NUM_WORKERS:-8}"

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
