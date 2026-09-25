#!/usr/bin/env bash
# Lichtenberg HPC: convert raw BP4D -> canonical layout (CPU job).
# 1) edit scripts/env_hpc.sh (paths/venv)   2) mkdir -p logs
# 3) sbatch scripts/hpc/submit_prepare.sbatch
#SBATCH --job-name=prep_bp4d
#SBATCH --partition=deflt             # CPU; 1 day max ('deflt_short' = 30 min)
# NB: the LUA plugin rejects --mem, so memory is --mem-per-cpu * cpus-per-task.
# --account is omitted on purpose: the plugin bills your default project.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G              # 4 cpus x 6G = 24G total
#SBATCH --time=03:00:00
#SBATCH --output=logs/prep_bp4d_%j.out
#SBATCH --error=logs/prep_bp4d_%j.err

set -euo pipefail
# sbatch COPIES this script into the spool dir and runs that copy, so $0 is
# /var/spool/slurmd/job<id>/slurm_script -- NOT the repo path. Resolve the
# profile from $SLURM_SUBMIT_DIR (submit from code/), and fall back to
# BASH_SOURCE only for direct `bash scripts/hpc/...` runs.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/env_hpc.sh" ]; then
    ENV_SH="$SLURM_SUBMIT_DIR/scripts/env_hpc.sh"
else
    ENV_SH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env_hpc.sh"
fi
if [ ! -f "$ENV_SH" ]; then
    echo "ERROR: env_hpc.sh not found (tried $ENV_SH); submit from code/: cd code && sbatch scripts/hpc/..." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_SH"
activate_project_env

cd "$CODE_DIR"
# PREPARE_SESSIONS: 0 = all sessions, N = only the first N (quick smoke)
python -u data/prepare_bp4d.py \
    --raw_root "$RAW_DATA_PATH" \
    --out_root "$DATA_PATH" \
    --limit_sessions "${PREPARE_SESSIONS:-0}" \
    --images copy

echo "Prepare finished -> $DATA_PATH"
