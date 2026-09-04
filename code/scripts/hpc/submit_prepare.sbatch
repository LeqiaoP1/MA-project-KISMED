#!/usr/bin/env bash
# Lichtenberg HPC: convert raw BP4D -> canonical layout (CPU job).
# 1) edit scripts/env_hpc.sh (paths/venv)   2) mkdir -p logs
# 3) sbatch scripts/hpc/submit_prepare.sbatch
#SBATCH --job-name=prep_bp4d
#SBATCH --partition=<cpu-partition>   # EDIT: e.g. standard / cpu
#SBATCH --account=<account>           # EDIT (optional)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/prep_bp4d_%j.out
#SBATCH --error=logs/prep_bp4d_%j.err

set -euo pipefail
ENV_SH="$(cd "$(dirname "$0")/.." && pwd)/env_hpc.sh"
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
