#!/usr/bin/env bash
# Lichtenberg HPC: data-pipeline smoke test (load 1-2 sessions, report shapes).
# 1) edit scripts/env_hpc.sh   2) mkdir -p logs
# 3) sbatch scripts/hpc/submit_inspect.sbatch
#SBATCH --job-name=inspect_bp4d
#SBATCH --partition=<gpu-partition>   # EDIT: e.g. gpu
#SBATCH --account=<account>           # EDIT (optional)
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/inspect_bp4d_%j.out
#SBATCH --error=logs/inspect_bp4d_%j.err

set -euo pipefail
ENV_SH="$(cd "$(dirname "$0")/.." && pwd)/env_hpc.sh"
# shellcheck disable=SC1090
source "$ENV_SH"
activate_project_env

cd "$CODE_DIR"
python -u runners/run_inspect_data.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR/inspect_data" \
    --max_sessions "${INSPECT_SESSIONS:-2}" \
    --max_entries "${INSPECT_ENTRIES:-2}" \
    --input_size 64 \
    --plot

echo "Inspection summary: $OUTPUT_DIR/inspect_data/inspect_summary.json"
