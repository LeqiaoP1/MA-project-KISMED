#!/usr/bin/env bash
# Lichtenberg HPC: data-pipeline smoke test (load 3 sessions, report shapes).
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
# Plot & save a PNG for every clip of train+val. Optional HPC overrides:
# INSPECT_SESSIONS, INSPECT_ENTRIES (0 = no cap / all clips),
# INSPECT_CLIP_DURATION, INSPECT_MAX_CLIPS (the latter two add a flag only when
# set; runner defaults otherwise: 10 s window, no per-session cap).
EXTRA_ARGS=()
if [ -n "${INSPECT_CLIP_DURATION:-}" ]; then
    EXTRA_ARGS+=(--clip_duration "$INSPECT_CLIP_DURATION")
fi
if [ -n "${INSPECT_MAX_CLIPS:-}" ]; then
    EXTRA_ARGS+=(--max_clips "$INSPECT_MAX_CLIPS")
fi

python -u runners/run_inspect_data.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR/inspect_data" \
    --max_sessions "${INSPECT_SESSIONS:-3}" \
    --max_entries "${INSPECT_ENTRIES:-0}" \
    --input_size 64 \
    --plot \
    "${EXTRA_ARGS[@]}"

echo "Inspection summary: $OUTPUT_DIR/inspect_data/inspect_summary.json"
