#!/usr/bin/env bash
# Lichtenberg HPC TEMPLATE: Stage-3 waveform fine-tuning (multi-GPU, single node).
# NOTE: only usable once the spatio-temporal video model front-end + a Stage-2
# checkpoint exist (see docs/ImplementationPlan.md / code/README.md).
# 1) edit scripts/env_hpc.sh   2) mkdir -p logs
# 3) TARGET=bp sbatch scripts/hpc/submit_waveform.sbatch
#SBATCH --job-name=waveform
#SBATCH --partition=acc               # GPU; 1 day max ('acc_long' = 7 days)
# NB: the LUA plugin rejects --mem, so memory is --mem-per-cpu * cpus-per-task.
# --account is omitted on purpose: the plugin bills your default project.
#SBATCH --gres=gpu:4                  # EDIT: GPUs per node (4 or 8 here)
#SBATCH --ntasks=4                    # EDIT: == number of GPUs
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G              # 4 tasks x 8 cpus x 2G = 64G total
#SBATCH --time=08:00:00
#SBATCH --output=logs/waveform_%j.out
#SBATCH --error=logs/waveform_%j.err

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
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)}"
export MASTER_PORT="${MASTER_PORT:-29500}"

TARGET="${TARGET:-bp}"   # bp | resp | eda
mkdir -p "$OUTPUT_DIR/finetune/$TARGET"

# one task per GPU; utils/dist.py reads SLURM_* env to init DDP
srun --ntasks="$SLURM_NTASKS" --ntasks-per-node="$SLURM_NTASKS_PER_NODE" \
    python -u runners/run_waveform.py \
    -c "configs/finetune/$TARGET.yaml" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR/finetune/$TARGET"

echo "Waveform ($TARGET) finished -> $OUTPUT_DIR/finetune/$TARGET"
