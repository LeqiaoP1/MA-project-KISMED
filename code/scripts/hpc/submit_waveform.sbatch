#!/usr/bin/env bash
# Lichtenberg HPC TEMPLATE: Stage-3 waveform fine-tuning (multi-GPU, single node).
# NOTE: only usable once the spatio-temporal video model front-end + a Stage-2
# checkpoint exist (see docs/ImplementationPlan.md / code/README.md).
# 1) edit scripts/env_hpc.sh   2) mkdir -p logs
# 3) TARGET=bvp sbatch scripts/hpc/submit_waveform.sbatch
#SBATCH --job-name=waveform
#SBATCH --partition=<gpu-partition>   # EDIT
#SBATCH --account=<account>           # EDIT (optional)
#SBATCH --gres=gpu:4                  # EDIT: GPUs per node
#SBATCH --ntasks=4                    # EDIT: == number of GPUs
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/waveform_%j.out
#SBATCH --error=logs/waveform_%j.err

set -euo pipefail
ENV_SH="$(cd "$(dirname "$0")/.." && pwd)/env_hpc.sh"
# shellcheck disable=SC1090
source "$ENV_SH"
activate_project_env

cd "$CODE_DIR"
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)}"
export MASTER_PORT="${MASTER_PORT:-29500}"

TARGET="${TARGET:-bvp}"   # bvp | resp | eda
mkdir -p "$OUTPUT_DIR/finetune/$TARGET"

# one task per GPU; utils/dist.py reads SLURM_* env to init DDP
srun --ntasks="$SLURM_NTASKS" --ntasks-per-node="$SLURM_NTASKS_PER_NODE" \
    python -u runners/run_waveform.py \
    -c "configs/finetune/$TARGET.yaml" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR/finetune/$TARGET"

echo "Waveform ($TARGET) finished -> $OUTPUT_DIR/finetune/$TARGET"
