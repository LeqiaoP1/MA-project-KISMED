#!/usr/bin/env bash
#SBATCH --job-name=adr001_sweep
#SBATCH --partition=acc_long
#SBATCH --gres=gpu:rtx6000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/adr001_sweep_%j.out
#SBATCH --error=logs/adr001_sweep_%j.err
#
# Wrapper for scripts/hpc/submit_sweep.sh. The five folds run SEQUENTIALLY
# inside this ONE allocation; the 4 GPUs are used by the fold that is running.
#
#   EXP_ROOT=$WORK_SCRATCH/protocol_v1 sbatch scripts/hpc/submit_sweep.sbatch
#   EXP_ROOT=... FOLDS="0 1" sbatch scripts/hpc/submit_sweep.sbatch
#
# There is deliberately NO --array: an array task would run the whole sweep, and
# the tasks would run in parallel — two folds would then share MASTER_PORT and,
# worse, a reader could pool a half-written sweep.
#
# `sbatch` does not expand variables inside #SBATCH lines, so the resource block
# above is literal; override it on the command line instead, e.g.
#   sbatch --partition=acc --time=24:00:00 scripts/hpc/submit_sweep.sbatch
set -euo pipefail

mkdir -p logs
if [ -f scripts/env_hpc.sh ]; then
  # shellcheck disable=SC1091
  source scripts/env_hpc.sh
fi

: "${EXP_ROOT:?set EXP_ROOT (e.g. EXP_ROOT=\$WORK_SCRATCH/protocol_v1)}"

echo "[sbatch] job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "[sbatch] python: $(command -v "${PY:-python}")"
echo "[sbatch] git   : $(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"

exec bash scripts/hpc/submit_sweep.sh
