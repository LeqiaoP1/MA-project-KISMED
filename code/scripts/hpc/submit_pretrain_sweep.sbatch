#!/usr/bin/env bash
#SBATCH --job-name=adr001_pretrain
#SBATCH --partition=acc_long
#SBATCH --gres=gpu:rtx6000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/adr001_pretrain_%j.out
#SBATCH --error=logs/adr001_pretrain_%j.err
#
# Wrapper for scripts/hpc/submit_pretrain_sweep.sh. STAGE 2 ONLY: it never runs
# run_waveform.py. Five folds run SEQUENTIALLY inside this ONE allocation.
#
#   EXP_ROOT=$WORK_SCRATCH/fourstream_v1 \
#   ROI_CACHE=... RGB2D_CACHE=... RESP_CACHE=... BP_ARTEFACT=... \
#       bash scripts/hpc/submit.sh scripts/hpc/submit_pretrain_sweep.sbatch
#
# There is deliberately NO --array: parallel folds would share MASTER_PORT and
# a reader could pool a half-written sweep.
#
# `sbatch` does not expand variables inside #SBATCH lines, so the resource block
# above is literal; override it on the command line instead, e.g.
#   sbatch --partition=acc --time=24:00:00 scripts/hpc/submit_pretrain_sweep.sbatch
set -euo pipefail

mkdir -p logs
if [ -f scripts/env_hpc.sh ]; then
  # shellcheck disable=SC1091
  source scripts/env_hpc.sh
  activate_project_env
fi

: "${EXP_ROOT:?set EXP_ROOT (e.g. EXP_ROOT=\$WORK_SCRATCH/fourstream_v1)}"

echo "[sbatch] job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "[sbatch] python: $(command -v "${PY:-python}")"
echo "[sbatch] git   : $(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"

exec bash scripts/hpc/submit_pretrain_sweep.sh
