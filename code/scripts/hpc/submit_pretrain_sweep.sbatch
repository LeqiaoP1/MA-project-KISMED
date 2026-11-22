#!/usr/bin/env bash
#SBATCH --job-name=adr001_pretrain
#SBATCH --partition=acc_long
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/adr001_pretrain_%j.out
#SBATCH --error=logs/adr001_pretrain_%j.err
#
# GPU TYPE. `sinfo -p acc_long` on 2026-11-21 lists ONLY a100:4, a100:8,
# h100:4, mi300x:8, v100:4 and pvc128g:4 -- there are NO rtx6000 nodes in
# acc_long (rtx6000 exists only in `acc` / `acc_short` / `special_comput`), so a
# rtx6000 directive here can never be satisfied. This build is CUDA-only, which
# also rules out v100 (sm_70 dropped in cu130), mi300x (ROCm) and pvc128g (XPU).
# a100 is the default; override per submission with GPU_TYPE / SBATCH_GRES.
#
# --mem-per-cpu IS MANDATORY here: the LUA job_submit plugin rejects a job with
# "Parameter --mem-per-cpu missing" (sbatch then reports only "Unspecified
# error"). 16 cpus/task covers num_workers=8 plus the main process; 4 tasks x
# 16 x 4G = 256G of the node's ~1T.
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
#
# Submit FROM code/ (both the `logs/` output paths and `scripts/env_hpc.sh`
# below are relative to the submit directory).
set -euo pipefail

mkdir -p logs
if [ ! -f scripts/env_hpc.sh ]; then
  echo "ERROR: submit from code/: scripts/env_hpc.sh not found in $PWD" >&2
  exit 1
fi
if [ -f scripts/env_hpc.sh ]; then
  # shellcheck disable=SC1091
  source scripts/env_hpc.sh
  activate_project_env
fi

: "${EXP_ROOT:?set EXP_ROOT (e.g. EXP_ROOT=\$WORK_SCRATCH/fourstream_v1)}"

echo "[sbatch] job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "[sbatch] python: $(command -v "${PY:-python}")"
echo "[sbatch] git   : $(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"

exec bash scripts/hpc/submit_pretrain_sweep.sh "$@"
