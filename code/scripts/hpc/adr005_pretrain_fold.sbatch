#!/usr/bin/env bash
#SBATCH --job-name=adr005_pretrain
#SBATCH --partition=acc
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/adr005_pretrain_%j.out
#SBATCH --error=logs/adr005_pretrain_%j.err
#
# ADR-005 three-stream curriculum arm -- Stage-2 pre-training, ONE fold, 4 GPUs.
#
# The directive block above is the PRODUCTION default (acc, 24 h). Override it
# on the command line or through SBATCH_* without editing this file:
#   SBATCH_PARTITION=acc_short SBATCH_TIMELIMIT=00:10:00 EPOCHS=1 \
#       EXP_ROOT=$WORK_SCRATCH/probe GPU_TYPE=a100 GPUS=4 \
#       bash scripts/hpc/submit.sh scripts/hpc/adr005_pretrain_fold.sbatch
#
# PARTITION CHOICE. `acc_short` has a 30-MINUTE wall clock, so it can only host
# a probe or a handful of epochs; `acc` allows 24 h and `acc_long` 7 days. Use
# `acc_short` to measure steps/epoch and s/step (the LR-schedule line prints the
# exact step count within minutes), then size the real job from that number.
#
# GPUS is the DDP world size: utils/dist.py takes it from SLURM_NTASKS, and
# submit.sh sets --gres/--ntasks together. Pass GPUS=4 for this file.
#
# --mem-per-cpu is mandatory (the site's job_submit plugin rejects a job that
# omits it); 16 cpus/task covers num_workers=8 plus the main process.
#
# Resumability: auto_resume_model continues from <output_dir>/checkpoints, so a
# re-submission with the SAME --epochs and --output_dir picks up where the wall
# clock stopped. Never change --epochs between submissions: the LR cosine is
# rebuilt from it.
set -euo pipefail

mkdir -p logs
if [ ! -f scripts/env_hpc.sh ]; then
  echo "ERROR: submit from code/: scripts/env_hpc.sh not found in $PWD" >&2
  exit 1
fi
# shellcheck disable=SC1091
source scripts/env_hpc.sh
activate_project_env

CFG="${CFG:-configs/pretrain/stage2_budget_3stream_curriculum.yaml}"
FOLD="${FOLD:-0}"
EPOCHS="${EPOCHS:-150}"
WARMUP="${WARMUP:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SCRATCH="${WORK_SCRATCH:-/tmp}"
EXP_ROOT="${EXP_ROOT:-${SCRATCH}/pretrain/adr005_3stream_fold${FOLD}}"

CACHE_ARGS=()
if [ -n "${ROI_CACHE:-}" ] && [ -d "${ROI_CACHE}" ]; then
  CACHE_ARGS+=(--roi_cache "${ROI_CACHE}")
fi
if [ -n "${RGB2D_CACHE:-}" ] && [ -d "${RGB2D_CACHE}" ]; then
  CACHE_ARGS+=(--rgb2d_cache "${RGB2D_CACHE}")
fi
if [ -n "${RESP_CACHE:-}" ] && [ -d "${RESP_CACHE}" ]; then
  CACHE_ARGS+=(--resp_cache "${RESP_CACHE}"
               --min_signal_spread 0 --rail_touch_v 0)
fi

echo "[sbatch] job ${SLURM_JOB_ID:-local} on $(hostname) partition ${SLURM_JOB_PARTITION:-?}"
echo "[sbatch] python   : $(command -v "${PY:-python}")"
echo "[sbatch] config   : ${CFG}"
echo "[sbatch] fold     : ${FOLD}   epochs: ${EPOCHS}   lr-warmup: ${WARMUP}"
echo "[sbatch] exp_root : ${EXP_ROOT}"
echo "[sbatch] world    : ${SLURM_NTASKS:-1} task(s), batch ${BATCH_SIZE}/gpu, workers ${NUM_WORKERS}"
echo "[sbatch] caches   : ${CACHE_ARGS[*]:-<none>}"
echo "[sbatch] gpus     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | tr '\n' ' ')"

exec python -u runners/run_pretrain.py \
  -c "${CFG}" \
  --exp_root "${EXP_ROOT}" --fold "${FOLD}" \
  --epochs "${EPOCHS}" --warmup_epochs "${WARMUP}" \
  --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
  --output_dir "${EXP_ROOT}/out" \
  "${CACHE_ARGS[@]}"
