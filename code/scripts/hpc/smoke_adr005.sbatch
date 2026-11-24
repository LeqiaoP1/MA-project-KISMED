#!/usr/bin/env bash
#SBATCH --job-name=adr005_smoke
#SBATCH --partition=acc_short
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:25:00
#SBATCH --output=logs/adr005_smoke_%j.out
#SBATCH --error=logs/adr005_smoke_%j.err
#
# ADR-005 §3.3 smoke run (Criterion 3): one epoch, warm-up disabled, corpus
# capped, single device, development partition.
#
# Why this file exists: the smoke is the cheapest way to prove the decision was
# executed -- constant 350-token input, the geometry of §2.2, every §2.8 record
# -- and it has to be runnable from a compute node, not from a login shell.
#
# Usage (from code/):
#   GPU_TYPE=a100 GPUS=1 bash scripts/hpc/submit.sh scripts/hpc/smoke_adr005.sbatch
#   GPU_TYPE=rtx6000 GPUS=1 bash scripts/hpc/submit.sh scripts/hpc/smoke_adr005.sbatch
#   MAX_ENTRIES=64 GPU_TYPE=a100 GPUS=1 \
#       bash scripts/hpc/submit.sh scripts/hpc/smoke_adr005.sbatch
#
# GPU_TYPE: acc_short has a100 / rtx6000 / h100. v100, mi300x and pvc128g are
# NOT usable (CUDA 13 dropped sm_70; the other two are not NVIDIA). submit.sh
# overrides the --gres directive above, so the wrapper's GPU_TYPE wins.
# The comparator arm is the same job with CFG=...rgb2d.yaml.
#
# --mem-per-cpu is mandatory: the site's job_submit plugin rejects a job that
# omits it. 16 cpus/task leaves room for num_workers=8 plus the main process.
#
# Caps that matter: --max_clips is PER SESSION, so it does NOT bound a run.
# --max_entries is the per-SPLIT corpus cap and is what makes this a smoke
# (without it an epoch is the whole corpus: 1650 steps at batch 8).
#
# The cache flags are only passed when the directories exist; with
# --resp_cache attached the legacy window filters must be OFF
# (--min_signal_spread 0 --rail_touch_v 0), because the cached per-sample mask
# is the only validity rule (data/tir_resp_dataset.py).
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
MAX_ENTRIES="${MAX_ENTRIES:-16}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SCRATCH="${WORK_SCRATCH:-/tmp}"
EXP_ROOT="${EXP_ROOT:-${SCRATCH}/adr005_smoke_${SLURM_JOB_ID:-0}}"

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

echo "[sbatch] job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "[sbatch] python   : $(command -v "${PY:-python}")"
echo "[sbatch] config   : ${CFG}"
echo "[sbatch] exp_root : ${EXP_ROOT}"
echo "[sbatch] caches   : ${CACHE_ARGS[*]:-<none>}"
echo "[sbatch] caps     : max_entries=${MAX_ENTRIES} batch=${BATCH_SIZE} workers=${NUM_WORKERS}"
echo "[sbatch] gpu      : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

exec python -u runners/run_pretrain.py \
  -c "${CFG}" \
  --exp_root "${EXP_ROOT}" --fold 0 --dev_smoke \
  --epochs 1 --warmup_epochs 0 --virtual_sensing_warmup_epochs 0 \
  --max_entries "${MAX_ENTRIES}" \
  --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
  --output_dir "${EXP_ROOT}/out" \
  "${CACHE_ARGS[@]}"
