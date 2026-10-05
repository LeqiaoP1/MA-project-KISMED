#!/usr/bin/env bash
# Lichtenberg HPC: Stage-2 pre-training END-TO-END SMOKE TEST (1 GPU, <= 30 min).
#
# Runs ONE epoch over a handful of clips so it exercises the full path that a
# real run depends on: Stage-1 weight loading -> raw-tree discovery -> ROI crop
# + respiration resampling -> masking -> model build -> forward/backward -> AMP
# scaler -> checkpoint save. It is the cheap gate to run BEFORE the multi-hour
# full-corpus job.
#
# It writes to $WORK_SCRATCH (or /tmp) under a job-id directory, so it can NEVER
# overwrite a real run directory.
#
# Prerequisites:
#   bash scripts/hpc/download_weights.sh     # ON A LOGIN NODE (no internet on compute)
#   mkdir -p logs                            # in code/
#
# Submit from code/:
#   sbatch scripts/hpc/smoke_pretrain.sbatch
#   ENTRIES=8 sbatch scripts/hpc/smoke_pretrain.sbatch
#   CONFIG=configs/pretrain/stage2_multimodal.yaml sbatch scripts/hpc/smoke_pretrain.sbatch
#
# Env overrides: CONFIG, ENTRIES, SESSIONS, BATCH_SIZE, WEIGHTS_DIR
#SBATCH --job-name=st2_smoke
#SBATCH --partition=acc_short         # GPU; 30 min max ('acc' = 1 day)
# NB: the LUA plugin rejects --mem, so memory is --mem-per-cpu * cpus-per-task.
# --account is omitted on purpose: the plugin bills your default project.
# GPU type pinned: `acc_short` mixes a100/h100/rtx6000 with v100 / Intel PVC /
# AMD MI300X, and this repo's torch is CUDA-only. a100 has the most headroom for
# a 1-GPU job (a 1-GPU request is not stuck behind the whole-node requests).
# Override: SBATCH_GRES=gpu:rtx6000:1 sbatch scripts/hpc/smoke_pretrain.sbatch
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G              # 4 cpus x 4G = 16G total
#SBATCH --time=00:25:00
#SBATCH --output=logs/st2_smoke_%j.out
#SBATCH --error=logs/st2_smoke_%j.err

set -euo pipefail
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

# Keep the weight cache in lockstep with download_weights.sh / submit_pretrain.sbatch
# (env_hpc.sh points it outside the repo; see the note in download_weights.sh).
export INITIAL_MODELS_DIR="${WEIGHTS_DIR:-$PROJ_DIR/models/initial}"

CONFIG="${CONFIG:-configs/pretrain/stage2_hpc_tir_roi_resp.yaml}"
ENTRIES="${ENTRIES:-4}"
SESSIONS="${SESSIONS:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
OUT_DIR="${OUT_DIR:-${WORK_SCRATCH:-/tmp}/kismed_smoke/$(basename "${CONFIG%.yaml}")_${SLURM_JOB_ID:-manual}}"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG (run from $CODE_DIR)" >&2
    exit 1
fi
if [ -z "$(ls -A "$INITIAL_MODELS_DIR" 2>/dev/null)" ]; then
    echo "ERROR: no Stage-1 weights in $INITIAL_MODELS_DIR." >&2
    echo "       Run this ON A LOGIN NODE first:" >&2
    echo "           cd $CODE_DIR && bash scripts/hpc/download_weights.sh" >&2
    exit 1
fi

# --warmup_epochs 0 is REQUIRED: epochs < warmup trips the pre-existing
# cosine_scheduler assert in utils/lr_sched.py.
# --batch_size <= --max_entries, otherwise drop_last=True empties the loader and
# check_loader_not_empty() aborts before the first step.
ARGS=(-c "$CONFIG"
      --raw_root "$RAW_DATA_PATH"
      --output_dir "$OUT_DIR"
      --epochs 1 --warmup_epochs 0
      --save_ckpt_freq 1
      --max_sessions "$SESSIONS" --max_entries "$ENTRIES"
      --batch_size "$BATCH_SIZE"
      --num_workers 0)
ARGS+=("$@")

mkdir -p "$OUT_DIR" "$CODE_DIR/logs"

export MASTER_PORT="${MASTER_PORT:-29511}"

echo "[smoke_pretrain] config   : $CONFIG"
echo "[smoke_pretrain] raw_root : $RAW_DATA_PATH"
echo "[smoke_pretrain] weights  : $INITIAL_MODELS_DIR"
echo "[smoke_pretrain] output   : $OUT_DIR"
echo "[smoke_pretrain] command  : python -u runners/run_pretrain.py ${ARGS[*]}"
echo

python -u runners/run_pretrain.py "${ARGS[@]}"

echo
echo "[smoke_pretrain] OK -- checkpoint written to:"
ls -lh "$OUT_DIR/checkpoints" 2>/dev/null || true
