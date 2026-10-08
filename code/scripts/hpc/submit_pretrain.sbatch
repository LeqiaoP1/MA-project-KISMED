#!/usr/bin/env bash
# Lichtenberg HPC: Stage-2 multimodal masked pre-training (multi-GPU, single node).
#
# Generic Slurm wrapper around runners/run_pretrain.py. This job was MISSING
# from scripts/hpc/ (only submit_prepare / submit_waveform / submit_inspect*
# existed), so every Stage-2 run so far had to be launched by hand.
#
# PREREQUISITES (both are quick and cheap to get wrong):
#   1) Stage-1 weights on disk -- run ONCE on a LOGIN node (compute nodes have
#      NO internet):
#          bash scripts/hpc/download_weights.sh
#      First sanity-check the whole path with the cheap smoke job:
#          sbatch scripts/hpc/smoke_pretrain.sbatch
#   2) mkdir -p logs        (sbatch resolves --output relative to the submit dir)
#
# Submit from code/:
#   sbatch scripts/hpc/submit_pretrain.sbatch
#   CONFIG=configs/pretrain/stage2_multimodal.yaml sbatch scripts/hpc/submit_pretrain.sbatch
#   EPOCHS=2 sbatch scripts/hpc/submit_pretrain.sbatch          # short pilot
#
# Env overrides (sbatch --export defaults to ALL, so exported vars propagate):
#   CONFIG      configs/pretrain/stage2_hpc_tir_roi_resp.yaml
#   RUN_NAME    output subdir (default: <CONFIG basename>)
#   OUT_ROOT    output root (default: $WORK_SCRATCH, else $OUTPUT_DIR)
#   WEIGHTS_DIR Stage-1 cache (default: $PROJ_DIR/models/initial)
#   SUBJECTS / TASKS / TASK_SET   corpus filters (''=all)
#   EPOCHS / BATCH_SIZE / INPUT_SIZE / CLIP_STRIDE / NUM_WORKERS
#   Anything after the script name is forwarded to the runner verbatim.
#
# !! RESUME IS WIRED: runners/run_pretrain.py calls auto_resume_model right
#    after the optimizer/scaler are built, so a job killed by the wall clock no
#    longer loses its progress -- RESUBMIT with the same OUT_DIR and the same
#    EPOCHS and it continues from OUT_DIR/checkpoints/latest_checkpoint.txt.
#      * keep EPOCHS IDENTICAL across submits: the LR schedule is rebuilt from
#        --epochs, NOT restored, so changing it moves the schedule position;
#      * the run exits with "nothing to do" when the checkpoint epoch is already
#        >= EPOCHS -- raise EPOCHS to train longer, or use a fresh OUT_DIR;
#      * checkpoints land every CONFIG's save_ckpt_freq epochs.
#
# NOTE ON OUTPUT LOCATION: env_hpc.sh sets OUTPUT_DIR=$PROJ_DIR/output, which is
# on $HOME's small quota, while a Stage-2 checkpoint is ~1.2 GB and this run
# writes dozens of them. OUT_ROOT therefore defaults to $WORK_SCRATCH.
#SBATCH --job-name=st2_pretrain
#SBATCH --partition=acc               # GPU; 1 day max ('acc_long' = 7 days)
# NB: the LUA plugin rejects --mem, so memory is --mem-per-cpu * cpus-per-task.
# --account is omitted on purpose: the plugin bills your default project.
# ---- GPU TYPE IS PINNED ON PURPOSE ---------------------------------------- #
# `acc` is HETEROGENEOUS (a100 40, h100 8, rtx6000 64, v100 12, mi300x 16,
# pvc128g 12). This repo's torch is CUDA-only (2.14.0+cu130), so an UNTYPED
# `--gres=gpu:4` can be scheduled onto a V100, an Intel PVC or an AMD MI300X
# node and fail outright -- and those are often the only idle nodes. Hence the
# typed default below.
# Slurm does NOT expand shell variables inside #SBATCH lines, so switching type
# goes through its env override -- VERIFIED on this cluster to beat the
# directive below (SBATCH_GRES=gpu:a100:1 overrode '#SBATCH --gres=gpu:1') -- or
# the friendlier wrapper scripts/hpc/submit.sh:
#     SBATCH_GRES=gpu:rtx6000:4 sbatch scripts/hpc/submit_pretrain.sbatch
#     GPU_TYPE=rtx6000 bash scripts/hpc/submit.sh scripts/hpc/submit_pretrain.sbatch
#     sbatch --gres=gpu:h100:4 scripts/hpc/submit_pretrain.sbatch
#SBATCH --gres=gpu:a100:4             # EDIT: GPUs per node (4 or 8); type pinned
#SBATCH --ntasks=4                    # EDIT: == number of GPUs
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16             # MUST cover num_workers (8) + the main
                                      # process, else the decode workers queue
                                      # behind each other. The loop is CPU-WORK
                                      # bound (measured): 1 GPU does the step in
                                      # 0.20 s of a 2.45 s iteration. 4 x 16 = 64
                                      # of the node's 128 cores; keep
                                      # ntasks x cpus-per-task <= 128.
#SBATCH --mem-per-cpu=4G              # 4 tasks x 16 cpus x 4G = 256G total
#SBATCH --time=24:00:00
#SBATCH --output=logs/st2_pretrain_%j.out
#SBATCH --error=logs/st2_pretrain_%j.err

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

# env_hpc.sh resolves INITIAL_MODELS_DIR to "$(dirname "$PROJ_DIR")/models/initial"
# == $HOME/models/initial, i.e. OUTSIDE the repo. Pin it to the in-repo cache so
# the login-node download (download_weights.sh) and this job agree; WEIGHTS_DIR
# is the explicit escape hatch.
export INITIAL_MODELS_DIR="${WEIGHTS_DIR:-$PROJ_DIR/models/initial}"

CONFIG="${CONFIG:-configs/pretrain/stage2_hpc_tir_roi_resp.yaml}"
RUN_NAME="${RUN_NAME:-$(basename "${CONFIG%.yaml}")}"
OUT_ROOT="${OUT_ROOT:-${WORK_SCRATCH:-$OUTPUT_DIR}}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/pretrain/$RUN_NAME}"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG (run from $CODE_DIR)" >&2
    exit 1
fi
if [ -z "$(ls -A "$INITIAL_MODELS_DIR" 2>/dev/null)" ]; then
    echo "ERROR: no Stage-1 weights in $INITIAL_MODELS_DIR." >&2
    echo "       Compute nodes are offline; pre-fetch on a LOGIN node with:" >&2
    echo "           cd $CODE_DIR && bash scripts/hpc/download_weights.sh" >&2
    exit 1
fi

# ---- runner args (config -> defaults; explicit flags override) ------------ #
ARGS=(-c "$CONFIG" --raw_root "$RAW_DATA_PATH" --output_dir "$OUT_DIR")
[ -n "${SUBJECTS:-}" ]    && ARGS+=(--subjects "$SUBJECTS")
[ -n "${TASKS:-}" ]       && ARGS+=(--tasks "$TASKS")
[ -n "${TASK_SET:-}" ]    && ARGS+=(--task_set "$TASK_SET")
[ -n "${EPOCHS:-}" ]      && ARGS+=(--epochs "$EPOCHS")
[ -n "${BATCH_SIZE:-}" ]  && ARGS+=(--batch_size "$BATCH_SIZE")
[ -n "${INPUT_SIZE:-}" ]  && ARGS+=(--input_size "$INPUT_SIZE")
[ -n "${CLIP_STRIDE:-}" ] && ARGS+=(--clip_stride "$CLIP_STRIDE")
[ -n "${NUM_WORKERS:-}" ] && ARGS+=(--num_workers "$NUM_WORKERS")
# You normally do NOT need --resume: run_pretrain.py auto-resumes from
# OUT_DIR/checkpoints. An explicit path can still be appended via "$@" below.
ARGS+=("$@")

mkdir -p "$OUT_DIR" "$CODE_DIR/logs"

export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)}"
export MASTER_PORT="${MASTER_PORT:-29500}"

echo "[submit_pretrain] config   : $CONFIG"
echo "[submit_pretrain] raw_root : $RAW_DATA_PATH"
echo "[submit_pretrain] weights  : $INITIAL_MODELS_DIR"
echo "[submit_pretrain] output   : $OUT_DIR"
echo "[submit_pretrain] world    : ${SLURM_NTASKS} task(s) on $(hostname)"
echo "[submit_pretrain] gres     : $(scontrol show job -o "$SLURM_JOB_ID" 2>/dev/null | grep -oE 'gres/gpu[a-z0-9/]*=[0-9]+' | tr '\n' ' ')"
echo "[submit_pretrain] command  : python -u runners/run_pretrain.py ${ARGS[*]}"
echo

# one task per GPU; utils/dist.py reads SLURM_* env to init DDP
srun --ntasks="$SLURM_NTASKS" --ntasks-per-node="$SLURM_NTASKS_PER_NODE" \
    python -u runners/run_pretrain.py "${ARGS[@]}"

echo "Stage-2 pre-training finished -> $OUT_DIR"
