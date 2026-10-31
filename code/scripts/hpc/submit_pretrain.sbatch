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
#   ROI_CACHE   offline TIR-ROI cache OUT_ROOT (default: $WORK_SCRATCH/tir_roi_cache).
#               The cache is NOT just a speedup: the pre-cache data path decodes
#               .wmv from /work/projects inside the DataLoader, and job 55368106
#               died at epoch 7 when one of those reads stalled 266 s and OpenCV
#               returned 'could not open .../F040/T9.wmv' (OSError in a worker).
#               The cache moves every pixel read to /work/scratch. ROI_CACHE=''
#               disables it (decode as before).
#   RESP_CACHE  offline RESPIRATION-grid cache OUT_ROOT (default:
#               $WORK_SCRATCH/resp_cache). Built once by
#               `runners/run_build_resp_cache.py --init --build` (seconds).
#               RESP_CACHE='' falls back to reading Resp_Volts.txt per epoch.
#               When active it also forces --min_signal_spread 0 and
#               --rail_touch_v 0: the cached per-sample mask is the ONLY
#               validity rule (the configs' legacy thresholds would raise).
#   FRESH=1      start from scratch: move OUT_DIR/checkpoints aside to
#               checkpoints.precache_<timestamp> instead of auto-resuming.
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
# NOTE ON OUTPUT LOCATION: a Stage-2 checkpoint is ~1.2 GB and this run writes
# dozens of them, so the output must not sit on $HOME's small quota. env_hpc.sh
# defines OUTPUT_DIR as $WORK_SCRATCH (mirroring OUT_ROOT below), and
# $PROJ_DIR/output is itself a symlink to the scratch, so both branches land on
# the scratch file system. OUT_ROOT still defaults to $WORK_SCRATCH explicitly.
#SBATCH --job-name=st2_pretrain
# NB: the partition is set further down (acc_long) -- a full fold sweep does not
# fit the 1-day `acc` cap; see the --time note.
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
# --time: one fold is ~4.5 h (measured), and the default FOLDS is the whole
# 0..4 sweep -> ~23 h, over the `acc` 1-day cap once ROI decode is included.
# A single fold (FOLDS=3) can use the short queue:
#     sbatch --partition=acc --time=24:00:00 ...
# A killed sweep is recoverable: each fold has its own checkpoint pointer.
#SBATCH --partition=acc_long          # 7-day max; see the --time note above
#SBATCH --time=2-00:00:00
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

# ---- fold sweep: SEQUENTIAL by design ------------------------------------- #
# All folds run one after another INSIDE THIS ONE JOB, each fold using all GPUs
# of the allocation (DDP is intra-fold only). No two folds ever run at the same
# time, so GPU memory, the ROI cache and the shared Stage-1 weight cache are
# never contended.
#
# Each fold MUST own its run dir: checkpoints are checkpoint-NNNN.pth and
# auto_resume_model() follows OUT_DIR/checkpoints/latest_checkpoint.txt, so two
# folds sharing ONE dir would resume each other. The test fold is excluded from
# the corpus by --pretrain_split fold.
#
#   FOLDS="0 1 2 3 4"  sbatch ...      # default: 0..N_FOLDS-1
#   FOLDS="3"          sbatch ...      # a single fold
#
# Do NOT submit this with --array: each array task would run the whole sweep.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: this script is a SEQUENTIAL fold sweep; drop --array." >&2
    echo "       Use FOLDS=\"0 1 2 3 4\" (the default) to run every fold in" >&2
    echo "       order inside one job, or FOLDS=\"<k>\" for a single fold." >&2
    exit 2
fi

N_FOLDS="${N_FOLDS:-5}"
# space-separated, so it prints on one line and splits cleanly in `for fold in`
FOLDS="${FOLDS:-$(seq 0 $((N_FOLDS - 1)) | tr '\n' ' ')}"
FOLDS="$(echo $FOLDS)"
OUT_ROOT="${OUT_ROOT:-${WORK_SCRATCH:-$OUTPUT_DIR}}"

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

# ---- ROI cache (see the header note: this is a RELIABILITY fix, not a nicety) --
ROI_CACHE="${ROI_CACHE-${WORK_SCRATCH:-$OUTPUT_DIR}/tir_roi_cache}"
if [ -n "$ROI_CACHE" ] && [ ! -d "$ROI_CACHE" ]; then
    echo "ERROR: ROI_CACHE=$ROI_CACHE does not exist." >&2
    echo "       Build it with:  python runners/run_build_roi_cache.py --init --build" >&2
    echo "       Or set ROI_CACHE='' to decode .wmv as before." >&2
    exit 1
fi

# ---- respiration-grid cache (optional; empty disables) -------------------- #
RESP_CACHE="${RESP_CACHE-${WORK_SCRATCH:-$OUTPUT_DIR}/resp_cache}"
if [ -n "$RESP_CACHE" ] && [ ! -d "$RESP_CACHE" ]; then
    echo "ERROR: RESP_CACHE=$RESP_CACHE does not exist." >&2
    echo "       Build it with:  python runners/run_build_resp_cache.py --init --build" >&2
    echo "       Or set RESP_CACHE='' to read Resp_Volts.txt as before." >&2
    exit 1
fi

# ---- runner args: assembled PER FOLD inside run_fold --------------------- #
# You normally do NOT need --resume: run_pretrain.py auto-resumes from
# OUT_DIR/checkpoints. An explicit path can still be appended via "$@" below.
mkdir -p "$CODE_DIR/logs"

# ---- FRESH=1: force a clean start (applies to every fold) ----------------- #
# auto_resume_model() keys off OUT_DIR/checkpoints/latest_checkpoint.txt. Moving
# the whole directory aside (rather than deleting the pointer) makes the start
# unambiguous AND keeps the old weights, which a bare `rm latest_checkpoint.txt`
# would silently overwrite at the first save_ckpt_freq epoch.
fresh_dir() {
    local out_dir="$1"
    if [ -n "${FRESH:-}" ] && [ -d "$out_dir/checkpoints" ]; then
        local archive="$out_dir/checkpoints.precache_$(date +%Y%m%d_%H%M%S)"
        mv "$out_dir/checkpoints" "$archive"
        echo "[submit_pretrain] FRESH=1: archived old checkpoints -> $archive"
    fi
}

export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostname "$SLURM_JOB_NODELIST" | sed -n '1p')}"
# Derive the port from the job id -- a hardcoded 29500 collides whenever two jobs
# land on the same node (Lichtenberg packs 4-GPU jobs 2-per-node), and the loser
# dies in barrier() with "NCCL ... Connection refused" (this cost the 2026-10-09
# batch sweep 2 of its 6 arms before any training ran). The modulus keeps the
# range at 20000-29999, i.e. BELOW the Linux ephemeral range (32768-60999), so
# the kernel can never hand this port to an outbound connection.
run_fold() {
    local fold="$1"
    shift          # the rest are runner args forwarded from the submit command
    local out_dir="${OUT_DIR:-$OUT_ROOT/pretrain/${RUN_NAME}_fold${fold}}"
    local args=()

    # Per-fold port offset: folds share this allocation, and re-binding the same
    # port right after the previous fold exits can hit a lingering TIME_WAIT
    # socket.
    export MASTER_PORT="$(( 20000 + (SLURM_JOB_ID * 17 + fold) % 10000 ))"

    args=(-c "$CONFIG" --raw_root "$RAW_DATA_PATH" --output_dir "$out_dir"
          --n_folds "$N_FOLDS" --fold "$fold" --pretrain_split fold)
    [ -n "${SUBJECTS:-}" ]    && args+=(--subjects "$SUBJECTS")
    [ -n "${TASKS:-}" ]       && args+=(--tasks "$TASKS")
    [ -n "${TASK_SET:-}" ]    && args+=(--task_set "$TASK_SET")
    [ -n "${EPOCHS:-}" ]      && args+=(--epochs "$EPOCHS")
    [ -n "${BATCH_SIZE:-}" ]  && args+=(--batch_size "$BATCH_SIZE")
    [ -n "${INPUT_SIZE:-}" ]  && args+=(--input_size "$INPUT_SIZE")
    [ -n "${CLIP_STRIDE:-}" ] && args+=(--clip_stride "$CLIP_STRIDE")
    [ -n "${NUM_WORKERS:-}" ] && args+=(--num_workers "$NUM_WORKERS")
    [ -n "$ROI_CACHE" ]       && args+=(--roi_cache "$ROI_CACHE")
    if [ -n "$RESP_CACHE" ]; then
        args+=(--resp_cache "$RESP_CACHE" --min_signal_spread 0 \
               --rail_touch_v 0)
    fi
    args+=("$@")

    fresh_dir "$out_dir"
    mkdir -p "$out_dir"

    echo
    echo "=================================================================="
    echo "[fold $fold] config   : $CONFIG"
    echo "[fold $fold] raw_root : $RAW_DATA_PATH"
    echo "[fold $fold] weights  : $INITIAL_MODELS_DIR"
    echo "[fold $fold] output   : $out_dir"
    echo "[fold $fold] roi_cache: ${ROI_CACHE:-<disabled: decode .wmv>}"
    echo "[fold $fold] resp_cache: ${RESP_CACHE:-<disabled: read Resp_Volts.txt>}"
    echo "[fold $fold] world    : ${SLURM_NTASKS} task(s) on $(hostname)"
    echo "[fold $fold] command  : python -u runners/run_pretrain.py ${args[*]}"
    echo "=================================================================="

    # one task per GPU; utils/dist.py reads SLURM_* env to init DDP
    srun --ntasks="$SLURM_NTASKS" --ntasks-per-node="$SLURM_NTASKS_PER_NODE" \
        python -u runners/run_pretrain.py "${args[@]}" || return 1
    echo "[fold $fold] finished -> $out_dir"
}

FAILED=""
for fold in $FOLDS; do
    echo "[submit_pretrain] SEQUENTIAL sweep: fold $fold (of $N_FOLDS)"
    if ! run_fold "$fold" "$@"; then
        echo "[fold $fold] FAILED -- continuing with the remaining folds" >&2
        FAILED="$FAILED $fold"
    fi
done

if [ -n "$FAILED" ]; then
    echo "ERROR: these folds failed:$FAILED -- see the per-fold logs above." >&2
    exit 1
fi
echo "All folds finished: $FOLDS"

# NOTE: OUT_DIR is only set when the CALLER pinned one directory; the normal
# per-fold layout is $OUT_ROOT/pretrain/${RUN_NAME}_fold<k> (see run_fold()).
# Referring to $OUT_DIR bare under `set -u` aborted this script with
# "line 255: OUT_DIR: unbound variable" AFTER every fold had trained, so jobs
# 55476920/55476921 were reported as FAILED 1:0 although their folds completed.
if [ -n "${OUT_DIR:-}" ]; then
    echo "Stage-2 pre-training finished -> $OUT_DIR"
else
    echo "Stage-2 pre-training finished -> $OUT_ROOT/pretrain/${RUN_NAME}_fold<k> (fold(s): $FOLDS)"
fi
