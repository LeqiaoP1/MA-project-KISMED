#!/usr/bin/env bash
# Lichtenberg HPC: Stage-3 waveform fine-tuning -- TIR-ROI -> RESPIRATION.
#
# Dedicated to the RAW-tree thermal-ROI branch. scripts/hpc/submit_waveform.sbatch
# is the CANONICAL-branch job (bp / resp / eda): it hardcodes
# `configs/finetune/$TARGET.yaml` and `--data_path "$DATA_PATH"` (= the canonical
# tree), so it cannot drive a `data_set: tir_roi_resp` config, which needs
# `--data_path "$RAW_DATA_PATH"`.
#
# PREREQUISITES:
#   1) a finished Stage-2 run -- scripts/hpc/submit_pretrain.sbatch
#   2) mkdir -p logs        (sbatch resolves --output relative to the submit dir)
#
# Submit from code/:
#   sbatch scripts/hpc/submit_waveform_tir_roi.sbatch            # folds 0..4, in order
#   FOLDS="0 1" sbatch scripts/hpc/submit_waveform_tir_roi.sbatch  # a subset, still sequential
#   FOLDS=3     sbatch scripts/hpc/submit_waveform_tir_roi.sbatch  # ONE fold
#   TASK_SET=low,moderate sbatch scripts/hpc/submit_waveform_tir_roi.sbatch
#   VAL_SUBJECT=F004 sbatch scripts/hpc/submit_waveform_tir_roi.sbatch   # legacy LOSO fold
#
# Folds are SEQUENTIAL: one Slurm job runs them one after another, each using all
# 4 GPUs (DDP is intra-fold). Do NOT use --array -- it would run the whole sweep
# once per array task, in parallel.
#
# Env overrides:
#   CONFIG      configs/finetune/resp_tir_roi_hpc.yaml
#   FINETUNE    Stage-2 checkpoint. Unset -> newest checkpoint-*.pth under the
#               matching Stage-2 run dir (logged so it is never a silent pick).
#   STAGE2_DIR  Stage-2 run dir to auto-resolve FINETUNE from
#               (default: $OUT_ROOT/pretrain/stage2_hpc_tir_roi_resp)
#   ROI_CACHE   offline TIR-ROI cache OUT_ROOT (default: $WORK_SCRATCH/tir_roi_cache).
#               The cache is SHARED with Stage 2 (same key: it is the PIXELS that
#               are cached, so the 2 s -> 1 s clip_stride change needs no rebuild).
#               ROI_CACHE='' disables it. See the note above ARGS below.
#   RUN_NAME / OUT_ROOT   output dir controls (default: $WORK_SCRATCH)
#   VAL_SUBJECT / TASK_SET / EPOCHS / BATCH_SIZE
#   Anything after the script name is forwarded to the runner verbatim.
#SBATCH --job-name=st3_resp_tir
# NB: the partition is set further down (acc_long) -- a full fold sweep does not
# fit the 1-day `acc` cap; see the --time note.
# NB: the LUA plugin rejects --mem, so memory is --mem-per-cpu * cpus-per-task.
# --account is omitted on purpose: the plugin bills your default project.
# ---- GPU TYPE IS PINNED ON PURPOSE ---------------------------------------- #
# Same reasoning as submit_pretrain.sbatch: `acc` mixes a100/h100/rtx6000 with
# v100 / Intel PVC / AMD MI300X, and this repo's torch is CUDA-only, so an
# untyped `--gres=gpu:4` can land on a node that cannot run the job.
# Override (sbatch does not expand vars in #SBATCH lines):
#     SBATCH_GRES=gpu:rtx6000:4 sbatch scripts/hpc/submit_waveform_tir_roi.sbatch
#     GPU_TYPE=rtx6000 bash scripts/hpc/submit.sh scripts/hpc/submit_waveform_tir_roi.sbatch
#SBATCH --gres=gpu:a100:4             # EDIT: GPUs per node (4 or 8); type pinned
#SBATCH --ntasks=4                    # EDIT: == number of GPUs
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G              # 4 tasks x 8 cpus x 4G = 128G total
# --time: a REAL fold run is ~7-9 h (`epochs: 150` plus a validation pass every
# epoch), and the default FOLDS is the whole 0..4 sweep -> up to ~45 h. `acc`
# caps at 1 day, so the sweep default needs `acc_long`. A single fold
# (FOLDS=3) can be submitted to the short queue with
#     sbatch --partition=acc --time=24:00:00 ...
# A killed sweep is recoverable: every fold has its own checkpoint pointer, so
# re-submitting resumes the fold it reached and skips the finished ones.
#SBATCH --partition=acc_long          # 7-day max; see the --time note above
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/st3_resp_tir_%j.out
#SBATCH --error=logs/st3_resp_tir_%j.err

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

# Keep the weight cache in lockstep with download_weights.sh / submit_pretrain.sbatch
# (env_hpc.sh points it outside the repo; see the note in download_weights.sh).
export INITIAL_MODELS_DIR="${WEIGHTS_DIR:-$PROJ_DIR/models/initial}"

CONFIG="${CONFIG:-configs/finetune/resp_tir_roi_hpc.yaml}"
RUN_NAME="${RUN_NAME:-$(basename "${CONFIG%.yaml}")}"

# ---- fold sweep: SEQUENTIAL by design ------------------------------------- #
# All folds run one after another INSIDE THIS ONE JOB, each fold using all
# GPUs of the allocation (DDP is intra-fold only). Nothing runs in parallel
# across folds, so the folds never compete for GPU memory, the ROI cache, or
# the shared Stage-1 weight cache.
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
if [ -z "${RAW_DATA_PATH:-}" ] || [ ! -d "$RAW_DATA_PATH" ]; then
    echo "ERROR: RAW_DATA_PATH is unset or not a directory ($RAW_DATA_PATH)." >&2
    echo "       The thermal-ROI branch needs the RAW tree, not the canonical one." >&2
    echo "       Check RAW_DATA_PATH in scripts/env_hpc.sh." >&2
    exit 1
fi

# ---- resolve the Stage-2 checkpoint --------------------------------------- #
# Per fold, INSIDE the loop: the Stage-2 tree a fold was pretrained from MUST be
# that fold's own run. A shared "newest checkpoint" directory would silently
# pair a fold with a Stage-2 model that already saw its test subjects -- exactly
# the leak the k-fold protocol exists to prevent. An explicit STAGE2_DIR or
# FINETUNE override still applies to every fold (the run-time fold contract
# check refuses a mismatched pair).

# ---- runner args (config -> defaults; explicit flags override) ------------- #
# --data_path MUST be the RAW tree for data_set: tir_roi_resp (the canonical
# tree has no Thermal/ or IRFeatures/ next to the session dirs).
#
# ROI CACHE: Stage 3 reads the SAME raw-tree thermal ROI as Stage 2, so it uses
# the SAME offline cache. The cache stores PIXELS, keyed on roi_landmarks /
# roi_padding / input_size / the corpus -- NEVER on clip_stride, so changing the
# window hop (1 s <-> 2 s) never needs a cache rebuild.
# This is not only speed: the pre-cache path decodes .wmv from /work/projects
# inside the DataLoader, and job 55368106 died at epoch 7 when one such read
# stalled 266 s and OpenCV raised 'could not open .../F040/T9.wmv'.
# ROI_CACHE='' disables the cache (decode as before).
ROI_CACHE="${ROI_CACHE-${WORK_SCRATCH:-$OUTPUT_DIR}/tir_roi_cache}"
if [ -n "$ROI_CACHE" ] && [ ! -d "$ROI_CACHE" ]; then
    echo "ERROR: ROI_CACHE=$ROI_CACHE does not exist." >&2
    echo "       Build it with:  python runners/run_build_roi_cache.py --init --build" >&2
    echo "       Or set ROI_CACHE='' to decode .wmv as before." >&2
    exit 1
fi

mkdir -p "$CODE_DIR/logs"

# Port derived from the job id: a hardcoded 29501 collides whenever two jobs land
# on the same node (Lichtenberg packs 4-GPU jobs 2-per-node) and the loser dies
# in barrier() with "NCCL ... Connection refused" (this cost the 2026-10-09
# batch sweep 2 of its 6 arms before any training ran). The modulus keeps the
# range at 20000-29999, i.e. BELOW the Linux ephemeral range (32768-60999), so
# the kernel can never hand this port to an outbound connection.
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostname "$SLURM_JOB_NODELIST" | sed -n '1p')}"

run_fold() {
    local fold="$1"
    shift          # the rest are runner args forwarded from the submit command
    local run_name="${RUN_NAME}_fold${fold}"
    local out_dir="${OUT_DIR:-$OUT_ROOT/finetune/$run_name}"
    local stage2_dir="${STAGE2_DIR:-$OUT_ROOT/pretrain/stage2_hpc_tir_roi_resp_fold${fold}}"
    local finetune="${FINETUNE:-}"
    local args=()

    # Per-fold port offset: the folds share this allocation, and re-binding the
    # same port a few seconds after the previous fold's process exits can hit a
    # lingering TIME_WAIT socket.
    export MASTER_PORT="$(( 20000 + (SLURM_JOB_ID * 17 + fold) % 10000 ))"

    if [ -z "$finetune" ]; then
        finetune="$(ls -1 "$stage2_dir"/checkpoints/checkpoint-*.pth 2>/dev/null \
                    | sort -V | tail -n 1 || true)"
        if [ -n "$finetune" ]; then
            echo "[fold $fold] FINETUNE not set -> newest Stage-2 ckpt: $finetune"
        fi
    fi
    if [ -z "$finetune" ] || [ ! -f "$finetune" ]; then
        echo "ERROR: fold $fold: no Stage-2 checkpoint resolved." >&2
        echo "       Looked in : ${stage2_dir}/checkpoints/" >&2
        echo "       Got       : '${finetune:-<unset>}'" >&2
        echo "       Run scripts/hpc/submit_pretrain.sbatch for this fold first," >&2
        echo "       or pass FINETUNE=/path/to/checkpoint-NNNN.pth." >&2
        return 1
    fi

    args=(-c "$CONFIG"
          --data_path "$RAW_DATA_PATH"
          --finetune "$finetune"
          --output_dir "$out_dir"
          --n_folds "$N_FOLDS"
          --fold "$fold")
    [ -n "${VAL_SUBJECT:-}" ] && args+=(--val_subject "$VAL_SUBJECT")
    [ -n "${TASK_SET:-}" ]    && args+=(--task_set "$TASK_SET")
    [ -n "${TASKS:-}" ]       && args+=(--tasks "$TASKS")
    [ -n "${EPOCHS:-}" ]      && args+=(--epochs "$EPOCHS")
    [ -n "${BATCH_SIZE:-}" ]  && args+=(--batch_size "$BATCH_SIZE")
    [ -n "${NUM_WORKERS:-}" ] && args+=(--num_workers "$NUM_WORKERS")
    [ -n "$ROI_CACHE" ]       && args+=(--roi_cache "$ROI_CACHE")
    args+=("$@")

    echo
    echo "=================================================================="
    echo "[fold $fold] config   : $CONFIG"
    echo "[fold $fold] raw_root : $RAW_DATA_PATH"
    echo "[fold $fold] finetune : $finetune"
    echo "[fold $fold] roi_cache: ${ROI_CACHE:-<disabled: decode .wmv>}"
    echo "[fold $fold] output   : $out_dir"
    echo "[fold $fold] world    : ${SLURM_NTASKS} task(s) on $(hostname)"
    echo "[fold $fold] command  : python -u runners/run_waveform.py ${args[*]}"
    echo "=================================================================="

    mkdir -p "$out_dir"
    # one task per GPU; utils/dist.py reads SLURM_* env to init DDP
    srun --ntasks="$SLURM_NTASKS" --ntasks-per-node="$SLURM_NTASKS_PER_NODE" \
        python -u runners/run_waveform.py "${args[@]}" || return 1
    echo "[fold $fold] finished -> $out_dir"
}

FAILED=""
for fold in $FOLDS; do
    echo "[submit_waveform_tir_roi] SEQUENTIAL sweep: fold $fold (of $N_FOLDS)"
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

