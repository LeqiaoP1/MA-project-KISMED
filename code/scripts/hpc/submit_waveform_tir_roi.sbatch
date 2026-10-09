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
#   sbatch scripts/hpc/submit_waveform_tir_roi.sbatch
#   FINETUNE=/work/scratch/$USER/pretrain/stage2_hpc_tir_roi_resp/checkpoints/checkpoint-0399.pth \
#       sbatch scripts/hpc/submit_waveform_tir_roi.sbatch
#   TASK_SET=low,moderate sbatch scripts/hpc/submit_waveform_tir_roi.sbatch
#   VAL_SUBJECT=F004 sbatch scripts/hpc/submit_waveform_tir_roi.sbatch   # LOSO fold
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
#SBATCH --partition=acc               # GPU; 1 day max ('acc_long' = 7 days)
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
# --time is 24 h, not 12 h: the config ships `epochs: 150`, and Stage 3 runs a
# VALIDATION pass every epoch (unlike Stage-2 pre-training), so ~150 epochs costs
# roughly 7-9 h. At a 12 h limit a slower-than-expected epoch would be killed
# mid-run, and a requeue costs another multi-day scheduling wait -- far more than
# the margin is worth. Stage 2 used 4.5 h of its 24 h for reference.
#SBATCH --time=24:00:00
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
OUT_ROOT="${OUT_ROOT:-${WORK_SCRATCH:-$OUTPUT_DIR}}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/finetune/$RUN_NAME}"
STAGE2_DIR="${STAGE2_DIR:-$OUT_ROOT/pretrain/stage2_hpc_tir_roi_resp}"

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
if [ -z "${FINETUNE:-}" ]; then
    FINETUNE="$(ls -1 "$STAGE2_DIR"/checkpoints/checkpoint-*.pth 2>/dev/null \
                | sort -V | tail -n 1 || true)"
    if [ -n "$FINETUNE" ]; then
        echo "[submit_waveform_tir_roi] FINETUNE not set -> newest Stage-2 ckpt:"
        echo "                          $FINETUNE"
    fi
fi
if [ -z "${FINETUNE:-}" ] || [ ! -f "$FINETUNE" ]; then
    echo "ERROR: no Stage-2 checkpoint resolved." >&2
    echo "       Looked in : ${STAGE2_DIR}/checkpoints/" >&2
    echo "       Got       : '${FINETUNE:-<unset>}'" >&2
    echo "       Run scripts/hpc/submit_pretrain.sbatch first, or pass" >&2
    echo "           FINETUNE=/path/to/checkpoint-NNNN.pth sbatch $0" >&2
    exit 1
fi

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

ARGS=(-c "$CONFIG"
      --data_path "$RAW_DATA_PATH"
      --finetune "$FINETUNE"
      --output_dir "$OUT_DIR")
[ -n "${VAL_SUBJECT:-}" ] && ARGS+=(--val_subject "$VAL_SUBJECT")
[ -n "${TASK_SET:-}" ]    && ARGS+=(--task_set "$TASK_SET")
[ -n "${TASKS:-}" ]       && ARGS+=(--tasks "$TASKS")
[ -n "${EPOCHS:-}" ]      && ARGS+=(--epochs "$EPOCHS")
[ -n "${BATCH_SIZE:-}" ]  && ARGS+=(--batch_size "$BATCH_SIZE")
[ -n "${NUM_WORKERS:-}" ] && ARGS+=(--num_workers "$NUM_WORKERS")
[ -n "$ROI_CACHE" ]      && ARGS+=(--roi_cache "$ROI_CACHE")
ARGS+=("$@")

mkdir -p "$OUT_DIR" "$CODE_DIR/logs"

# Port derived from the job id: a hardcoded 29501 collides whenever two jobs land
# on the same node (Lichtenberg packs 4-GPU jobs 2-per-node) and the loser dies
# in barrier() with "NCCL ... Connection refused" (this cost the 2026-10-09
# batch sweep 2 of its 6 arms before any training ran). The modulus keeps the
# range at 20000-29999, i.e. BELOW the Linux ephemeral range (32768-60999), so
# the kernel can never hand this port to an outbound connection.
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostname "$SLURM_JOB_NODELIST" | sed -n '1p')}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + (SLURM_JOB_ID * 17) % 10000))}"

echo "[submit_waveform_tir_roi] config   : $CONFIG"
echo "[submit_waveform_tir_roi] raw_root : $RAW_DATA_PATH"
echo "[submit_waveform_tir_roi] finetune : $FINETUNE"
echo "[submit_waveform_tir_roi] roi_cache: ${ROI_CACHE:-<disabled: decode .wmv>}"
echo "[submit_waveform_tir_roi] output   : $OUT_DIR"
echo "[submit_waveform_tir_roi] world    : ${SLURM_NTASKS} task(s) on $(hostname)"
echo "[submit_waveform_tir_roi] gres     : $(scontrol show job -o "$SLURM_JOB_ID" 2>/dev/null | grep -oE 'gres/gpu[a-z0-9/]*=[0-9]+' | tr '\n' ' ')"
echo "[submit_waveform_tir_roi] command  : python -u runners/run_waveform.py ${ARGS[*]}"
echo

# one task per GPU; utils/dist.py reads SLURM_* env to init DDP
srun --ntasks="$SLURM_NTASKS" --ntasks-per-node="$SLURM_NTASKS_PER_NODE" \
    python -u runners/run_waveform.py "${ARGS[@]}"

echo "Stage-3 (resp / tir_roi) finished -> $OUT_DIR"
