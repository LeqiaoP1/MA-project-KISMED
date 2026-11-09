#!/usr/bin/env bash
# Lichtenberg HPC: Stage-2 AMMM (arm A: TIR-ROI + RESP) k-fold sweep --
# ONE JOB PER FOLD. DRY-RUN by default; pass --submit to actually queue.
#
# WHY A DRIVER
#   * submit_pretrain.sbatch already knows how to run a fold (it pins
#     `--n_folds/--fold/--pretrain_split fold`, resolves the ROI + respiration
#     caches and derives MASTER_PORT per fold). This script only PREFLIGHTS and
#     then submits ONE job per fold with an explicit per-fold OUT_DIR, so a
#     stale exported OUT_DIR can never point two folds at one directory.
#   * A per-fold job fits the 1-day `acc` cap (rtx6000 fold ~4.5 h at 150
#     epochs, a100 ~9 h), backfills sooner than the script's 2-day
#     `acc_long` default, and gets its own log + checkpoint pointer.
#   * The folds are independent by construction (subject-disjoint test blocks),
#     so they may run concurrently on different nodes.
#
# USAGE (from code/)
#   bash scripts/hpc/submit_ammm_pretrain_kfold.sh                    # dry run, 5 folds
#   bash scripts/hpc/submit_ammm_pretrain_kfold.sh --submit           # queue 5 folds
#   bash scripts/hpc/submit_ammm_pretrain_kfold.sh --pilot 20 --submit
#        -> fold 0 only, 20 epochs, run dir ..._ammm_..._pilot  (measure FIRST)
#   FOLDS="0 1" EPOCHS=150 bash scripts/hpc/submit_ammm_pretrain_kfold.sh --submit
#
# ENV KNOBS
#   CONFIG     configs/pretrain/stage2_ammm_hpc_tir_roi_resp.yaml
#   RUN_NAME   run dir basename (default: config basename)
#   N_FOLDS    default 5            FOLDS  default 0..N_FOLDS-1
#   EPOCHS     forwarded to the runner (default: whatever the config ships)
#   OUT_ROOT   default $WORK_SCRATCH (repo output/ is a symlink to it)
#   GPU_TYPE   rtx6000 (default) | a100 | h100   GPUS  default 4
#   PARTITION  acc (default)        TIME   06:00:00 (rtx6000); a100 needs 12 h
#   ROI_CACHE / RESP_CACHE   '' disables (do NOT: the caches are the label
#                            space for Stage 3 -- on when the twin was run)
#   RESUME=1   allow a run dir that already holds checkpoints (auto-resume)
#   PILOT      same as --pilot <EPOCHS>
#
# BUDGET EVIDENCE -- measured on the RATIO twin in this exact 96 px lineage
# (cache ON, global batch 64, 4 ranks; parse a run's log with
#  notes/eval/tir_resp/ammm_pilot_report.py):
#   4 x RTX PRO 6000: 255 steps/epoch, 65 s/epoch (0.255 s/it)
#                     => 150 epochs ~= 2.8 h/fold, 5 folds ~= 14 GPU-hours
#   4 x A100-40GB   : 242 steps/epoch, 184 s/epoch (0.760 s/it)
#                     => 150 epochs ~= 7.8 h/fold  ->  TIME=12:00:00
# AMMM is expected to be SLOWER per step than the ratio twin: its video cap is
# a KEPT cap (0.25 of 36 patches = up to 450 visible tokens vs 200 at mask
# 0.90), so the encoder sees up to ~2x more tokens and the visible count
# changes every step. That -- not the schedule -- is what the pilot measures.
set -uo pipefail

SUBMIT=0
PILOT="${PILOT:-}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --submit)  SUBMIT=1; shift ;;
        --pilot)   [ "$#" -ge 2 ] || { echo "ERROR: --pilot needs an epoch count" >&2; exit 2; }
                   PILOT="$2"; shift 2 ;;
        -h|--help) sed -n '2,50p' "$0"; exit 0 ;;
        *)         echo "ERROR: unknown argument '$1' (use --submit / --pilot <EPOCHS>)" >&2; exit 2 ;;
    esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$HERE/../.." && pwd)"
cd "$CODE_DIR"

# The profile exports template values (PARTITION=gpu, GPU_TYPE=...), so the
# caller's knobs are captured FIRST and restored after sourcing.
CALLER_GPU_TYPE="${GPU_TYPE-}"
CALLER_PARTITION="${PARTITION-}"
CALLER_TIME="${TIME-}"

set +u
# shellcheck disable=SC1091
source "$CODE_DIR/scripts/env_hpc.sh"
set -u

CONFIG="${CONFIG:-configs/pretrain/stage2_ammm_hpc_tir_roi_resp.yaml}"
OUT_ROOT="${OUT_ROOT:-${WORK_SCRATCH:-$OUTPUT_DIR}}"
N_FOLDS="${N_FOLDS:-5}"
EPOCHS="${EPOCHS:-}"
GPU_TYPE="${CALLER_GPU_TYPE:-rtx6000}"
GPUS="${GPUS:-4}"
PARTITION="${CALLER_PARTITION:-acc}"
TIME="${CALLER_TIME:-06:00:00}"
ROI_CACHE="${ROI_CACHE-${WORK_SCRATCH:-$OUTPUT_DIR}/tir_roi_cache}"
RESP_CACHE="${RESP_CACHE-${WORK_SCRATCH:-$OUTPUT_DIR}/resp_cache}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$CODE_DIR/../models/initial}"
JOB_SCRIPT="scripts/hpc/submit_pretrain.sbatch"

fail() { echo "ERROR: $*" >&2; exit 1; }

# --------------------------------------------------------------------------- #
# pilot mode: fold 0 only, a short epoch budget, its own run name
# --------------------------------------------------------------------------- #
SUFFIX=""
if [ -n "$PILOT" ]; then
    case "$PILOT" in (''|*[!0-9]*) fail "--pilot needs an integer epoch count" ;; esac
    [ "$PILOT" -gt 0 ] || fail "--pilot needs a positive epoch count"
    FOLDS="${FOLDS:-0}"
    EPOCHS="$PILOT"
    SUFFIX="_pilot"
fi
FOLDS="${FOLDS:-$(seq 0 $((N_FOLDS - 1)) | tr '\n' ' ')}"
FOLDS="$(echo $FOLDS)"
RUN_NAME="${RUN_NAME:-$(basename "${CONFIG%.yaml}")}${SUFFIX}"

# --------------------------------------------------------------------------- #
# PREFLIGHT (no GPU needed)
# --------------------------------------------------------------------------- #
[ -f "$CONFIG" ]     || fail "config not found: $CONFIG"
[ -f "$JOB_SCRIPT" ] || fail "job script not found: $JOB_SCRIPT"

# is it really the AMMM arm, and are the AMMM knobs declared? (a typo in a key
# is a hard error inside the runner, but only AFTER the job started)
grep -qE '^masking:[[:space:]]*ammm[[:space:]]*$' "$CONFIG" \
    || fail "$CONFIG: masking is not 'ammm' -- this launcher is for the AMMM arm"
for key in dirichlet_alpha video_kept_cap physio_kept_cap min_kept; do
    grep -qE "^${key}:" "$CONFIG" || fail "$CONFIG: missing AMMM knob '$key'"
done

# geometry / schedule digest, so the log says what was launched
# (strip the inline YAML comment -- every knob in these configs carries one)
cfg_get() {
    sed -nE "s/^$1:[[:space:]]*([^#]*).*/\1/p" "$CONFIG" \
        | head -n 1 | sed -E 's/[[:space:]]+$//; s/^["'\'']//; s/["'\'']$//'
}
echo "[ammm-kfold] config      : $CONFIG"
echo "[ammm-kfold] masking     : $(cfg_get masking)  alpha=$(cfg_get dirichlet_alpha)  video_cap=$(cfg_get video_kept_cap)"
echo "[ammm-kfold] geometry    : input_size=$(cfg_get input_size) roi_padding=$(cfg_get roi_padding) clip_stride=$(cfg_get clip_stride) tubelet=$(cfg_get tubelet)"
echo "[ammm-kfold] schedule    : epochs=${EPOCHS:-$(cfg_get epochs)} batch_size=$(cfg_get batch_size) blr=$(cfg_get blr) warmup=$(cfg_get warmup_epochs)"
echo "[ammm-kfold] run         : $RUN_NAME (folds: $FOLDS of $N_FOLDS)"
echo "[ammm-kfold] resources   : ${GPU_TYPE} x ${GPUS} on ${PARTITION}, ${TIME}"

[ -z "$(ls -A "$WEIGHTS_DIR" 2>/dev/null)" ] && fail \
    "no Stage-1 weights in $WEIGHTS_DIR -- pre-fetch on a LOGIN node:
       cd $CODE_DIR && bash scripts/hpc/download_weights.sh"
if [ -n "$ROI_CACHE" ] && [ ! -d "$ROI_CACHE" ]; then
    fail "ROI_CACHE=$ROI_CACHE missing. Build it:
       python runners/run_build_roi_cache.py --init --build
       (or ROI_CACHE='' to decode .wmv -- much slower and less reliable)"
fi
if [ -n "$RESP_CACHE" ] && [ ! -d "$RESP_CACHE" ]; then
    fail "RESP_CACHE=$RESP_CACHE missing. Build it:
       python runners/run_build_resp_cache.py --init --build
       (do NOT silently fall back: the cache IS the Stage-3 label space)"
fi
[ "$GPUS" -eq 4 ] || echo "[ammm-kfold] WARNING: GPUS=$GPUS changes the global batch (64 at 4x16), hence the LR and the comparability with the ratio twin." >&2

mkdir -p logs || fail "cannot create logs/"

# never reuse a run dir that already holds weights unless RESUME=1 was asked
for fold in $FOLDS; do
    out_dir="$OUT_ROOT/pretrain/${RUN_NAME}_fold${fold}"
    if [ -z "${RESUME:-}" ] && compgen -G "$out_dir/checkpoints/*.pth" > /dev/null; then
        fail "$out_dir already holds checkpoints. Pick another RUN_NAME, or
       RESUME=1 to continue it (auto-resume; keep EPOCHS identical)."
    fi
done

# --------------------------------------------------------------------------- #
# SUBMIT one job per fold
# --------------------------------------------------------------------------- #
submitted=""
for fold in $FOLDS; do
    out_dir="$OUT_ROOT/pretrain/${RUN_NAME}_fold${fold}"
    args=(RUN_NAME="$RUN_NAME" CONFIG="$CONFIG" FOLDS="$fold" N_FOLDS="$N_FOLDS"
          OUT_DIR="$out_dir" SBATCH_PARTITION="$PARTITION"
          SBATCH_TIMELIMIT="$TIME" SBATCH_JOB_NAME="st2_ammm_f${fold}${SUFFIX}")
    [ -n "$EPOCHS" ] && args+=(EPOCHS="$EPOCHS")

    if [ "$SUBMIT" -eq 0 ]; then
        echo "[dry-run] fold $fold -> $out_dir"
        echo "          env ${args[*]} GPU_TYPE=$GPU_TYPE GPUS=$GPUS \\"
        echo "              bash scripts/hpc/submit.sh $JOB_SCRIPT"
        continue
    fi
    out="$(env "${args[@]}" GPU_TYPE="$GPU_TYPE" GPUS="$GPUS" \
             bash scripts/hpc/submit.sh "$JOB_SCRIPT" 2>&1)"
    job="$(printf '%s\n' "$out" | sed -nE 's/.*Submitted batch job ([0-9]+).*/\1/p' | tail -n 1)"
    if [ -z "$job" ]; then
        printf '%s\n' "$out" >&2
        fail "fold $fold: submission produced no job id"
    fi
    echo "[ammm-kfold] fold $fold queued as job $job -> $out_dir"
    submitted="$submitted $fold:$job"
done

if [ "$SUBMIT" -eq 0 ]; then
    echo
    echo "[ammm-kfold] DRY RUN. Re-run with --submit."
    echo "[ammm-kfold] FIRST TIME? run the pilot instead of the full sweep:"
    echo "             bash scripts/hpc/submit_ammm_pretrain_kfold.sh --pilot 20 --submit"
    echo "             then read the epoch time / loss curve with:"
    echo "             python notes/eval/tir_resp/ammm_pilot_report.py \\"
    echo "                 $CODE_DIR/logs/st2_pretrain_<jobid>.out"
else
    echo "[ammm-kfold] queued:$submitted"
    echo "[ammm-kfold] logs: $CODE_DIR/logs/st2_pretrain_<jobid>.{out,err}"
fi
