#!/usr/bin/env bash
# Pretrain-only ADR-001 sweep: Stage 2 for the four-stream (or any) arm.
#
# Per fold it runs exactly two steps, in order:
#   1. runners/run_partition.py  -> <EXP_ROOT>/fold<k>/partition.json
#   2. runners/run_pretrain.py   -> <EXP_ROOT>/fold<k>/stage2/
#
# It deliberately does NOT run run_waveform.py (Stage 3). Use submit_sweep.sh
# for the combined Stage-2 / Stage-3 protocol.
#
# Usage (from code/):
#   EXP_ROOT=$WORK_SCRATCH/fourstream_v1 \
#   ROI_CACHE=... RGB2D_CACHE=... RESP_CACHE=... BP_ARTEFACT=... \
#       bash scripts/hpc/submit_pretrain_sweep.sh
#
# Submission (from code/):
#   EXP_ROOT=... ROI_CACHE=... RGB2D_CACHE=... RESP_CACHE=... BP_ARTEFACT=... \
#       bash scripts/hpc/submit.sh scripts/hpc/submit_pretrain_sweep.sbatch
#
# A hardware probe uses EPOCHS=1 AND its own EXP_ROOT: the LR schedule is
# rebuilt from --epochs on every submit, so a probe followed by the real run in
# the SAME exp_root would silently move the schedule.
#
# Args after the .sbatch path (and any "$@" here) are forwarded to
# run_pretrain.py verbatim; a bare --epochs there has the same caveat.
#
# DRY_RUN=1 prints the exact commands without executing them.
set -euo pipefail
cd "$(dirname "$0")/../.."

: "${EXP_ROOT:?set EXP_ROOT (e.g. EXP_ROOT=\$WORK_SCRATCH/fourstream_v1)}"

FOLDS="${FOLDS:-0 1 2 3 4}"
PY="${PY:-python}"
# DDP WORLD SIZE. utils/dist.py takes world_size from SLURM_NTASKS, so the
# runner must be spawned with `srun --ntasks=$SLURM_NTASKS`: one plain `python`
# in a 4-task allocation would initialise a 4-rank process group from a single
# process and block in init_process_group() until the wall clock kills it.
NTASKS="${SLURM_NTASKS:-1}"
SRUN="${SRUN:-srun}"
RAW_ROOT="${RAW_ROOT:-${RAW_DATA_PATH:-}}"
CFG2="${CFG2:-configs/pretrain/stage2_fourstream_tir_rgb2d_bp_resp_small_random.yaml}"
DATA_SET2="${DATA_SET2:-tir_roi_resp_rgb2d_bp}"
SPLIT_SEED="${SPLIT_SEED:-42}"
SPLIT_STRATIFY="${SPLIT_STRATIFY:-sex}"

ROI_CACHE="${ROI_CACHE:-}"
RGB2D_CACHE="${RGB2D_CACHE:-}"
RESP_CACHE="${RESP_CACHE:-}"
BP_ARTEFACT="${BP_ARTEFACT:-}"
EPOCHS="${EPOCHS:-}"

if [ -z "$RAW_ROOT" ]; then
  echo "ERROR: RAW_ROOT/\$RAW_DATA_PATH is empty; the subject universe cannot be enumerated." >&2
  exit 2
fi

# One task per GPU. Refuse to guess when the allocation asks for >1 task but no
# launcher exists: a plain python there hangs in init_process_group() instead of
# failing, which costs the whole wall-clock budget.
if [ "$NTASKS" -gt 1 ] && ! command -v "$SRUN" >/dev/null 2>&1; then
  echo "ERROR: SLURM_NTASKS=$NTASKS but '$SRUN' is unavailable; cannot start $NTASKS ranks." >&2
  echo "       Resubmit with GPUS=1, or run inside a Slurm allocation." >&2
  exit 2
fi

# Rendezvous address. srun/--ntasks gives each rank the same MASTER_ADDR, but a
# batch step must resolve the first host of its own nodelist explicitly.
if [ -n "${SLURM_JOB_NODELIST:-}" ] && command -v scontrol >/dev/null 2>&1; then
  export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostname "$SLURM_JOB_NODELIST" | sed -n '1p')}"
fi

# The four-stream arm needs all four artefacts. Other arms can omit them.
if [ "$DATA_SET2" = "tir_roi_resp_rgb2d_bp" ]; then
  : "${ROI_CACHE:?set ROI_CACHE (TIR-ROI cache OUT_ROOT)}"
  : "${RGB2D_CACHE:?set RGB2D_CACHE (2D-texture cache OUT_ROOT)}"
  : "${RESP_CACHE:?set RESP_CACHE (1000 Hz respiration cache OUT_ROOT)}"
  : "${BP_ARTEFACT:?set BP_ARTEFACT (1000 Hz BP artefact OUT_ROOT)}"
fi

CACHE_ARGS=()
[ -n "$ROI_CACHE" ] && CACHE_ARGS+=(--roi_cache "$ROI_CACHE")
[ -n "$RGB2D_CACHE" ] && CACHE_ARGS+=(--rgb2d_cache "$RGB2D_CACHE")
[ -n "$RESP_CACHE" ] && CACHE_ARGS+=(--resp_cache "$RESP_CACHE")
[ -n "$BP_ARTEFACT" ] && CACHE_ARGS+=(--bp_artefact "$BP_ARTEFACT")
[ -n "$EPOCHS" ] && CACHE_ARGS+=(--epochs "$EPOCHS")

run() {
  printf '[pretrain-sweep]'
  printf ' %q' "$@"
  printf '\n'
  if [ "${DRY_RUN:-0}" = "1" ]; then
    return 0
  fi
  "$@"
}

echo "[pretrain-sweep] exp_root : $EXP_ROOT"
echo "[pretrain-sweep] folds    : $FOLDS (sequential)"
echo "[pretrain-sweep] config   : $CFG2"
echo "[pretrain-sweep] data_set : $DATA_SET2"
echo "[pretrain-sweep] raw_root : $RAW_ROOT"
echo "[pretrain-sweep] dry_run  : ${DRY_RUN:-0}"

FAILED=()
for k in $FOLDS; do
  dir="$EXP_ROOT/fold$k"
  mkdir -p "$dir/stage2"
  # Per-fold rendezvous port, derived from the JOB id: a hard-coded 29500 range
  # collides whenever two of these jobs land on the same node (this cluster
  # packs 4-GPU jobs 2-per-node), and the loser dies in barrier() with an NCCL
  # connection error. Below 20000-29999 keeps clear of the ephemeral range
  # (32768-60999).
  export MASTER_PORT="$(( 20000 + (${SLURM_JOB_ID:-0} * 17 + k) % 10000 ))"

  echo "================ fold $k ================"
  if ! run "$PY" -u runners/run_partition.py \
        --exp_root "$EXP_ROOT" --fold "$k" \
        --data_set "$DATA_SET2" --raw_root "$RAW_ROOT" \
        --split_seed "$SPLIT_SEED" --split_stratify "$SPLIT_STRATIFY"; then
    echo "ERROR: fold $k: partition record failed" >&2
    FAILED+=("$k")
    continue
  fi

  # Only the TRAINING step is multi-rank: the partition record must be written
  # once, not once per rank.
  LAUNCH=()
  if [ "$NTASKS" -gt 1 ]; then
    LAUNCH=("$SRUN" --ntasks="$NTASKS" \
            --ntasks-per-node="${SLURM_NTASKS_PER_NODE:-$NTASKS}")
  fi

  if ! run ${LAUNCH[@]+"${LAUNCH[@]}"} "$PY" -u runners/run_pretrain.py -c "$CFG2" \
        --exp_root "$EXP_ROOT" --fold "$k" \
        --data_set "$DATA_SET2" --raw_root "$RAW_ROOT" \
        --output_dir "$dir/stage2" \
        ${CACHE_ARGS[@]+"${CACHE_ARGS[@]}"} ${1+"$@"}; then
    echo "ERROR: fold $k: Stage 2 failed" >&2
    FAILED+=("$k")
    continue
  fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "[pretrain-sweep] these folds failed: ${FAILED[*]}" >&2
  exit 1
fi
echo "[pretrain-sweep] all requested folds finished"
echo "[pretrain-sweep] checkpoints: $EXP_ROOT/fold<k>/stage2/checkpoints/"
