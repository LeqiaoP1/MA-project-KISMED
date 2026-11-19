#!/usr/bin/env bash
# ADR-001 five-fold sweep — one allocation, folds run SEQUENTIALLY.
#
#   EXP_ROOT=$WORK_SCRATCH/protocol_v1 bash scripts/hpc/submit_sweep.sh
#   FOLDS="0 1" EXP_ROOT=... bash scripts/hpc/submit_sweep.sh     # subset
#
# Per fold k it runs exactly the protocol's four steps, in order:
#
#   1. runners/run_partition.py     -> <EXP_ROOT>/fold<k>/partition.json  (once)
#   2. runners/run_pretrain.py      -> <EXP_ROOT>/fold<k>/stage2/        (train+val)
#   3. runners/run_waveform.py      -> <EXP_ROOT>/fold<k>/stage3/        (train+val)
#   4. runners/run_waveform.py --eval_role test -> <EXP_ROOT>/fold<k>/stage3/test/
#
# Step 1 is idempotent: an existing record with the SAME digest is reused, a
# different one aborts the fold, so a resumed sweep cannot silently switch
# partitions. Step 4 refuses to run twice (see --overwrite_test).
#
# Run from `code/` (the wrapper cds there). Nothing here is fold-parallel: the
# GPUs inside one fold are used by that fold's own DDP run.
set -euo pipefail
cd "$(dirname "$0")/../.."

: "${EXP_ROOT:?set EXP_ROOT, e.g. EXP_ROOT=\$WORK_SCRATCH/protocol_v1}"

FOLDS="${FOLDS:-0 1 2 3 4}"
PY="${PY:-python}"
RAW_ROOT="${RAW_ROOT:-${RAW_DATA_PATH:-}}"
CFG2="${CFG2:-configs/pretrain/stage2_budget_tir_roi_resp_rgb2d.yaml}"
CFG3="${CFG3:-configs/finetune/resp_tir_roi_hpc.yaml}"
DATA_SET2="${DATA_SET2:-tir_roi_resp_rgb2d}"
DATA_SET3="${DATA_SET3:-tir_roi_resp}"
SPLIT_SEED="${SPLIT_SEED:-42}"
SPLIT_STRATIFY="${SPLIT_STRATIFY:-sex}"

if [ -z "$RAW_ROOT" ]; then
  echo "ERROR: RAW_ROOT/\$RAW_DATA_PATH is empty; the subject universe cannot be enumerated." >&2
  exit 2
fi

newest_ckpt() {
  ls -1 "$1"/checkpoints/checkpoint-*.pth 2>/dev/null | sort | tail -1
}

echo "[sweep] exp_root : $EXP_ROOT"
echo "[sweep] folds    : $FOLDS   (sequential)"
echo "[sweep] configs  : $CFG2  +  $CFG3"
echo "[sweep] raw_root : $RAW_ROOT"

FAILED=()
for k in $FOLDS; do
  dir="$EXP_ROOT/fold$k"
  mkdir -p "$dir/stage2" "$dir/stage3"
  # per-fold rendezvous port: a requeued fold must not collide with a socket
  # still in TIME_WAIT from the previous attempt
  export MASTER_PORT=$((29500 + 10 * k))

  echo "================ fold $k ================"
  if ! "$PY" runners/run_partition.py \
        --exp_root "$EXP_ROOT" --fold "$k" \
        --data_set "$DATA_SET2" --raw_root "$RAW_ROOT" \
        --split_seed "$SPLIT_SEED" --split_stratify "$SPLIT_STRATIFY"; then
    echo "ERROR: fold $k: partition record failed" >&2
    FAILED+=("$k")
    continue
  fi

  if ! "$PY" runners/run_pretrain.py -c "$CFG2" \
        --exp_root "$EXP_ROOT" --fold "$k" \
        --data_set "$DATA_SET2" --raw_root "$RAW_ROOT" \
        --output_dir "$dir/stage2"; then
    echo "ERROR: fold $k: Stage 2 failed" >&2
    FAILED+=("$k")
    continue
  fi

  s2="$(newest_ckpt "$dir/stage2")"
  if [ -z "$s2" ]; then
    echo "ERROR: fold $k: no Stage-2 checkpoint under $dir/stage2/checkpoints" >&2
    FAILED+=("$k")
    continue
  fi

  if ! "$PY" runners/run_waveform.py -c "$CFG3" \
        --exp_root "$EXP_ROOT" --fold "$k" \
        --data_set "$DATA_SET3" --raw_root "$RAW_ROOT" --finetune "$s2" \
        --output_dir "$dir/stage3"; then
    echo "ERROR: fold $k: Stage 3 failed" >&2
    FAILED+=("$k")
    continue
  fi

  # Report time: a separate process, scored once (I5).
  if ! "$PY" runners/run_waveform.py -c "$CFG3" \
        --exp_root "$EXP_ROOT" --fold "$k" \
        --data_set "$DATA_SET3" --raw_root "$RAW_ROOT" --finetune "$s2" \
        --output_dir "$dir/stage3" --eval_role test; then
    echo "ERROR: fold $k: test scoring failed" >&2
    FAILED+=("$k")
    continue
  fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "[sweep] these folds failed: ${FAILED[*]}" >&2
  exit 1
fi
echo "[sweep] all requested folds finished; pool with:"
echo "        $PY runners/run_evaluate_kfold.py --exp_root $EXP_ROOT"
