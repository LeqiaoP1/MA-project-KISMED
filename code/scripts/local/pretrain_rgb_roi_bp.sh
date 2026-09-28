#!/usr/bin/env bash
# Stage-2 LOCAL pre-training: RGB face ROI (visual) -> BP (1-D target).
#
# Wrapper around
#     runners/run_pretrain.py -c configs/pretrain/stage2_local_rgb_roi_bp.yaml
# which selects the ADD-ON `data_set: rgb_roi` path (data/rgb_roi_dataset.py):
# the visual stream is the LANDMARK-DEFINED FACE CROP of the visible-light
# frames and the 1-D target is Physiology/<S>/<T>/BP_mmHg.txt (1000 Hz -> 100 Hz).
# A (subject, task) pair whose 2DFeatures frame count differs from the jpeg count
# is excluded as `unusable` (local corpus: F003_T8 only -> 39 usable sessions).
#
# Usage (from anywhere):
#     bash scripts/local/pretrain_rgb_roi_bp.sh              # full local run
#     SMOKE=1 bash scripts/local/pretrain_rgb_roi_bp.sh      # 2-clip smoke, /tmp
#     bash scripts/local/pretrain_rgb_roi_bp.sh --epochs 5    # extra runner args
#     SUBJECTS=F001 TASKS=T1,T2 bash scripts/local/pretrain_rgb_roi_bp.sh
#     CONFIG=configs/pretrain/<other>.yaml bash scripts/local/pretrain_rgb_roi_bp.sh
#
# Anything after the script name is passed to the runner VERBATIM and therefore
# overrides the config (argparse: later wins), e.g.
#     ... --input_size 64        # cheap local smoke (DIFFERENT geometry: the
#                                # pos-embed shape changes, so the checkpoint is
#                                # a separate lineage -- smoke only)
#     ... --decode_scale 2       # halve the JPEG decode cost (DIFFERENT
#                                # front-end: Stage 3 must match)
#     ... --physio_mask span --mask_span_s bp=1.0
#
# The SMOKE path always runs in /tmp so it can never overwrite a real run dir
# (a smoke without --output_dir writes a ~1.2 GB checkpoint into the real run).
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env_local.sh

# Interpreter: $PYTHON wins, else the project venv at <repo>/.venv, else "python".
if [ -z "${PYTHON:-}" ] && [ -x "../.venv/bin/python" ]; then
    PYTHON="../.venv/bin/python"
fi
PY="${PYTHON:-python}"

CONFIG="${CONFIG:-configs/pretrain/stage2_local_rgb_roi_bp.yaml}"
OUT_DIR="${OUT_DIR:-../output/pretrain/stage2_local_rgb_roi_bp}"
SMOKE="${SMOKE:-0}"
SUBJECTS="${SUBJECTS:-}"
TASKS="${TASKS:-}"

ARGS=(-c "$CONFIG")
[ -n "$SUBJECTS" ] && ARGS+=(--subjects "$SUBJECTS")
[ -n "$TASKS" ] && ARGS+=(--tasks "$TASKS")

case "$SMOKE" in
    1|true|TRUE|yes|YES)
        SMOKE_DIR="${SMOKE_DIR:-/tmp/rgb_roi_bp_smoke}"
        # --warmup_epochs 0 is REQUIRED: epochs < warmup trips the pre-existing
        # cosine_scheduler assert in utils/lr_sched.py.
        ARGS+=(--epochs 1 --warmup_epochs 0 --max_entries 2 --num_workers 0
               --output_dir "$SMOKE_DIR")
        OUT_DIR="$SMOKE_DIR"
        ;;
esac

ARGS+=("$@")
LOG="${LOG:-$OUT_DIR/train.log}"
mkdir -p "$(dirname "$LOG")"

echo "[pretrain_rgb_roi_bp] config  : $CONFIG"
echo "[pretrain_rgb_roi_bp] output  : $OUT_DIR"
echo "[pretrain_rgb_roi_bp] log     : $LOG"
echo "[pretrain_rgb_roi_bp] command : $PY -u runners/run_pretrain.py ${ARGS[*]}"
echo

# -u is MANDATORY: without it stdout is block-buffered through `tee`, so an
# interrupted run leaves a log with NO epoch lines (an 11-epoch divergence was
# invisible that way once already).
"$PY" -u runners/run_pretrain.py "${ARGS[@]}" 2>&1 | tee "$LOG"

echo
echo "Log:      $LOG"
echo "Dump all: ../output/pretrain/stage2_local_rgb_roi_bp/"
