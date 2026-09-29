#!/usr/bin/env bash
# Inspect RGBRoiDataset clips: the visible-light face ROI + the 1000 Hz 1-D
# window the dataset hands to a model, straight out of the RAW tree (no
# canonical decode, no prepared session copy).
#
# Per clip it draws (runners/run_inspect_rgb_bp.py)
#   * the clip's first RGB frame with the ONE crop box shared by all clip_frames
#     frames, plus the target landmarks (LANDMARKS=face -> all 49) colour-coded
#     by anatomy group and the same points of every frame faintly behind them,
#     which is what makes the box's min/max visible; the rectangle is drawn in
#     the DECODED pixels with the dataset's own floor/ceil/clamp arithmetic;
#   * the ROI patch EXACTLY as it leaves the dataset ([C, T, H, W] slice);
#   * the BP_mmHg.txt window in mmHg and, unless --norm none, the per-clip
#     z-score the model computes internally (norm=clip);
#   * (SKIN=1, the default) the two-stage SKIN mask -- per-frame brow/eye/mouth
#     landmark exclusion + the YCrCb skin gate -- tinted red on the frame AND on
#     the patch, with the per-frame retained fraction and the retained vs dropped
#     mean RGB in the JSON. The mask is REPORTED here, not applied to the tensor
#     (wiring it into the provider is a separate step).
# and writes clip_<n>.png + clip_<n>.json + dataset_summary.json +
# rgb_bp_index.json into $OUTPUT_DIR/inspect_data/rgb_bp/.
#
# The SAME clip is run through the dataset's own checks (keys, dtypes, [0,1]
# range, the jpg==mat unusable gate, ROI box containment / clip-static box /
# "the tensor is the resize of that crop", the raw-file signal slice, and
# T/fps == L/phys_fs); a failing clip makes the script exit non-zero.
#
#   SUBJECT=F001 TASK=T1  bash scripts/local/inspect_rgb_bp.sh
#   bash scripts/local/inspect_rgb_bp.sh --list                 # what is on disk?
#   CLIPS=0,7             bash scripts/local/inspect_rgb_bp.sh
#   LANDMARKS=nose_mouth  bash scripts/local/inspect_rgb_bp.sh
#   SIZE=64 DECODE_SCALE=4 PADDING=0.2 bash scripts/local/inspect_rgb_bp.sh
#   SKIN=0                bash scripts/local/inspect_rgb_bp.sh   # no skin mask
#   bash scripts/local/inspect_rgb_bp.sh --task all --no-plot
#
# SIZE=224 (default) is the provider's default; the local Stage-2/3 runs use 64.
# PADDING defaults to 0.1 (+20% overall) -- deliberately tighter than the RGB
# provider's 0.2 (+40%), which is what the configs pin; pass PADDING=0.2 to
# reproduce a training run. CLIPS is the clip index (or comma list) INSIDE the
# selection: F001_T1 (1612 frames, 8 s, non-overlapping) has 8 clips, so
# CLIPS=0,7 is its first and its last window. A native (DECODE_SCALE=1) clip
# costs a full 1392x1040 JPEG decode per frame, so DECODE_SCALE=4 is the fast
# look (the box is rescaled accordingly and both sizes land in the JSON).
#
# Anything after the script name is passed to the runner verbatim.
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env_local.sh

# Interpreter: $PYTHON wins, else the project venv at <repo>/.venv, else "python".
if [ -z "${PYTHON:-}" ] && [ -x "../.venv/bin/python" ]; then
    PYTHON="../.venv/bin/python"
fi
PY="${PYTHON:-python}"

SUBJECT="${SUBJECT:-F001}"
TASK="${TASK:-T1}"
SIZE="${SIZE:-224}"
CLIP_SECONDS="${CLIP_SECONDS:-8}"
# CLIPS is the documented name; CLIP_INDEX is accepted as an alias for it.
CLIPS="${CLIPS:-${CLIP_INDEX:-0}}"
CLIP_STRIDE="${CLIP_STRIDE:-0}"
LANDMARKS="${LANDMARKS:-face}"
SIGNAL="${SIGNAL:-bp}"
DECODE_SCALE="${DECODE_SCALE:-1}"
NORM="${NORM:-clip}"
# The inspector's default; the RGB provider / the configs use 0.2.
PADDING="${PADDING:-0.1}"
# SKIN=1 adds the two-stage skin mask (report + figure tint); SKIN=0 is the
# plain ROI view. It costs one extra clip decode.
SKIN="${SKIN:-1}"
MAX_ENTRIES="${MAX_ENTRIES:-0}"
PLOT="${PLOT:-1}"

if [ "$#" -gt 0 ]; then
    # explicit CLI wins over the env-style overrides above
    "$PY" -u runners/run_inspect_rgb_bp.py "$@"
else
    ARGS=(--subject "$SUBJECT" --task "$TASK" --signal "$SIGNAL"
          --input_size "$SIZE" --clip_seconds "$CLIP_SECONDS"
          --clip_index "$CLIPS" --landmarks "$LANDMARKS"
          --decode_scale "$DECODE_SCALE" --norm "$NORM"
          --roi_padding "$PADDING")
    [ "$CLIP_STRIDE" != "0" ] && ARGS+=(--clip_stride "$CLIP_STRIDE")
    [ "$MAX_ENTRIES" != "0" ] && ARGS+=(--max_entries "$MAX_ENTRIES")
    case "$SKIN" in
        0|false|FALSE|no|NO) ;;
        *) ARGS+=(--skin_mask) ;;
    esac
    case "$PLOT" in
        0|false|FALSE|no|NO) ARGS+=(--no-plot) ;;
    esac
    "$PY" -u runners/run_inspect_rgb_bp.py "${ARGS[@]}"
fi

echo
echo "Figures + JSON: $OUTPUT_DIR/inspect_data/rgb_bp/<subject>_<task>/"
echo "Index:          $OUTPUT_DIR/inspect_data/rgb_bp/rgb_bp_index.json"
