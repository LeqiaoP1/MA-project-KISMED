#!/usr/bin/env bash
# Inspect / verify the RGB face-ROI dataset provider (data/rgb_roi_dataset.py).
#
# Nothing is decoded from the canonical prepared tree: this reads the RAW layout
# directly (2D+3D/<S>/<T>/*.jpg, 2DFeatures/<S>_<T>.mat, Physiology/<S>/<T>/...),
# exactly what Stage 2/3 will consume.
#
# It prints the discovery verdict for every (subject, task) -- including the
# UNUSABLE ones, i.e. pairs whose 2DFeatures frame count differs from the jpeg
# count, which are excluded whole so their 1-D physiology is never used -- and
# then runs the per-clip data checks:
#   * keys / dtypes / shapes and the [0, 1] pixel range;
#   * the ROI box lies inside the frame AND contains every target landmark;
#   * rgb[:, 0] IS the cv2.resize of that crop (not a re-centred crop);
#   * each 1-D window == the brute-force raw slice y[s0 : s0+L];
#   * T/fps == L/phys_fs.
# A failing clip makes the script exit non-zero.
#
#   bash scripts/local/inspect_rgb_roi.sh --list            # what is on disk?
#   bash scripts/local/inspect_rgb_roi.sh                   # F001/T1, first 3 clips
#   SUBJECT=F003 TASK=T8 bash scripts/local/inspect_rgb_roi.sh   # the UNUSABLE pair
#   LANDMARKS=nose_mouth bash scripts/local/inspect_rgb_roi.sh
#   SIGNALS=bp,resp SIZE=224 DECODE_SCALE=4 CHECKS=2 bash scripts/local/inspect_rgb_roi.sh
#   bash scripts/local/inspect_rgb_roi.sh --subject all --task all --n_check 1
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
CLIP_STRIDE="${CLIP_STRIDE:-0}"
LANDMARKS="${LANDMARKS:-face}"
DECODE_SCALE="${DECODE_SCALE:-1}"
SIGNALS="${SIGNALS:-bp}"
CHECKS="${CHECKS:-3}"

if [ "$#" -gt 0 ]; then
    # explicit CLI wins over the env-style overrides above
    "$PY" -u data/rgb_roi_dataset.py "$@"
else
    ARGS=(--subject "$SUBJECT" --task "$TASK" --signals "$SIGNALS"
          --input_size "$SIZE" --clip_seconds "$CLIP_SECONDS"
          --landmarks "$LANDMARKS" --decode_scale "$DECODE_SCALE"
          --n_check "$CHECKS")
    [ "$CLIP_STRIDE" != "0" ] && ARGS+=(--clip_stride "$CLIP_STRIDE")
    "$PY" -u data/rgb_roi_dataset.py "${ARGS[@]}"
fi

echo
echo "Tip: '--list' prints every session verdict (usable / UNUSABLE)."
echo "     '--landmarks face|nose_mouth|nostrils|nostril_mouth|nose_tip|1,2,3'"
echo "     '--norm none|clip|session'   '--roi_padding 0.2'  '--roi_quantile 0.05'"
