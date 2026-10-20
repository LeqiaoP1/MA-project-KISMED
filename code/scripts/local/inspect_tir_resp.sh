#!/usr/bin/env bash
# Inspect BP4DPlusTIRRespDataset clips: the thermal ROI + the 1000 Hz respiration
# window the dataset hands to a model, straight out of the RAW tree (no
# canonical decode, no prepared session copy).
#
# Per clip it draws (runners/run_inspect_tir_resp.py)
#   * the clip's first thermal frame with the ONE crop box shared by all
#     clip_frames frames, plus the 12 mouth+nose landmarks (1-indexed user-guide
#     labels 9,10,11,12,13,20,21,22,23,24,25,26) that produced it -- nose
#     yellow, mouth magenta, and the same points of every frame faintly behind
#     them, which is what makes the box's min/max visible;
#   * the ROI patch EXACTLY as it leaves the dataset ([C, T, H, W] slice);
#   * the Resp_Volts.txt respiration window in RAW VOLTS -- the dataset's
#     resp_signal ALSO holds raw volts (norm='none', no per-clip z-score).
# and writes clip_<n>.png + clip_<n>.gif (the whole clip as an animation, with
# the frame, the patch and a respiration playhead) + clip_<n>.json +
# dataset_summary.json + tir_resp_index.json into
# $OUTPUT_DIR/inspect_data/tir_resp/.
#
# The SAME clip is run through the dataset's own checks (shapes, dtypes, [0,1]
# range, per-clip z-score, ROI box containment / clip-static box / "the tensor
# is the crop of that box", temporal alignment, and the respiration window
# against a brute-force slice of the raw file); a failing clip makes the script
# exit non-zero.
#
#   SUBJECT=F001 TASK=T1  bash scripts/local/inspect_tir_resp.sh
#   bash scripts/local/inspect_tir_resp.sh --list              # what is on disk?
#   CLIPS=0,1,7           bash scripts/local/inspect_tir_resp.sh
#   SIZE=224 CLIP_SECONDS=4 CLIP_STRIDE=2 bash scripts/local/inspect_tir_resp.sh
#   bash scripts/local/inspect_tir_resp.sh --task all --no-plot
#   NORM=clip             bash scripts/local/inspect_tir_resp.sh   # opt-in z-score
#   VIDEO=none            bash scripts/local/inspect_tir_resp.sh   # figure only
#
# SIZE=112 (default) is the INSPECTION patch size: the native mouth+nose box is
# ~73x89 px, so 112 up-samples it enough to read in the figure. The TRAINING
# geometry for the local Stage-2/3 runs is input_size 64 -- pass `SIZE=64`
# (env) or `--input_size 64` (CLI) to inspect exactly what the model receives.
# CLIPS is the clip index (or comma list) INSIDE the selection -- not a count and not a
# timestamp: F001_T1 (1612 frames, 8 s, non-overlapping) has 8 clips, so
# CLIPS=0,7 is its first and its last window. CLIP_SECONDS is the window length;
# CLIP_STRIDE>0 makes the windows overlap (0 = non-overlapping, the default).
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
SIZE="${SIZE:-112}"
CLIP_SECONDS="${CLIP_SECONDS:-8}"
# CLIPS is the documented name; CLIP_INDEX is accepted as an alias for it.
CLIPS="${CLIPS:-${CLIP_INDEX:-0}}"
CLIP_STRIDE="${CLIP_STRIDE:-0}"
MAX_ENTRIES="${MAX_ENTRIES:-0}"
NORM="${NORM:-none}"
VIDEO="${VIDEO:-gif}"
PLOT="${PLOT:-1}"

if [ "$#" -gt 0 ]; then
    # explicit CLI wins over the env-style overrides above
    "$PY" -u runners/run_inspect_tir_resp.py "$@"
else
    ARGS=(--subject "$SUBJECT" --task "$TASK" --input_size "$SIZE"
          --clip_seconds "$CLIP_SECONDS" --clip_index "$CLIPS"
          --norm "$NORM" --video "$VIDEO")
    [ "$CLIP_STRIDE" != "0" ] && ARGS+=(--clip_stride "$CLIP_STRIDE")
    [ "$MAX_ENTRIES" != "0" ] && ARGS+=(--max_entries "$MAX_ENTRIES")
    case "$PLOT" in
        0|false|FALSE|no|NO) ARGS+=(--no-plot) ;;
    esac
    "$PY" -u runners/run_inspect_tir_resp.py "${ARGS[@]}"
fi

echo
echo "Figures + JSON: $OUTPUT_DIR/inspect_data/tir_resp/<subject>_<task>/"
echo "Index:          $OUTPUT_DIR/inspect_data/tir_resp/tir_resp_index.json"
