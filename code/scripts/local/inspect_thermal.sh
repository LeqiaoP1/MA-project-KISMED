#!/usr/bin/env bash
# Raw-thermal-video inspection (NO canonical decode): read the single BP4D
# Thermal/<subject>/<task>.wmv straight out of the raw tree, report its
# container / stream / decode / chroma / luma / crop geometry and write a frame
# grid, an overview figure and TIR.json into
# <repo>/output/inspect_data/<subject>_<task>/. Run from code/ (or anywhere; the
# script cd's into code/).
#
# Every panel of TIR_frames.png carries the session's 28 IRFeatures landmarks
# (IRFeatures/<subject>_<task>.txt, one line per thermal frame), colour-coded by
# region: brow=cyan, eye=green, nose=yellow, mouth=magenta.
#
# THE IRFeatures ZERO GATE IS ON. A target whose IRFeatures track contains ANY
# line that is entirely 0.0 -- the corpus' undocumented missing-data sentinel,
# written when the tracker loses the face (head turned away; F001_T8 has 112/227
# such lines) -- is IGNORED COMPLETELY: nothing is decoded and no artifact is
# written, the session is recorded as "skipped" in thermal_index.json, and the
# run still exits 0 because a skip is the requested behaviour and not a failure.
# ACCEPT_ZERO_IR=1 (or --accept-zero-ir) overrides the gate; that session is
# then inspected and the all-zero lines simply contribute no landmarks.
# --list shows which targets the gate would ignore BEFORE running anything.
#
#   SUBJECT=F001 TASK=T1                  bash scripts/local/inspect_thermal.sh
#   SUBJECT=F001 TASK=all                 bash scripts/local/inspect_thermal.sh
#   bash scripts/local/inspect_thermal.sh --list           # what is on disk?
#   SUBJECT=F001 TASK=T8 ACCEPT_ZERO_IR=1 bash scripts/local/inspect_thermal.sh
#   DECODER=opencv FRAMES=16              bash scripts/local/inspect_thermal.sh
#
# DECODER=auto (default; PyAV -> OpenCV -> decord) is the informative one: a
# single pass yields the presentation timestamps AND the raw YUV chroma planes.
# DECODER=opencv reproduces what training sees (minus timestamps/planes).
# MAX_FRAMES caps the decode per session (0 = all, the default: the DECODED
# frame count is the authoritative one, so a cap makes every count a lower
# bound).
#
# Anything after the script name is passed to the runner verbatim, so flags can
# also be given directly:
#   bash scripts/local/inspect_thermal.sh --subject F001 --task T1 --frames 16
set -e

cd "$(dirname "$0")/../.."
source scripts/env_local.sh

SUBJECT="${SUBJECT:-F001}"
TASK="${TASK:-T1}"
FRAMES="${FRAMES:-6}"
DECODER="${DECODER:-auto}"
CROP_SIZE="${CROP_SIZE:-224}"
MAX_FRAMES="${MAX_FRAMES:-0}"
ACCEPT_ZERO_IR="${ACCEPT_ZERO_IR:-0}"

if [ "$#" -gt 0 ]; then
    # explicit CLI wins over the env-style overrides above
    python runners/run_inspect_thermal.py "$@"
else
    # --output_dir is left to the runner: it defaults to $OUTPUT_DIR/inspect_data
    ARGS=(--subject "$SUBJECT" --task "$TASK" --decoder "$DECODER"
          --frames "$FRAMES" --crop_size "$CROP_SIZE")
    [ "$MAX_FRAMES" != "0" ] && ARGS+=(--max_frames "$MAX_FRAMES")
    case "$ACCEPT_ZERO_IR" in
        1|true|TRUE|yes|YES) ARGS+=(--accept-zero-ir) ;;
    esac
    python runners/run_inspect_thermal.py "${ARGS[@]}"
fi

echo
echo "Figures + JSON: $OUTPUT_DIR/inspect_data/<subject>_<task>/"
echo "Index:          $OUTPUT_DIR/inspect_data/thermal_index.json"
