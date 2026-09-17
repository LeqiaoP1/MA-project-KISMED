#!/usr/bin/env bash
# AU occurrence coding inspection (raw BP4D+ FACS AUCoding/AU_OCC): how often
# every AU fires, how the AUs co-occur, how long their activation segments are
# and where the coded blocks sit in the videos. Writes the figures + AU_summary.json
# into <repo>/output/inspect_data/AUCoding/. Run from code/ (or anywhere; the
# script cd's into code/).
#
#   bash scripts/local/inspect_au.sh                       # whole corpus
#   bash scripts/local/inspect_au.sh --list                # what is on disk?
#   TASK=T7 bash scripts/local/inspect_au.sh               # one task only
#   bash scripts/local/inspect_au.sh --task T1 --no-plot   # JSON only
#   bash scripts/local/inspect_au.sh --max_files 20        # smoke run
#
# Anything after the script name is passed to the runner verbatim, so flags can
# also be given directly: bash scripts/local/inspect_au.sh --au_list 6,7,10,12,14
set -e

cd "$(dirname "$0")/../.."
source scripts/env_local.sh

TASK="${TASK:-all}"
SUBJECT="${SUBJECT:-}"
FPS="${FPS:-25}"

if [ "$#" -gt 0 ]; then
    # explicit CLI wins over the env-style overrides above
    python runners/run_inspect_au.py "$@"
else
    # --output_dir is left to the runner: it defaults to $OUTPUT_DIR/inspect_data/AUCoding
    ARGS=(--task "$TASK" --fps "$FPS")
    [ -n "$SUBJECT" ] && ARGS+=(--subject "$SUBJECT")
    python runners/run_inspect_au.py "${ARGS[@]}"
fi

echo
echo "Figures + JSON: $OUTPUT_DIR/inspect_data/AUCoding/"
