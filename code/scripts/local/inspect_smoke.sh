#!/usr/bin/env bash
# Local inspect: load aligned RGB+TIR+signals from the canonical sessions and
# save a PNG preview for EVERY clip of train+val (bounded by max_sessions /
# max_clips). Run from code/ (or anywhere; the script cd's into code/).
set -e

cd "$(dirname "$0")/../.."
source scripts/env_local.sh

# Optional overrides (accept short or INSPECT_* names), e.g.
#   MAX_SESSIONS=6 MAX_CLIPS=2 CLIP_DURATION=2 bash scripts/local/inspect_smoke.sh
# max_entries 0 (default) => no cap: every clip of each split gets a PNG.
MAX_SESSIONS="${MAX_SESSIONS:-${INSPECT_SESSIONS:-3}}"
MAX_ENTRIES="${MAX_ENTRIES:-${INSPECT_ENTRIES:-0}}"
INPUT_SIZE="${INPUT_SIZE:-${INSPECT_INPUT:-64}}"
CLIP_DURATION="${CLIP_DURATION:-}"
MAX_CLIPS="${MAX_CLIPS:-}"

args=(--data_path "$DATA_PATH"
      --output_dir "$OUTPUT_DIR/inspect_data"
      --max_sessions "$MAX_SESSIONS"
      --max_entries "$MAX_ENTRIES"
      --input_size "$INPUT_SIZE"
      --plot)
# --clip_duration / --max_clips are added only when set, so the runner defaults
# apply otherwise (10 s window / no per-session cap).
[ -n "$CLIP_DURATION" ] && args+=(--clip_duration "$CLIP_DURATION")
[ -n "$MAX_CLIPS" ] && args+=(--max_clips "$MAX_CLIPS")

python runners/run_inspect_data.py "${args[@]}"

echo
echo "Inspection summary: $OUTPUT_DIR/inspect_data/inspect_summary.json"
echo "Preview figures:   $OUTPUT_DIR/inspect_data/figures/<split>/"
