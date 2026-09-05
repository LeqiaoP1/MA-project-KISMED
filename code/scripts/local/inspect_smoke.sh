#!/usr/bin/env bash
# Local smoke: load aligned RGB+TIR+signals from the canonical sessions
# (1-2 sessions, capped clips) and print/report tensor statistics + a plot.
# Run from code/ (or from anywhere; the script cd's into code/).
set -e

cd "$(dirname "$0")/../.."
source scripts/env_local.sh

python runners/run_inspect_data.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR/inspect_data" \
    --max_sessions 3 \
    --max_entries 2 \
    --input_size 64 \
    --plot

echo
echo "Inspection summary: $OUTPUT_DIR/inspect_data/inspect_summary.json"
echo "Figure:            $OUTPUT_DIR/inspect_data/fig_*.png"
