#!/usr/bin/env bash
# Local smoke: convert the first N (default 2) raw BP4D sessions into the
# canonical per-session layout that PairedSessionDataset consumes.
# Run from code/ (or from anywhere; the script cd's into code/).
set -e

cd "$(dirname "$0")/../.."
source scripts/env_local.sh

N="${1:-2}"   # number of sessions to convert (pass as argv[1])

mkdir -p "$(dirname "$DATA_PATH")"
python data/prepare_bp4d.py \
    --raw_root "$RAW_DATA_PATH" \
    --out_root "$DATA_PATH" \
    --limit_sessions "$N" \
    --images copy

echo
echo "Done. Canonical sessions are under: $DATA_PATH"
echo "Next: scripts/local/inspect_smoke.sh"
