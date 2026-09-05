#!/usr/bin/env bash
# Local smoke: convert the first N (default 2) raw BP4D sessions into the
# canonical per-session layout that PairedSessionDataset consumes.
# Run from code/ (or from anywhere; the script cd's into code/).
#
# Usage:
#   $0 [N] [--force]
#
#   N             number of sessions to convert (default: 2)
#   -f, --force   add --force to prepare_bp4d.py so existing sessions are
#                 re-converted instead of skipped
set -e

cd "$(dirname "$0")/../.."
source scripts/env_local.sh

N=2          # number of sessions to convert (overridable by positional arg)
FORCE=       # empty, or "--force" when -f/--force is given

usage() {
    cat <<EOF
Usage: $0 [N] [--force]

  N             number of sessions to convert (default: 2)
  -f, --force   re-convert existing sessions (adds --force to prepare_bp4d.py)
  -h, --help    show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--force) FORCE="--force"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) N="$1"; shift ;;   # positional: number of sessions
    esac
done

mkdir -p "$(dirname "$DATA_PATH")"
python data/prepare_bp4d.py \
    --raw_root "$RAW_DATA_PATH" \
    --out_root "$DATA_PATH" \
    --limit_sessions "$N" \
    --images copy \
    $FORCE

echo
echo "Done. Canonical sessions are under: $DATA_PATH"
echo "Next: scripts/local/inspect_smoke.sh"
