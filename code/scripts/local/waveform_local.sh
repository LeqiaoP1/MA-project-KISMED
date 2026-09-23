#!/usr/bin/env bash
# Local Stage-3 fine-tuning: RGB-only -> one target waveform (bp | resp).
#
# Usage (from code/, or from the repo root via scripts/local/...):
#     source scripts/env_local.sh
#     TARGET=bp   scripts/local/waveform_local.sh              # the BP branch
#     TARGET=resp scripts/local/waveform_local.sh              # the RESP branch
#     TARGET=bp   scripts/local/waveform_local.sh --epochs 1 --warmup_epochs 0 \
#                      --batch_size 2 --max_sessions 2 --max_entries 2   # smoke
#
# Anything after the script name is passed straight to runners/run_waveform.py
# (CLI beats the YAML, which beats the env defaults). Use --resume to continue
# an interrupted run from <output_dir>/checkpoints/latest_checkpoint.txt.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # code/
# Interpreter: $PYTHON wins, else the project venv at <repo>/.venv, else "python".
if [ -z "${PYTHON:-}" ] && [ -x "../.venv/bin/python" ]; then
    PYTHON="../.venv/bin/python"
fi
PY="${PYTHON:-python}"

TARGET="${TARGET:-bp}"
case "$TARGET" in
    bp|resp) ;;
    *) echo "[waveform_local] TARGET must be 'bp' or 'resp' (got '$TARGET')" >&2; exit 2 ;;
esac

CONFIG="configs/finetune/${TARGET}_local.yaml"
if [ ! -f "$CONFIG" ]; then
    echo "[waveform_local] missing $CONFIG" >&2; exit 2
fi

echo "[waveform_local] interpreter = $PY"
echo "[waveform_local] TARGET      = $TARGET"
echo "[waveform_local] config      = $CONFIG"

# Criterion: WaveformJointLoss = alpha*L1 + beta*(1-Pearson) + gamma*MR-STFT
# Masking: none (the sensor-failure protocol feeds RGB only, and the 1-D target
# is the label, never an input).
NUM_WORKERS="${NUM_WORKERS:-4}" "$PY" -u runners/run_waveform.py -c "$CONFIG" "$@"
