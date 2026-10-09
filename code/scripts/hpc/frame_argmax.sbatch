#!/usr/bin/env bash
# Lichtenberg HPC: WHERE is the whole-frame gate's best pixel? (open Q1)
#
# Runs analysis/tir_resp/frame_argmax.py, which decodes the WHOLE thermal frame
# (NOT the ROI crop) and locates the argmax pixel of the gate statistic (max over
# pixels x lags of |r| vs the respiration belt). `roi_sweep.py` v2 found the
# `frame` box detects the belt MORE strongly than any face box while the two
# off-face background controls stay clean; since `frame` is a superset of them,
# the winning pixel must be off-face AND off-both-background -- this job says
# exactly where, before any positive in the line can be believed.
#
# CPU-ONLY, and it DECODES the raw .wmv, so it needs raw-tree access. Runs on
# `deflt` (separate queue from the GPU `acc` partition). Do NOT run on the login
# node: a full-frame correlation sweep is heavy (one session burned 64 CPU-min
# in 69 s wall there).
#
# Submit from code/:
#   sbatch scripts/hpc/frame_argmax.sbatch
#   SESSIONS=F001_T3,F002_T1 NPERM=50 sbatch scripts/hpc/frame_argmax.sbatch
#SBATCH --job-name=frame_argmax
#SBATCH --partition=deflt              # CPU-only partition; no GPU needed
# NB: the LUA plugin rejects --mem, so memory is --mem-per-cpu * cpus-per-task.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G               # 32 x 4G = 128G (full native frame is ~8G)
#SBATCH --time=04:00:00
#SBATCH --output=logs/frame_argmax_%j.out
#SBATCH --error=logs/frame_argmax_%j.err

set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/env_hpc.sh" ]; then
    ENV_SH="$SLURM_SUBMIT_DIR/scripts/env_hpc.sh"
else
    ENV_SH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env_hpc.sh"
fi
if [ ! -f "$ENV_SH" ]; then
    echo "ERROR: env_hpc.sh not found (tried $ENV_SH); submit from code/." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_SH"
activate_project_env
cd "$CODE_DIR"

RAW_ROOT="${RAW_ROOT:-${RAW_DATA_PATH:-/work/projects/l0003511/test}}"
# The roi_sweep v2 winner (frame 0.909 > face 0.813 > nose_mouth 0.804 with the
# two background controls clean) -- the case that opened Q1.
SESSIONS="${SESSIONS:-F001_T3}"
# 0 = locate only (significance already established by the gate); >0 adds the
# phase-scrambled null of the whole-frame max|r|, i.e. a p-value for the location.
NPERM="${NPERM:-50}"
DOWNSCALE="${DOWNSCALE:-1}"
OUT="${OUT:-analysis/tir_resp/frame_argmax.json}"
FIG="${FIG:-analysis/tir_resp/frame_argmax.png}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

echo "[argmax] node      : $(hostname)"
echo "[argmax] raw_root  : $RAW_ROOT"
echo "[argmax] sessions  : $SESSIONS"
echo "[argmax] downscale : $DOWNSCALE | nperm $NPERM"
echo "[argmax] threads   : $OMP_NUM_THREADS"
echo

set +e
python -u analysis/tir_resp/frame_argmax.py \
    --raw_root "$RAW_ROOT" \
    --sessions "$SESSIONS" \
    --downscale "$DOWNSCALE" \
    --nperm "$NPERM" \
    --device cpu \
    --out "$OUT" \
    --figure "$FIG"
RC=$?
set -e

echo
echo "[argmax] exit code: $RC"
echo "[argmax] INTERPRETATION:"
echo "  * the ARGMAX (x, y) + its 'regions' say where the winning pixel is;"
echo "  * regions == ['frame'] only  => OFF-FACE and OFF both background"
echo "    controls -> the whole-frame win is NOT a face signal: keep digging"
echo "    (belt buckle? camera housing? overlay? hot object?);"
echo "  * regions includes 'face'/'nose_mouth' => the landmark box WAS the wrong"
echo "    region and the line genuinely reopens;"
echo "  * compare the printed per-region max|r| (frame vs face vs bg_*): the"
echo "    face-vs-off-face statement now holds at ONE native resolution."
exit "$RC"
