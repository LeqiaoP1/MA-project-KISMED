#!/usr/bin/env bash
# Lichtenberg HPC: respiration gate measured in THE MODEL'S OWN INPUT SPACE.
#
# Runs analysis/tir_resp/gate_model_input.py, which reads the OFFLINE ROI CACHE
# -- the exact pixels the encoder consumes -- instead of re-deriving a scalar
# proxy from the raw .wmv. That matters because the recorded gate's pixel
# coordinate (HSV hue) had a palette LUT that was LOST with its /tmp scripts, so
# its absolute numbers cannot be compared to anything measured today.
#
# No video decode is needed (the cache is already decoded), so this is far
# cheaper than scripts/hpc/gate_native.sbatch and scales to many sessions.
#
# CPU-ONLY: the statistic is one matmul per permutation. Runs on `deflt`, a
# SEPARATE partition/queue from the GPU jobs, so it makes progress while a
# training job waits on `acc`.
#
# Submit from code/:
#   sbatch scripts/hpc/gate_model_input.sbatch
#   SESSIONS=F001_T1,F004_T9 NPERM=50 sbatch scripts/hpc/gate_model_input.sbatch
#SBATCH --job-name=gate_modelin
#SBATCH --partition=deflt             # CPU-only partition; no GPU needed
# NB: the LUA plugin rejects --mem, so memory is --mem-per-cpu * cpus-per-task.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G              # 32 x 4G = 128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/gate_modelin_%j.out
#SBATCH --error=logs/gate_modelin_%j.err

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
CACHE_ROOT="${CACHE_ROOT:-${WORK_SCRATCH:-$OUTPUT_DIR}/tir_roi_cache}"

# The recorded gate's 8 sessions by default, so these rows are directly
# comparable to the recorded 0/8.
SESSIONS="${SESSIONS:-F001_T1,F001_T3,F002_T1,F003_T2,F004_T2,F004_T3,F004_T9,F004_T8}"
# Stratified mode: draw SAMPLE_PER_TASK session-tasks PER TASK LABEL (all 10),
# subject-spread. Use this to ask whether the TASK (or the motion it proxies) is
# the lever -- an explicit session list cannot answer that.
SAMPLE_PER_TASK="${SAMPLE_PER_TASK:-0}"
# NB: do NOT name this GROUPS -- that is a BASH-RESERVED array (the user's group
# IDs), so `${GROUPS:-default}` never applies its default and the per-GROUP tally
# silently collapses to '?'. It must be TASK_GROUPS.
TASK_GROUPS="${TASK_GROUPS:-low=T2,T3;moderate=T4,T7,T8,T10;high=T1,T5,T6,T9}"
NPERM="${NPERM:-200}"
# `112` is the model's input_size; `box` is the cached native box with NO resize,
# so the pair separates "the resize destroyed it" from "the representation did".
SIZES="${SIZES:-112,box}"
OUT="${OUT:-analysis/tir_resp/gate_model_input.json}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

if [ "$SAMPLE_PER_TASK" -gt 0 ]; then
    SEL=(--sample-per-task "$SAMPLE_PER_TASK")
    echo "[gate] mode     : STRATIFIED, $SAMPLE_PER_TASK per task label"
else
    SEL=(--sessions "$SESSIONS")
    echo "[gate] mode     : explicit session list"
fi

echo "[gate] node     : $(hostname)"
echo "[gate] raw_root : $RAW_ROOT"
echo "[gate] cache    : $CACHE_ROOT"
echo "[gate] sizes    : $SIZES | nperm $NPERM"
echo "[gate] groups   : $TASK_GROUPS"
echo "[gate] threads  : $OMP_NUM_THREADS"
echo

set +e
python -u analysis/tir_resp/gate_model_input.py \
    --raw_root "$RAW_ROOT" \
    --cache_root "$CACHE_ROOT" \
    "${SEL[@]}" \
    --groups "$TASK_GROUPS" \
    --sizes "$SIZES" \
    --nperm "$NPERM" \
    --device cpu \
    --out "$OUT"
RC=$?
set -e

echo
echo "[gate] exit code: $RC"
echo "[gate] INTERPRETATION:"
echo "  * every 'vs motion_x' row MUST read DETECTED, else the pipeline has no"
echo "    power and no other row means anything;"
echo "  * a DETECTED 'vs belt' row at raw OR 0.1-0.6 says the pixels carry"
echo "    respiration -- which would REOPEN the line and justify a loss sweep;"
echo "  * '112x112' vs 'box' isolates the resize from the representation."
exit "$RC"
