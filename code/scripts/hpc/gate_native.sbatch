#!/usr/bin/env bash
# Lichtenberg HPC: pixel-level positive-control GATE at the model's own input
# resolution -- the decisive test for whether the TIR-ROI -> RESP line can work
# at all.
#
# WHY. `TirROI_Resp_plan.md` section 12 closed the TIR-ROI -> RESP waveform line:
# the recorded gate (2026-09-26) found respiration undetectable in the thermal ROI
# (0/8 sessions, observed best |r| BELOW the phase-scrambled null median) while
# the SAME pipeline detects head motion 8/8 at p < 0.005. But that gate ran at
# 64x64, which is a ~1.6x DOWNSAMPLE of the dataset's own ~72x93 native ROI box,
# so "the resize averaged away a LOCALISED signal" was a real, UNTESTED
# alternative to "there is no signal".
#
# This job tests that alternative directly: the same statistic at native
# resolution (no resize at all) and at 112 -- the `input_size` the Stage-2
# pre-train and the Stage-3 fine-tune actually consume, and the native ceiling
# per the analyser's own docstring. If respiration is absent at NATIVE, no
# architecture, loss, ROI or hyperparameter change can recover it, because
# nothing else in the pipeline can create phase information the pixels lack.
#
# CPU-ONLY ON PURPOSE: the statistic is one matmul per permutation
# ([P,F] @ [F,L]), so it needs no GPU -- and it burns ~1 CPU-hour PER SESSION at
# nperm 20, which is why it runs here on a compute node rather than on a login
# node.
#
# The header of analysis/tir_resp/native_resolution_gate.py documents the
# statistic, the reconstructed-palette caveat, and the synthetic `inject`
# control that quantifies how much a localised signal loses in the resize.
#
# Submit from code/:
#   sbatch scripts/hpc/gate_native.sbatch
#   SESSIONS=F001_T1,F004_T2 NPERM=50 sbatch scripts/hpc/gate_native.sbatch
#SBATCH --job-name=gate_native
#SBATCH --partition=deflt             # CPU-only partition; no GPU needed
# NB: the LUA plugin rejects --mem, so memory is --mem-per-cpu * cpus-per-task.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gate_native_%j.out
#SBATCH --error=logs/gate_native_%j.err

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

# The analyser defaults to '../data/raw/BP4D', which is the LOCAL layout -- the
# HPC corpus lives on /work/projects. RAW_DATA_PATH is the same value
# submit_pretrain.sbatch passes as --raw_root.
RAW_ROOT="${RAW_ROOT:-${RAW_DATA_PATH:-/work/projects/l0003511/test}}"
if [ ! -d "$RAW_ROOT" ]; then
    echo "ERROR: raw root is not a directory: $RAW_ROOT" >&2
    exit 1
fi

# The recorded gate's 8 sessions, so the 112/native rows are directly
# comparable to the recorded 0/8.
SESSIONS="${SESSIONS:-F001_T1,F001_T3,F002_T1,F003_T2,F004_T2,F004_T3,F004_T9,F004_T8}"
SIZES="${SIZES:-native,112,64}"
NPERM="${NPERM:-200}"
OUT="${OUT:-analysis/tir_resp/gate_native_112.json}"

# Cap the thread pool: the permutation loop is BLAS-bound and would otherwise
# claim every core on the node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

echo "[gate] node     : $(hostname)"
echo "[gate] raw_root : $RAW_ROOT"
echo "[gate] sessions : $SESSIONS"
echo "[gate] sizes    : $SIZES"
echo "[gate] nperm    : $NPERM"
echo "[gate] threads  : $OMP_NUM_THREADS"
echo "[gate] out      : $OUT"
echo

set +e
python -u analysis/tir_resp/native_resolution_gate.py \
    --raw_root "$RAW_ROOT" \
    --sessions "$SESSIONS" \
    --sizes "$SIZES" \
    --nperm "$NPERM" \
    --roi_padding 0.2 \
    --device cpu \
    --out "$OUT"
RC=$?
set -e

echo
echo "[gate] exit code: $RC"
echo "[gate] the verdict line per session is the 'vs belt' row:"
echo "       'DETECTED' = p < 0.05, i.e. the ROI carries respiration phase."
echo "       'raw'      = no band-pass; '0.1-0.6' = the respiration band."
echo "       The 'vs motion_x' rows are the POSITIVE CONTROL and must say"
echo "       DETECTED, else the pipeline itself is broken."
exit "$RC"
