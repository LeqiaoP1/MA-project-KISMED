#!/usr/bin/env bash
# Lichtenberg HPC: build the offline TIR-ROI pixel cache (one array element).
#
# WHAT IT DOES -- caches the NATIVE-resolution union crop of every subject-task
# in <raw_root>, so the training loader never has to decode a .wmv again. Serves
# Stage 2 (2 s hop), Stage 3 (1 s hop) and both the 112 px and 64 px lineages
# from ONE ~92 GB artifact.
#
# CPU-ONLY: the work is decode + crop + write, no GPU is involved. Do NOT add a
# --gres line -- an 'acc' slot would be wasted (our own 4-GPU job waited 17 h).
#
# PREREQUISITES: mkdir -p logs   (in code/)
#
# Submit from code/:
#   sbatch scripts/hpc/submit_roi_cache.sbatch                  # default 0-31
#   sbatch --array=0-63 scripts/hpc/submit_roi_cache.sbatch     # 64 ways
#   CACHE_LIMIT=40 sbatch scripts/hpc/submit_roi_cache.sbatch   # smoke build
#
# AFTER the array completes (both are cheap, run them on the login node):
#   python -u runners/run_build_roi_cache.py --status --out_root "$CACHE_OUT_ROOT"
#   python -u runners/run_build_roi_cache.py --index  --out_root "$CACHE_OUT_ROOT"
#   python -u runners/run_build_roi_cache.py --check  --out_root "$CACHE_OUT_ROOT"
#
# Env-style overrides (propagated by sbatch's default --export=ALL). NOTE: every
# one of these changes the cache KEY, so --init and --build must agree on them:
#   CACHE_OUT_ROOT     destination                default: $WORK_SCRATCH/tir_roi_cache
#   CACHE_ROI_PADDING  ROI margin                  default: 0.2
#   CACHE_ROI_LANDMARKS '' / preset / CSV          default: '' (nose_mouth)
#   CACHE_SUBJECTS     comma list ('' = all)
#   CACHE_LIMIT        first N subject-tasks (0 = all)
# Anything after the script name is forwarded to the runner verbatim.
#SBATCH --job-name=roicache
#SBATCH --partition=deflt             # CPU-only; 1 day max ('deflt_short' = 30 min)
# NB: the LUA plugin rejects --mem, so memory is --mem-per-cpu * cpus-per-task.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
# 2 x 8G = 16G. The writer decodes in 256-frame chunks so the peak is a chunk
# (~0.27 GB) plus the output crop, but the read_all FALLBACK for a container
# that over-reports its frame count re-materialises the whole task: measured
# 11.1 GB RSS on the corpus' longest (5149 frames). --max_task_gb guards that.
# The first attempt asked for 8G and 8 of 32 elements were OOM-killed (0:125).
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-31
#SBATCH --output=logs/roi_cache_%A_%a.out
#SBATCH --error=logs/roi_cache_%A_%a.err

set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/env_hpc.sh" ]; then
    ENV_SH="$SLURM_SUBMIT_DIR/scripts/env_hpc.sh"
else
    ENV_SH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env_hpc.sh"
fi
if [ ! -f "$ENV_SH" ]; then
    echo "ERROR: env_hpc.sh not found (tried $ENV_SH); submit from code/: cd code && sbatch scripts/hpc/..." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_SH"
activate_project_env

cd "$CODE_DIR"

if [ -z "${RAW_DATA_PATH:-}" ] || [ ! -d "$RAW_DATA_PATH" ]; then
    echo "ERROR: RAW_DATA_PATH is unset or not a directory ($RAW_DATA_PATH)." >&2
    exit 1
fi

OUT_ROOT="${CACHE_OUT_ROOT:-${WORK_SCRATCH:-$OUTPUT_DIR}/tir_roi_cache}"
mkdir -p "$OUT_ROOT" "$CODE_DIR/logs"

ARGS=(--raw_root "$RAW_DATA_PATH" --out_root "$OUT_ROOT")
[ -n "${CACHE_ROI_PADDING:-}" ]   && ARGS+=(--roi_padding "$CACHE_ROI_PADDING")
[ -n "${CACHE_ROI_LANDMARKS:-}" ] && ARGS+=(--roi_landmarks "$CACHE_ROI_LANDMARKS")
[ -n "${CACHE_SUBJECTS:-}" ]      && ARGS+=(--subjects "$CACHE_SUBJECTS")
[ -n "${CACHE_LIMIT:-}" ]         && ARGS+=(--limit "$CACHE_LIMIT")
[ -n "${CACHE_COMPRESS:-}" ]      && ARGS+=(--compress)
[ -n "${CACHE_MAX_TASK_GB:-}" ]   && ARGS+=(--max_task_gb "$CACHE_MAX_TASK_GB")

echo "[roi_cache] raw_root  : $RAW_DATA_PATH"
echo "[roi_cache] out_root  : $OUT_ROOT"
echo "[roi_cache] array     : ${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}"
echo "[roi_cache] node      : $(hostname)"
echo

# --init is idempotent and races harmlessly (atomic os.replace), so every array
# element may call it. It probes the source frame size, which is a KEY field, and
# it fails loudly if the existing params.json disagrees with these arguments.
python -u runners/run_build_roi_cache.py --init "${ARGS[@]}"
echo

python -u runners/run_build_roi_cache.py --build "${ARGS[@]}"

echo
echo "[roi_cache] element ${SLURM_ARRAY_TASK_ID:-0} done."
echo "[roi_cache] when the whole array has finished, run:"
echo "  python -u runners/run_build_roi_cache.py --index --out_root '$OUT_ROOT'"
echo "  python -u runners/run_build_roi_cache.py --check --out_root '$OUT_ROOT'"
