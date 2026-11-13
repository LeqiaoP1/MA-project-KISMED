#!/usr/bin/env bash
# Lichtenberg HPC: build the offline 2D-texture face-ROI cache (`rgb2d`).
#
# WHAT IT DOES -- caches the NATIVE-resolution SQUARE HULL crop of every
# subject-task under <raw_root>, so the training loader never has to decode a
# 1392x1040 JPEG again. Because the pixels are stored native, ONE artifact
# serves every hop, every fold and every --input_size (64/96/112).
#
# CPU-ONLY: the work is decode + crop + pad + write, no GPU is involved. Do NOT
# add a --gres line -- an 'acc' slot would be wasted.
#
# SIZE (MEASURED 2026-11-10 on subject F001, see
# code/notes/eval/rgb2d/build_preflight.md): 1.21 MB/frame average and
# 1.37 GB/session, because the stored box is the union over every stride-1 8 s
# window of the WHOLE task (621-1036 px on a 1392x1040 source). Forecast for the
# 1390-session corpus: ~1.9 TB of frame data, ~2.25 TB on disk -- versus 623 GB
# used of the 40 TB scratch quota today, i.e. ~6 %. Check `--status` bytes after
# the FIRST element instead of waiting for the whole array.
#
# TIME (MEASURED): 26.8 s/session on one core -> ~43 sessions per element of
# --array=0-31 -> ~19 min. --time is set to 2 h, ~6x that, for Lustre
# contention when 32 elements read the same archives.
#
# PREREQUISITES: mkdir -p logs   (in code/)
#
# Submit from code/:
#   sbatch scripts/hpc/submit_2d_roi_cache.sbatch                # default 0-31
#   sbatch --array=0-63 scripts/hpc/submit_2d_roi_cache.sbatch   # 64 ways
#   CACHE_LIMIT=40 sbatch scripts/hpc/submit_2d_roi_cache.sbatch # smoke build
#
# AFTER the array completes (cheap; run on the login node):
#   python -u runners/run_build_2d_roi_cache.py --status --out_root "$CACHE_OUT_ROOT"
#   python -u runners/run_build_2d_roi_cache.py --index  --out_root "$CACHE_OUT_ROOT"
#   python -u runners/run_build_2d_roi_cache.py --check 5 --out_root "$CACHE_OUT_ROOT"
# `--index` must run AFTER all elements: it rebuilds manifest.json from what is
# on disk. `--status` should report `missing 0`. An archive that is still being
# copied in is reported as "not readable yet" and its subject is skipped for
# that run ONLY (nothing permanent is recorded, so a later run picks it up);
# run notes/eval/rgb2d/zip_audit.py first if the corpus may still be arriving.
#
# Env-style overrides (propagated by sbatch's default --export=ALL). NOTE: every
# one of these changes the cache KEY, so --init and --build must agree on them:
#   CACHE_OUT_ROOT        destination                default: $WORK_SCRATCH/rgb2d_roi_cache
#   CACHE_ROI_PADDING     ROI margin                  default: unset -> 0.10
#   CACHE_ROI_LANDMARKS   'face' / preset / CSV       default: 'face' (all 49)
#   CACHE_CROP_RULE       box shape                   default: 'square'
#   CACHE_CLIP_SECONDS    window the hull unions over  default: 8
#   CACHE_FPS             frames per second            default: 25
#   CACHE_SUBJECTS        comma list ('' = all)
#   CACHE_LIMIT           first N subject-tasks (0 = all)
#   CACHE_MAX_TASK_GB     refuse a larger single task  default: unset -> 12
#   CACHE_STORE           'native' (source pixels; one artefact serves every
#                         input_size) or a pixel size such as 112. A pixel
#                         store is resized at BUILD time: 7.5 MB/clip instead
#                         of 316 MB (42x less I/O, measured 2026-11-12), but
#                         it is a KEY field, so input_size must be <= that size.
#                         The two stores live in DIFFERENT cache dirs.
#
# Anything after the script name is forwarded to the runner verbatim.
#SBATCH --job-name=rgb2dcache
#SBATCH --partition=deflt             # CPU-only; 1 day max ('deflt_short' = 30 min)
# NB: the LUA plugin rejects --mem, so memory is --mem-per-cpu * cpus-per-task.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
# 2 x 8G = 16G. The writer streams one 32-frame block (~38 MB at the measured
# 1.2 MB/frame) plus one decoded source frame (4.3 MB), so the peak is O(50 MB)
# -- this is generous on purpose, because a zip member decode holds the
# compressed bytes too.
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-31
#SBATCH --output=logs/rgb2d_cache_%A_%a.out
#SBATCH --error=logs/rgb2d_cache_%A_%a.err

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

OUT_ROOT="${CACHE_OUT_ROOT:-${WORK_SCRATCH:-$OUTPUT_DIR}/rgb2d_roi_cache}"
mkdir -p "$OUT_ROOT" "$CODE_DIR/logs"

ARGS=(--raw_root "$RAW_DATA_PATH" --out_root "$OUT_ROOT")
[ -n "${CACHE_ROI_PADDING:-}" ]   && ARGS+=(--roi_padding "$CACHE_ROI_PADDING")
[ -n "${CACHE_ROI_LANDMARKS:-}" ] && ARGS+=(--roi_landmarks "$CACHE_ROI_LANDMARKS")
ARGS+=(--crop_rule "${CACHE_CROP_RULE:-square}")
ARGS+=(--clip_seconds "${CACHE_CLIP_SECONDS:-8}")
ARGS+=(--fps "${CACHE_FPS:-25}")
[ -n "${CACHE_SUBJECTS:-}" ]      && ARGS+=(--subjects "$CACHE_SUBJECTS")
[ -n "${CACHE_LIMIT:-}" ]         && ARGS+=(--limit "$CACHE_LIMIT")
[ -n "${CACHE_MAX_TASK_GB:-}" ]   && ARGS+=(--max_task_gb "$CACHE_MAX_TASK_GB")
ARGS+=(--store_resolution "${CACHE_STORE:-native}")

echo "[rgb2d_cache] raw_root  : $RAW_DATA_PATH"
echo "[rgb2d_cache] out_root  : $OUT_ROOT"
echo "[rgb2d_cache] store     : ${CACHE_STORE:-native}"
echo "[rgb2d_cache] array     : ${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}"
echo "[rgb2d_cache] node      : $(hostname)"
echo

# --init is idempotent and races harmlessly (atomic os.replace), so every array
# element may call it. It probes the source frame size, which is a KEY field, and
# it fails loudly if the existing params.json disagrees with these arguments.
python -u runners/run_build_2d_roi_cache.py --init "${ARGS[@]}"
echo

python -u runners/run_build_2d_roi_cache.py --build "${ARGS[@]}"

echo
echo "[rgb2d_cache] element ${SLURM_ARRAY_TASK_ID:-0} done."
echo "[rgb2d_cache] check bytes now (do not wait for the array):"
echo "  python -u runners/run_build_2d_roi_cache.py --status --out_root '$OUT_ROOT'"
