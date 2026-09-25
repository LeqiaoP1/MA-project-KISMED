#!/usr/bin/env bash
# Lichtenberg HPC: RAW 1-D physiology inspection (BP4D Physiology/<subj>/<task>/*.txt).
# CPU-only -> NO GPU requested: the runner draws with matplotlib and only reaches
# torch through ``import data`` (``data/__init__.py``), which it never calls.
# 1) edit scripts/env_hpc.sh   2) mkdir -p logs
# 3) sbatch scripts/hpc/submit_inspect_physio.sbatch
#
# Env-style overrides (propagated by sbatch's default --export=ALL):
#   INSPECT_SUBJECTS  subject id(s)          default F001
#   INSPECT_TASKS     task id(s) or "all"    default T1
#   INSPECT_CHANNEL   BP|Resp|EDA|HR|all     default all
#   PHYS_FS           raw sample rate in Hz; 0 = estimate from RGB frames
#   INSPECT_OUTPUT_DIR  write under $WORK instead of the $HOME default
# Anything after the script name is forwarded to the runner verbatim (a trailing
# --output_dir overrides INSPECT_OUTPUT_DIR):
#   sbatch scripts/hpc/submit_inspect_physio.sbatch --list
#   sbatch scripts/hpc/submit_inspect_physio.sbatch --no-plot
#   sbatch scripts/hpc/submit_inspect_physio.sbatch --band Resp:0.15,0.8
#SBATCH --job-name=inspect_physio
#SBATCH --partition=deflt_short       # CPU-only, no GPU; 30 min max ('deflt' = 1 day)
# NB: the LUA plugin rejects --mem, so memory is --mem-per-cpu * cpus-per-task.
# --account is omitted on purpose: the plugin bills your default project.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G              # 4 cpus x 2G = 8G total
#SBATCH --time=00:30:00
#SBATCH --output=logs/inspect_physio_%j.out
#SBATCH --error=logs/inspect_physio_%j.err

set -euo pipefail
# sbatch COPIES this script into the spool dir and runs that copy, so $0 is
# /var/spool/slurmd/job<id>/slurm_script -- NOT the repo path. Resolve the
# profile from $SLURM_SUBMIT_DIR (submit from code/), and fall back to
# BASH_SOURCE only for direct `bash scripts/hpc/...` runs.
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
# $HOME has a quota and a full session writes tens of MB of full-rate PNGs, so
# INSPECT_OUTPUT_DIR (e.g. "$WORK_PROJ/inspect_physio") can move them to $WORK.
EXTRA_ARGS=()
if [ -n "${INSPECT_OUTPUT_DIR:-}" ]; then
    EXTRA_ARGS+=(--output_dir "$INSPECT_OUTPUT_DIR")
fi
OUT_ROOT="${INSPECT_OUTPUT_DIR:-$OUTPUT_DIR/inspect_data}"

# No alignment, no filtering, no resampling: the ORIGINAL samples are analysed
# and plotted 1:1 (--phys_fs is used for the time axis and Hz <-> /min only).
python -u runners/run_inspect_physio.py \
    --subject "${INSPECT_SUBJECTS:-F001}" \
    --task "${INSPECT_TASKS:-T1}" \
    --channel "${INSPECT_CHANNEL:-all}" \
    --phys_fs "${PHYS_FS:-1000}" \
    --fps_rgb "${FPS_RGB:-25}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "Figures + JSON: $OUT_ROOT/<subject>_<task>/"
echo "Index:          $OUT_ROOT/physio_index.json"
