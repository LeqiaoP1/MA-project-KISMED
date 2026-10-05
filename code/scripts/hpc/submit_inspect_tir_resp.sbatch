#!/usr/bin/env bash
# Lichtenberg HPC: inspect the TIR-ROI + RESPIRATION clips (CPU-ONLY, no GPU).
#
# WHAT IT DOES -- per clip of one session, straight from the RAW tree:
#   * the clip's first thermal frame + the ONE crop box shared by all frames,
#     with the landmarks that produced it (nose / mouth colour-coded);
#   * the ROI patch exactly as the dataset returns it ([3, T, H, W] slice);
#   * the Resp_Volts.txt window in volts and its per-clip z-scored target.
# It also re-runs the dataset's own correctness checks on the same clip and
# exits non-zero if any fail, so a green job is a real verification.
#
# NO GPU IS NEEDED and none is requested: the runner only uses numpy,
# matplotlib (Agg, no display) and the dataset (CPU tensors). Do NOT add a
# --gres line here. `deflt`/`deflt_short` are the CPU partitions.
#
# PREREQUISITES: mkdir -p logs   (in code/)
#
# Submit from code/:
#   sbatch scripts/hpc/submit_inspect_tir_resp.sbatch
#   sbatch scripts/hpc/submit_inspect_tir_resp.sbatch --list
#   sbatch scripts/hpc/submit_inspect_tir_resp.sbatch --subject F001 --task T1 --clip_index 0
#
# Env-style overrides (propagated by sbatch's default --export=ALL):
#   INSPECT_SUBJECT   subject id(s)        default F001
#   INSPECT_TASK      task id(s) or "all"  default T1
#   INSPECT_TASK_SET  task group(s)        default '' (none)
#   INSPECT_CLIPS     clip index, or list  default 0
#   INSPECT_SIZE      ROI side in px       default 112  (the HPC config value)
#   INSPECT_ROI_PADDING  per-side ROI margin, default 0.2 -- MUST stay in sync
#                       with the configs: data/tir_resp_dataset.DEFAULT_ROI_PADDING
#                       is 0.1, so the runner's own default would draw a
#                       DIFFERENT crop than the model actually receives.
#   INSPECT_LANDMARKS '' = 12-pt nose+mouth preset; or nose_tip / CSV
#   INSPECT_SPREAD    --min_signal_spread  default 0.01 (matches the HPC configs)
#   INSPECT_OUTPUT_DIR write under $WORK instead of the $HOME default
#   INSPECT_LIST=1    discovery only (no decode, no plotting)
# Anything after the script name is forwarded to the runner verbatim.
#
# OUTPUT (default): $OUTPUT_DIR/inspect_data/tir_resp/<subject>_<task>/
#   clip_<n>.png + clip_<n>.json, dataset_summary.json, tir_resp_index.json
# $HOME has a small quota and a full sweep writes many PNGs -- set
# INSPECT_OUTPUT_DIR="$WORK_SCRATCH/inspect_tir_resp" to move them to scratch.
#SBATCH --job-name=insp_tirresp
#SBATCH --partition=deflt_short       # CPU-only, no GPU; 30 min max ('deflt' = 1 day)
# NB: the LUA plugin rejects --mem, so memory is --mem-per-cpu * cpus-per-task.
# --account is omitted on purpose: the plugin bills your default project.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G              # 4 cpus x 2G = 8G total
#SBATCH --time=00:30:00
#SBATCH --output=logs/inspect_tir_resp_%j.out
#SBATCH --error=logs/inspect_tir_resp_%j.err

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

# This path reads the RAW tree (Thermal/, IRFeatures/, Physiology/Resp_Volts.txt),
# NOT data/processed/bp4d_canonical.
if [ -z "${RAW_DATA_PATH:-}" ] || [ ! -d "$RAW_DATA_PATH" ]; then
    echo "ERROR: RAW_DATA_PATH is unset or not a directory ($RAW_DATA_PATH)." >&2
    echo "       Check RAW_DATA_PATH in scripts/env_hpc.sh." >&2
    exit 1
fi

OUT_DIR="${INSPECT_OUTPUT_DIR:-$OUTPUT_DIR/inspect_data/tir_resp}"
mkdir -p "$OUT_DIR" "$CODE_DIR/logs"

ARGS=(--raw_root "$RAW_DATA_PATH"
      --subject "${INSPECT_SUBJECT:-F001}"
      --task "${INSPECT_TASK:-T1}"
      --input_size "${INSPECT_SIZE:-112}"
      --roi_padding "${INSPECT_ROI_PADDING:-0.2}"
      --roi_landmarks "${INSPECT_LANDMARKS:-}"
      --min_signal_spread "${INSPECT_SPREAD:-0.01}"
      --output_dir "$OUT_DIR")
[ -n "${INSPECT_TASK_SET:-}" ] && ARGS+=(--task_set "$INSPECT_TASK_SET")
[ -n "${INSPECT_CLIPS:-}" ]    && ARGS+=(--clip_index "$INSPECT_CLIPS")
case "${INSPECT_LIST:-0}" in
    1|true|TRUE|yes|YES) ARGS+=(--list) ;;
esac
ARGS+=("$@")

echo "[inspect_tir_resp] raw_root : $RAW_DATA_PATH"
echo "[inspect_tir_resp] output   : $OUT_DIR"
echo "[inspect_tir_resp] command  : python -u runners/run_inspect_tir_resp.py ${ARGS[*]}"
echo

python -u runners/run_inspect_tir_resp.py "${ARGS[@]}"

echo
echo "Figures + JSON: $OUT_DIR/<subject>_<task>/"
echo "Index:          $OUT_DIR/tir_resp_index.json"
