#!/usr/bin/env bash
# Lichtenberg HPC: WHOLE-CORPUS quality survey for the TIR-ROI + RESP pipeline.
#
# WHAT IT DOES -- for EVERY <raw_root>/Thermal/<subject>/<task> session (1400 in
# the full tree) it reports the dataset's own usability gates (missing/invalid
# IRFeatures, missing Resp_Volts, resp too short, undecodable video, all clips
# dropped), the 8 s clip yield, and respiration-signal QUALITY: sample count,
# invalid samples, min/max/spread and the fraction of samples pinned at the
# -10.0000 V dead-channel rail -- per file AND per clip.
#
# CPU-ONLY, no GPU: numpy + an OpenCV *container probe* (frame count / fps);
# frames are never decoded. Do NOT add a --gres line.
#
# PREREQUISITES: mkdir -p logs   (in code/)
#
# Submit from code/:
#   sbatch scripts/hpc/submit_survey_tir_resp.sbatch
#   SURVEY_WORKERS=32 sbatch scripts/hpc/submit_survey_tir_resp.sbatch
#   SURVEY_LIMIT_SUBJECTS=8 sbatch scripts/hpc/submit_survey_tir_resp.sbatch
#
# Env-style overrides (propagated by sbatch's default --export=ALL):
#   SURVEY_WORKERS        parallel subject workers   default: cpus-per-task
#   SURVEY_OUTPUT_DIR     JSON/CSV destination       default: $WORK_SCRATCH/tir_resp_survey
#   SURVEY_SUBJECTS       comma list ('' = all)
#   SURVEY_LIMIT_SUBJECTS cap the subject count (0 = no cap)
#   SURVEY_TASK_GROUPS    group definitions (default = the tir_roi config's)
# Anything after the script name is forwarded to the runner verbatim.
#SBATCH --job-name=srv_tirresp
#SBATCH --partition=deflt             # CPU-only; 1 day max ('deflt_short' = 30 min)
# NB: the LUA plugin rejects --mem, so memory is --mem-per-cpu * cpus-per-task.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G              # 32 cpus x 2G = 64G total
#SBATCH --time=04:00:00
#SBATCH --output=logs/survey_tir_resp_%j.out
#SBATCH --error=logs/survey_tir_resp_%j.err

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

OUT_DIR="${SURVEY_OUTPUT_DIR:-${WORK_SCRATCH:-$OUTPUT_DIR}/tir_resp_survey}"
mkdir -p "$OUT_DIR" "$CODE_DIR/logs"

ARGS=(--raw_root "$RAW_DATA_PATH"
      --output_dir "$OUT_DIR"
      --workers "${SURVEY_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}")
[ -n "${SURVEY_SUBJECTS:-}" ]       && ARGS+=(--subjects "$SURVEY_SUBJECTS")
[ -n "${SURVEY_LIMIT_SUBJECTS:-}" ] && ARGS+=(--limit_subjects "$SURVEY_LIMIT_SUBJECTS")
[ -n "${SURVEY_TASK_GROUPS:-}" ]    && ARGS+=(--task_groups "$SURVEY_TASK_GROUPS")
ARGS+=("$@")

echo "[survey_tir_resp] raw_root : $RAW_DATA_PATH"
echo "[survey_tir_resp] output   : $OUT_DIR"
echo "[survey_tir_resp] command  : python -u runners/run_survey_tir_resp.py ${ARGS[*]}"
echo

python -u runners/run_survey_tir_resp.py "${ARGS[@]}"

echo
echo "Report: $OUT_DIR/tir_resp_survey.json + tir_resp_survey_sessions.csv"
