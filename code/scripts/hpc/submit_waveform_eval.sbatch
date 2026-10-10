#!/usr/bin/env bash
# Lichtenberg HPC: re-run the FINAL Stage-3 evaluation of an ALREADY-TRAINED
# run, SINGLE-PROCESS, to get a complete val-split prediction dump.
#
# WHY THIS EXISTS. run_waveform.py evaluates on a ``DistributedSampler``
# whenever it is launched with more than one task, and the final artefact dump
# (preds.npy / targets.npy / entries.json) is written by ``is_main_process()``
# alone -- no cross-rank gather. A 4-GPU run therefore left only rank 0's shard:
# job 55392202 reported "5143 dataset entries vs 1286 predictions" (= 5143/4),
# so both the dumped arrays and every printed val metric (pearson ~0.496,
# best_pearson 0.4988) covered 25% of the val split. runners/
# run_evaluate_session.py needs the WHOLE split to assemble per-session
# waveforms, so this job re-runs just the evaluation at world size 1.
#
# ``--eval_only`` loads the checkpoint out of "$OUT_DIR/checkpoints" (via
# auto_resume_model, i.e. latest_checkpoint.txt) and exits after the validation
# pass + dump; there is NO training and no LR schedule. It REFUSES to run above
# world size 1, which is the whole point.
#
# PREREQUISITES:
#   1) a finished Stage-3 run in $OUT_DIR (checkpoints/ + latest_checkpoint.txt)
#   2) mkdir -p logs        (sbatch resolves --output relative to the submit dir)
#
# Submit from code/ -- note GPUS=1:
#   GPU_TYPE=rtx6000 GPUS=1 bash scripts/hpc/submit.sh scripts/hpc/submit_waveform_eval.sbatch
#   OUT_DIR=/work/scratch/$USER/finetune/resp_tir_roi_hpc \
#       GPU_TYPE=rtx6000 GPUS=1 bash scripts/hpc/submit.sh scripts/hpc/submit_waveform_eval.sbatch
#
# Env overrides:
#   CONFIG     configs/finetune/resp_tir_roi_hpc.yaml (must MATCH the trained run)
#   OUT_DIR    the TRAINED run's dir (default $OUT_ROOT/finetune/$RUN_NAME)
#   FINETUNE   Stage-2 encoder checkpoint; only used to re-check the ROI contract
#   ROI_CACHE  offline TIR-ROI cache OUT_ROOT (default $WORK_SCRATCH/tir_roi_cache)
#   Anything after the script name is forwarded to the runner verbatim.
#SBATCH --job-name=st3_resp_eval
#SBATCH --partition=acc
# 1 GPU / 1 task ON PURPOSE: this is the single-process path. A 5143-clip val
# pass is ~5-15 min with the ROI cache attached, so the limit is margin, not
# need (a killed pass would just have to be re-run).
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/st3_resp_eval_%j.out
#SBATCH --error=logs/st3_resp_eval_%j.err

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

export INITIAL_MODELS_DIR="${WEIGHTS_DIR:-$PROJ_DIR/models/initial}"

CONFIG="${CONFIG:-configs/finetune/resp_tir_roi_hpc.yaml}"
RUN_NAME="${RUN_NAME:-$(basename "${CONFIG%.yaml}")}"
OUT_ROOT="${OUT_ROOT:-${WORK_SCRATCH:-$OUTPUT_DIR}}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/finetune/$RUN_NAME}"
STAGE2_DIR="${STAGE2_DIR:-$OUT_ROOT/pretrain/stage2_hpc_tir_roi_resp}"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG (run from $CODE_DIR)" >&2
    exit 1
fi
if [ -z "${RAW_DATA_PATH:-}" ] || [ ! -d "$RAW_DATA_PATH" ]; then
    echo "ERROR: RAW_DATA_PATH is unset or not a directory ($RAW_DATA_PATH)." >&2
    echo "       The thermal-ROI branch needs the RAW tree, not the canonical one." >&2
    exit 1
fi
# --eval_only reads its weights from OUT_DIR/checkpoints, so THAT is the hard
# prerequisite. Fail here rather than after model construction.
if [ ! -f "$OUT_DIR/checkpoints/latest_checkpoint.txt" ]; then
    echo "ERROR: $OUT_DIR/checkpoints/latest_checkpoint.txt not found." >&2
    echo "       --eval_only evaluates the checkpoint of an ALREADY-TRAINED run;" >&2
    echo "       point OUT_DIR at the run directory (or check RUN_NAME/CONFIG)." >&2
    exit 1
fi
echo "[submit_waveform_eval] resume    : $(cat "$OUT_DIR/checkpoints/latest_checkpoint.txt")"

# Optional: only for the Stage-2 ROI-contract re-check. --eval_only does not need
# it (auto_resume_model restores the full Stage-3 state dict), so a missing
# Stage-2 checkpoint is a warning, never a failure.
FINETUNE="${FINETUNE:-}"
if [ -z "$FINETUNE" ]; then
    FINETUNE="$(ls -1 "$STAGE2_DIR"/checkpoints/checkpoint-*.pth 2>/dev/null \
                | sort -V | tail -n 1 || true)"
fi
if [ -z "$FINETUNE" ] || [ ! -f "$FINETUNE" ]; then
    echo "[submit_waveform_eval] NOTE: no Stage-2 checkpoint resolved -- the ROI"
    echo "                       contract re-check will be SKIPPED (weights still"
    echo "                       come from $OUT_DIR/checkpoints)."
    FINETUNE=""
fi

ROI_CACHE="${ROI_CACHE-${WORK_SCRATCH:-$OUTPUT_DIR}/tir_roi_cache}"
if [ -n "$ROI_CACHE" ] && [ ! -d "$ROI_CACHE" ]; then
    echo "ERROR: ROI_CACHE=$ROI_CACHE does not exist (use ROI_CACHE='' to disable)." >&2
    exit 1
fi

ARGS=(-c "$CONFIG"
      --data_path "$RAW_DATA_PATH"
      --output_dir "$OUT_DIR"
      --eval_only
      --save_preds)
[ -n "$FINETUNE" ]       && ARGS+=(--finetune "$FINETUNE")
[ -n "${VAL_SUBJECT:-}" ] && ARGS+=(--val_subject "$VAL_SUBJECT")
[ -n "${TASK_SET:-}" ]    && ARGS+=(--task_set "$TASK_SET")
[ -n "${TASKS:-}" ]       && ARGS+=(--tasks "$TASKS")
[ -n "${BATCH_SIZE:-}" ]  && ARGS+=(--batch_size "$BATCH_SIZE")
[ -n "${NUM_WORKERS:-}" ] && ARGS+=(--num_workers "$NUM_WORKERS")
[ -n "$ROI_CACHE" ]      && ARGS+=(--roi_cache "$ROI_CACHE")
ARGS+=("$@")

if [ "${SLURM_NTASKS:-1}" != "1" ]; then
    echo "ERROR: this job needs --ntasks=1 (got ${SLURM_NTASKS}). Submit with" >&2
    echo "       GPU_TYPE=rtx6000 GPUS=1 bash scripts/hpc/submit.sh $0" >&2
    exit 1
fi

mkdir -p "$CODE_DIR/logs"

# Job-derived port, kept below the ephemeral range (32768-60999): Slurm packs
# 1-GPU jobs together, and the default 29500 would collide between two of them
# on one node (NCCL bind failure). See submit_waveform_tir_roi.sbatch.
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostname "$SLURM_JOB_NODELIST" | sed -n '1p')}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + (SLURM_JOB_ID * 17) % 10000))}"

echo "[submit_waveform_eval] config   : $CONFIG"
echo "[submit_waveform_eval] raw_root : $RAW_DATA_PATH"
echo "[submit_waveform_eval] finetune : ${FINETUNE:-<none: ROI check skipped>}"
echo "[submit_waveform_eval] roi_cache: ${ROI_CACHE:-<disabled: decode .wmv>}"
echo "[submit_waveform_eval] output   : $OUT_DIR"
echo "[submit_waveform_eval] world    : ${SLURM_NTASKS} task(s) on $(hostname)"
echo "[submit_waveform_eval] gres     : $(scontrol show job -o "$SLURM_JOB_ID" 2>/dev/null | grep -oE 'gres/gpu[a-z0-9/]*=[0-9]+' | tr '\n' ' ')"
echo "[submit_waveform_eval] command  : python -u runners/run_waveform.py ${ARGS[*]}"
echo

# NO srun: python is launched directly so this stays a single process. Slurm
# still exports SLURM_PROCID for a 1-task step, which utils/dist.py turns into a
# world_size-1 process group -- DistributedSampler(num_replicas=1) is then the
# identity, so the ENTIRE val split is evaluated and dumped.
python -u runners/run_waveform.py "${ARGS[@]}"

echo "Stage-3 (resp / tir_roi) EVAL finished -> $OUT_DIR"
