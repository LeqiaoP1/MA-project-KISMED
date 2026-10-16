#!/usr/bin/env bash
# Lichtenberg HPC: AU-occurrence probe over the Stage-2 TIR-ROI + Resp encoder.
#
# Protocol of record: analysis/tir_resp/AU_OccurenceProbe.md.
# Diagnostic only (no Stage-1/2/3 path is touched). Single process on purpose:
# the AU engine's DDP path is code-correct but not yet exercised on-GPU, and the
# linear probe is forward-bound, so one GPU is plenty.
#
# PREREQUISITES:
#   1) a finished Stage-2 TIR-ROI run -- scripts/hpc/submit_pretrain.sbatch
#   2) mkdir -p logs        (sbatch resolves --output relative to the submit dir)
#   3) the RAW tree (Thermal/ + IRFeatures/ + Physiology/ + AUCoding/AU_OCC)
#
# Submit from code/:
#   sbatch scripts/hpc/submit_au_probe.sbatch
#   FINETUNE=/work/scratch/$USER/pretrain/stage2_hpc_tir_roi_resp/checkpoints/checkpoint-0149.pth \
#       sbatch scripts/hpc/submit_au_probe.sbatch
#   SMOKE=1 sbatch scripts/hpc/submit_au_probe.sbatch                # capped run
#   TEST_SUBJECT=F004 TUNE=1 sbatch scripts/hpc/submit_au_probe.sbatch
#
# Env overrides:
#   CONFIG      configs/finetune/au_tir_roi_hpc.yaml
#   FINETUNE    Stage-2 checkpoint. Unset -> newest checkpoint-*.pth under STAGE2_DIR.
#   STAGE2_DIR  Stage-2 run dir (default: $OUT_ROOT/pretrain/stage2_hpc_tir_roi_resp)
#   AU_ROOT     AU_OCC dir (default: $RAW_DATA_PATH/AUCoding/AU_OCC)
#   ROI_CACHE   offline TIR-ROI cache OUT_ROOT (default: $WORK_SCRATCH/tir_roi_cache);
#               same ROI contract as Stage 2, so the existing cache is reused.
#               ROI_CACHE='' disables it.
#   RUN_NAME / OUT_ROOT   output dir controls (default: $WORK_SCRATCH)
#   SMOKE=1     add dev caps (--max_entries 256 --max_sessions 4)
#   TEST_SUBJECT / TUNE   leakage-free threshold tuning (protocol §10.1)
#   Anything after the script name is forwarded to the runner verbatim.
#SBATCH --job-name=au_probe_tir
#SBATCH --partition=acc               # GPU partition
# NB: the LUA plugin rejects --mem, so memory is --mem-per-cpu * cpus-per-task.
# --account is omitted on purpose: the plugin bills your default project.
# ONE GPU, ONE task: the probe runs single-process (see the header).
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8            # covers num_workers (8) + the trainer
#SBATCH --mem-per-cpu=8G             # 1 x 8 x 8G = 64G
# Keep the wall time TIGHT: sched/backfill only slips a job into a SHORT free
# window, so a 12 h limit can sit pending for days (seen 2026-10-14). The linear
# probe runs in minutes; override per run, e.g. `sbatch --time=00:30:00 ...`.
#SBATCH --time=04:00:00
#SBATCH --output=logs/au_probe_tir_%j.out
#SBATCH --error=logs/au_probe_tir_%j.err

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

# Keep the weight cache in lockstep with submit_pretrain.sbatch / download_weights.sh.
export INITIAL_MODELS_DIR="${WEIGHTS_DIR:-$PROJ_DIR/models/initial}"

CONFIG="${CONFIG:-configs/finetune/au_tir_roi_hpc.yaml}"
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
    echo "       The thermal-ROI branch needs the RAW tree. Check env_hpc.sh." >&2
    exit 1
fi
AU_ROOT="${AU_ROOT:-$RAW_DATA_PATH/AUCoding/AU_OCC}"
if [ ! -d "$AU_ROOT" ]; then
    echo "ERROR: AU coding dir not found: $AU_ROOT" >&2
    echo "       Set AU_ROOT=... or check RAW_DATA_PATH in env_hpc.sh." >&2
    exit 1
fi

# ---- resolve the Stage-2 checkpoint --------------------------------------- #
if [ -z "${FINETUNE:-}" ]; then
    FINETUNE="$(ls -1 "$STAGE2_DIR"/checkpoints/checkpoint-*.pth 2>/dev/null \
                | sort -V | tail -n 1 || true)"
    if [ -n "$FINETUNE" ]; then
        echo "[submit_au_probe] FINETUNE not set -> newest Stage-2 ckpt:"
        echo "                   $FINETUNE"
    fi
fi
if [ -z "${FINETUNE:-}" ] || [ ! -f "$FINETUNE" ]; then
    echo "ERROR: no Stage-2 checkpoint resolved." >&2
    echo "       Looked in : ${STAGE2_DIR}/checkpoints/" >&2
    echo "       Got       : '${FINETUNE:-<unset>}'" >&2
    echo "       Run scripts/hpc/submit_pretrain.sbatch first, or pass" >&2
    echo "           FINETUNE=/path/to/checkpoint-NNNN.pth sbatch $0" >&2
    exit 1
fi

# ---- runner args (config -> defaults; explicit flags override) ------------- #
# --raw_root / --au_root MUST be the RAW tree (the config's relative paths are
# for the local repo layout and do not exist on the cluster).
ARGS=(python -u runners/run_au_probe.py -c "$CONFIG"
      --raw_root "$RAW_DATA_PATH" --au_root "$AU_ROOT"
      --finetune "$FINETUNE" --output_dir "$OUT_DIR"
      --num_workers "${NUM_WORKERS:-8}")

# offline ROI cache (optional; config-independent, supplied by the env)
ROI_CACHE="${ROI_CACHE-$WORK_SCRATCH/tir_roi_cache}"
if [ -n "${ROI_CACHE:-}" ] && [ -d "$ROI_CACHE" ]; then
    ARGS+=(--roi_cache "$ROI_CACHE")
else
    echo "[submit_au_probe] ROI cache disabled (ROI_CACHE='${ROI_CACHE:-}')"
fi

# dev caps (small batch: 4 subjects yield only a handful of clips, and the
# default batch_size 16 with drop_last=True would give ZERO training batches)
if [ -n "${SMOKE:-}" ]; then
    ARGS+=(--max_entries 256 --max_sessions 4 --batch_size 4)
fi

# leakage-free per-AU threshold tuning (protocol §10.1)
if [ -n "${TEST_SUBJECT:-}" ]; then
    ARGS+=(--test_subjects "$TEST_SUBJECT")
fi
if [ -n "${TUNE:-}" ]; then
    ARGS+=(--tune_threshold)
fi

echo "[submit_au_probe] config   : $CONFIG"
echo "[submit_au_probe] raw_root : $RAW_DATA_PATH"
echo "[submit_au_probe] au_root  : $AU_ROOT"
echo "[submit_au_probe] ckpt     : $FINETUNE"
echo "[submit_au_probe] out_dir  : $OUT_DIR"
echo "[submit_au_probe] roi_cache: ${ROI_CACHE:-<off>}"
echo "[submit_au_probe] command  : ${ARGS[*]} $*"

mkdir -p "$OUT_DIR"
# Tee the run into the OUTPUT dir too: Slurm's own .out/.err occasionally go
# unreadable (stale lustre directory entries -- seen 2026-10-15), which would
# lose the F1 numbers. `set -o pipefail` (above) keeps the exit code honest.
"${ARGS[@]}" "$@" 2>&1 | tee "$OUT_DIR/train.log"
