#!/usr/bin/env bash
# Lichtenberg HPC: Stage-3 vs Stage-2 STEP-TIME A/B on ONE node.
#
# Answers: "is Stage-3 fine-tuning faster than Stage-2 pre-training?" with a
# MEASUREMENT. Both arms run the REAL runners for a bounded number of optimizer
# steps, back to back in the SAME job, so node / GPU / batch / world size / cache
# state are held constant -- only the training objective differs.
#
# WHY THE ANSWER IS NOT OBVIOUS. The tempting argument for "Stage 3 is faster"
# is (1) no 1-D physio stream is fed and (2) there is no random masking. Point
# (2) is BACKWARDS for the encoder: this is a masked autoencoder, and
# `MultiModalMAE.forward` GATHERS ONLY THE VISIBLE TOKENS before `enc_blocks`
# (core/multimae.py, "gather the VISIBLE tokens of every ENCODER stream only").
# At `mask_ratio_tir 0.90` the Stage-2 encoder therefore sees ~10 % of the 2450
# video tokens (~270 with resp), while Stage-3 feeds ALL 2450 unmasked. The
# encoder is 12 layers of d=768, so that difference dominates; the Stage-2
# decoder that masking ADDS is only 2 layers (dec_depth default 2) and is not
# obviously large enough to close the gap. Stage-3 additionally pays a
# per-epoch VALIDATION pass that Stage-2 pre-training does not have.
#
# DO NOT hand-roll the step: a custom harness misses DDP, the masking generator,
# the AMP scaler and the real dataset. Both arms here are the production entry
# points, so the numbers transfer to real runs.
#
# COMPARABILITY RULES (both arms share all of them):
#   * same GPU type/count, same node, same hour;
#   * BATCH_SIZE per rank, hence the same global batch;
#   * WORLD ranks, so the collectives are paid the same way;
#   * ENTRIES, so both measure the same number of optimizer steps;
#   * ROI cache ON for both (the data path must not be the variable);
#   * same clip geometry: input_size 112 / 8 s clip / stride 2.0 / tubelet 2,16,16.
# NOTE both configs now use `clip_stride: 2.0`, so an epoch covers the SAME
# clip windows in both stages -- the per-epoch comparison is meaningful.
#
# Submit from code/:
#   GPU_TYPE=rtx6000 GPUS=4 ALLOW_MULTI_GPU=1 \
#       bash scripts/hpc/submit.sh scripts/hpc/bench_stage3_step.sbatch
#
# Env overrides: ENTRIES (6400 = 100 steps at batch 16 x 4 ranks), BATCH_SIZE,
# WORKERS, FINETUNE, CACHE_ROOT, OUT_ROOT, CONFIG_S2, CONFIG_S3.
#SBATCH --job-name=bench_s3_vs_s2
#SBATCH --partition=acc_short         # GPU; 30 min max ('acc' = 1 day)
# NB: the LUA plugin rejects --mem, so memory is --mem-per-cpu * cpus-per-task.
# GPU type pinned: the partitions mix a100/h100/rtx6000 with v100 / Intel PVC /
# AMD MI300X and this repo's torch is CUDA-only. Override with SBATCH_GRES.
#SBATCH --gres=gpu:rtx6000:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8             # == num_workers (8) + the main process
#SBATCH --mem-per-cpu=4G              # 4 x 8 x 4G = 128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/bench_s3_%j.out
#SBATCH --error=logs/bench_s3_%j.err

set -euo pipefail
# sbatch COPIES this script into the spool dir and runs that copy, so $0 is
# /var/spool/slurmd/job<id>/slurm_script -- NOT the repo path. Resolve the
# profile from $SLURM_SUBMIT_DIR (submit from code/).
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

CONFIG_S2="${CONFIG_S2:-configs/pretrain/stage2_hpc_tir_roi_resp.yaml}"
CONFIG_S3="${CONFIG_S3:-configs/finetune/resp_tir_roi_hpc.yaml}"
BATCH_SIZE="${BATCH_SIZE:-16}"        # per rank; both configs ship 16
WORKERS="${WORKERS:-8}"
ENTRIES="${ENTRIES:-6400}"            # 6400 / (16 x 4 ranks) = 100 optimizer steps
STAGE2_RUN="${STAGE2_RUN:-${WORK_SCRATCH:-$OUTPUT_DIR}/pretrain/stage2_hpc_tir_roi_resp}"
CACHE_ROOT="${CACHE_ROOT:-${WORK_SCRATCH:-$OUTPUT_DIR}/tir_roi_cache}"
# Checkpoints are ~1.2-1.3 GB and BOTH arms write one (epoch 0 % save_ckpt_freq
# == 0), so this MUST be a real file system -- node-local /tmp is too small.
OUT_ROOT="${OUT_ROOT:-${WORK_SCRATCH:-${LOCAL_TMP:-/tmp}}}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/kismed_bench/s3_vs_s2_${SLURM_JOB_ID:-manual}}"

# A DDP run measures a DIFFERENT quantity than a 1-GPU one, so requiring 4 ranks
# has to be DELIBERATE -- a mis-aimed --gres would silently produce a number that
# cannot be compared to the production Stage-2 rate (0.3224 s/it at 4 ranks).
if [ "${SLURM_NTASKS:-1}" != "${ALLOWED_RANKS:-4}" ] && [ "${ALLOW_MULTI_GPU:-0}" != "1" ]; then
    echo "ERROR: this A/B expects ${ALLOWED_RANKS:-4} ranks (SLURM_NTASKS=${SLURM_NTASKS})." >&2
    echo "       Pass ALLOW_MULTI_GPU=1 to run it anyway." >&2
    exit 1
fi

for f in "$CONFIG_S2" "$CONFIG_S3"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: config not found: $f (run from $CODE_DIR)" >&2
        exit 1
    fi
done
if [ ! -d "$CACHE_ROOT" ]; then
    echo "ERROR: ROI_CACHE=$CACHE_ROOT does not exist (both arms must use it)." >&2
    exit 1
fi
if [ -z "${FINETUNE:-}" ]; then
    FINETUNE="$(ls -1 "$STAGE2_RUN"/checkpoints/checkpoint-*.pth 2>/dev/null \
                | sort -V | tail -n 1 || true)"
fi
if [ -z "${FINETUNE:-}" ] || [ ! -f "$FINETUNE" ]; then
    echo "ERROR: no Stage-2 checkpoint for the Stage-3 arm (got '${FINETUNE:-<unset>}')." >&2
    exit 1
fi

mkdir -p "$OUT_DIR" "$CODE_DIR/logs"

export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostname "$SLURM_JOB_NODELIST" | sed -n '1p')}"
# Below the ephemeral range (32768-60999) so the kernel never steals the port.
export MASTER_PORT="${MASTER_PORT:-$((20000 + (SLURM_JOB_ID * 17) % 10000))}"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed -n '1p')"
[ -n "$GPU_NAME" ] || GPU_NAME=unknown

echo "[bench_ab] node     : $(hostname)"
echo "[bench_ab] gpu      : $GPU_NAME"
echo "[bench_ab] gres     : $(scontrol show job -o "$SLURM_JOB_ID" 2>/dev/null | grep -oE 'gres/gpu[a-z0-9/]*=[0-9]+' | tr '\n' ' ')"
echo "[bench_ab] world    : ${SLURM_NTASKS} rank(s), ${SLURM_CPUS_PER_TASK:-?} cpu(s) each"
echo "[bench_ab] batch    : $BATCH_SIZE per rank -> $((BATCH_SIZE * SLURM_NTASKS)) global"
echo "[bench_ab] entries  : $ENTRIES -> ~$((ENTRIES / (BATCH_SIZE * SLURM_NTASKS))) optimizer steps/rank"
echo "[bench_ab] cache    : $CACHE_ROOT"
echo "[bench_ab] finetune : $FINETUNE"
echo "[bench_ab] output   : $OUT_DIR"
echo

# --warmup_epochs 0 is REQUIRED for a 1-epoch bench. Both configs ship a warmup
# (Stage-2 10, Stage-3 5) and cosine_scheduler asserts
# `len(schedule) == epochs * niter_per_ep`; with warmup_epochs >= epochs the
# schedule is entirely warmup, so it ABORTS BEFORE any step runs and the job
# "completes" in ~50 s with no timings at all. Observed 2026-10-09: Stage-2 died
# with "1000 != 100", Stage-3 with "--warmup_epochs (5) must be < --epochs (1)".
# Warmup changes only the LR VALUES, never the step cost, so this does not
# affect the measured s/it.
ARGS_COMMON=(--num_workers "$WORKERS"
             --batch_size "$BATCH_SIZE"
             --max_entries "$ENTRIES"
             --epochs 1
             --warmup_epochs 0
             --roi_cache "$CACHE_ROOT")

run_arm () {           # run_arm <label> <runner.py> <config> <outdir> [extra...]
    local label="$1" runner="$2" cfg="$3" out="$4"; shift 4
    local log="$OUT_DIR/${label}.log"
    mkdir -p "$out"
    echo "=================================================================="
    echo "[bench_ab] ARM $label -> $runner -c $cfg"
    echo "=================================================================="
    set +e
    srun --ntasks="$SLURM_NTASKS" --ntasks-per-node="$SLURM_NTASKS_PER_NODE" \
        python -u "runners/$runner" -c "$cfg" \
            --output_dir "$out" "${ARGS_COMMON[@]}" "$@" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    echo "[bench_ab] ARM $label exit code: $rc"
    return "$rc"
}

# ---- arm 1: Stage-2 masked pre-training -------------------------------- #
run_arm stage2 run_pretrain.py "$CONFIG_S2" "$OUT_DIR/out_s2" \
        --raw_root "$RAW_DATA_PATH" || echo "[bench_ab] stage2 arm returned non-zero"
echo

# ---- arm 2: Stage-3 waveform fine-tuning (no masking) ------------------ #
run_arm stage3 run_waveform.py "$CONFIG_S3" "$OUT_DIR/out_s3" \
        --data_path "$RAW_DATA_PATH" --finetune "$FINETUNE" \
    || echo "[bench_ab] stage3 arm returned non-zero"

# ---- report ------------------------------------------------------------ #
# `time:` is the total s/it and `data:` the part spent WAITING for the loader,
# so the pure compute+comm cost is time - data. Both are windowed means from
# utils/logger.py; the Epoch line is the whole-epoch mean, which is what the
# production Stage-2 log reports (0.3224 s/it).
echo
echo "=================================================================="
echo "[bench_ab] ===== RESULT ($(hostname), $GPU_NAME) ====="
echo "=================================================================="
for label in stage2 stage3; do
    log="$OUT_DIR/${label}.log"
    [ -f "$log" ] || { printf '%-8s <no log>\n' "$label"; continue; }
    line=$(grep -E '^Epoch: \[0\] Total time' "$log" | tail -1 || true)
    last=$(grep -E '^Epoch: \[0\]  \[' "$log" | tail -1 || true)
    printf '%-8s epoch: %s\n' "$label" "${line:-<none>}"
    printf '%-8s last : %s\n' "$label" "${last:-<none>}"
done
echo
echo "[bench_ab] cache line (both arms must show it):"
grep -h '\[data\] roi_cache' "$OUT_DIR"/stage2.log "$OUT_DIR"/stage3.log 2>/dev/null || true
echo "[bench_ab] logs: $OUT_DIR"
