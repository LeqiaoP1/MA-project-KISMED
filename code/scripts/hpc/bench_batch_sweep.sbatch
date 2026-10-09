#!/usr/bin/env bash
# Lichtenberg HPC: BATCH-SIZE sweep for the Stage-2 TIR-ROI loop, with the
# offline ROI cache ON (and one cache-OFF reference arm).
#
# QUESTION. Does a larger batch reduce training time? The answer depends on
# whether the loop is GPU-bound or overhead-bound, and that is measurable:
#
#   * GPU-bound   -> s/step scales ~linearly with batch -> total time is FLAT
#                    (the same 21,231 clips/epoch cost the same whatever the
#                    grouping), so a bigger batch buys ~nothing.
#   * overhead-bound -> s/step is sub-linear -> fewer optimizer steps and fewer
#                    DDP all-reduce rounds per epoch, so there IS a real win.
#
# The knee of that curve is the answer. The sweep also reports PEAK GPU MEMORY,
# which settles whether the larger GRAM is even in play (the pre-cache evidence
# says it is not: the GPU step was 0.20 s inside a 2.45 s iteration).
#
# WHY 4 RANKS. DDP with 4 ranks is the production configuration, so the
# collectives are paid exactly as in the real run. A 1-GPU arm has no allreduce
# and would UNDERSTATE the benefit of a larger batch.
#
# WHY THE CACHE. It removes the ~1.3-2.3 s/iter of decode that previously hid the
# GPU cost; measuring the batch curve on the OLD (decode-bound) loop would just
# re-measure the decoder. Arm 5 keeps one cache-OFF reference at batch 16 so the
# cache's own contribution stays visible next to the published 2.54 s/it.
#
# STEPS ARE FIXED, NOT ENTRIES: every arm runs the same number of optimizer
# steps, so s/step is directly comparable and every arm pays the same warm-up.
# ENTRIES is derived as STEPS x BATCH x RANKS.
#
# Submit from code/:
#   GPU_TYPE=rtx6000 GPUS=4 ALLOW_MULTI_GPU=1 \
#       bash scripts/hpc/submit.sh scripts/hpc/bench_batch_sweep.sbatch
#
# Env overrides: STEPS (30), WORKERS (8), CACHE_ROOT, CONFIG, SWEEP_ROOT, THREADS.
#SBATCH --job-name=bsweep
#SBATCH --partition=acc_short         # GPU; 30 min max ('acc' = 1 day)
# NB: the LUA plugin rejects --mem, so memory is --mem-per-cpu * cpus-per-task.
#SBATCH --gres=gpu:rtx6000:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8            # 4 ranks x 8 = 32 CPUs on the node
#SBATCH --mem-per-cpu=8G             # 32 x 8G = 256G (the cache LRU is per rank)
#SBATCH --time=00:30:00
#SBATCH --array=0-5
#SBATCH --output=logs/bsweep_%A_%a.out
#SBATCH --error=logs/bsweep_%A_%a.err

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

CONFIG="${CONFIG:-configs/pretrain/stage2_hpc_tir_roi_resp.yaml}"
STEPS="${STEPS:-30}"
WORKERS="${WORKERS:-8}"
RANKS="${SLURM_NTASKS:-4}"
CACHE_ROOT="${CACHE_ROOT:-${WORK_SCRATCH:-$OUTPUT_DIR}/tir_roi_cache}"
SWEEP_ROOT="${SWEEP_ROOT:-${WORK_SCRATCH:-$OUTPUT_DIR}/batch_sweep}"

# ---- the arms ------------------------------------------------------------ #
BATCHES=(8 16 32 64 128 16)
CACHES=(on on on on on off)          # arm 5 = the cache-OFF reference
IDX="${SLURM_ARRAY_TASK_ID:-0}"
BATCH="${BATCHES[$IDX]}"
CACHE="${CACHES[$IDX]}"
ENTRIES=$((STEPS * BATCH * RANKS))   # every arm runs the same STEP count

OUT_DIR="$SWEEP_ROOT/arm_${IDX}_b${BATCH}_cache${CACHE}"
mkdir -p "$OUT_DIR" "$CODE_DIR/logs"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG (run from $CODE_DIR)" >&2
    exit 1
fi
if [ "$CACHE" = "on" ] && [ ! -d "$CACHE_ROOT" ]; then
    echo "ERROR: CACHE_ROOT not found: $CACHE_ROOT (build it with submit_roi_cache.sbatch)" >&2
    exit 1
fi

# --warmup_epochs 0 is REQUIRED (epochs < warmup trips the cosine assert).
# --epochs 1 still writes ONE checkpoint (epoch % freq == 0 matches epoch 0),
# so OUT must have ~1.2 GB free; that is why SWEEP_ROOT defaults to scratch.
ARGS=(-c "$CONFIG"
      --raw_root "$RAW_DATA_PATH"
      --output_dir "$OUT_DIR"
      --epochs 1 --warmup_epochs 0
      --save_ckpt_freq 100
      --batch_size "$BATCH"
      --num_workers "$WORKERS"
      --max_entries "$ENTRIES")
[ "$CACHE" = "on" ] && ARGS+=(--roi_cache "$CACHE_ROOT")
ARGS+=("$@")

export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)}"
# MASTER_PORT MUST be unique per RUNNING element. Array elements share a node
# (2 x 4 GPUs on an 8-GPU box), and a hardcoded port makes the second one fail to
# bind -> the others die in `barrier()` with
#   ncclOsSocketPollConnect: connect to <ip> returned Connection refused
# That killed arms 2 and 4 of the 2026-10-09 sweep BEFORE any training ran, so
# their peak-memory readings (795 MiB) were meaningless. Derive it per element.
# The modulus keeps the range at 20000-29999, i.e. BELOW the Linux ephemeral
# range (32768-60999), so the kernel can never hand this port to an outbound
# connection while the job holds it.
export MASTER_PORT="${MASTER_PORT:-$((20000 + (SLURM_JOB_ID * 17 + ${SLURM_ARRAY_TASK_ID:-0}) % 10000))}"

if [ -n "${THREADS:-}" ]; then
    export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
           OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
fi

# NOTE: do NOT write `nvidia-smi ... | head -1 || echo unknown`. `head` closes the
# pipe, nvidia-smi dies of SIGPIPE (nonzero), so the `||` fallback ALSO runs and
# the variable ends up as "<value>\nunknown" -- which is what injected a stray
# line into meta.txt. `sed -n 1p` drains the input, so there is no SIGPIPE and the
# fallback stays a separate statement.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed -n '1p')"
[ -n "$GPU_NAME" ] || GPU_NAME=unknown
GPU_MEM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sed -n '1p')"
[ -n "$GPU_MEM" ] || GPU_MEM=0
echo "[bsweep] arm      : $IDX  batch=$BATCH  cache=$CACHE"
echo "[bsweep] node     : $(hostname)   gpu: ${GPU_NAME:-unknown} (${GPU_MEM:-?} MiB)"
echo "[bsweep] world    : $RANKS rank(s) x ${SLURM_CPUS_PER_TASK:-?} cpu(s)"
echo "[bsweep] schedule : $STEPS steps -> $ENTRIES entries total, $WORKERS worker(s)/rank"
echo "[bsweep] out      : $OUT_DIR"
echo

# Peak GPU memory: sample every 2 s and keep the maximum across all visible GPUs.
# A single nvidia-smi reading is timing-dependent and would miss the peak.
( while :; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | sort -n | tail -1
      sleep 2
  done ) > "$OUT_DIR/mem_samples.txt" &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null || true' EXIT

set +e
srun --ntasks="$RANKS" --ntasks-per-node="$RANKS" \
    python -u runners/run_pretrain.py "${ARGS[@]}" 2>&1 | tee "$OUT_DIR/run.log"
PY_RC=${PIPESTATUS[0]}
set -e

kill "$SAMPLER" 2>/dev/null || true
wait "$SAMPLER" 2>/dev/null || true

PEAK_MEM_MIB="$(awk 'max < $1 {max = $1} END {print max + 0}' "$OUT_DIR/mem_samples.txt" 2>/dev/null || echo 0)"
{
    echo "arm=$IDX"
    echo "batch=$BATCH"
    echo "cache=$CACHE"
    echo "ranks=$RANKS"
    echo "steps=$STEPS"
    echo "entries=$ENTRIES"
    echo "workers=$WORKERS"
    echo "gpu=$GPU_NAME"
    echo "gpu_total_mib=$GPU_MEM"
    echo "peak_gpu_mib=$PEAK_MEM_MIB"
    echo "python_rc=$PY_RC"
} > "$OUT_DIR/meta.txt"

echo
echo "[bsweep] ===== ARM $IDX RESULT (batch $BATCH, cache $CACHE) ====="
grep -E '^Epoch: \[0\]  \[' "$OUT_DIR/run.log" | tail -2 || true
echo "[bsweep] peak GPU memory used: ${PEAK_MEM_MIB} MiB of ${GPU_MEM} MiB"
echo "[bsweep] meta: $OUT_DIR/meta.txt"
echo "[bsweep] python exit code: $PY_RC"
exit "$PY_RC"
