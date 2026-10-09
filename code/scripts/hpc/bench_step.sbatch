#!/usr/bin/env bash
# Lichtenberg HPC: single-GPU STEP-TIME micro-benchmark (GPU-type A/B).
#
# Purpose: answer "is Job X faster on a100 or rtx6000?" with a MEASUREMENT
# instead of a spec-sheet guess. It runs the REAL Stage-2 runner
# (runners/run_pretrain.py) with production geometry, just for a short epoch --
# so the number it prints (`time:` s/it) is directly comparable to the `time:`
# column of a full-corpus run (see logs/st2_pretrain_55368106.out: 2.54 s/it at
# batch 16/GPU, 4 x rtx6000).
#
# It deliberately does NOT re-implement the training step: a hand-rolled
# benchmark harness measures a different code path (no DDP wrapper, no masking
# generator, no AMP scaler, no real dataset) and its numbers do not transfer.
#
# COMPARABILITY RULES -- only `--gres` may differ between two submissions:
#   * same CONFIG (geometry is a contract: input_size 112 / clip 8 s / stride 2)
#   * same BATCH_SIZE (16, i.e. what one rank sees in the real 4-GPU run)
#   * same WORKERS (4, the value the real run was launched with)
#   * same ENTRIES, so both measure the same number of optimizer steps
# A 1-GPU job has NO allreduce, so its s/it is a FLOOR for the 4-GPU rate; the
# A/B delta is still valid because both sides pay the same absence.
#
# It writes under $WORK_SCRATCH by DEFAULT, because --epochs 1 still triggers ONE
# save_model call (epoch % save_ckpt_freq == 0 matches epoch 0) and node-local
# /tmp is too small for the ~1.2 GB checkpoint: both arms of the first run died
# inside torch.save with
#     RuntimeError: [enforce fail at inline_container.cc:672]
#         . unexpected pos 533554048 vs 533553940   (+ basic_ios::clear)
# i.e. the write was cut off at ~509 MiB. The 80 training steps had already
# completed and are unaffected -- only the post-training save aborted.
#
# Submit from code/ -- GPUS=1 MATTERS: submit.sh passes --gres as a sbatch
# COMMAND-LINE option, which beats SBATCH_GRES (verified: SBATCH_GRES=gpu:a100:1
# + submit.sh still produced gres/gpu:a100:4 from GPU_TYPE/GPUS defaults):
#   GPU_TYPE=a100   GPUS=1 bash scripts/hpc/submit.sh scripts/hpc/bench_step.sbatch
#   GPU_TYPE=rtx6000 GPUS=1 bash scripts/hpc/submit.sh scripts/hpc/bench_step.sbatch
# Or bypass the wrapper (SBATCH_GRES DOES beat the in-script directive):
#   SBATCH_GRES=gpu:h100:1 sbatch --gres=gpu:h100:1 scripts/hpc/bench_step.sbatch
#
# Env overrides: CONFIG, ENTRIES, BATCH_SIZE, WORKERS, OUT_ROOT, plus anything
# after the script name (forwarded to the runner verbatim).
#SBATCH --job-name=st2_bench
#SBATCH --partition=acc_short         # GPU; 30 min max ('acc' = 1 day)
# NB: the LUA plugin rejects --mem, so memory is --mem-per-cpu * cpus-per-task.
# --account is omitted on purpose: the plugin bills your default project.
# GPU type pinned: acc_short mixes a100/h100/rtx6000 with v100 / Intel PVC /
# AMD MI300X and this repo's torch is CUDA-only. Override with SBATCH_GRES.
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G              # 8 cpus x 4G = 32G total
#SBATCH --time=00:20:00
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err

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

# Keep the weight cache in lockstep with download_weights.sh / submit_pretrain.sbatch
# (env_hpc.sh points it outside the repo; see the note in download_weights.sh).
export INITIAL_MODELS_DIR="${WEIGHTS_DIR:-$PROJ_DIR/models/initial}"

CONFIG="${CONFIG:-configs/pretrain/stage2_hpc_tir_roi_resp.yaml}"
BATCH_SIZE="${BATCH_SIZE:-16}"        # what ONE rank sees in the real 4-GPU run
WORKERS="${WORKERS:-4}"               # the value the real run was launched with
ENTRIES="${ENTRIES:-1280}"            # 1280 / 16 = 80 optimizer steps
OUT_ROOT="${OUT_ROOT:-${WORK_SCRATCH:-${LOCAL_TMP:-/tmp}}}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/kismed_bench/stage2_${SLURM_JOB_ID:-manual}}"
# The checkpoint write is 1.2 GB: warn BEFORE burning GPU time on a step whose
# result cannot be saved.
# !! POSIX df ONLY. `df -P` and `--output` are MUTUALLY EXCLUSIVE in coreutils
# ("df: options -P and --output are mutually exclusive", exit 1), and because
# this script runs under `set -o pipefail` with the df stderr sent to /dev/null,
# a failing `AVAIL_MB=$(df ...)` aborts the WHOLE script with no message at all.
# That is how the first scaling run died: 3 jobs, 0-3 s wall, one line of
# output, EMPTY stderr. A space check is a diagnostic -- it must degrade to
# "unknown", never become the reason a job fails before it starts.
CKPT_MB=1400
AVAIL_MB=$(df -Pm "$OUT_ROOT" 2>/dev/null | awk 'NR==2 {print $4}') || AVAIL_MB=""
case "$AVAIL_MB" in ''|*[!0-9]*) AVAIL_MB="" ;; esac
if [ -n "$AVAIL_MB" ] && [ "$AVAIL_MB" -lt "$CKPT_MB" ]; then
    echo "ERROR: $OUT_ROOT has only ${AVAIL_MB} MiB free; the checkpoint needs ~${CKPT_MB} MiB." >&2
    echo "       Set OUT_ROOT to a bigger file system (e.g. OUT_ROOT=\$WORK_SCRATCH)." >&2
    exit 1
fi

# A DDP run measures a DIFFERENT quantity (collectives amortised over N ranks),
# so a multi-GPU submission has to be DELIBERATE -- otherwise a mis-aimed
# --gres/SBATCH_GRES silently produces a number that cannot be compared to the
# other arm. Seen in practice: a `SBATCH_GRES=gpu:a100:1` request going through
# submit.sh became a silent 4-GPU submission. Pass ALLOW_MULTI_GPU=1 for a
# scaling run (GPUS=4) and always compare it against a GPUS=1 arm of the same
# build -- NOT against the full-corpus production number.
if [ "${SLURM_NTASKS:-1}" != "1" ] && [ "${ALLOW_MULTI_GPU:-0}" != "1" ]; then
    echo "ERROR: this micro-benchmark defaults to single-GPU, but SLURM_NTASKS=${SLURM_NTASKS}." >&2
    echo "       Single GPU : GPU_TYPE=<type> GPUS=1 bash scripts/hpc/submit.sh scripts/hpc/bench_step.sbatch" >&2
    echo "       Scaling arm: GPU_TYPE=<type> GPUS=4 ALLOW_MULTI_GPU=1 bash scripts/hpc/submit.sh scripts/hpc/bench_step.sbatch" >&2
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG (run from $CODE_DIR)" >&2
    exit 1
fi
if [ -z "$(ls -A "$INITIAL_MODELS_DIR" 2>/dev/null)" ]; then
    echo "ERROR: no Stage-1 weights in $INITIAL_MODELS_DIR." >&2
    echo "       Run this ON A LOGIN NODE first:" >&2
    echo "           cd $CODE_DIR && bash scripts/hpc/download_weights.sh" >&2
    exit 1
fi

# --warmup_epochs 0 is REQUIRED: epochs < warmup trips the pre-existing
# cosine_scheduler assert in utils/lr_sched.py.
# --save_ckpt_freq 100 + --epochs 1 still writes checkpoint-0000 (epoch % freq
# == 0 matches epoch 0), which is why OUT_ROOT must have room for ~1.2 GB.
ARGS=(-c "$CONFIG"
      --raw_root "$RAW_DATA_PATH"
      --output_dir "$OUT_DIR"
      --epochs 1 --warmup_epochs 0
      --save_ckpt_freq 100
      --batch_size "$BATCH_SIZE"
      --num_workers "$WORKERS"
      --max_entries "$ENTRIES")
ARGS+=("$@")

mkdir -p "$OUT_DIR" "$CODE_DIR/logs"

# one task per GPU; utils/dist.py reads the SLURM_* env to init DDP (same wiring
# as submit_pretrain.sbatch).
# NOTE the PORT is derived from the job id, not hardcoded: two concurrent jobs on
# the same node with the same port make the second fail to bind and the first die
# in barrier() with "NCCL ... Connection refused" (cost the 2026-10-09 batch sweep
# 2 of its 6 arms, before any training ran). The modulus keeps the range at
# 20000-29999, BELOW the ephemeral range (32768-60999), so the kernel never hands
# this port to an outbound connection.
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostname "$SLURM_JOB_NODELIST" | sed -n '1p')}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + (SLURM_JOB_ID * 17) % 10000))}"

# THREADS caps the per-rank compute threads. This is the lever the scaling arm
# exists to test: torch's intra-op pool defaults to the CPU count, so N ranks on
# one node can ask for N x cores threads and oversubscribe it -- the prime
# suspect for ~2.4 s/step in the 4-GPU production run vs 0.20 s/step at 1 GPU.
# The Linux pip wheels link OpenMP, so OMP_NUM_THREADS does resize torch's pool.
if [ -n "${THREADS:-}" ]; then
    export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
           OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed -n '1p')"
[ -n "$GPU_NAME" ] || GPU_NAME=unknown
echo "[bench_step] node     : $(hostname)"
echo "[bench_step] gpu      : ${GPU_NAME:-unknown}"
echo "[bench_step] gres     : $(scontrol show job -o "$SLURM_JOB_ID" 2>/dev/null | grep -oE 'gres/gpu[a-z0-9/]*=[0-9]+' | tr '\n' ' ')"
echo "[bench_step] world    : ${SLURM_NTASKS:-1} rank(s), ${SLURM_CPUS_PER_TASK:-?} cpu(s) each"
echo "[bench_step] config   : $CONFIG"
echo "[bench_step] raw_root : $RAW_DATA_PATH"
echo "[bench_step] output   : $OUT_DIR"
echo "[bench_step] steps    : $((ENTRIES / BATCH_SIZE)) batch(es) at batch $BATCH_SIZE, $WORKERS worker(s)/rank"
echo "[bench_step] disk     : ${AVAIL_MB:-?} MiB free on $OUT_ROOT"
# What a RANK actually sees: if this prints ~128 threads while 4 ranks share the
# node, the threads are oversubscribed and THREADS=<cpus-per-task> is the fix.
python -c "import os, torch; print('[bench_step] rank visibility: cpus=%d torch_threads=%d OMP_NUM_THREADS=%s' % (len(os.sched_getaffinity(0)), torch.get_num_threads(), os.environ.get('OMP_NUM_THREADS', 'unset')))" 2>/dev/null || true
echo "[bench_step] command  : srun --ntasks=${SLURM_NTASKS:-1} python -u runners/run_pretrain.py ${ARGS[*]}"
echo

# tee so the timing lines can be re-printed in a parseable RESULT block without
# racing Slurm's own stdout redirection. The exit status is captured rather than
# propagated immediately, so a failure AFTER training (e.g. the checkpoint write)
# still prints the measured numbers -- the first run's save error swallowed its
# own RESULT block and made a successful 80-step measurement look like a crash.
set +e
srun --ntasks="${SLURM_NTASKS:-1}" --ntasks-per-node="${SLURM_NTASKS_PER_NODE:-${SLURM_NTASKS:-1}}" \
    python -u runners/run_pretrain.py "${ARGS[@]}" 2>&1 | tee "$OUT_DIR/run.log"
PY_RC=${PIPESTATUS[0]}
set -e

echo
echo "[bench_step] ===== RESULT ($(hostname), ${GPU_NAME:-unknown}) ====="
# `time:` is total s/it and `data:` is the s/it spent WAITING for the loader, so
# the pure step cost is time - data. Both are 10-iteration windowed means, and
# the last window before the epoch end is the steady-state one.
grep -E "^Epoch: \[0\]  \[ *(60|70|79|80|160)/|Total time|Averaged stats" "$OUT_DIR/run.log" | grep -vE "^\[rank" || true
echo "[bench_step] log: $OUT_DIR/run.log"
echo "[bench_step] python exit code: $PY_RC"
exit "$PY_RC"
