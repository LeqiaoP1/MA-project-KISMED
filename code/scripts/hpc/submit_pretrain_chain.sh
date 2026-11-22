#!/usr/bin/env bash
# N-link CHAIN for a long Stage-2 run on the 30-minute queue.
#
# WHY THIS EXISTS. On this cluster `acc`/`acc_long` carry PriorityJobFactor=10
# while `acc_short` carries 100, so a multi-hour GPU job in `acc` practically
# never wins the main cycle against other tenants' 30-minute jobs (measured
# 2026-11-22: identical 2x A100 requests start in ~20 s on acc_short, and stayed
# PENDING for 13 h on acc_long). The way to get a long run done is therefore a
# CHAIN of 30-minute allocations that each resume from the previous link.
#
# Each link: runners/run_partition.py (write-once, idempotent) then
# runners/run_pretrain.py, which calls auto_resume_model() and starts at
# checkpoint_epoch + 1. `--dependency=afterany:<prev>` is what makes a walltime
# KILL (not just a clean exit) hand off to the next link.
#
# Full rationale (partitions, priority factors, dependency mechanics, why
# `afterany` and not `afterok`): notes/explanation/hpc_why_long_gpu_jobs_never_start.md
#
# PROGRESS GUARANTEE: a link can only hand work on if it wrote a checkpoint, so
# this script appends `--save_ckpt_freq` (default 5) unless the caller passes
# their own. A link covers ~24 epochs at 4x A100 (measured 66 s/epoch), so 5
# leaves ~4 checkpoints per link.
#
# Usage (from code/):
#   EXP_ROOT=$WORK_SCRATCH/fourstream_v1 FOLDS=0 \
#   ROI_CACHE=... RGB2D_CACHE=... RESP_CACHE=... BP_ARTEFACT=... \
#       bash scripts/hpc/submit_pretrain_chain.sh
#
# Knobs: LINKS (7) | TIME (00:30:00) | PARTITION (acc_short) | GPU_TYPE (a100)
#        GPUS (4) | MEM_PER_CPU (1G) | SAVE_FREQ (5) | DRY_RUN (0)
# `--epochs` must be IDENTICAL in every link: the LR schedule is rebuilt from it.
set -euo pipefail
cd "$(dirname "$0")/../.."

: "${EXP_ROOT:?set EXP_ROOT (e.g. EXP_ROOT=\$WORK_SCRATCH/fourstream_v1)}"
: "${FOLDS:?set FOLDS (e.g. FOLDS=0); the chain resumes one exp_root, so one fold}"

LINKS="${LINKS:-7}"
TIME="${TIME:-00:30:00}"
PARTITION="${PARTITION:-acc_short}"
GPU_TYPE="${GPU_TYPE:-a100}"
GPUS="${GPUS:-4}"
MEM_PER_CPU="${MEM_PER_CPU:-1G}"
SAVE_FREQ="${SAVE_FREQ:-5}"
DRY_RUN="${DRY_RUN:-0}"

# Progress guarantee: only add the flag when the caller did not set one.
EXTRA_ARGS=("$@")
case " ${EXTRA_ARGS[*]:-} " in
  *" --save_ckpt_freq "*) : ;;
  *) EXTRA_ARGS+=("--save_ckpt_freq" "$SAVE_FREQ") ;;
esac

echo "[chain] exp_root : $EXP_ROOT"
echo "[chain] folds    : $FOLDS (one chain per exp_root)"
echo "[chain] links    : $LINKS x $TIME on $PARTITION, ${GPUS}x${GPU_TYPE}"
echo "[chain] extra    : ${EXTRA_ARGS[*]:-<none>}"
echo "[chain] dry_run  : $DRY_RUN"

prev=""
for i in $(seq 1 "$LINKS"); do
  args=(--nodes=1
        --gres="gpu:${GPU_TYPE}:${GPUS}"
        --ntasks="$GPUS" --ntasks-per-node="$GPUS"
        --partition="$PARTITION" --time="$TIME"
        --mem-per-cpu="$MEM_PER_CPU"
        --parsable)
  if [ -n "$prev" ]; then
    args+=("--dependency=afterany:${prev}")
  fi

  cmd=(sbatch "${args[@]}" scripts/hpc/submit_pretrain_sweep.sbatch
       ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
  if [ "$DRY_RUN" = "1" ]; then
    printf '[chain] %q ' "${cmd[@]}"; printf '\n'
    prev="DRYRUN${i}"
    continue
  fi
  prev=$( "${cmd[@]}" | tail -1 | cut -d';' -f1 )
  echo "[chain] link $i/$LINKS -> job $prev"
done

echo "[chain] submitted $LINKS link(s); first link runs immediately, the rest"
echo "[chain] start when their predecessor ends (walltime kill included)."
