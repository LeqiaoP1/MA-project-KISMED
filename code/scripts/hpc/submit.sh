#!/usr/bin/env bash
# Lichtenberg HPC: thin sbatch submitter that makes the GPU TYPE a plain env var.
#
# WHY THIS EXISTS. `#SBATCH --gres=...` lines are parsed by sbatch BEFORE the
# job script runs, and Slurm does NOT expand shell variables inside them, so a
# `#SBATCH --gres=gpu:${GPU_TYPE}:4` line is simply impossible. Slurm's own
# env-var override is SBATCH_GRES, which DOES take precedence over the in-script
# directive (verified on this cluster: SBATCH_GRES=gpu:a100:1 overrode
# '#SBATCH --gres=gpu:1'). This wrapper just maps the friendlier GPU_TYPE / GPUS
# names onto that flag.
#
# WHY PIN AT ALL. The `acc` / `acc_short` / `acc_long` partitions are
# HETEROGENEOUS: a100 40, h100 8, rtx6000 64, v100 12, mi300x 16, pvc128g 12
# GPUs. This repo's torch is CUDA-only (2.14.0+cu130), so an untyped
# `--gres=gpu:N` can be scheduled onto:
#   * an AMD MI300X (ROCm) or Intel PVC (XPU) node -> no CUDA device at all;
#   * a V100 -> CUDA 13 dropped Volta/sm_70, so the cu130 build cannot run it;
#   * and those are often the ONLY idle nodes, so it is a likely outcome.
# Usable types here: a100 (sm_80), h100 (sm_90), rtx6000.
#
# Usage (from code/):
#   bash scripts/hpc/submit.sh scripts/hpc/submit_pretrain.sbatch
#   GPU_TYPE=rtx6000 bash scripts/hpc/submit.sh scripts/hpc/submit_pretrain.sbatch
#   GPU_TYPE=rtx6000 GPUS=8 bash scripts/hpc/submit.sh scripts/hpc/submit_pretrain.sbatch
#   GPUS=1 bash scripts/hpc/submit.sh scripts/hpc/smoke_pretrain.sbatch
#
# GPUS IS THE DDP WORLD SIZE, not just a gres count. utils/dist.py derives
# world_size from SLURM_NTASKS, so this wrapper sets --gres, --ntasks and
# --ntasks-per-node TOGETHER (single node: every job script here is 1-node).
# Pass GPUS=1 for the smoke job -- its own directive is --ntasks=1.
#
# THE PER-JOB CAP IS NODES, NOT GPUs: partition `acc` sets MaxNodes=4 and has
# NO TRES/MaxTRES cap (QoS `normal` is unlimited, the account has no MaxTRES),
# so one job may ask for up to 4 nodes x 8 GPUs = 32 GPUs. Verified: a request
# for 4 nodes x rtx6000:8 was accepted. To spread across nodes pass
# SBATCH_NODELIST/--nodes via the job script instead of x8 on one node.
#
# ARG POSITION MATTERS. sbatch treats everything AFTER the script name as
# arguments to the JOB SCRIPT, not as options to itself -- so args you pass after
# the .sbatch path here reach the python runner, exactly as they would with a
# plain `sbatch`:
#   bash scripts/hpc/submit.sh scripts/hpc/smoke_pretrain.sbatch --epochs 1 --max_entries 4
# Any other SCHEDULING knob (partition, time, dependency, ...) goes through the
# native SBATCH_* environment variables, which is also how they can override the
# #SBATCH directives inside the job scripts:
#   SBATCH_TIMELIMIT=06:00:00 SBATCH_PARTITION=acc \
#       bash scripts/hpc/submit.sh scripts/hpc/submit_pretrain.sbatch
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: [GPU_TYPE=a100] [GPUS=4] bash scripts/hpc/submit.sh <job.sbatch> [job-script args...]" >&2
    echo "       types: a100 | h100 | rtx6000   (v100/mi300x/pvc128g are NOT usable: see the header)" >&2
    exit 2
fi

SCRIPT="$1"; shift
GPU_TYPE="${GPU_TYPE:-a100}"
GPUS="${GPUS:-4}"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: job script not found: $SCRIPT (run this from code/)" >&2
    exit 1
fi

case "$GPU_TYPE" in
    a100|h100|rtx6000) ;;
    v100|mi300x|pvc128g)
        echo "WARNING: '$GPU_TYPE' cannot run this CUDA-only build (see the header)." >&2
        ;;
    *)
        echo "WARNING: '$GPU_TYPE' is not one of the known acc types (a100|h100|rtx6000)." >&2
        ;;
esac

echo "[submit.sh] $SCRIPT -> 1 node, --gres=gpu:${GPU_TYPE}:${GPUS}, --ntasks=${GPUS}"
if [ "$#" -gt 0 ]; then
    echo "[submit.sh] forwarded to the job script: $*"
fi

# --gres / --ntasks / --ntasks-per-node are set TOGETHER on purpose: changing the
# GPU count without the task count would leave GPUs idle or bind ranks to the
# wrong devices (utils/dist.py takes world_size from SLURM_NTASKS).
# "$@" stays AFTER the script on purpose: sbatch hands it to the job script.
exec sbatch --nodes=1 \
    --gres="gpu:${GPU_TYPE}:${GPUS}" \
    --ntasks="${GPUS}" --ntasks-per-node="${GPUS}" \
    "$SCRIPT" "$@"
