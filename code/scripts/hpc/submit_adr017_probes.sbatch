#!/usr/bin/env bash
# ADR-017 fold-0 zero-shot probe matrix (Gates G2 / G3 + remote-sensing controls).
#
# WHY A BATCH JOB: the zero-shot probes are GPU work. They must NOT run on a
# login node -- the `log*` hosts share ONE communal T4 with other tenants and
# kill terminal-attached processes on session teardown with no traceback
# (notes/spec/04_configs_and_execution.md §4.2.1; measured 2026-11-28: a
# login-node chain died after `preloaded 1742 clip(s) into host RAM`).
#
# Runs runners/run_eval_recon.py over the five probe conditions of
# code/notes/plan/experiments/exp_20261127_adr017_tir_resp_masked_span.md §8.3,
# sequentially in ONE allocation, each writing its own report + log:
#
#   respmax_masked_span   G2  --kept_tir 0.111 --kept_resp 0.40 masked_span
#   mid_masked_span       G2  --kept_tir 0.139 --kept_resp 0.30 masked_span
#   mode1                 G3  --kept_tir 1.0   --kept_resp 0.0  hidden
#   remote_sensing        TIR-900, budget-exempt
#   remote_sensing_tir_rolled   TIR-900 with the TIR stream rolled (content control)
#
# All conditions share --split test --repeats 5 --seed 0 --preload, so within
# an arm the masks are IDENTICAL across conditions. --interp_baseline adds the
# no-video np.interp / constant predictors on the SAME masks (the G0 floor).
#
# Env overrides:
#   CKPT      Stage-2 checkpoint   (default: the ADR-017 fold-0 checkpoint-0149)
#   OUT_DIR   results directory    (default: output/eval/exp_20261127_adr017_fold0)
#   REPEATS   mask draws per clip  (default: 5)
#   BATCH     eval batch size      (default: 16)
#
# Submit FROM code/:  sbatch scripts/hpc/submit_adr017_probes.sbatch
#SBATCH --job-name=adr017_probe
#SBATCH --partition=acc_short
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/adr017_probe_%j.out
#SBATCH --error=logs/adr017_probe_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/env_hpc.sh" ]; then
    ENV_SH="$SLURM_SUBMIT_DIR/scripts/env_hpc.sh"
else
    ENV_SH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env_hpc.sh"
fi
# shellcheck disable=SC1090
source "$ENV_SH"
activate_project_env
cd "$CODE_DIR"

mkdir -p logs
CKPT="${CKPT:-${WORK_SCRATCH:-$OUTPUT_DIR}/exp_20261127_adr017_tir_resp/fold0/stage2/checkpoints/checkpoint-0149.pth}"
OUT_DIR="${OUT_DIR:-${PROJ_DIR}/output/eval/exp_20261127_adr017_fold0}"
REPEATS="${REPEATS:-5}"
BATCH="${BATCH:-16}"
mkdir -p "$OUT_DIR"

: "${CKPT:?set CKPT}"
[ -f "$CKPT" ] || { echo "[adr017] checkpoint not found: $CKPT" >&2; exit 1; }

echo "[adr017] job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "[adr017] python: $(command -v python)"
echo "[adr017] ckpt  : $CKPT"
echo "[adr017] out   : $OUT_DIR"
python -c "import torch;print('[adr017] cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

COMMON=(--checkpoint "$CKPT" --split test --repeats "$REPEATS" --batch_size "$BATCH"
        --preload --seed 0 --device cuda --pretext_metrics)

run() {  # run <log-name> <label> <out.json> [extra args...]
    local name="$1" label="$2" out="$3"; shift 3
    echo "[adr017] $(date +%T) START $name"
    python -u runners/run_eval_recon.py "${COMMON[@]}" "$@" \
        --label "$label" --out "$OUT_DIR/$out" > "$OUT_DIR/$name.log" 2>&1
    local rc=$?
    echo "[adr017] $(date +%T) END   $name rc=$rc -> $OUT_DIR/$name.log"
    return $rc
}

rc=0
run respmax_masked_span adr017_fold0_respmax_masked_span respmax_masked_span_fold0.json \
    --kept_tir 0.111 --kept_resp 0.40 --resp_probe masked_span --interp_baseline || rc=$?
run mid_masked_span adr017_fold0_mid_masked_span mid_masked_span_fold0.json \
    --kept_tir 0.139 --kept_resp 0.30 --resp_probe masked_span --interp_baseline || rc=$?
run mode1 adr017_fold0_mode1 mode1_fold0.json \
    --kept_tir 1.0 --kept_resp 0.0 --resp_probe hidden || rc=$?
run remote_sensing adr017_fold0_remote_sensing remote_sensing_fold0.json \
    --kept_tir 1.0 --resp_probe hidden --exempt_budget || rc=$?
run remote_sensing_tir_rolled adr017_fold0_remote_sensing_tir_rolled \
    remote_sensing_tir_rolled_fold0.json \
    --kept_tir 1.0 --resp_probe hidden --exempt_budget \
    --ablate_stream tir --ablate_mode roll || rc=$?

echo "[adr017] $(date +%T) DONE rc=$rc"
ls -l "$OUT_DIR"/*.json 2>/dev/null || true
exit $rc
