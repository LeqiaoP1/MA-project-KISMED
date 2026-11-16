#!/usr/bin/env bash
# Lichtenberg HPC: ZERO-SHOT respiration reconstruction from FROZEN Stage-2
# checkpoints -- does the extra RGB2D stream help?
#
# Runs runners/run_eval_recon.py over a fixed matrix of arms on the SAME
# held-out fold-0 subject set. No fine-tuning, no Stage-3: the checkpoints are
# used exactly as saved.
#
# WHY AN ARRAY: one arm is ~35 min of GPU (3099 clips x 5 mask draws), so the 8
# arms are independent ELEMENTS instead of a sequential loop (that overran the
# wall clock). Every element needs 1 GPU and its own log; two elements never
# share an output file (`$OUT_DIR/<label>.json` is per arm).
#
# ARMS (all share --seed 0, so within a model the masks are IDENTICAL between
# arms that differ only in --kept_rgb2d -> an exactly paired ablation):
#
#   tri  random on/off   resp 50 % scattered (the TRAINING condition): the
#                        model can interpolate from neighbours, r ~ 0.998 ->
#                        a ceiling reference, not a discriminator
#   tri  span   on/off   resp 50 % as ONE contiguous 4 s block: NOT locally
#                        interpolatable -> the informative comparison
#   tri  hidden on/off   resp completely masked: no cross-modal prior at all
#                        (never seen in training) -> floor reference
#   twin random / span   the TIR+RESP checkpoint on the SAME clips
#                        (--data_checkpoint pins the dataset), i.e. "trained
#                        without rgb2d" vs "trained with rgb2d"
#
# Env overrides (sbatch exports them):
#   TRI_CKPT   tri-modal weights   default .../stage2_hpc_tir_roi_resp_rgb2d_fold0/checkpoints/checkpoint-0149.pth
#   TWIN_CKPT  TIR+RESP weights    default .../stage2_hpc_tir_roi_resp_96respgrid_fold0/checkpoints/checkpoint-0149.pth
#   OUT_DIR    results dir         default $WORK_SCRATCH/eval/resp_recon_rgb2d
#   REPEATS    mask draws / clip   default 5
#   MAX_CLIPS  cap (0 = all)       default 0
#   BATCH      eval batch size     default 16
#
# Submit from code/:
#   sbatch scripts/hpc/submit_recon_eval.sbatch              # all 8 arms
#   sbatch --array=2,3 scripts/hpc/submit_recon_eval.sbatch  # just those arms
#   MAX_CLIPS=400 sbatch scripts/hpc/submit_recon_eval.sbatch     # quick pass
#SBATCH --job-name=resprecon
#SBATCH --partition=acc
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:30:00
#SBATCH --array=0-7
#SBATCH --output=logs/recon_eval_%A_%a.out
#SBATCH --error=logs/recon_eval_%A_%a.err

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

PT="${WORK_SCRATCH:-$OUTPUT_DIR}/pretrain"
TRI_CKPT="${TRI_CKPT:-$PT/stage2_hpc_tir_roi_resp_rgb2d_fold0/checkpoints/checkpoint-0149.pth}"
TWIN_CKPT="${TWIN_CKPT:-$PT/stage2_hpc_tir_roi_resp_96respgrid_fold0/checkpoints/checkpoint-0149.pth}"
OUT_DIR="${OUT_DIR:-${WORK_SCRATCH:-$OUTPUT_DIR}/eval/resp_recon_rgb2d}"
REPEATS="${REPEATS:-5}"
MAX_CLIPS="${MAX_CLIPS:-0}"
BATCH="${BATCH:-16}"

for f in "$TRI_CKPT" "$TWIN_CKPT"; do
    [ -f "$f" ] || { echo "ERROR: checkpoint not found: $f" >&2; exit 1; }
done
mkdir -p "$OUT_DIR" "$CODE_DIR/logs"

COMMON=(--split test --seed 0 --device cuda --repeats "$REPEATS"
        --batch_size "$BATCH" --kept_tir 0.10 --kept_resp 0.50)
[ "$MAX_CLIPS" != "0" ] && COMMON+=(--max_clips "$MAX_CLIPS")

run() {   # run <label> <checkpoint> <extra args...>
    local label="$1" ckpt="$2"
    shift 2
    echo "[recon] ARM $label -> $OUT_DIR/$label.json"
    echo "[recon] gpu : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    python -u runners/run_eval_recon.py --checkpoint "$ckpt" \
        --label "$label" --out "$OUT_DIR/$label.json" \
        "${COMMON[@]}" "$@"
}

IDX="${SLURM_ARRAY_TASK_ID:-0}"
case "$IDX" in
    0) run tri_random_rgb2d_on  "$TRI_CKPT" --resp_probe random --kept_rgb2d 0.50 ;;
    1) run tri_random_rgb2d_off "$TRI_CKPT" --resp_probe random --kept_rgb2d 0.00 ;;
    2) run tri_span_rgb2d_on    "$TRI_CKPT" --resp_probe span   --kept_rgb2d 0.50 ;;
    3) run tri_span_rgb2d_off   "$TRI_CKPT" --resp_probe span   --kept_rgb2d 0.00 ;;
    4) run tri_hidden_rgb2d_on  "$TRI_CKPT" --resp_probe hidden --kept_rgb2d 0.50 ;;
    5) run tri_hidden_rgb2d_off "$TRI_CKPT" --resp_probe hidden --kept_rgb2d 0.00 ;;
    6) run twin_random          "$TWIN_CKPT" --data_checkpoint "$TRI_CKPT" --resp_probe random ;;
    7) run twin_span            "$TWIN_CKPT" --data_checkpoint "$TRI_CKPT" --resp_probe span ;;
    *) echo "ERROR: unknown array index $IDX (0-7)" >&2; exit 2 ;;
esac

echo "[recon] element $IDX done -> $OUT_DIR"
