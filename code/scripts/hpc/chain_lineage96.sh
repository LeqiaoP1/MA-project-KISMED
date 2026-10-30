#!/usr/bin/env bash
# Submit the TIR-ROI 96 px / roi_padding 0.10 lineage END TO END under NEW run
# names, so that no existing run directory is resumed into or overwritten.
#
# WHY THIS EXISTS. Both pipeline stages derive their output directory from the
# CONFIG BASENAME (`RUN_NAME=$(basename "${CONFIG%.yaml}")`), and that name does
# NOT encode the geometry. `run_pretrain.py` then auto-resumes from
# <out_dir>/checkpoints/latest_checkpoint.txt (utils/checkpoint.py). So a restart
# with the shipped defaults targets
#     $WORK_SCRATCH/pretrain/stage2_hpc_tir_roi_resp_fold<k>   (31 .pth each)
# and either dies on the pos_embed shape (2450 tokens at 112 px vs 1800 at 96)
# or -- once that resume pointer is gone -- overwrites those files name for name.
# Naming the lineage *_96 keeps the 112 px tree intact and auditable.
#
# CHAIN
#   Stage-2 : 5 folds, SEQUENTIAL inside one acc_long job
#   Stage-3 : 5 folds, SEQUENTIAL inside one acc_long job, started after Stage-2
#   eval    : one single-GPU acc job per fold, after Stage-3, each writing the
#             COMPLETE val-split dump (preds.npy / targets.npy / entries.json)
#
# USAGE (from code/)
#   bash scripts/hpc/chain_lineage96.sh                # submit the whole chain
#   DRYRUN=1 bash scripts/hpc/chain_lineage96.sh       # print, do not submit
#   FOLDS="0" bash scripts/hpc/chain_lineage96.sh      # a single fold
#   TAG=96b bash scripts/hpc/chain_lineage96.sh        # a second, separate copy
#   SUBMIT_EVAL=0 bash scripts/hpc/chain_lineage96.sh  # Stage-2 + Stage-3 only
#
# The ROI cache needs no flag: ROI_CACHE defaults to $WORK_SCRATCH/tir_roi_cache,
# which currently holds exactly one key (pad0.10 / clip 8 s / native), and the
# loader matches by key, not by directory position.
set -eo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CODE_DIR"

# env_hpc.sh predates `set -u` discipline, so enable nounset only after it.
# shellcheck disable=SC1091
source scripts/env_hpc.sh
set -u

TAG="${TAG:-96}"
N_FOLDS="${N_FOLDS:-5}"
FOLDS="${FOLDS:-$(seq 0 $((N_FOLDS - 1)) | tr '\n' ' ')}"
FOLDS="$(echo $FOLDS)"
OUT_ROOT="${OUT_ROOT:-${WORK_SCRATCH:-$OUTPUT_DIR}}"
SUBMIT_EVAL="${SUBMIT_EVAL:-1}"
DRYRUN="${DRYRUN:-}"
ST2_CONFIG="${ST2_CONFIG:-configs/pretrain/stage2_hpc_tir_roi_resp.yaml}"
ST3_CONFIG="${ST3_CONFIG:-configs/finetune/resp_tir_roi_hpc.yaml}"

S2_RUN="${S2_RUN:-stage2_hpc_tir_roi_resp_${TAG}}"
S3_RUN="${S3_RUN:-resp_tir_roi_hpc_${TAG}}"

die() { echo "ERROR: $*" >&2; exit 1; }

# --------------------------------------------------------------------------- #
# guard 1 -- the configs really are the 96 px / 0.10 contract
# --------------------------------------------------------------------------- #
for f in "$ST2_CONFIG" "$ST3_CONFIG"; do
    [ -f "$f" ] || die "missing config $f"
    grep -qE '^input_size:[[:space:]]*96([[:space:]]|$)' "$f" \
        || die "$f: input_size is not 96 -- this launcher is for the 96 px lineage"
    grep -qE '^roi_padding:[[:space:]]*0\.10?([[:space:]]|$)' "$f" \
        || die "$f: roi_padding is not 0.10"
done

# --------------------------------------------------------------------------- #
# guard 2 -- never reuse a run directory that already holds weights
# --------------------------------------------------------------------------- #
collide() {                     # collide <run base> <pretrain|finetune>
    local base="$1" stage="$2" d
    for d in "$OUT_ROOT/$stage/$base" "$OUT_ROOT/$stage/$base"_fold*/; do
        [ -d "$d" ] || continue
        if compgen -G "${d%/}/checkpoints/*.pth" > /dev/null; then
            die "${d%/} already holds checkpoints. This launcher never reuses a
       run dir -- pick another TAG (e.g. TAG=${TAG}b)."
        fi
    done
}
collide "$S2_RUN" pretrain
collide "$S3_RUN" finetune

# --------------------------------------------------------------------------- #
# submission
# --------------------------------------------------------------------------- #
run_sbatch() {                  # run_sbatch <label> <script> <dep|-> [VAR=...]
    local label="$1" script="$2" dep="$3"
    shift 3
    local -a envs=( "$@" )
    local -a flags=( --parsable )
    [ "$dep" != "-" ] && flags+=( "--dependency=afterok:$dep" )
    if [ -n "$DRYRUN" ]; then
        # stderr: the caller captures stdout as the job id
        printf '[dry-run] %-12s env %s sbatch %s %s\n' \
               "$label" "${envs[*]:-<none>}" "${flags[*]}" "$script" >&2
        printf '%s\n' '<dry-run>'
        return 0
    fi
    local out
    if [ ${#envs[@]} -gt 0 ]; then
        out="$(env "${envs[@]}" sbatch "${flags[@]}" "$script")"
    else
        out="$(sbatch "${flags[@]}" "$script")"
    fi
    printf '%s\n' "${out%%;*}"
}

mkdir -p logs

JOB2="$(run_sbatch Stage-2 scripts/hpc/submit_pretrain.sbatch - \
        RUN_NAME="$S2_RUN" CONFIG="$ST2_CONFIG" \
        FOLDS="$FOLDS" N_FOLDS="$N_FOLDS")"
[ -n "$JOB2" ] || die "Stage-2 submission produced no job id"
echo "[chain] Stage-2 : job $JOB2 -> $OUT_ROOT/pretrain/${S2_RUN}_fold<k>"

JOB3="$(run_sbatch Stage-3 scripts/hpc/submit_waveform_tir_roi.sbatch "$JOB2" \
        RUN_NAME="$S3_RUN" CONFIG="$ST3_CONFIG" STAGE2_RUN_NAME="$S2_RUN" \
        FOLDS="$FOLDS" N_FOLDS="$N_FOLDS")"
[ -n "$JOB3" ] || die "Stage-3 submission produced no job id"
echo "[chain] Stage-3 : job $JOB3 -> $OUT_ROOT/finetune/${S3_RUN}_fold<k>"

if [ "$SUBMIT_EVAL" = "1" ]; then
    for fold in $FOLDS; do
        J="$(run_sbatch "eval f$fold" scripts/hpc/submit_waveform_eval.sbatch "$JOB3" \
             RUN_NAME="${S3_RUN}_fold${fold}" CONFIG="$ST3_CONFIG" \
             OUT_DIR="$OUT_ROOT/finetune/${S3_RUN}_fold${fold}" \
             STAGE2_DIR="$OUT_ROOT/pretrain/${S2_RUN}_fold${fold}")"
        [ -n "$J" ] || die "eval submission for fold $fold produced no job id"
        echo "[chain] eval f$fold : job $J"
    done
fi

cat <<EOF

[chain] done. New output roots (no pre-existing run dir is reused):
  Stage-2 : $OUT_ROOT/pretrain/${S2_RUN}_fold<k>
  Stage-3 : $OUT_ROOT/finetune/${S3_RUN}_fold<k>
  eval    : those same dirs, rewritten at world size 1 with the COMPLETE
            val split (preds.npy / targets.npy / entries.json)

Deliberately untouched -- the 112 px lineage, 31 .pth per dir:
  $OUT_ROOT/pretrain/stage2_hpc_tir_roi_resp{,_fold0..4}
  $OUT_ROOT/finetune/resp_tir_roi_hpc_fold0..4   (+ their entries.json)

AFTER the eval stage, sync the five new entries.json dumps into
output/finetune/${S3_RUN}_fold<k>/ and re-run the geometry gate:

  cd code && python notes/eval/tir_resp/roi_geometry_ledger.py \\
      --raw_root "\$RAW_DATA_PATH" \\
      --entries "../output/finetune/${S3_RUN}_fold*/entries.json"

It SHOULD reproduce the current EXPECT figures unchanged: the clip population is
set by the sentinel / rail / flat / motion gates, none of which depend on
input_size. A mismatch means the new run's corpus drifted from the contract.
EOF
