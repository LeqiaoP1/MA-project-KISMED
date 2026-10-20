#!/usr/bin/env bash
# Lichtenberg HPC: Stage-3 (TIR-ROI -> RESPIRATION) 5-fold sweep -- ONE JOB PER
# FOLD. DRY-RUN by default; pass --submit to actually queue.
#
# WHY A DRIVER INSTEAD OF ONE BIG SWEEP JOB
#   * submit_waveform_tir_roi.sbatch can run all folds sequentially in ONE job,
#     but its header is sized for that (`acc_long`, 5 days), and an `acc_long`
#     request has been refused on this cluster ("Nodes required ... reserved for
#     jobs in higher priority partitions") while the identical request in `acc`
#     started within the hour.
#   * A per-fold job is ~8 h (measured: 3:09/epoch x 150 epochs on 4x rtx6000
#     with the ROI cache), so it fits `acc` (1-day cap) with margin, backfills
#     sooner than a 5-day request, and gets its own log + checkpoint pointer.
#   * Nothing is duplicated here: this script only PREFLIGHTS and then calls
#     `scripts/hpc/submit.sh scripts/hpc/submit_waveform_tir_roi.sbatch`, so all
#     fold/run-dir/contract logic stays in that one job script.
#
# USAGE (from code/):
#   bash scripts/hpc/submit_stage3_kfold.sh                     # dry run: print the commands
#   bash scripts/hpc/submit_stage3_kfold.sh --submit            # queue all ready folds
#   FOLDS="0 1 2 3" bash scripts/hpc/submit_stage3_kfold.sh --submit
#   EPOCHS=150 bash scripts/hpc/submit_stage3_kfold.sh --submit # pin the budget
#
# PREFLIGHT (per fold, no GPU needed):
#   * the Stage-2 run dir exists and has checkpoints;
#   * its newest checkpoint is FINAL, i.e. `epoch + 1 >= args.epochs`. This is
#     what stops the sweep from starting Stage 3 on a fold whose Stage-2 is
#     still training (fold 4 while job 55495409 runs) -- `latest_checkpoint.txt`
#     alone points at a half-trained encoder;
#   * the checkpoint RECORDS the fold it belongs to (`args.n_folds`/`args.fold`).
#     run_waveform.py hard-exits on a mismatch; catching it here turns a wasted
#     queue slot into an instant error;
#   * the ROI cache out_root and the Stage-3 config exist.
#
# RESOURCES (override by env): GPU_TYPE=rtx6000 GPUS=4 PARTITION=acc
#   TIME=12:00:00. rtx6000 is the right type: a100 is ~2.6x slower per step on
#   this workload (measured 0.31 -> 0.82 s/it), which would push a fold to ~20 h.
#   Fallback if the rtx6000 pool is congested:
#     GPU_TYPE=a100 GPUS=4 TIME=24:00:00 bash scripts/hpc/submit_stage3_kfold.sh --submit
#
# NOTE: GPU count changes the GLOBAL BATCH (and therefore the LR), so keep
# GPUS=4 to stay comparable with the Stage-2 folds and with any earlier run.
set -uo pipefail

SUBMIT=0
ALLOW_PARTIAL=0
for arg in "$@"; do
    case "$arg" in
        --submit)        SUBMIT=1 ;;
        --allow-partial) ALLOW_PARTIAL=1 ;;
        -h|--help)       sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "ERROR: unknown argument '$arg' (use --submit / --allow-partial)" >&2; exit 2 ;;
    esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$HERE/../.." && pwd)"
cd "$CODE_DIR"

# The profile BELOW exports template values (PARTITION=gpu,
# GPU_TYPE=a100:4gb?), so the caller's knobs are captured first and restored
# after sourcing -- otherwise `GPU_TYPE=rtx6000 bash ...` would be overridden.
CALLER_GPU_TYPE="${GPU_TYPE-}"
CALLER_PARTITION="${PARTITION-}"
CALLER_TIME="${TIME-}"

# Profile for OUT_ROOT / RAW_DATA_PATH / VENV. Sourced with nounset OFF: it
# references site variables (WORK_PROJ) that a login shell may not export.
set +u
# shellcheck disable=SC1091
source "$CODE_DIR/scripts/env_hpc.sh"
set -u

CONFIG="${CONFIG:-configs/finetune/resp_tir_roi_hpc.yaml}"
N_FOLDS="${N_FOLDS:-5}"
FOLDS="${FOLDS:-$(seq 0 $((N_FOLDS - 1)) | tr '\n' ' ')}"
FOLDS="$(echo $FOLDS)"
STAGE2_ROOT="${STAGE2_ROOT:-${OUT_ROOT:-${WORK_SCRATCH:-$OUTPUT_DIR}}/pretrain}"
STAGE2_NAME="${STAGE2_NAME:-stage2_hpc_tir_roi_resp}"
# ---- RESULT ISOLATION ---------------------------------------------------- #
# Fold k's artefacts must never share a directory with fold j. The job script
# already names the run `${RUN_NAME}_fold<k>`, but `OUT_DIR` OVERRIDES that, and
# sbatch exports the submitting environment -- a stale `OUT_DIR` in the caller's
# shell would therefore point all five jobs at ONE directory. Two of them would
# then write metrics.jsonl / checkpoints/ / latest_checkpoint.txt concurrently
# (the cross-fold resume guard only fires on the NEXT restart). So OUT_DIR is
# pinned per fold HERE, and the value is passed explicitly to every sbatch.
FIN_ROOT="${FIN_ROOT:-${OUT_ROOT:-${WORK_SCRATCH:-$OUTPUT_DIR}}/finetune}"
RUN_NAME="${RUN_NAME:-$(basename "${CONFIG%.yaml}")}"
JOB_SCRIPT="scripts/hpc/submit_waveform_tir_roi.sbatch"

GPU_TYPE="${CALLER_GPU_TYPE:-rtx6000}"
GPUS="${GPUS:-4}"
PARTITION="${CALLER_PARTITION:-acc}"
TIME="${CALLER_TIME:-12:00:00}"
ROI_CACHE="${ROI_CACHE-${WORK_SCRATCH:-$OUTPUT_DIR}/tir_roi_cache}"

fail() { echo "ERROR: $*" >&2; exit 1; }

[ -f "$CONFIG" ]          || fail "Stage-3 config not found: $CONFIG"
[ -f "$JOB_SCRIPT" ]      || fail "job script not found: $JOB_SCRIPT"
mkdir -p logs             || fail "cannot create logs/"
[ -d "$STAGE2_ROOT" ]     || fail "Stage-2 root not found: $STAGE2_ROOT"
if [ -n "$ROI_CACHE" ] && [ ! -d "$ROI_CACHE" ]; then
    fail "ROI_CACHE=$ROI_CACHE does not exist (set ROI_CACHE='' to decode .wmv)."
fi

PY="${VENV:-}/bin/python"
[ -x "$PY" ] || PY="$(command -v python3 || true)"
[ -n "$PY" ] || fail "no python found (set VENV or activate an env)"

echo "=================================================================="
echo "Stage-3 k-fold submit plan"
echo "  config      : $CONFIG"
echo "  stage2 root : $STAGE2_ROOT   (dir pattern: ${STAGE2_NAME}_fold<k>)"
echo "  result root : $FIN_ROOT   (dir pattern: ${RUN_NAME}_fold<k>)"
echo "  folds       : $FOLDS  (of $N_FOLDS)"
echo "  roi_cache   : ${ROI_CACHE:-<disabled>}"
echo "  gres        : gpu:${GPU_TYPE}:${GPUS}   partition=$PARTITION  time=$TIME"
echo "  mode        : $([ "$SUBMIT" -eq 1 ] && echo SUBMIT || echo 'DRY RUN (pass --submit)')"
echo "=================================================================="

READY=()
NOTREADY=()
for fold in $FOLDS; do
    s2="$STAGE2_ROOT/${STAGE2_NAME}_fold${fold}"
    # shellcheck disable=SC2016
    out="$("$PY" - "$s2" "$fold" "$N_FOLDS" <<'PY'
import glob, os, sys
s2, fold, n_folds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ckpts = sorted(glob.glob(os.path.join(s2, "checkpoints", "checkpoint-*.pth")))
if not ckpts:
    print(f"NOT_READY no checkpoints under {s2}/checkpoints"); raise SystemExit(0)
newest = ckpts[-1]
try:
    import torch
    ck = torch.load(newest, map_location="cpu", weights_only=False, mmap=True)
except Exception as exc:                                   # pragma: no cover
    print(f"NOT_READY cannot read {newest}: {type(exc).__name__}: {exc}")
    raise SystemExit(0)
a = vars(ck.get("args")) if ck.get("args") is not None else {}
ep, tot = int(ck.get("epoch", -1)), int(a.get("epochs", 0) or 0)
sf, sn = a.get("fold"), int(a.get("n_folds", 0) or 0)
print(f"CHECK {newest} epoch={ep} epochs={tot} recorded_fold={sf}/{sn}")
if tot and ep + 1 < tot:
    print(f"NOT_READY Stage-2 is mid-run (epoch {ep + 1}/{tot})"); raise SystemExit(0)
if sn and (int(sf) != fold or sn != n_folds):
    print(f"NOT_READY checkpoint belongs to fold {sf}/{sn}, not {fold}/{n_folds}")
    raise SystemExit(0)
print("READY")
PY
)"
    status="$(echo "$out" | tail -1)"
    detail="$(echo "$out" | head -1)"
    out_dir="$FIN_ROOT/${RUN_NAME}_fold${fold}"
    # A pre-existing TARGET dir that belongs to another fold is a collision that
    # already happened; refuse instead of appending to it.
    if compgen -G "$out_dir/checkpoints/checkpoint-*.pth" > /dev/null; then
        prev="$("$PY" - "$out_dir" "$fold" <<'PY'
import glob, sys, torch
out_dir, fold = sys.argv[1], int(sys.argv[2])
ck = sorted(glob.glob(out_dir + "/checkpoints/checkpoint-*.pth"))[-1]
try:
    a = vars(torch.load(ck, map_location="cpu", weights_only=False, mmap=True).get("args"))
except Exception:
    print("UNKNOWN"); raise SystemExit(0)
print(f"{a.get('fold')}@{ck}" if a.get('fold') is not None else "NOFOLD")
PY
)"
        case "$prev" in
            "$fold"@*) echo "[fold $fold] target dir holds THIS fold's state (resume): $out_dir" ;;
            *) fail "$out_dir already contains another fold's checkpoint ($prev).
       Refusing to mix folds -- move it aside or set FIN_ROOT/RUN_NAME." ;;
        esac
    fi
    if [ "$status" = "READY" ]; then
        echo "[fold $fold] READY     $(echo "$detail" | sed 's/^CHECK //')"
        READY+=("$fold")
    else
        echo "[fold $fold] NOT READY $(echo "$status" | sed 's/^NOT_READY //')"
        NOTREADY+=("$fold")
    fi
done

echo "------------------------------------------------------------------"
if [ "${#READY[@]}" -eq 0 ]; then
    fail "no fold has a finished Stage-2 checkpoint -- nothing to submit."
fi
if [ "${#NOTREADY[@]}" -gt 0 ] && [ "$ALLOW_PARTIAL" -eq 0 ]; then
    fail "fold(s) ${NOTREADY[*]} are not ready. Wait for their Stage-2 run, or pass
       --allow-partial to queue only the ready folds (${READY[*]})."
fi

# Belt and braces: every queued fold must own a DIFFERENT result directory.
SEEN_DIRS=" "
for fold in "${READY[@]}"; do
    d="$FIN_ROOT/${RUN_NAME}_fold${fold}"
    case "$SEEN_DIRS" in
        *" $d "*) fail "two folds resolved to the same result dir: $d" ;;
    esac
    SEEN_DIRS="$SEEN_DIRS$d "
done

for fold in "${READY[@]}"; do
    out_dir="$FIN_ROOT/${RUN_NAME}_fold${fold}"
    cmd=(env "SBATCH_PARTITION=$PARTITION" "SBATCH_TIMELIMIT=$TIME"
         "SBATCH_JOB_NAME=st3_resp_tir_fold${fold}"
         "GPU_TYPE=$GPU_TYPE" "GPUS=$GPUS" "FOLDS=$fold"
         "OUT_DIR=$out_dir")
    [ -n "${EPOCHS:-}" ] && cmd+=("EPOCHS=$EPOCHS")
    cmd+=(bash scripts/hpc/submit.sh "$JOB_SCRIPT")
    echo "fold $fold -> $out_dir"
    echo "          ${cmd[*]}"
    if [ "$SUBMIT" -eq 1 ]; then
        "${cmd[@]}" || fail "sbatch failed for fold $fold"
    fi
done

if [ "$SUBMIT" -eq 1 ]; then
    echo "------------------------------------------------------------------"
    echo "queued: ${READY[*]}   (logs: code/logs/st3_resp_tir_<jobid>.{out,err})"
else
    echo "------------------------------------------------------------------"
    echo "dry run only -- nothing submitted. Re-run with --submit to queue."
fi
