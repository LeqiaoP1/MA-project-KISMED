#!/usr/bin/env bash
# Raw-physiology inspection (NO canonical resampling): read the BP4D
# Physiology/<subject>/<task>/<channel>.txt files straight out of the raw tree,
# print length / min / max / estimated frequency and write one figure + JSON per
# channel into <repo>/output/inspect_data/<subject>_<task>/. Run from code/ (or
# anywhere; the script cd's into code/).
#
#   SUBJECT=F001 TASK=T1 CHANNEL=all            bash scripts/local/inspect_physio.sh
#   SUBJECT=F001,F002 TASK=T1,T2 CHANNEL=Resp,EDA bash scripts/local/inspect_physio.sh
#   bash scripts/local/inspect_physio.sh --list            # what is on disk?
#
# Channel names are case-insensitive (BVP | Resp | EDA | all; BVP<-BP_mmHg.txt,
# Resp<-Resp_Volts.txt, EDA<-EDA_microsiemens.txt). Anything after the script
# name is passed to the runner verbatim, so flags can also be given directly:
#   bash scripts/local/inspect_physio.sh --subject F001 --task T1 --channel EDA
set -e

cd "$(dirname "$0")/../.."
source scripts/env_local.sh

SUBJECT="${SUBJECT:-F001}"
TASK="${TASK:-T1}"
CHANNEL="${CHANNEL:-${INSPECT_CHANNEL:-all}}"
# sample rate: BP4D nominal (1000 Hz). PHY_FS=0 estimates it per session from
# the RGB frame count and FPS_RGB instead. The original samples are kept either
# way - nothing is resampled.
PHY_FS="${PHY_FS:-${PHYS_FS:-1000}}"
FPS_RGB="${FPS_RGB:-25}"

if [ "$#" -gt 0 ]; then
    # explicit CLI wins over the env-style overrides above
    python runners/run_inspect_physio.py "$@"
else
    # --output_dir is left to the runner: it defaults to $OUTPUT_DIR/inspect_data
    python runners/run_inspect_physio.py \
        --subject "$SUBJECT" \
        --task "$TASK" \
        --channel "$CHANNEL" \
        --phys_fs "$PHY_FS" \
        --fps_rgb "$FPS_RGB"
fi

echo
echo "Figures + JSON: $OUTPUT_DIR/inspect_data/<subject>_<task>/"
echo "Index:          $OUTPUT_DIR/inspect_data/physio_index.json"
