#!/usr/bin/env bash
# Local AU-occurrence probe smoke (dev only: F001..F004, capped samples).
# Run from code/ after `source scripts/env_local.sh`:
#     bash scripts/local/au_smoke.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # code/
PY="${PYTHON:-python}"                       # use the project .venv, e.g. PYTHON=.venv/bin/python

# C2 (Stage-2 encoder) linear probe -- 768-d Stage-2 run (au_local.yaml probes
# the SSV2-init encoder, au_local_pretrained.yaml the K400-init one).
# 4 workers by default (the configs also set num_workers: 4; YAML beats
# $NUM_WORKERS, so override a single run with --num_workers N).
NUM_WORKERS=4 "$PY" runners/run_au_probe.py -c configs/finetune/au_local.yaml "$@"

# Optional: the K400-init Stage-2 run (768-d) linear probe:
# NUM_WORKERS=4 "$PY" runners/run_au_probe.py -c configs/finetune/au_local_pretrained.yaml "$@"

# Controls (same geometry/config, change only the checkpoint; blank is REJECTED):
#   C1 Stage-1: ... -c configs/finetune/au_local_pretrained.yaml \
#                  --finetune videomae:k400   # downloads VideoMAE ViT-B into ../models/initial/
#                  --finetune videomae:ssv2   # the SSV2 corpus instead
