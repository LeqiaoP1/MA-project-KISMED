#!/usr/bin/env bash
# Local AU-occurrence probe smoke (dev only: F001..F004, capped samples).
# Run from code/ after `source scripts/env_local.sh`:
#     bash scripts/local/au_smoke.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # code/
PY="${PYTHON:-python}"                       # use the project .venv, e.g. PYTHON=.venv/bin/python

# C2 (Stage-2 from-scratch encoder) linear probe -- tiny 192-d Stage-2 run.
# 4 workers by default (the configs also set num_workers: 4; YAML beats
# $NUM_WORKERS, so override a single run with --num_workers N).
NUM_WORKERS=4 "$PY" runners/run_au_probe.py -c configs/finetune/au_local.yaml "$@"

# Optional: the ViT-Base Stage-2 run (768-d) linear probe:
# NUM_WORKERS=4 "$PY" runners/run_au_probe.py -c configs/finetune/au_local_pretrained.yaml "$@"

# Controls (same geometry/config, change only the checkpoint):
#   C0 random:  ... -c configs/finetune/au_local.yaml --finetune ''
#   C1 Stage-1: ... -c configs/finetune/au_local_pretrained.yaml \
#                  --finetune base      # downloads MAE ViT-B into ../models/initial/
