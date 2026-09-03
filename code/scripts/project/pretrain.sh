#!/usr/bin/env bash
# Example SLURM/torch.distributed launcher for pre-training.
# Run from the code/ directory.
# Adapt the VideoMAE scripts under tmp/videomae/scripts/<dataset>/<variant>/
# for dataset-specific flags.
set -e

OUTPUT_DIR='output/pretrain/example'
DATA_PATH='/path/to/your/dataset'
CFG='configs/pretrain/example.yaml'

mkdir -p "$OUTPUT_DIR"

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port 12320 \
    runners/run_pretrain.py \
    -c "$CFG" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR"
