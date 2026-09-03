#!/usr/bin/env bash
# Example SLURM/torch.distributed launcher for fine-tuning.
# Run from the code/ directory.
set -e

OUTPUT_DIR='output/finetune/example'
DATA_PATH='/path/to/your/dataset'
MODEL_PATH='output/pretrain/example/checkpoints/checkpoint-0799.pth'
CFG='configs/finetune/example.yaml'

mkdir -p "$OUTPUT_DIR"

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port 12321 \
    runners/run_finetune.py \
    -c "$CFG" \
    --data_path "$DATA_PATH" \
    --finetune "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR"
