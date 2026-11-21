# Four-stream Stage-2 smoke — 2026-11-21

- **Commit**: `b193414`
- **Config**: `configs/pretrain/stage2_fourstream_tir_rgb2d_bp_resp_small_random.yaml`
- **Hardware**: 1× Tesla T4, 15,360 MiB (local validation only; not the target A100/H100/RTX6000)
- **PyTorch**: `2.14.0+cu130`
- **Verdict**: **PASS** (plumbing + budget + finiteness)

## Command

```bash
EXP_ROOT=$WORK_SCRATCH/fourstream_smoke_exp_20261121_01
python -u runners/run_pretrain.py \
  -c configs/pretrain/stage2_fourstream_tir_rgb2d_bp_resp_small_random.yaml \
  --exp_root "$EXP_ROOT" --fold 0 --dev_smoke \
  --data_set tir_roi_resp_rgb2d_bp --raw_root "$RAW_DATA_PATH" \
  --epochs 1 --warmup_epochs 0 --num_workers 0 \
  --max_clips 1 --max_entries 2 --batch_size 1 \
  --roi_cache "$WORK_SCRATCH/tir_roi_cache" \
  --rgb2d_cache "$WORK_SCRATCH/rgb2d_roi_cache" \
  --resp_cache "$WORK_SCRATCH/resp_cache" \
  --bp_artefact "$WORK_SCRATCH/bp_artefact" \
  --output_dir "$EXP_ROOT/fold0/stage2"
```

## Configuration actually exercised

- `data_set: tir_roi_resp_rgb2d_bp`
- `model: project_multimae_small` — **30,503,488** trainable parameters
- `pretrained_encoder: none` — `[stage1] random init`
- `streams: tir,rgb2d,bp,resp`
- Geometry: 8 s window, 100 frames after `temporal_stride=2`,
  `tubelet=4,16,16`, `input_size=96`, `seq_len=8000`, `fs=1000`,
  `sig_kernel=32`
- Masking: `n_budget=350`, visual cap 9 tubes, physio cap 125 tokens,
  `video_min_tubes=2`, `p_virtual_sensing=0.5`
- Mask units: `tir (36, 25)`, `rgb2d (36, 25)`, `bp (250, 1)`,
  `resp (250, 1)`

## Data and partition

- Dev partition: `fold 0/5: train=97, val=14, test=28`,
  digest `dc80d46cf8e7`
- Train dataset: 2 clips after `--max_clips 1 --max_entries 2`
- BP filter: 2 of 2 kept, 0 skipped, 0 unusable
- Validation: 2 clips

## Budget verification

The logger prints the **lower median** of the step values in the first column
and the global average in parentheses. The two plans are therefore:

| Step | Mode | `tir` | `rgb2d` | `bp` | `resp` | Sum |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 175 | 75 | 76 | 24 | **350** |
| 1 | 1 | 150 | 200 | 0 | 0 | **350** |

The `kept_rgb2d` line `75.0000 (137.5000)` is the lower median 75 and the
two-step average 137.5 of values 75 and 200, not a budget violation.

## Training metrics

| Metric | Step 0 | Step 1 | Epoch average |
|---|---:|---:|---:|
| `loss` | 5.7045 | 5.7045 | 5.7173 |
| `mse_tir` | 1.3439 | 1.3439 | 1.3475 |
| `mse_rgb2d` | 1.3372 | 1.3296 | 1.3334 |
| `mse_bp` | 3.3518 | 3.3518 | 3.3941 |
| `mse_resp` | 2.6950 | 2.6623 | 2.6787 |
| `grad_norm` | 35.60 | 35.18 | 35.39 |
| `n_batch` | 350 | 350 | 350 |

Validation: `val_loss = 5.7712` over 2 samples. Selection:
`fixed_epoch`, epoch 0, `joint_loss = 5.7711575`.

`mse_bp` and `mse_resp` are **monitors**, not reconstruction-quality metrics.
The smoke proves the plumbing and the constant budget; it makes no accuracy
claim.

## Artefacts

- `checkpoint-0000.pth` — 366,287,071 bytes
- `metrics.jsonl` — validation/selection metrics
- `selection.json` — fixed-epoch selection record
- `checkpoints/latest_checkpoint.txt`

## Resource gate

One forward + backward at batch 1, 96 px, four streams:

- peak **allocated**: **0.484 GB**
- peak **reserved**: **0.570 GB**
- device: Tesla T4, 15,360 MiB

This is a batch-1 lower bound. The full run uses batch 8 and optimizer state, so
the 24 GB per-GPU cap remains provisional until the first target-hardware run.

## Gate result

| Gate | Result |
|---|---|
| G1 `n_batch == 350` | **PASS** on both steps |
| G2 sum == 350, caps/floors | **PASS** (plan reproduction: step 0 = 350, step 1 = 350) |
| G3 both modes observed | **PASS** (mode 2 and mode 1) |
| G4 finite loss / MSE | **PASS** |
| G5 peak VRAM | **PASS** (0.484 GB on a 15 GB T4; batch-8 cap still provisional) |

## Notes

- The smoke LR is `5.859e-07` because `batch_size=1`, `world_size=1`.
  The full run uses the derived large-batch LR.
- `partition.dev.json` is a dev-smoke record and must never be pooled.
- The T4 is not a target GPU; this run validates code and data plumbing, not
  throughput. The first target-hardware test is the one-fold HPC smoke.
