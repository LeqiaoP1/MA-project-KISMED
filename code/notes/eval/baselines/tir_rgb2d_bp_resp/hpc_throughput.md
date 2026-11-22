# Four-stream Stage-2 throughput on Lichtenberg — 2026-11-22

- **Config**: `configs/pretrain/stage2_fourstream_tir_rgb2d_bp_resp_small_random.yaml`
- **Purpose**: decide the execution width and shape for fold 0 on the HPC cluster,
  and record the numbers the chain arithmetic rests on.
- **Mechanism** (why a chain rather than one long job):
  `code/notes/explanation/hpc_why_long_gpu_jobs_never_start.md`.

All runs below used the real fold-0 partition (6648 train / 1106 val clips) and
the four real caches. Benchmarks wrote to `$WORK_SCRATCH/fourstream_bench*`, never
to `fourstream_v1`.

## 1. Single-job startup (the limiting factor, not throughput)

Identical 2×a100 requests, same hour:

| Walltime → partition | State | Start latency |
|---|---|---|
| 5 min → `acc_short` (job 56950920) | RUNNING | **10 s** |
| 2 h / 6 h / 12 h / 24 h → `acc` (56951161/62/63, 56950960) | PENDING | never started; cancelled after 8 min |
| 5 d → `acc_long` (56904546) | PENDING | **13 h 41 m, no start**; cancelled |

4×a100 requests (probes 56954651/56954652, 5 min): started after **6–8 min**;
the scheduler's own `squeue --start` estimate had said 09:46 and they started at
09:23/09:25. Treat backfill estimates as optimistic.

## 2. Measured throughput (global batch held at 32 in both rows)

| Width | Per-GPU batch | LR | Steps/epoch | Epoch time | s/it |
|---|---|---|---|---|---|
| 2×a100 (job 56952225) | 16 | `1.875e-05` | 208 | 1:54 / 1:47 / 1:50 / 1:48 / 1:47 / 1:48 | 0.5508 / 0.5197 / 0.5351 / 0.5222 / 0.5214 / 0.5235 |
| 4×a100 (job 56956021) | 8 | `1.875e-05` | 208 | 1:06 / 1:07 / 1:06 | 0.3224 / 0.3242 / 0.3206 |

- 4 GPUs is **1.64×** faster per epoch, not 2× — dataloader and per-step overhead
  do not halve.
- Bootstrap (env → partition → dataset build → first step) is **~180 s**, mostly
  the dataset build, and it is paid by **every** link of a chain.
- Steady state on 2×a100: ~108 s/epoch; on 4×a100: ~66 s/epoch.

## 3. Chain arithmetic

30-minute link, 4×a100: `(1800 − 180) / 66 ≈ 24` epochs.
150 epochs ⇒ **7 links** with margin; ~2.75 h of compute plus 7 × 180 s bootstrap.

## 4. Validation behaviour (found here, fixed here)

The first 2-GPU benchmark reported `val/loss 3.5044 over 553 samples`, the 4-GPU
one `3.5006 over 277 samples`: `evaluate_pretrain` ran on the DDP-sharded val
loader and nothing all-gathered, so rank 0's number covered only its own slice
(1/world of the role). Fixed in `engines/pretrain.py` by all-gathering
`(loss_sum, n_samples, n_batches, components, kept, mode)` and merging; covered by
`tests/test_val_gather.py`. Expect `over 1106 sample(s)` from the next run.

## 5. What this does not establish

- Nothing about convergence: the benchmarks were cut by their walltime at 15–16
  epochs. Loss trend in that window (train mean 4.41 at epoch 0 → `val/loss`
  3.504 → 2.715 at epoch 5) is a sanity signal, not a result.
- Nothing about other folds: all timings are fold 0 (6648 of 6284–6785 clips
  across folds, so within ±7 %).
