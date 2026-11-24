# Three-stream Stage-2 execution on Lichtenberg — measured chain (2026-11-24)

- **Config**: `configs/pretrain/stage2_budget_3stream_curriculum.yaml`
- **Run**: `/work/scratch/ne95ocyg/pretrain/adr005_3stream_fold0` (fold 0)
- **Purpose**: record the numbers the chain arithmetic actually produced for
  this arm.
- **Mechanism** (why a chain rather than one long job):
  `code/notes/explanation/hpc_why_long_gpu_jobs_never_start.md`.

## 1. Execution shape

| Item | Value |
|---|---|
| GPUs | 4 × A100, `--ntasks=4 --ntasks-per-node=4` |
| Partition / walltime | `acc_short`, 30 min per link |
| Global batch | 16 per GPU × 4 = **64** |
| Steps per epoch | **105** (`floor(6762 / 64)`) |
| Total steps | 15,750 (150 epochs) |
| LR | `blr 1.5e-4 × 64 / 256 = 3.750e-05`, cosine, 1,050 warm-up steps |
| Precision | FP32 (`amp_dtype=none`) |

## 2. Measured epoch throughput

| Item | Value |
|---|---:|
| Epoch executions logged | 184 (150 distinct + 34 re-runs) |
| Mean `Total time` per epoch | **93.1 s** |
| Min / max epoch | 80 s / 310 s (first epoch after each bootstrap, cold caches) |
| Mean per-step time | **0.886 s/it** |
| Samples per second | **72.2** |
| Epoch compute, summed | **4.76 h** |

The steady-state step is 0.72–0.89 s, of which the logger attributes ~0.20 s to
data wait; 64 samples every 0.886 s is 72.2 samples/s on 4×A100. The cold first
step of a link is much slower (`time 8.04 s, data 5.69 s` at epoch 148).

## 3. Chain overhead

| Component | Value |
|---|---:|
| Links | **13** (12 TIMEOUT + 1 COMPLETED at 28:36) |
| Sum of link elapsed times | **6 h 33 min** |
| Epoch compute | 4 h 46 min |
| Remaining overhead | ≈ 1 h 47 min |
| — of which re-run epochs (5-epoch resume cadence) | 34 × 93 s ≈ **53 min** |
| — of which bootstrap + cold-cache first epoch | ≈ **54 min** (13 × ~4 min) |

`save_ckpt_freq 5` is the reason for the re-runs: a 30-minute link advances
~18 epochs, so 0–5 epochs past the last checkpoint are repeated by the next link.
At `save_ckpt_freq 1` the re-run cost would disappear but add ~25 s per epoch of
checkpoint writes; at `save_ckpt_freq 10` it would roughly double to ~1.8 h of
re-runs. A longer walltime partition would remove both the re-runs and most of the
13 bootstraps, which is the main throughput lever for the remaining folds.

## 4. Storage

| Item | Value |
|---|---:|
| Checkpoint size | **1.2 GB** each |
| Checkpoints written | 31 (`0000, 0005, …, 0145, 0149`) |
| Checkpoint tree | **37 GB** per fold |

Five folds at this cadence and 1.2 GB/checkpoint is ~185 GB; budget for it before
launching a five-fold sweep.

## 5. What this does not establish

- Nothing about convergence; throughput is independent of the loss.
- Nothing about other folds or other hardware. Fold 0 has 6,762 training clips,
  which is within ±8 % of the other folds' pre-BP counts, so the step arithmetic
  transfers but was not measured.
- The FP32 path is what ran; an AMP run (`--amp_dtype bf16`) would change both
  s/it and peak memory and has not been benchmarked for this arm.
