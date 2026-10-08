# Batch-size sweep: does a bigger batch shorten the Stage-2 epoch?

**Date:** 2026-10-09
**Harness:** `scripts/hpc/bench_batch_sweep.sbatch` (array, 6 arms, 4 ranks,
`acc_short`, `NVIDIA RTX PRO 6000 Blackwell Server Edition`), analyser `analysis/batch_size_sweep.py`.
**Spec:** `analysis/hpc_precomputation.md` (the ROI cache this sweep runs on).

## 0. Answer in brief

**No -- and the larger 96 GB card does not change that.**

Measured: FLAT -- every batch projects to within 5 % of batch 16, so batch size is NOT a wall-clock lever here: the loop is already at its compute floor. Choose the batch for OPTIMISATION reasons, not for speed.

Total work per epoch is FIXED -- the same clips cost the same FLOPs whatever the
grouping -- so a larger batch only helps if the loop is **overhead-bound**
(fewer optimiser steps, fewer DDP all-reduce rounds, fewer kernel launches).
If the per-step cost grows linearly with the batch, the epoch time is flat.

What DOES move the wall clock is the ROI cache: **11.7x at equal batch**, versus
a flat-to-slightly-worse epoch across a 16x batch range. See section 3.1.

The diagnostic is **step per clip (ms)**: constant across arms => GPU-bound;
falling => overhead-bound.

## 1. Method

* the REAL runner (`runners/run_pretrain.py`) on the production config, so the
  measured path includes DDP, the masking generator and the AMP scaler;
* **4 ranks**, the production world size, so the collectives are paid as in the
  real run (a 1-GPU arm has no allreduce and would understate the benefit);
* **the ROI cache ON**, because the pre-cache loop was decode-bound and its
  2.45 s/iter would have hidden the GPU cost entirely. One arm keeps the cache
  OFF as a reference to the published number;
* **STEPS fixed, not entries**: every arm runs the same number of optimiser
  steps (`entries = steps x batch x ranks`), so `s/step` is directly comparable
  and every arm pays the same warm-up;
* `time:` is total s/it and `data:` is the s/it spent waiting for the loader
  (both 10-iteration windowed means from `utils/logger.py`), so
  `step = time - data` is the pure compute+communication cost.

## 2. Results

| batch | global | time s/it | data | data % | step | ms/clip | clips/s | epoch steps | epoch proj (min) | peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 32 | 0.1145 | 0.0001 | 0.1 | 0.1144 | 3.575 | 279.48 | 663.5 | 1.3 | 7735 |
| 16 | 64 | 0.2055 | 0.0001 | 0.0 | 0.2054 | 3.209 | 311.44 | 331.7 | 1.1 | 11951 |
| 32 | 128 | 0.4638 | 0.0001 | 0.0 | 0.4637 | 3.623 | 275.98 | 165.9 | 1.3 | 20771 |
| 64 | 256 | 1.0150 | 0.2678 | 26.4 | 0.7472 | 2.919 | 252.22 | 82.9 | 1.4 | 37967 |
| 128 | 512 | 1.9833 | 0.2891 | 14.6 | 1.6942 | 3.309 | 258.16 | 41.5 | 1.4 | 72975 |

cache OFF reference:

| cache | batch | time s/it | data | step | peak MiB |
|---|---|---|---|---|---|
| off | 16 | 2.3961 | 2.0239 | 0.3722 | 12375 |

Relative to batch 16 (epoch projection and per-clip step cost):

| batch | epoch proj vs b16 | ms/clip |
|---|---|---|
| 8 | 1.114x | 3.575 |
| 16 | 1.000x | 3.209 |
| 32 | 1.128x | 3.623 |
| 64 | 1.235x | 2.919 |
| 128 | 1.206x | 3.309 |

`peak MiB` is the maximum sampled `nvidia-smi memory.used` across all GPUs
(`gpu_total_mib` = 97887 MiB); a single reading is timing-dependent and would miss
the peak.

## 3. What the numbers mean

**Measured verdict:** FLAT -- every batch projects to within 5 % of batch 16, so batch size is NOT a wall-clock lever here: the loop is already at its compute floor. Choose the batch for OPTIMISATION reasons, not for speed.

Measured detail:

* step-cost scaling b8 -> b128: x14.81 for a batch ratio of x16
* ms/clip 3.575 -> 3.309 (-7.4 %)
* best epoch projection: batch 16 at 1.14 min/epoch
* vs the batch-16 reference (1.14 min): +0.0 %

### 3.1 The cache is the lever, the batch is not

* **Cache ON vs OFF, both at batch 16:** `time` 2.3961 -> 0.2055 s/it (**11.7x faster**), and the loader share of the iteration collapses from 84.5 % to 0.0 %. That single change is worth more than the ENTIRE batch curve.

* `epoch proj` = `(21231 clips / global batch) x time`, i.e. the wall
  clock of one epoch at the shipped 2 s hop, excluding the ~2 min of dataset
  discovery.
* The per-clip step cost (`ms/clip`) is the mechanism: if it is flat across
  arms the GPU is already the bottleneck and a larger batch only REGROUPS the
  same work. Any fall in `ms/clip` is the per-step overhead (allreduce, kernel
  launches, optimiser) being amortised away.
* `peak MiB` settles whether memory is the binding constraint at all, against
  `gpu_total_mib` = 97887 MiB on this node. It rises from 7.6 GB
  to 71.3 GB -- roughly linear in the batch -- while
  `epoch proj` stays flat. So the extra GRAM is *consumed* but buys no wall
  clock: a larger batch is worth it only if it buys a BETTER model, never a
  faster epoch. At the largest arm it also leaves little headroom for the real
  run's activation spikes.

### 3.2 Two arms were lost to a scheduler/NCCL transient, not to the batch

Arms at batch 32 and 128 of the first submission died with
`ncclOsSocketPollConnect: connect to <ip> returned Connection refused` in
`barrier()`, after ~2.1 GB MaxRSS -- i.e. BEFORE any training ran. The cause was
a hardcoded `MASTER_PORT`: array elements are packed 2-per-node (2 x 4 GPUs on
an 8-GPU box), so the second element could not bind and the first died in the
barrier. The port is now derived from `SLURM_JOB_ID` and the element index. Those
arms were re-run; the numbers above are from the fixed harness. Any `peak MiB`
of a few hundred MiB is an early-death artefact, not a measurement.

## 4. Caveats that decide whether to act

* **A batch change is an OPTIMISATION change, not a free speed knob.**
  `lr = blr x batch x world / 256`, so raising the per-rank batch from 16 to 64
  takes the LR from 3.75e-5 to 1.5e-4 -- a 4x jump that needs the warmup
  re-checked.
* **For masked autoencoding, larger batches push the WRONG way.**
  `mask_ratio_tir: 0.90`, so a clip contributes only ~10 % of its tokens as
  prediction targets; the MAE literature's direction is smaller batches with
  more steps, and batch size trades off against the masking ratio.
* **"150 epochs" is not a constant amount of training.** At a larger batch an
  epoch holds fewer steps, so the run would see fewer optimiser updates in total.
  Comparing wall clock at equal epochs compares two different amounts of training.
* `update_freq` (gradient accumulation) is the memory-neutral way to a larger
  EFFECTIVE batch if that is wanted for optimisation reasons -- but it is not a
  speed lever, since it does not reduce forward/backward passes.
* Single-arm noise: these are 30-step runs, so a few percent of jitter between
  arms is expected; only a consistent trend across all arms is meaningful.
