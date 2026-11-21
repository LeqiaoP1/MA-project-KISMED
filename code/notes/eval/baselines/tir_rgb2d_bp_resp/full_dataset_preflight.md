# Four-stream full-dataset preflight — 2026-11-21

- **Config**: `configs/pretrain/stage2_fourstream_tir_rgb2d_bp_resp_small_random.yaml`
- **Caches**: TIR-ROI, RGB2D `px112`, RESP 1000 Hz, BP 1000 Hz (see
  `cache_preflight.md`)
- **Scope**: measure the real per-fold clip counts and the batch-8 step on the
  **full** partition, i.e. the two items the smoke could not cover.
- **Verdict**: **the one-fold run is unblocked**; the five-fold sweep is not yet
  sanctioned (see §4).

## 1. Per-fold clip counts (no dev caps)

Built with `build_pretraining_dataset` for `train` and `val` only — Stage 2
refuses the `test` role by design. Partition: `split_seed=42`,
`split_stratify=sex`.

| Fold | Digest (12) | `train` clips | parent | BP dropped | `val` clips | parent | BP dropped |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | `dc80d46cf8e7` | 6648 | 6762 | 114 (1.69 %) | 1106 | 1127 | 21 (1.86 %) |
| 1 | `138cf3775967` | 6386 | 6512 | 126 (1.94 %) | 854 | 864 | 10 (1.16 %) |
| 2 | `566120cf7af4` | 6785 | 6868 | 83 (1.21 %) | 942 | 980 | 38 (3.88 %) |
| 3 | `6ffb80f07e1d` | 6307 | 6438 | 131 (2.03 %) | 977 | 992 | 15 (1.51 %) |
| 4 | `419f6a6b28b4` | 6284 | 6415 | 131 (2.04 %) | 971 | 974 | 3 (0.31 %) |

- `clips_dropped_bp_skipped = 0` and `sessions_bp_skipped = 0` on every fold:
  every session has a BP artefact, so the drop is purely the per-window
  `bp_windows.usable` gate.
- No fold has an empty or thin split; `drop_last` at `batch_size=8` cannot eat a
  role (smallest val is 854 clips).
- The corpus-level BP window yield reported in `cache_preflight.md` (98.14 % of
  8 s windows) is consistent: the per-clip loss is 0.3–2.0 % because a clip is
  dropped only when its own window is unusable.
- Dataset construction is CPU-only and read-mostly: 55–125 s per fold for
  `train` + `val` together.

## 2. Batch-8 step (real fold-0 partition)

| Item | Value |
|---|---|
| `batch_size` | 8 |
| `num_workers` | 8 |
| peak allocated / reserved | **2.897 GB / 3.271 GB** (Tesla T4, 15 GB) |
| derived LR (`blr·batch·world/256`, world=1) | 4.6875e-06 |
| steady-state iteration | **1.48 s/it** (data 0.42 s/it, first step 5.3 s incl. cache warm-up) |
| steps per epoch, fold 0 | `floor(6648/8) = 831` |
| masked budget | `n_batch = 350` on every step; both modes observed |

The batch-8 footprint is 5.7× the batch-1 smoke figure (0.484 / 0.570 GB) and
leaves >85 % headroom on a 24 GB cap, so `batch_size: 8` is safe for ViT-Small.

## 3. Runtime arithmetic (not a promise)

`831 steps × 1.48 s ≈ 20.5 min/epoch` on a **T4**; `150 epochs ≈ 51 h/fold`.
That is a lower bound for the target GPU, not a target-hardware measurement.
`save_ckpt_freq=10` writes 15 checkpoints × 366 MB ≈ 5.5 GB per fold, i.e.
≈ 27 GB across five folds — check the `EXP_ROOT` quota before the sweep.

## 4. What remains unverified

- Target-hardware (RTX 6000 / A100 / H100) throughput and VRAM per rank.
- Stability over a real epoch at the derived peak LR (only 5 steps seen).
- DDP scaling: every number above is single-process.

These are precisely what the one-fold target-hardware run is for; they are the
reason the five-fold sweep stays blocked (`exp_20261121_fourstream_vits_random.md`
§9, Stop/Retire Rule).
