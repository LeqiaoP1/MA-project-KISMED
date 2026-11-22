# Fold-0 four-stream Stage-2 pre-training — 150 epochs complete (2026-11-22)

- **Experiment**: `EXP-20261121-01` (`notes/plan/experiments/exp_20261121_fourstream_vits_random.md`)
- **Config**: `configs/pretrain/stage2_fourstream_tir_rgb2d_bp_resp_small_random.yaml`
- **EXP_ROOT**: `/work/scratch/ne95ocyg/fourstream_v1` (fold 0 only)
- **Verdict**: **completed as designed** — 150/150 epochs, budget invariants held, no NaN,
  `val/loss` 3.5040 → 2.1051.

## 1. What ran

| Item | Value |
|---|---|
| Data set / streams | `tir_roi_resp_rgb2d_bp` (`tir`, `rgb2d`, `bp`, `resp`) |
| Model | `project_multimae_small`, 30,503,488 params, **random init** (`pretrained_encoder: none`) |
| Geometry | 8 s clip, 4 s hop, 100 frames, `tubelet 4,16,16`, `input_size 96`, `fs 1000`, `seq_len 8000` |
| Masking | ADR-002 constant budget: `n_budget 350`, caps 0.25 / 0.50, `video_min_tubes 2`, `p_virtual_sensing 0.5` |
| Optim | AdamW, `blr 1.5e-4` → derived `1.875e-05` (batch 8/GPU × world 4), cosine to 0, 10-epoch warmup |
| Partition (data) | fold 0: 6648 train / 1106 val clips, digest `dc80d46c…` |

## 2. Execution: 7-link `acc_short` chain

| Link | Job | Resume epoch | Epochs in link | State | Elapsed | Ended |
|---|---|---|---|---|---|---|
| 1 | 56957081 | 0 | 25 | TIMEOUT | 30:30 | 10:38:44 |
| 2 | 56957082 | 21 | 26 | TIMEOUT | 30:20 | 11:09:17 |
| 3 | 56957083 | 46 | 26 | TIMEOUT | 30:05 | 11:39:52 |
| 4 | 56957084 | 71 | 27 | TIMEOUT | 30:27 | 12:10:26 |
| 5 | 56957085 | 96 | 27 | TIMEOUT | 30:26 | 12:41:14 |
| 6 | 56957086 | 121 | 26 | TIMEOUT | 30:16 | 13:11:48 |
| 7 | 56957087 | 146 | 4 | **COMPLETED** | 06:39 | 13:19:00 |

- Wall clock: **3 h 11 min** for 150 epochs (~4.7 min/epoch including seven ~180 s
  bootstraps), 4×a100 throughout, **zero queue gaps** between links.
- Each 30-minute link advanced **25 epochs** = 5 checkpoints at `save_ckpt_freq 5`;
  the 3–4 epochs between the last checkpoint and the walltime kill are re-run by the
  next link (link 1 finished 25 epochs but resumed at 21).
- Link 7 exited itself after epoch 149; it had 20 epochs of allocation to spare.

## 3. Validation trajectory (full val role, 1108 clips / 14 subjects)

| Epoch | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 |
|---|---|---|---|---|---|---|---|---|---|
| `val/loss` | 3.5040 | 2.5786 | 2.4487 | 2.3950 | 2.3586 | 2.3322 | 2.3242 | 2.3169 | 2.3031 |

| Epoch | 85 | 90 | 95 | 100 | 105 | 110 | 115 |
|---|---|---|---|---|---|---|---|
| `val/loss` | 2.2895 | 2.2712 | 2.2108 | 2.1562 | 2.1351 | 2.1261 | 2.1224 |

| Epoch | 120 | 125 | 130 | 135 | 140 | 145 | 149 |
|---|---|---|---|---|---|---|---|
| `val/loss` | 2.1110 | 2.1096 | 2.1076 | 2.1061 | 2.1053 | 2.1052 | **2.1051** |

Shape: fast early descent → **plateau at epochs 55–80** (−0.001 … −0.008 per 5 epochs) →
**re-acceleration at 85–105** (−0.060, −0.055 over 90→95→100, the largest intervals since
epoch 20, while the LR was already decaying) → final flattening (−0.0002 at 145→149).

The end of the curve is flat, so within the scheduled budget the run **converged under this
monitor**. That is not evidence that the representation stopped improving — see §6.

## 4. Artefacts

| Item | Path |
|---|---|
| Checkpoints | `<EXP_ROOT>/fold0/stage2/checkpoints/checkpoint-{0000,0005,…,0145,0149}.pth` — 31 files, **11 GB** |
| Resume pointer | `…/checkpoints/latest_checkpoint.txt` → `checkpoint-0149.pth` |
| Metric records | `…/stage2/metrics.jsonl` — 31 eval records (`eval_freq 5`) |
| Selection | `…/stage2/selection.json` — `rule: fixed_epoch`, `selected_epoch: 149`, `joint_loss 2.1050849` |
| Partition | `<EXP_ROOT>/fold0/partition.json`, digest `dc80d46c…` |
| Runner logs | `code/logs/adr001_pretrain_5695708{1..7}.out` |
| W&B | `…/stage2/wandb/` — **7 offline runs** (one per link); sync from a login node with `wandb sync …/offline-run-*` |

## 5. Gate results (plan §5.3)

| Gate | Result |
|---|---|
| G1 `n_batch == 350` | held on every logged step |
| G2 sum/caps/floors | held |
| G3 both mask modes | observed |
| G4 finite loss / MSE | no `NaN`/`Inf` in 150 epochs |
| G5 peak VRAM | ~2.9 GB/rank at batch 8 (24 GB cap, ample) |
| Tests | `800 passed` at launch time |

## 6. What this does NOT establish

- It is **fold 0 only**; folds 1–4 are not run.
- `val/loss` is a **monitor** (masked reconstruction MSE), not the thesis metric. The
  per-stream decoder has no cross-modal path when a stream is fully masked, so BP is not
  expected to improve RESP reconstruction.
- A flat tail under this monitor is not proof that more epochs would not help the
  *representation*: the decisive test is a frozen-encoder probe on checkpoints at, e.g.,
  epochs 60 / 95 / 149 (`core/au_probe.py`, `runners/run_eval_recon.py`). Not done.
- Extending beyond 150 would mean a **new schedule** (cosine is defined over `epochs`),
  i.e. a new run, not a continuation of this one.
