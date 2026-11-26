# Fold-0 three-stream Stage-2 pre-training — 150 epochs complete (2026-11-24)

- **Design of record**: `code/notes/design/archive/ADR-005-asymmetric-multimodal-masking-3streams.md`
- **Build plan**: `code/notes/plan/features/archive/ADR-005-asymmetric-multimodal-masking-3streams-plan.md`
- **Config**: `configs/pretrain/stage2_budget_3stream_curriculum.yaml`
- **EXP_ROOT**: `/work/scratch/ne95ocyg/pretrain/adr005_3stream_fold0` (fold 0 only)
- **Code**: git `1a47e7e`
- **Verdict**: **completed as designed** — 150/150 epochs, budget and caps held on
  every logged step, no `NaN`, curriculum boundary visible in the logs. The
  validation monitor is **not** comparable across epoch 50 (its mask distribution
  changes by design).

## 1. What ran

| Item | Value |
|---|---|
| Data set / streams | `tir_roi_resp_rgb2d` (`tir`, `rgb2d`, `resp`) |
| Model | `project_multimae_base`, **105,581,600** params |
| Initialisation | VideoMAE `k400` → **TIR** adapter verbatim (148 encoder tensors, 0 skipped, 0 shape-mismatched); `rgb2d` adapter **random** (ADR-005 I2 debt, recorded in `masking.json`) |
| Geometry | 8 s clip, 4 s hop, decimation 4 → **50 frames**, `tubelet 2,16,16`, `input_size 96`, `fs 1000`, `sig_kernel 32`, `seq_len 8000` |
| Token geometry | 36 spatial tubes × 25 tokens = 900 per visual; 250 per signal; `T = 25`, `R = 10` |
| Masking | ADR-005 constant budget `n_budget 350`, caps 0.25 / 0.50, `video_min_tubes 2`, `physio_placement scattered` |
| Curriculum | `p_warm 0.10` for epochs 0–49, `p_target 0.50` for epochs 50–149 (`E_warm 50`) |
| Conditioning | decoder **prefixing** (`masking.json: conditioning.mechanism = prefix`, contract 7a) |
| Optim | AdamW, `blr 1.5e-4` → derived **3.750e-05** (batch 16/GPU × world 4 → global 64), cosine to 0, 10-epoch LR warm-up (1,050 steps) |
| Partition (data) | fold 0: 97 / 14 / 28 subjects, digest `dc80d46cf8e7` |
| Clips | train **6,762** from 838 sessions (of 7,205 candidates: 6 invalid, 241 sway, 196 sessions excluded); val **1,128** samples / 14 subjects; test pool **1,594** clips / 28 subjects |
| Precision | FP32 (`amp_dtype=none`) |

Two launch-time overrides of the YAML are part of the record:

- `--min_signal_spread 0 --rail_touch_v 0` — the raw-voltage guards are **off**;
  validity comes from the respiration cache's per-sample mask (saturation /
  flatline / short-run), as the `[data] stage2 corpus cleaning` line states.
- `--save_ckpt_freq 5` — the YAML says 10; the chain uses 5 so a 30-minute
  walltime kill loses at most four epochs.

## 2. Execution: 13-link `acc_short` chain

| Link | Job | Epoch range logged | Epochs | State | Elapsed |
|---:|---|---:|---:|---|---:|
| 1 | 57105538 | 0–15 | 16 | TIMEOUT | 30:03 |
| 2 | 57105539 | 16–31 | 16 | TIMEOUT | 30:11 |
| 3 | 57105540 | 31–45 | 15 | TIMEOUT | 30:02 |
| 4 | 57105541 | 41–54 | 14 | TIMEOUT | 30:24 |
| 5 | 57105542 | 51–66 | 16 | TIMEOUT | 30:10 |
| 6 | 57105543 | 66–80 | 15 | TIMEOUT | 30:13 |
| 7 | 57105544 | 76–89 | 14 | TIMEOUT | 30:02 |
| 8 | 57105545 | 86–98 | 13 | TIMEOUT | 30:24 |
| 9 | 57106063 | 96–108 | 13 | TIMEOUT | 30:21 |
| 10 | 57106064 | 106–119 | 14 | TIMEOUT | 30:02 |
| 11 | 57115448 | 116–125 | 10 | TIMEOUT | 30:03 |
| 12 | 57115449 | 126–139 | 14 | TIMEOUT | 30:05 |
| 13 | 57117055 | 136–149 | 14 | **COMPLETED** | 28:36 |

- Wall clock **6 h 33 min**, 4×A100, no queue gaps longer than ~30 s between links.
- **184 epoch executions for 150 epochs**: a link resumes from the last
  5-epoch checkpoint, so 0–5 epochs at the start of each link are re-run
  (34 duplicates, 22.7 % overhead). Epoch compute totals **4.76 h** at a mean
  **93.1 s/epoch**; the remaining ~1.8 h is the 13 × ~180 s bootstrap and the
  first-epoch cold-cache warm-up (one link's first epoch took 310 s).
- Link 13 exited itself after epoch 149 with ~1.5 min of allocation to spare.

## 3. Training trajectory (per-epoch means over 105 steps)

| Phase | `loss` first → last | `mse_tir` | `mse_rgb2d` | `mse_resp` | Mode-1 share |
|---|---|---|---:|---:|---:|---:|
| I (0–49) | 4.2380 → 1.2125 | 1.6697 → 0.6158 | 1.6520 → 0.5094 | 1.8327 → 0.1745 | 0.067–0.190 (mean 0.110) |
| II (50–149) | 1.3375 → 1.0013 | 0.6067 → 0.4379 | 0.5036 → 0.3563 | 0.4544 → 0.4143 | 0.333–0.590 (mean 0.493) |

The loss **steps up at epoch 50** (1.2125 → 1.3375) although the weights are
continuous: the mask distribution changes ($p$ 0.10 → 0.50), so the objective's
expected value changes. Within Phase II it falls monotonically to ~1.00. The
`sig_kernel 32` respiration term enters the loss with `signal_weight 0.5`, so
`loss = mse_tir + mse_rgb2d + 0.5 · mse_resp`.

## 4. Validation trajectory (1,128 samples / 14 subjects, `eval_freq 5`)

Validation masks follow the **phase's own** $p(e)$ with the same fixed-seed mask
sequence, so the curve is a within-phase observable (ADR-005 §2.7).

**Phase I — grounding, $p = 0.10$**

| Epoch | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 49 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `joint_loss` | 3.1512 | 2.2224 | 2.0369 | 1.8177 | 1.5496 | 1.4812 | 1.4313 | 1.4073 | 1.3960 | **1.3919** | — |
| `mse_resp` | 1.2952 | 1.0072 | 0.9207 | 0.5949 | 0.1602 | 0.0834 | 0.0477 | 0.0403 | 0.0332 | 0.0337 | — |

**Phase II — deployment, $p = 0.50$**

| Epoch | 50 | 55 | 60 | 65 | 70 | 75 | 80 | 85 | 90 | 95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `joint_loss` | 1.5946 | 1.5819 | 1.5859 | **1.5718** | 1.5810 | 1.5764 | 1.5758 | 1.5863 | 1.5820 | 1.5841 |

| Epoch | 100 | 105 | 110 | 115 | 120 | 125 | 130 | 135 | 140 | 145 | 149 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `joint_loss` | 1.5943 | 1.5996 | 1.6028 | 1.6034 | 1.6045 | 1.6033 | 1.6035 | 1.6047 | 1.6038 | 1.6039 | **1.6041** |

Final components (epoch 149): `mse_tir 0.7881`, `mse_rgb2d 0.5894`,
`mse_resp 0.4531`; `joint_loss = 0.7881 + 0.5894 + 0.5 · 0.4531 = 1.6041`.

The Phase-II curve is **flat from epoch 65 onward** (±0.003). That is a monitor
under a fixed mask mixture, not evidence that the representation stopped
improving — the decisive test is the frozen-encoder probe in
`resp_recon_arms_ab.md`, not a lower validation MSE.

The discontinuity at the boundary is **not a regression**: Phase I validation
sees ~74 respiration tokens on 89 % of steps, Phase II sees them on ~50 % of
steps (and ~74 on average when it does), so the objective is a different mixture.

## 5. Artefacts

| Item | Path |
|---|---|
| Checkpoints | `…/fold0/stage2/checkpoints/checkpoint-{0000,0005,…,0145,0149}.pth` — 31 files, **37 GB** (1.2 GB each) |
| Resume pointer | `…/checkpoints/latest_checkpoint.txt` → `checkpoint-0149.pth` |
| Metric records | `…/stage2/metrics.jsonl` — 31 eval records (`eval_freq 5`) |
| Selection | `…/stage2/selection.json` — `rule: fixed_epoch`, `selected_epoch: 149`, `joint_loss 1.6040643` |
| Masking provenance | `…/stage2/masking.json` — geometry, policy tuple, kernel recipient, conditioning, `mask_base 42` |
| Partition | `…/fold0/partition.json`, digest `dc80d46c…` |
| Runner logs | `code/logs/adr001_pretrain_5710{5538…5545,6063,6064}*.out`, `…5711{5448,5449,7055}*.out` |
| W&B | no `…/stage2/wandb/` directory — this run did not log to W&B |

## 6. Gate results

| Gate | Result |
|---|---|
| ADR-005 C1.1 exact sum `n_batch == 350` | **PASS** on 15,750 / 15,750 replayed steps; `n_batch = 350.0` in every logged epoch |
| ADR-005 C1.2 bounds / discreteness | **PASS** (visuals 2–9 tubes; respiration 0–125 in steps of 25) |
| ADR-005 C1.3 mode iff $K_{\text{resp}} = 0$ | **PASS** |
| ADR-005 C1.4 schedule | **PASS** — 11.01 % [10.19, 11.88] in Phase I, 49.33 % [48.38, 50.29] in Phase II |
| ADR-005 C1.5 replay from `(mask_base, epoch, step)` | **PASS** — every epoch mean matches the log to 0.000e+00 |
| G4 finite loss | **PASS** — no `NaN`/`Inf` in 150 epochs |
| Peak memory | not instrumented on the HPC run; the local smoke measured 6.18 GB allocated / 7.22 GB reserved at batch 8 (FP32, T4) |

The full Criterion-1 record and the kept-ratio figures are in
`kept_ratio_distribution.md` (same folder).

## 7. What this does NOT establish

- **Fold 0 only**; folds 1–4 are not run, so nothing here is a five-fold result.
- `joint_loss` is a **monitor** (masked reconstruction MSE under a phase-dependent
  mask mixture), not the thesis metric. It is not comparable across epoch 50.
- **No Stage-3 fine-tune** and no band-limited waveform metric has been computed;
  the headline of ADR-005 §3.2 remains open.
- The `rgb2d` tokenizer is **random-initialised** (I2): a null result for that
  stream cannot separate "carries nothing" from "tokenizer never converged".
- The run uses only one fold, one seed and one curriculum boundary (50/150). No
  ablation of $p_{\text{warm}}$, $E_{\text{warm}}$, the tubelet or the stream
  set exists in this record.
