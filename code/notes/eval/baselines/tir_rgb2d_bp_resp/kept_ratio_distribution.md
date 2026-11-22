# Kept-ratio distribution of the four modalities — fold-0 pre-training (2026-11-22)

**Question.** What fraction of each modality's tokens was actually visible to the
encoder across the 150-epoch fold-0 run, and how is that split between the two
mask modes?

**Answer in one line.** Visuals averaged **15.6 % / 15.7 %** visible (140.8 / 141.3
of 900 tokens) and the physiologies **14.5 % / 12.7 %** (36.2 / 31.7 of 250), with
the two modes splitting the run almost exactly in half; in mode 1 both
physiologies are fully hidden, so their mean is carried entirely by mode 2.

## 1. Method (exact replay, not log sampling)

The per-step counts are a deterministic function of
`counts_generator(mask_base, epoch, step)`, so they were replayed offline for every
training step — **150 epochs × 207 steps = 31,050 draws** — with the checkpoint's
own `args` (`mask_base = val_seed = 42`, `n_budget = 350`, caps 0.25/0.50,
`video_min_tubes = 2`).

Validation that the replay is faithful: the epoch-0 mean it produces is
`kept_tir = 140.0966`, and the training log records
`'kept_tir': 140.09661835748793`. Exact to four decimals.

Token totals per step: `tir` 900 (36 tubes × 25), `rgb2d` 900, `bp` 250, `resp` 250.

## 2. Mode split

| Mode | Steps | Share |
|---|---:|---:|
| 1 — virtual sensing (physiology fully hidden) | 15,587 | **50.2 %** |
| 2 — physiology visible (capped), visuals get the residual | 15,463 | **49.8 %** |

Expected from `p_virtual_sensing: 0.5`; the run is balanced to within 0.4 pp.

## 3. Per-modality kept statistics over all 31,050 steps

| Stream | Total | min | q25 | median | q75 | max | mean | mean ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `tir` | 900 | 50 | 100 | 150 | 175 | 225 | 140.82 | **0.156** |
| `rgb2d` | 900 | 50 | 100 | 150 | 175 | 225 | 141.31 | **0.157** |
| `bp` | 250 | 0 | 0 | 0 | 74 | 125 | 36.17 | **0.145** |
| `resp` | 250 | 0 | 0 | 0 | 63 | 125 | 31.69 | **0.127** |

Bounds are exactly the configured polices: visuals never leave `[2, 9]` tubes of
36 (= 50–225 tokens), physiology never exceeds the 125-token cap. No step
violates the budget: the four columns sum to 350 on every draw.

## 4. Visual streams — kept tubes of 36

| tubes | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `tir` | 10.3 % | 10.5 % | 9.2 % | **17.3 %** | 15.7 % | 13.8 % | 12.0 % | 11.3 % |
| `rgb2d` | 10.1 % | 10.6 % | 9.1 % | **17.1 %** | 15.5 % | 13.8 % | 12.3 % | 11.4 % |

The two visuals are statistically indistinguishable — the apportioner splits the
visual residual evenly, so masking is symmetric between TIR and RGB2D. The
distribution is **not** uniform: it peaks at 5 tubes and is slightly
counter-intuitive at 4 (the least frequent value), because the two modes put
different mass there (mode 1 gives ~7 tubes each, mode 2 ~4.3).

## 5. Physiological streams — kept tokens of 250

| Stream | exactly 0 | 1–124 | exactly 125 (cap) | nonzero mean |
|---|---:|---:|---:|---:|
| `bp` | **50.2 %** | spread over 126 distinct values | 0.82 % | 72.6 |
| `resp` | **50.6 %** | spread over 126 distinct values | 0.47 % | 63.6 |

Half the steps hide each physiology completely (mode 1); in mode 2 the count is
drawn per stream and is close to uniform over `[0, 125]`, which is why the
histogram is a spike at zero plus a low plateau up to the cap.

## 6. Mode-conditional means (this is the shape ADR-002 intends)

| Mode | `tir` | `rgb2d` | `bp` | `resp` | Sum |
|---|---:|---:|---:|---:|---:|
| 1 (n = 15,587) | 174.8 (0.19) | 175.2 (0.19) | 0.0 | 0.0 | 350 |
| 2 (n = 15,463) | 106.5 (0.12) | 107.2 (0.12) | 72.6 (0.29) | 63.6 (0.25) | 350 |

Mode 1 spends the entire budget on 14 visual tubes split 7/7 — the encoder never
sees a physiological token, which is exactly the "virtual sensing" condition
evaluated in `resp_recon_arms_ab.md`. Mode 2 shifts ~136 tokens to the
physiologies and leaves ~214 (≈ 4.3 tubes each) to the visuals.

## 7. Figures

- `kept_ratio_hist.png` — per-modality histograms over all 31,050 steps.
- `kept_ratio_by_mode.png` — mean kept ratio per modality, split by mode.

## 8. Caveats

- Counts are **per step and batch-uniform** (ADR-002 contract 6): every sample in
  a step shares the count; only the *placement* differs per sample. So this
  histogram is a property of the policy, not of individual clips.
- It says nothing about **which** tokens were kept (tube placement / scatter),
  which is drawn per sample and per rank.
- The replay uses the checkpoint's stored `args`; a run whose `mask_base`,
  `n_budget` or caps differ would produce a different histogram.
