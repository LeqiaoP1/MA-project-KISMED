# Kept-ratio distribution of the three streams — fold-0 pre-training (2026-11-24)

**Question.** What fraction of each stream was visible to the encoder across the
150-epoch ADR-005 fold-0 run, how is that split between the two modes, and does
the virtual-sensing curriculum show up in the realised mask sequence?

**Answer in one line.** The two visuals averaged **16.8 %** visible (151.4 / 151.4
of 900 tokens) and respiration **18.9 %** (47.2 of 250), with the resolved Mode-1
share **11.0 %** in Phase I and **49.3 %** in Phase II — both inside the ADR-005
$\pm 0.02$ tolerance — and the replay reproduces every logged epoch mean to
0.000e+00.

## 1. Method (exact replay, not log sampling)

The counts are a deterministic function of
`counts_generator(mask_base, epoch, global_step)` followed by
`ConstantBudgetApportioner.sample_step(epoch=e)`, so they were replayed for every
training step — **150 epochs × 105 steps = 15,750 draws** — with the checkpoint's
own `args` (`mask_base = val_seed = 42`, `n_budget = 350`, caps 0.25 / 0.50,
`video_min_tubes = 2`, `p_warm = 0.10`, `E_warm = 50`, `p_target = 0.50`).

- **Script**: `code/scripts/project/mask_policy_report.py`
- **Record**: `policy_replay.json` (this folder)
- **Reproduce**:

```bash
python -m scripts.project.mask_policy_report \
  --checkpoint /work/scratch/ne95ocyg/pretrain/adr005_3stream_fold0/fold0/stage2/checkpoints/checkpoint-0149.pth \
  --logs 'logs/adr001_pretrain_571*.out' \
  --out-dir notes/eval/baselines/tir_rgb2d_resp
```

**Replay fidelity.** Comparing the replay's mean over each epoch's 105 steps with
the run's own `Epoch <e> stats:` line:

| Key | max absolute difference over 150 epochs |
|---|---:|
| `kept_tir` | **0.000e+00** |
| `kept_rgb2d` | **0.000e+00** |
| `kept_resp` | **0.000e+00** |
| `is_virtual` | **0.000e+00** |

The replay is exact, so the histograms below are the run's policy, not an
approximation of it.

Token totals per step: `tir` 900 (36 tubes × 25), `rgb2d` 900, `resp` 250.

## 2. Mode split

The ADR-005 policy has two labels: the **drawn** mode and the **resolved** mode of
contract 4 (resolved Mode 1 iff $K_{\text{resp}} = 0$). A Mode-2 draw with
$\tilde K_{\text{resp}} = 0$ therefore resolves to Mode 1.

| Label | Mode 1 | Mode 2 | Mode-1 share |
|---|---:|---:|---:|
| drawn | 5,695 | 10,055 | 36.16 % |
| **resolved** | **5,758** | **9,992** | **36.56 %** |

The observed share of Mode-2 draws that resolved to Mode 1 is
$0.626 \pm 0.079\%$ (s.e.), against the predicted $1/126 = 0.794\%$ — one draw in
160, i.e. consistent with the design.

The run-average 36.6 % is not a target: it is the mixture of 50 epochs at
$p = 0.10$ and 100 at $p = 0.50$, whose expected resolved share is
$(50 \cdot 0.1071 + 100 \cdot 0.5040) / 150 = 37.17\%$.

## 3. Per-stream kept statistics over all 15,750 steps

| Stream | Total | min | q25 | median | q75 | max | mean | mean ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `tir` | 900 | 50 | 125 | 150 | 200 | 225 | 151.41 | **0.168** |
| `rgb2d` | 900 | 50 | 125 | 150 | 200 | 225 | 151.36 | **0.168** |
| `resp` | 250 | 0 | 0 | 50 | 75 | 125 | 47.23 | **0.189** |

Bounds are exactly the configured policy: visuals never leave `[2, 9]` tubes of
36 (= 50–225 tokens); respiration never exceeds the 125-token cap. No step
violates the budget: the three columns sum to 350 on every draw (Criterion 1.1).

## 4. Visual streams — kept tubes of 36

| tubes | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `tir` | 5.5 % | 7.5 % | 9.2 % | **16.7 %** | 16.5 % | **17.0 %** | 14.7 % | 12.9 % |
| `rgb2d` | 5.3 % | 7.5 % | 9.5 % | **16.5 %** | 16.9 % | 16.6 % | 14.9 % | 12.8 % |

The two visuals are statistically indistinguishable: the apportioner splits the
visual residual evenly and the first stream gets only the integer remainder. The
distribution is a **mixture of the two modes** — Mode 1 splits 14 tubes as 5–9
each, Mode 2 draws 9–14 total — which flattens it relative to either mode alone.

## 5. Respiration — kept tokens of 250

| exactly 0 | exactly 125 (cap) | distinct nonzero values | nonzero mean |
|---:|---:|---:|---:|
| **36.56 %** | **12.27 %** | 5 (`25, 50, 75, 100, 125`) | 74.44 |

The Mode-2 rule charges the tube-alignment remainder to respiration, so its
counts are confined to the five multiples of $T = 25$ — coarser than the
250-token grid. **This is the intended consequence of ADR-005 §2.4**, which
spends the remainder on the only physiological stream rather than leaking it into
a visual stream.

## 6. Mode-conditional means (the shape ADR-005 intends)

| Resolved mode | `tir` | `rgb2d` | `resp` | Sum |
|---|---:|---:|---:|---:|
| 1 — virtual sensing (n = 5,758) | 174.7 (0.194) | 175.3 (0.195) | 0.0 | 350 |
| 2 — waveform inpainted (n = 9,992) | 138.0 (0.153) | 137.5 (0.153) | 74.4 (0.298) | 350 |

Mode 1 spends the entire budget on 14 visual tubes split 7/7 on average — the
encoder never sees a respiration token, which is exactly the condition probed in
`resp_recon_arms_ab.md`. Mode 2 moves on average 74 respiration tokens into the
budget and leaves ≈ 5.5 tubes to each visual.

## 7. Curriculum schedule (ADR-005 §2.7 / Criterion 1.4)

| Phase | Epochs | $p(e)$ | Mode-1 share (resolved) | Wilson 95 % CI | Expected |
|---|---|---:|---:|---|---:|
| I — grounding | 0–49 | 0.10 | **11.01 %** | [10.19, 11.88] | 10.71 % |
| II — deployment | 50–149 | 0.50 | **49.33 %** | [48.38, 50.29] | 50.40 % |

Per-epoch `is_virtual` lies in `[0.067, 0.190]` in Phase I and `[0.333, 0.590]`
in Phase II; with only 105 draws per epoch the per-epoch spread is expected
(s.d. ≈ 0.049 at $p = 0.5$). Against the ADR-005 §3.1.4 tolerance
($0.10 \pm 0.02$ / $0.50 \pm 0.02$), **both phases pass**. Phase II sits
1.07 pp below the deterministic expectation of 50.40 % (which includes the
$\tilde K = 0$ draw), i.e. 2.2 standard errors over 10,500 draws — sampling
noise, not a policy deviation.

Because `p(e)` changes at epoch 50, **the mask distribution is discontinuous
there by construction**: no mask statistic — and no validation loss — may be
pooled across that boundary (ADR-005 §2.7).

## 8. Criterion 1 conformance (from `policy_replay.json`)

| Criterion 1 item | Result |
|---|---|
| 1. Exact sum = 350 | **PASS** — 15,750 / 15,750 steps |
| 2. Visual counts ∈ {50, …, 225}, multiples of 25 | **PASS** (both visuals) |
| 2. Respiration ∈ {0, 25, …, 125} | **PASS** |
| 3. Mode 1 iff $K_{\text{resp}} = 0$ | **PASS** |
| 3. Mode-2 $\tilde K = 0$ share ≈ 1/126 | 0.626 % vs 0.794 % expected |
| 4. Phase Mode-1 fractions | **PASS** (table §7) |
| 5. Batch uniformity / replay from `(seed, epoch, step)` | **PASS** (exact vs logs) |
| 6. Gradient path with $K_{\text{resp}} = 0$ | not a policy-replay item; covered by the build gates (`tests/test_curriculum_schedule.py`) |
| 7. Feasibility rejection | build-time, not replay; covered by the test suite |

Criterion 1.5's *placement* half (per-sample, per-rank) is not observable from
the counts; it is a property of `core/mask_placement.py::placement_generator` and
is covered by the test suite, not by this replay.

## 9. Figures

- `kept_ratio_hist.png` — per-stream kept-token histograms over all 15,750 steps.
- `kept_ratio_by_mode.png` — mean kept ratio per stream, split by resolved mode.

## 10. Caveats

- Counts are **per step and batch-uniform** (contract 8): every sample of a step
  shares the count; only the *placement* differs per sample. This histogram is a
  property of the policy, not of individual clips.
- It says nothing about **which** tokens were kept (tube placement / scattered
  indices), which is drawn per sample and per rank.
- The replay uses the checkpoint's stored `args`; a run whose `mask_base`,
  `n_budget`, caps or curriculum keys differ would produce a different histogram.
- The run-average mode shares are a mixture of the two curriculum phases; use the
  phase rows of §7 for a like-for-like comparison with any other arm.
