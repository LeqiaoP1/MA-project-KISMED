# ADR-008 — Fold-0 Mode-1 pretext evaluation (Criterion 2)

**Folder:** `code/notes/eval/baselines/tir_resp/` (see [`README.md`](README.md) for the run of record).
**Status:** measured 2026-11-26. **Non-gating** for Stage-3 adoption (ADR-008 §3.2); this file is the
evidence for that criterion, not a Go/No-Go decision. The gating Criterion 3 (five-fold Stage-3
band-limited Pearson, `>= +0.02` over the matched pure-MSE ADR-007 encoder) has **not** been run — the
comparator does not exist yet.

## What was evaluated

| | |
| :--- | :--- |
| Checkpoint | `/work/scratch/ne95ocyg/pretrain/adr008_2stream_fold0/fold0/stage2/checkpoints/checkpoint-0149.pth` (epoch 149 = all 150) |
| Arm | `configs/pretrain/stage2_budget_tir_resp_2stream_anatomy_adr008.yaml` (`alpha/beta/gamma = 1/2/1`, `lambda_resp 5.0`, `lambda_align 0.5`) |
| Probe | **Mode 1** (`--resp_probe hidden --kept_resp 0.0 --kept_tir 1.0`): `K_resp = 0`, `K_tir = 200`, `S_core = 1`, `S_periph = 7`, `regime = single_visual` |
| Role | fold 0 **test** (subject-disjoint, `partition.json`), 27 held-out subjects, 213 sessions, **1742 clips** × 5 mask draws |
| Command | `runners/run_eval_recon.py --split test --resp_probe hidden --kept_resp 0.0 --kept_tir 1.0 --repeats 5 --pretext_metrics --device cuda --out output/eval/adr008_pretext/mode1_fold0.json` |
| Report | `output/eval/adr008_pretext/mode1_fold0.json` (816 KB, with per-clip rows); trimmed copy + figure in [`artifacts/`](artifacts/) |

`--pretext_metrics` was added for this evaluation (variance retention + band-limited spectral peak
error, per ADR-008 §3.2). It is opt-in; every pre-existing report is unchanged.

## Criterion 2 — measured vs. the ADR's bands

| Indicator | Measured (subject median, 95 % CI) | ADR failure threshold | ADR success target | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Variance retention** `Var(ŷ)` | **0.472** `[0.457, 0.530]` | `< 0.10` | `>= 0.60` | above failure, **below target** |
| **Spectral peak error** | **4.92 bpm** `[4.01, 5.61]` | `> 3.0 bpm` | `<= 1.5 bpm` | **fails the failure threshold** |
| **Zero-shot masked `r`** | **+0.3256** `[+0.2444, +0.4061]` | `< +0.10` | `>= +0.35` | above failure, **just below target** |
| **Gradient balance** (median over 130 Phase-II epochs) | **66.5** (p25 44.9, p75 83.2, range 1.0–216.5) | `< 1e-3` | `[0.05, 0.50]` | clears failure, **far above the band** |

Supporting numbers: `mse 0.979` mean / `0.963` median (z², masked positions), `rmse 0.953`,
`mae 0.777`, per-clip `r_masked` mean `+0.3400` (sd 0.307, n = 1742). `Var(y) = 1` by the dataset's
per-clip z-score, so `Var(ŷ) = 0.47` with `r = 0.33` is internally consistent
(`r² = 0.109 ≈ 1 - MSE`).

## Comparison against the predecessors (the collapse this ADR was written to fix)

| arm | Mode-1 subject-median `r` | Mode-1 MSE |
| :--- | ---: | ---: |
| four-stream, pure MSE (`eval/baselines/tir_rgb2d_bp_resp/resp_recon_arms_ab.md`) | `+0.0062` | `1.025 z²` |
| three-stream, pure MSE (`eval/baselines/tir_rgb2d_resp/README.md`) | `+0.113` | `1.041 z²` |
| **ADR-008 two-stream composite (this file)** | **`+0.326`** | **`0.963 z²`** |

The conditional-mean collapse is **gone**: the subject-median correlation rose by a factor of ~53 over
the four-stream arm and ~2.9× over the three-stream arm, and `Var(ŷ)` is no longer ~0. Per ADR-008 §4's
rollback wording ("if Fold-0 pretext diagnostics still collapse, `Var < 0.10` or median `r < +0.10`"),
**the rollback trigger is not met** — but two success targets are also not met (see below).

## Honest reading of the three misses

1. **Spectral peak error 4.92 bpm fails, and it is not a resolution artefact.** The estimator's grid is
   `Δf = 0.0305 Hz = 1.83 bpm`, so 4.92 bpm is ≈ **2.7 bins** of genuine rate error (~0.082 Hz). Both
   `peak_hat` and `peak_y` are quantised the same way, so the *difference* in bins is meaningful. The
   predicted waveform carries the right *shape* (r = 0.33) but the wrong *rate* on a 8 s window — that
   is the expected failure mode of a correlation-shaped objective that has no phase/rate penalty beyond
   the band-limited PSD term.
2. **`r = +0.326` misses the `+0.35` target by 0.024**, with a CI that straddles it (`[0.244, 0.406]`).
   Calling this a miss or a hit is not decidable from one fold; it needs the other four folds.
3. **Gradient balance 66.5 vs. the `[0.05, 0.50]` band is almost certainly an ADR calibration error, not
   a training defect.** The matched pure-MSE comparator measured **4.47** on the same quantity in the
   1-epoch smoke — already ~9× above the band's top — so the band's lower/upper edges do not describe
   this implementation's scale. `mse_tir` never starved (1.96 → 0.78), which is the outcome the band
   exists to protect. **Do not tune `lambda_resp` on this number until the comparator's converged value
   is measured the same way.**

## Where the spread comes from: subject or task?

Each clip is one 8 s window from one **`<subject>_<task>`** session (the entry carries `subject`,
`task` (T1–T10), `task_level`, `t_start`). The per-subject panel of the figure is a median over **all**
of that subject's tasks — it does not isolate a task. Decomposing the 1742 clips:

| grouping | sd of medians | range |
| :--- | ---: | ---: |
| **subject** (27) | **0.203** | −0.18 (F028) … +0.65 (M058) |
| **task** (10, pooled clips) | 0.101 | +0.18 (T4) … +0.56 (T3) |
| (subject, task) session (213) | 0.238 | −0.38 … +0.84 |

**Subject identity dominates, at about twice the task effect**, and the session spread exceeds both —
i.e. window-to-window variation *within* one recording is a third, real component. A single clip inside
the best task can still fail: the figure's weak example is F053/**T3** at `r = −0.40` while T3 is the
best task overall.

Per task: `T1 +0.25 · T2 +0.39 · T3 +0.56 · T4 +0.18 · T5 +0.34 · T6 +0.26 · T7 +0.29 · T8 +0.32 ·
T9 +0.36 · T10 +0.30`. Grouped by the config's tiers: passive viewing `{T2,T3,T5} ≈ 0.43` >
high-distortion `{T6…T10} ≈ 0.31` > expression/startle `{T1,T4} ≈ 0.21` — physically sensible, since
laughter/expression moves the face and the thermal plume far more than passive viewing.

Three of 27 subjects sit near collapse (median `r < +0.10`): F028 `−0.18` (n = 4 clips), M026 `−0.08`
(n = 18), F019 `+0.01` (n = 7). Two caveats: the design is **unbalanced** (clip counts per task range
from 26 in T4 to 459 in T9 because the resp-cache validity filters drop different sessions), so this is
a descriptive decomposition of medians, not a two-way ANOVA; and these are **per-clip 8 s** Pearsons,
not the §3.3 band-limited *session* metric, which may reorder subjects and tasks once clips are
assembled into continuous sessions.

## What this justifies doing next

Per ADR-008 §5 (deferred, non-blocking) and §4 (rejection conditions), in order of information per
GPU-hour:

1. **Matched pure-MSE ADR-007 comparator, fold 0** — identical recipe, `lambda_resp 0.5`,
   `EXP_ROOT=.../adr007_2stream_fold0`. Without its converged `grad_ratio` and its own Mode-1 pretext
   row, rows 1–4 above have no baseline and Criterion 3 has no denominator.
2. If the comparator reproduces the same `grad_ratio` scale, **correct ADR-008 §3.2's band** rather than
   retuning the arm.
3. If the rate error persists across folds, the ADR's deferred `lambda_resp ∈ {2, 5, 10}` /
   `lambda_align ∈ {0.1, 0.5, 1.0}` sweep is the documented remedy — with the spectral term's weight
   (`gamma`) the more plausible lever for a *rate* error than `lambda_resp`.

## Reproduce

```bash
# the evaluation (writes the full per-clip JSON)
cd code && python -u runners/run_eval_recon.py \
    --checkpoint <EXP_ROOT>/fold0/stage2/checkpoints/checkpoint-0149.pth \
    --split test --resp_probe hidden --kept_resp 0.0 --kept_tir 1.0 \
    --repeats 5 --batch_size 16 --device cuda --pretext_metrics \
    --label adr008_fold0_mode1 --out ../output/eval/adr008_pretext/mode1_fold0.json

# the figure (this folder; --max_clips 96 for a ~20 s preview)
cd code && python notes/eval/baselines/tir_resp/figure_resp_reconstruction.py \
    --checkpoint <EXP_ROOT>/fold0/stage2/checkpoints/checkpoint-0149.pth \
    --out ../presentation/figures/resp-reconstruction-mode1-fold0.png
```
