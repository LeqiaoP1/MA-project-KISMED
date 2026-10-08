# physio normalisation: clip-level vs session-level (measured)

**Date:** 2026-10-08
**Scope:** the 1-D respiration target of the TIR-ROI branch
(`data_set: tir_roi`, `--physio_norm {none, clip|zscore, session}`).
**Cluster:** Lichtenberg — CPU only, **no video decoded** (metadata + text files).
**Question:** the HPC pair ships `physio_norm: session` ("one target space for the
whole task"); every local tir_roi/rgb_roi config ships `clip`. What does that
choice actually cost?

---

## 1. How it was produced (reproducible)

```bash
# from code/  (login node; no GPU, nothing decoded)
python analysis/tir_resp/physio_norm_impact.py --workers 8 \
    --csv "$WORK_SCRATCH/physio_norm_impact/per_session.csv"
```

| | |
|---|---|
| tool | `analysis/tir_resp/physio_norm_impact.py` |
| per-session CSV | `$WORK_SCRATCH/physio_norm_impact/per_session.csv` |
| corpus | **1260 subject-task(s), 21 231 clip(s)** — identical to the training corpus |
| windows | built by the production dataset (`clip_seconds 8`, `clip_stride 2.0` s, `rail_touch_v 9.9`, `min_signal_spread 0.1`), so the kept windows are exactly the training ones |
| boundaries measured | 19 971 |

The statistics the two conventions use:

* `session` — one (mu, sigma) per **whole `Resp_Volts.txt`**, computed once by
  `_resp_full` and cached; every clip of the task shares it.
* `clip` — one (mu, sigma) per 8 s window, so each clip gets its own affine map.
* The model never re-normalises in this lineage: `run_pretrain.pin_roi_target_norm`
  forces `target_norm='none'` (line 405, called at 445).

---

## 2. Results

### 2.1 Where the variance lives (within a session)

| component | median | p90 |
|---|---|---|
| between clips (their means differ) | **10.7 %** | 38.9 % |
| within a clip (the waveform itself) | **89.3 %** | 98.3 % |

Only ~11 % of the kept samples' variance is clip-to-clip level drift — that is
the share a `clip` map removes on top of what `session` already removed.

### 2.2 Does the session map squash the surviving clips?

`compression = std(kept windows) / std(whole raw file)`:

| | value |
|---|---|
| median | **x0.980** |
| p10 | x0.790 |
| min | **x0.076** |
| sessions below x0.9 | 24.6 % |
| sessions below x0.7 | **6.1 %** |
| sessions below x0.5 | **2.2 %** |

Clip union coverage of the file: median **95.9 %**; window overlap factor x3.1
(8 s window on a 2 s hop). Rail fraction: 0.00 % median in *both* the file and the
kept windows — so this is **not** a rail problem.

**Mechanism.** The session (mu, sigma) is taken over the whole file, which still
contains the samples of the windows the rail/spread filters REJECTED. A few wild
windows therefore set sigma for the entire task, and every surviving clip is
squashed: 2.2 % of sessions keep less than half their amplitude, 6.1 % less than
70 %, and the worst keeps **7.6 %** — effectively no learning signal.

### 2.3 Per-clip target std under `session`

(1.0 == what the `clip` convention forces by construction)

| median | p10 | p90 |
|---|---|---|
| **0.856** | 0.636 | 1.069 |

This confirms, at corpus scale, the caveat already in the config comments: under
`session` the target std is NOT ~1, so **`alpha*StdLoss` is a genuine amplitude
term** rather than `|std(pred) - 1|`, and the deviation is systematic (median
below 1), not just noisy.

### 2.4 Boundary continuity of the assembled waveform

Normalised step between the last sample of one window and the first of the next
(one hop apart, so both are non-zero); reference: a 1 ms step inside a clip is
0.00104:

| convention | median step |
|---|---|
| `session` (signal change only) | 0.828 |
| `clip` (signal + map change) | 1.244 |
| **excess added by `clip`** | **x1.37** |

So the "per-clip z-scoring breaks the series at clip boundaries" argument is
real but **quantitatively modest** — consistent with the 11 % between-clip
variance share.

---

## 3. What this means for the decision

**The session-level choice is well-founded, and the cost of the alternative is
measurable but smaller than the phrasing suggests.** The `clip` convention adds
a x1.37 step at boundaries and discards the ~11 % between-clip level drift; the
`session` convention preserves cross-clip comparability and gives one coherent
target series per task, which is what a whole-task waveform reconstruction task
needs.

**Its real cost is concentration, not continuity:** with whole-file statistics,
**2–6 % of subject-tasks** have their surviving clips compressed below half to
70 % amplitude. Those clips still occupy sampler slots and still contribute
loss, but carry almost no signal — a small, silent, systematic loss of training
data, plus the StdLoss semantics change in 2.3.

**Surgical fix that keeps the rationale.** Compute the session (mu, sigma) from
the **kept-window samples only** (equivalently: drop the samples belonging to
windows the filters rejected) before z-scoring. That is the same one-map-per-task
convention — so cross-clip comparability and assembled-series continuity are
untouched — while removing the compression pathology for the affected 2–6 %. The
cost is one pass over the respiration file per session, already cached.

---

## 4. Stage 3: verified, not assumed

**1. Stage 3 does NOT re-normalise — the `session` map is effective.**
The Stage-3 model is `MultiModalWaveformRegressor`, built by
`build_waveform_model` (`core/waveform_model.py:319`) — a DIFFERENT class from
Stage-2's `MultiModalMAE`. It contains no per-token or per-clip z-score at all
(the only `norm` in that file is the encoder's `nn.LayerNorm`), and neither
`build_waveform_model` nor `run_waveform.py` passes a `target_norm` (no such
argument exists in the Stage-3 runner). The dataset-side map therefore survives
verbatim, so both stages really do share ONE target space. The
`run_waveform.py` help text's "(the model is pinned to identity)" is accurate,
but it is identity *by construction*, not by an explicit pin.

The algebra in §5 applies to `MultiModalMAE._targets` — i.e. Stage-2 pretrain
and `run_finetune.py` — **not** to Stage 3.

**2. The spectral term is NOT `MultiModalMAE`'s — so the config combination is
legal.** Stage 3 uses `WaveformJointLoss(alpha, beta, gamma, fft_sizes)`
(`core/waveform_losses.py`, built at `run_waveform.py:549`), so the
`MultiModalMAE` guard (`spectral_weight > 0` requires `target_norm='clip'`,
`core/multimae.py:396`) cannot fire from `run_waveform.py`, which builds the
regressor. `resp_tir_roi_hpc.yaml`'s `physio_norm: session` +
`fft_sizes: '128,256,512'` + `alpha: 1.0` is a valid combination.

**Caveat that does survive.** `alpha/beta/gamma` were chosen against a target of
std ~1. Under `session` the target std is **0.856 median / 0.636 p10** (§2.3), so
the time-domain term changes scale with the target while the MR-STFT term is
scale-sensitive as well — the loss balance is not the one those defaults were
chosen for. The runner's help text flags this; treat `alpha` as re-tunable per
normalisation rather than inherited across conventions.

**3. Still open — no cross-stage check.** `run_waveform.py` accepts
`--physio_norm` but never compares it against the value recorded in the
checkpoint it loads (`--finetune`), unlike `ROI_CONTRACT_KEYS`, which aborts on
mismatch. A Stage-3 run normalised differently from its Stage-2 pre-training is
therefore SILENT: it loads fine and trains, with a target space whose scale the
encoder never saw. This is the one genuine gap left — and the cheap fix is to
record `physio_norm` in the same contract.

---

## 5. Why a dataset-side map is cancelled by a model-side z-score

Reference for the Stage-2 (`run_pretrain.py` / `run_finetune.py`) path, and the
reason item 1 in §4 above had to be checked rather than assumed. For a clip's
target vector `x`:

```
z_token((x - mu)/sigma) = (x - mean_token(x)) / std_token(x)
```

The `sigma` cancels and `mu` is subtracted away, so a **per-token** model-side
z-score is exactly invariant to any dataset-side affine map, and a **per-clip**
model-side z-score behaves the same way. A dataset-side normalisation therefore
only has an effect while the model-side normalisation is identity -- which is
exactly why `run_pretrain.pin_roi_target_norm` exists for the MAE lineages, and
why Stage 3 had to be verified separately (it is identity, §4.1).
