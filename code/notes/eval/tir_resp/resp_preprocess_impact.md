# Respiration session preprocessing — impact of the cached grid (2026-10-31)

What was measured when `design/RESP_session_preprocess.md` was implemented
(plan: `plan/RESP_session_preprocess_implementation_plan.md`). Two things are
recorded: the **artefact** (what the cache contains) and what it does to the
**corpus accounting** at the shipped 8 s / 2 s hop. No GPU was spent; the numbers
below are dataset statistics, not model results.

> **This is a new lineage.** The label space changed from raw whole-file volts to
> **run-mean-removed** volts on an anti-aliased 100 Hz grid, so every previous
> number refers to a different label. Any retrain must use a new run prefix.

## 1. The artefact (current contract: saturation `< 1 s` kept, flatline `< 0.3 V`)

Built on Lichtenberg (single process, `deflt`, **0.4 min** for the whole corpus):

```
python runners/run_build_resp_cache.py --init --build \
    --out_root $WORK_SCRATCH/resp_cache --raw_root /work/projects/l0003511/test
```

| item | value |
|---|---|
| cache directory | `respgrid_src1000_fs100_sat9.95-1s_flat0.3-5s_run8s_v2_24852055a5` |
| sessions | **1399 shards + 1 skipped** (`F082_T1`, no `Resp_Volts.txt`) |
| size | 41.7 MB uncompressed `npz` |
| `--check 20` (bit-identical re-preprocess) | **PASS** (20/20) |
| `valid_frac` | mean **0.850**, median **1.000**, min **0.000**, max 1.000 |
| exactly `valid_frac == 1.0` | **1049 / 1399** sessions |
| invalid-sample share | saturation **2.8 %**, flatline **9.2 %**, short run **2.6 %**, non-finite **0.0 %** |
| sessions with `valid_frac == 0` | **149** (still present — the build never rules a session unusable) |

The whole-artefact identity is the **directory name** (a hash of the frozen
parameters), which is what the runners record as `resp_cache_key`.

To look at one session's mask, colour the waveform per sample:

```
python code/notes/eval/tir_resp/plot_resp_mask.py --session M033_T9 \
    --resp_cache "$WORK_SCRATCH/resp_cache" --zoom 60,76 \
    --out output/eval/resp_mask/M033_T9_resp_mask.png
```

green = `mask == True`; grey span/ribbon = `mask == False` (the stored value is
`NaN`, so no value is drawn); the lower rows show the raw 1 kHz samples, split
into kept / kept-at-rail / rejected.

### 1.1 Parameter history (so the numbers stay traceable)

| build | `sat_min_s` | `flat_ptp_v` | cache key | `valid_frac` mean | == 1.0 | == 0 |
|---|---|---|---|---|---|---|
| #1 (first contract) | 2.0 s | 0.05 V | `…_sat9.95-2s_flat0.05-5s_run8s_v2_f44043f526` | 0.940 | 1239 | 46 |
| #2 (design revision 2026-10-31) | 1.0 s | 0.5 V | `…_sat9.95-1s_flat0.5-5s_run8s_v2_1b22028f9e` | 0.802 | 946 | 191 |
| #3 (**decided 2026-10-31**) | **1.0 s** | **0.3 V** | `…_sat9.95-1s_flat0.3-5s_run8s_v2_24852055a5` | **0.850** | **1049** | **149** |

All three directories are kept on scratch, so the contracts can be A/B'd by
pointing `--resp_cache` at the out root with the knobs set to match.

A separate fix (`v1 -> v2`) derived the per-run resample length from the session
grid (`floor((n_src-1)*fs_out/fs_in)+1`) instead of `round(len*fs_out/fs_in)`,
so a run reaching EOF fills the LAST grid sample instead of leaving it `NaN`.

### 1.2 Why the flatline probe sits at 0.3 V

The flatline threshold was raised twice on 2026-10-31 (0.05 -> 0.5 -> 0.3 V).
The 0.05 V value only caught a channel **pinned** just inside the rail (the old
`min_signal_spread = 0.1 V` dead-signal guard measured every such window at a
median `+9.1 V`), and the corpus notes put genuine breathing at `>= 0.3 V` p-p
with a p10 of ~0.75 V. A 5 s peak-to-peak probe at **0.3 V** rejects a frozen
channel *and* a channel that only rings a little, while leaving the bottom of
the real-breathing distribution (M058's reference std is ~0.12 V, i.e. ~0.3-0.5 V
p-p) inside the mask. It is a calibrated compromise: p-p alone cannot separate
shallow breathing from a quiet dead channel, so the exact value is a policy
choice, not a measurement.

What each step cost, same corpus, measured:

| metric | #1 (0.05 V) | #2 (0.5 V) | #3 (0.3 V) |
|---|---|---|---|
| sessions with LOWER `valid_frac` than #1 | — | 395 of 1399 | **292** |
| … of those, fully invalid | — | 145 | **103** |
| median `valid_frac` drop among the affected | — | 0.426 | **0.324** |
| sessions with `valid_frac == 0` | 46 | 191 | **149** |
| flatline-dropped samples | 163 870 | 808 119 | **572 271** |

Going 0.5 -> 0.3 V **rescues 42 sessions** that #2 had declared fully invalid,
and every subject that #2 wiped out comes back: `M058` (152 clips, 9 sessions),
`M037` (186), `M055` (107), `M026` (37), `F011`/`F014`/`F064`/`M004`/`M014`
back to 1.000 where they had gone to 0.000. `F024` regains 3 clips. Six subjects
stay without a usable 8 s clip: `F010`, `F023`, `M017`, `M025` (mask empty) and
`F021`, `F027` (their valid stretches are too fragmented for an 8 s window even
though `valid_frac > 0`).

The residual cost is concentrated where the channel really is frozen: 149
sessions keep `valid_frac == 0`, 103 of them were usable under 0.05 V. Treat #3
as "only windows with visible modulation", and state that in any result that
uses it.

## 2. Corpus accounting at 8 s / 2 s (all 140 subjects)

| metric | legacy raw path | mask 2 s / 0.05 V | mask 1 s / 0.5 V | mask 1 s / 0.3 V (current) |
|---|---|---|---|---|
| subjects | 138 | 139 | 132 | **133** |
| sessions | 1260 | 1291 | 1146 | **1191** |
| clips | 21 231 | 22 144 | 18 575 | **19 769** |
| `all_clips_dropped` sessions | 118 | 87 | 232 | **187** |
| `missing_ir_features` | 15 | 15 | 15 | 15 |
| `resp_too_short` | 6 | 6 | 6 | 6 |
| missing `Resp_Volts.txt` | 1 | 1 | 1 | 1 |

Window-level drops under the current contract: **1148** sentinel, **835** head
sway, **1022** saturation hole, **2624** flatline hole, **560** short run,
**0** non-finite. (`resp_valid_fraction` over the used sessions: **0.849**.)

Clips per task: T1 1201, T2 1135, T3 3830, T4 354, T5 1226, T6 2269, T7 1354,
T8 2466, T9 5400, T10 534.

**The net is -6.9 % against the legacy raw path (21 231 -> 19 769) and -10.7 %
against build #1 (22 144 -> 19 769).** Six subjects that build #1 kept have no
usable 8 s clip at 0.3 V (`F010`, `F021`, `F023`, `M017`, `M025`, `F027`), which
is why the count is 133 subjects and not 139; `F024` came back at #3 with 3
clips. The flatline probe is still the dominant cause (`2624` flatline drops
versus `1022` saturation drops) but no longer the only story: brief rail grazes
return to the corpus, and a `>= 1 s` saturation event removes every window
covering it.

### `F004` — still the rail rule's headline case

`F004` was excluded *entirely* by the legacy rail rule (its channel is clamped at
`-10.000 V` for 40-80 % of samples, made of many short grazes). Under build #2 it
yields **6 sessions / 26 clips** (build #1: 51 clips), because the 1 s alarm now
also rejects the longer grazes: `F004_T9` keeps only `valid_frac` **0.125**,
`F004_T8` 0.920.

## 3. Analytic gate (G2): the anti-alias path is real

30 s synthetic session, 0.3 Hz respiration + a 0.1 V out-of-band tone, at
1000 Hz -> 100 Hz (threshold-independent):

| tone | `resample_poly` residual RMS | `np.interp` error RMS |
|---|---|---|
| 30 Hz (inside the 50 Hz output passband) | 7.071e-02 V | 7.071e-02 V |
| **130 Hz** (above the 50 Hz Nyquist) | **2.449e-04 V** | **7.071e-02 V** |

A tone above the output Nyquist is attenuated by ~49 dB by the artefact and is
folded into the band **unchanged** by point sampling.

Other pinned properties (`tests/test_resp_preprocess.py`, 24 cases):

* per-run DC removal is exact: adding 4.5 V to a session does not change the
  stored signal (`< 1e-5 V`);
* 0.3 Hz amplitude survives (std 0.707 +/- 1 %);
* a 40 ms invalid gap leaves `False` samples on the 100 Hz grid and **no 8 s
  window straddles it**;
* `np.isnan(signal) == ~mask` in both the writer and the loader;
* at the current contract a **0.5 s** rail contact is KEPT and a **1.5 s** one is
  invalidated; a **0.6 V p-p** ripple is kept and a **0.4 V p-p** one is flat.

## 4. Data path (G5)

Synthetic 8 s clip through `TirRoiRespPretrainDataset`: `tir [3, 100, 112, 112]`,
`resp [1, 800]`. The per-epoch `np.interp` is gone (the label is an integer slice).

Verified on the real corpus with BOTH caches (`TirRoiRespFinetuneDataset`, `F001`,
the shipped Stage-3 geometry: 8 s / 2 s hop / `input_size 96` / `roi_padding
0.10` / `physio_norm session`): `tir [3, 200, 96, 96]`, `wav [800]`,
**12.9 ms/sample**.

## 5. What must not be read from this note

* No model was retrained, so **no pearson / MAE number here is comparable** to
  `resp_5fold_cv.md`.
* The build #1 and build #2 accounting are **different corpora**: 1146 sessions
  vs 1291. Never compare a model trained on one against one trained on the other.
* `valid_frac` is a *coverage* statistic; a high `valid_frac` says nothing about
  whether the remaining signal is respiration.
* The 191 all-invalid sessions are still shards. Downstream they produce no
  clips, which is why they land in `all_clips_dropped`, not in the skip list.
