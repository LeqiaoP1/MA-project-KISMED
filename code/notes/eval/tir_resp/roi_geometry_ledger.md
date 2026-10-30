# ROI geometry ledger — measured 2026-10-30

Contract of record: `roi_padding 0.10` + `input_size 96` + `max_sway_px 40`
(decided 2026-10-30; the retired pair was `0.20 / 112`).
Probe: `roi_geometry_ledger.py`. Verdict: **PASS, 0 mismatches** — every geometry
figure in `notes/design/ROI_crop_and_rescaling.md` reproduces from the raw tracks.

```bash
# from code/
python notes/eval/tir_resp/roi_geometry_ledger.py \
    --raw_root "$RAW_DATA_PATH" \
    --json ../output/eval/tir_resp/roi_geometry_ledger.json
```

~31 s, CPU only: no video decode, no cache, no GPU. It exits non-zero if a
published figure stops matching, so it is a gate, not a printout.

**Population.** The 21 231 emitted clips = the union of the five subject-disjoint
Stage-3 fold dumps (`output/finetune/resp_tir_roi_hpc_fold*/entries.json`), i.e. the
corpus AFTER the sentinel / rail / flat gates and BEFORE the motion gate. The probe
asserts the folds are session-disjoint, then reads each session's `IRFeatures` track
and applies the production box function. Source geometry is the corpus constant
726 × 480; clip 8 s @ 25 fps = 200 frames.

## Square crop side (`crop_rule='square'`, shipped `p = 0.10`)

| `roi_padding` | | pre-gate | after `max_sway_px = 40` |
|---|---|---|---|
| **0.10 (shipped)** | median | **95** | **95** |
| | p95 | **131** | **125** |
| | max | **353** | **174** |
| 0.20 (retired) | median | 111 | 110 |
| | p95 | 153 | 145 |
| | max | 413 | 202 |

The gate removes the 2× tail the rescale would otherwise have to absorb, in both
paddings. What the smaller padding removes is the ~15 px of margin that every box
carried — which is what lifts the face from 30.3 % to 41.1 % of the patch.

## Motion-gate attrition

| `max_sway_px` | clips kept | sessions kept | subjects kept | worst task |
|---|---|---|---|---|
| 30 | 19 952 (94.0 %) | 1227 / 1260 | 138 / 138 | T4 71.0 % |
| **40** | **20 558 (96.8 %)** | **1238 / 1260** | **138 / 138** | **T4 80.2 %** |
| 60 | 21 000 (98.9 %) | 1252 / 1260 | 138 / 138 | T4 91.6 % |
| 100 | 21 210 (99.9 %) | 1260 / 1260 | 138 / 138 | T4 98.8 % |

Per task at 40 px — **quote this whenever a per-task number is compared across the
gate**, because the gate changes the corpus, not just its quality:

| task | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 |
|---|---|---|---|---|---|---|---|---|---|---|
| kept % | 93.6 | 98.0 | 99.8 | **80.2** | 99.0 | 94.6 | 94.3 | 97.7 | 97.4 | 96.3 |

No subject loses all clips at any threshold, so a subject-level split stays valid.

## Isotropicity — σ = s / side

| padding | `s` | median σ | downscaled | upsampled | exactly native |
|---|---|---|---|---|---|
| 0.2 | 64 | 0.58 | 100.0 % | 0.0 % | 0.0 % |
| 0.2 | 96 | 0.86 | 83.3 % | 14.6 % | 2.1 % |
| 0.2 | 112 | 1.01 | 45.8 % | 51.4 % | 2.8 % |
| 0.2 | 128 | 1.15 | 18.7 % | 80.6 % | 0.7 % |
| 0.15 | 96 | 0.93 | 67.0 % | 30.3 % | 2.7 % |
| **0.1** | 64 | **0.67** | 100.0 % | 0.0 % | 0.0 % |
| 0.1 | 80 | 0.84 | 88.6 % | 9.8 % | 1.6 % |
| **0.1** | **96** | **1.01** | **46.1 %** | 50.8 % | 3.1 % |
| 0.1 | 112 | 1.18 | 16.4 % | 82.3 % | 1.3 % |
| 0.1 | 128 | 1.35 | 5.9 % | 93.6 % | 0.5 % |
| 0.0 | 80 | 1.00 | 46.7 % | 49.8 % | 3.5 % |

**σ is a density, not a size.** It is `s / side`, so `roi_padding` and `input_size`
are ONE decision: σ ≈ 1 sits at `(0.2, 112)`, at `(0.1, 96)` and at `(0.0, 80)`.
Changing one without the other moves the density, and density is the only thing this
column measures. The retired `0.2/112` and the shipped `0.1/96` share a density; they
differ in how much of the patch is face (30.3 % vs 41.1 %) and in tokens per frame
(49 vs 36).

## How much of the patch is actually FACE (2026-10-30)

`--face_sweep` answers "the crop looks too loose": it reports ONE frame's target
envelope as a fraction of the square box, which separates the three contributors a
loose crop is made of — the face, the head travel inside the window, and the
padding.

| `max_sway_px` | padding | clips | median side | face % of patch | σ @ paired `s` | upsampled > 1.25 |
|---|---|---|---|---|---|---|
| **40** | **0.10** | 20 558 | 95 | **41.1 %** | **1.01 @ 96** | **5.3 %** |
| 30 | 0.10 | 19 952 | 94 | 41.4 % | 1.02 @ 96 | 5.5 % |
| 20 | 0.10 | 18 473 | 93 | 42.1 % | 1.03 @ 96 | 5.9 % |
| 40 | 0.20 | 20 558 | 110 | 30.3 % | 1.02 @ 112 | 6.3 % |
| 40 | 0.15 | 20 558 | 103 | 35.1 % | 0.93 @ 96 | 0.9 % |
| 40 | 0.10 | 20 558 | 95 | 41.1 % | 1.18 @ **112** | 33.8 % |
| 40 | 0.00 | 20 558 | 79 | 59.0 % | 1.00 @ 80 | 40.5 % |

The last-but-one row is the mistake this ledger exists to prevent: same padding as
the shipped row, but `input_size` left at 112, so σ jumps to 1.18 and a third of the
corpus is upsampled past 1.25×.

Corpus-wide occupancy at the shipped setting (`p = 0.10`): p5 25.5 %,
**p50 41.1 %**, p95 52.6 %; only 0.9 % of clips below 20 %, against 6.8 % at
`p = 0.20` (p5 18.9 %, p50 30.3 %, p95 38.9 %). The worst kept clip of `M035_T6`
sits at 14.5 % — that session is the corpus' worst-cropped and is what the motion
gate exists for.

Read the two coverage columns together with σ, never alone. `face % of patch` says
how much of the grid the face covers; on its own it is not information. The face has
a fixed number of sensor pixels (~3600 in the median clip, ~60 × 60), so it is the
DENSITY (σ) that decides whether those photons are measured once or spread over more
grid points. At `p = 0.10 / s = 96` the median clip maps one sensor pixel to 1.01
grid points — the face spans ~61 grid points, exactly what it spans at
`0.20 / 112`. So the tighter padding buys FEWER TOKENS ON BACKGROUND (70 % → 59 %
of the patch) at the SAME resolution, plus 13 fewer tokens per frame. It does not
buy more resolution, and it does not cost any.

Three conclusions, in the order they matter:

1. **The motion gate is not an occupancy lever.** At `p = 0.10`, 40 → 30 → 20 px
   moves the median 41.1 → 41.4 → 42.1 % and only trims the tail (max side
   174 → 158), for 2.8 and 9.8 pp of the corpus. It bounds the pathological
   window, it does not tighten the typical crop. **Keep `max_sway_px: 40` under
   any padding**: 40 → 10 px costs ~30 pp of the corpus for 2–4 pp of occupancy.
2. **Padding is the only real lever on what the tokens look at — and it must move
   together with `input_size`.** Every padding has a paired grid that keeps σ ≈ 1
   (`0.20 → 112`, `0.10 → 96`, `0.00 → 80`). Shrinking padding alone (`0.10` at
   `s = 112`) buys nothing: occupancy rises, but so does interpolation (82.3 %
   upsampled, 33.8 % past 1.25×). Shrinking padding *and* the grid — the shipped
   `0.10 / 96` — raises occupancy 30.3 → 41.1 % at unchanged sampling density and
   drops the token count 49 → 36 per frame.
3. **Trimming the landmark set does not help either.** The square rule re-pads the
   short axis and the side is set by the WIDEST span, so dropping the nose bridge
   and lips (`nostril_mouth`) leaves the side at 115 px while the envelope halves
   (occupancy 48.9 % → 29.0 % on `M035_T6`); `nose_tip` / `nostrils` give 94 px but
   lower occupancy again (24.3 % / 10.7 %).

Consequence for the contract: **`roi_padding: 0.10`, `input_size: 96`,
`max_sway_px: 40`** (decided 2026-10-30). It is a different model front-end
(lineage-locked), so it needs the cache rebuilt and the encoder re-pretrained. Every
figure above is recorded for BOTH paddings, so the retired `0.20 / 112` geometry is
still reproducible from this script alone — but its pixel caches were **deleted on
2026-10-30** (the `pad0.10` build had passed `--check` and the parity probe), so
re-training that lineage would need a rebuild, not a re-read.

## Caveats

* The two shares do not sum to 100 %: 3.1 % of clips sit at exactly σ = 1
  (`side == s`), which is neither a down- nor an up-sample.
* `sway` here is the production statistic over the same 12 target landmarks; the
  numbers therefore move if `roi_landmarks` changes.
* The side is the production INTEGER side (`floor`/`ceil + 1` convention), which
  matches the design's figures to the pixel — no convention drift.
