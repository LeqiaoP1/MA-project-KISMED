# ROI geometry ledger — measured 2026-10-30

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

## Square crop side (production function, `crop_rule='square'`, `p = 0.2`)

| | pre-gate | after `max_sway_px = 40` |
|---|---|---|
| median | **111** | **110** |
| p95 | **153** | **145** |
| max | **413** | **202** |

The gate removes the 2× tail the rescale would otherwise have to absorb.

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
| **0.2** | **112** | **1.01** | **45.8 %** | 51.4 % | 2.8 % |
| 0.2 | 128 | 1.15 | 18.7 % | 80.6 % | 0.7 % |
| 0.1 | 64 | 0.67 | 100.0 % | 0.0 % | 0.0 % |
| 0.1 | 112 | 1.18 | 16.4 % | 82.3 % | 1.3 % |
| 0.1 | 128 | 1.35 | 5.9 % | 93.6 % | 0.5 % |

`p = 0.2, s = 112` puts the median clip at σ ≈ 1.0, i.e. the corpus is natively
~112 px — that is why the HPC lineage uses 112. A smaller padding (`0.1`) makes
every lineage a heavier upsampler, so there is nothing to buy by shrinking it.

## Caveats

* The two shares do not sum to 100 %: 2.8 % of clips sit at exactly σ = 1
  (`side == s`), which is neither a down- nor an up-sample.
* `sway` here is the production statistic over the same 12 target landmarks; the
  numbers therefore move if `roi_landmarks` changes.
* The side is the production INTEGER side (`floor`/`ceil + 1` convention), which
  matches the design's figures to the pixel — no convention drift.
