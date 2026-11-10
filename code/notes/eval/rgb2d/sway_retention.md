# rgb2d / `rgb2d_max_sway_px` — Phase 0 measurement

**Measured** 2026-11-10 on the full 2D corpus, 1390 sessions / 7093 windows.
Script: `sway_retention_probe.py` (raw output `sway_retention.json`).

```
cd code
../.env/bin/python notes/eval/rgb2d/sway_retention_probe.py \
    --raw_root "$RAW_DATA_PATH"          # ~28 s, no video decode
```

## What is measured

The gate statistic is **`tir_resp_dataset.landmark_sway`** — the very function
the dataset calls: the centroid excursion
`max_t ‖c_t − median_τ(c_τ)‖₂` of the target landmarks inside one
`clip_seconds = 8 s` (200 frame) window. It is computed on the 49-point `'face'`
track at source resolution (1392×1040).

Two quantities are reported, and they answer **different questions**:

| | what it is | what it decides |
|---|---|---|
| `sway` | centroid excursion | **the gate** — may the model see this clip? |
| `span` | largest single-landmark excursion over the window | a LOWER BOUND on the box side; the stored box also spans the whole 49-point face and the whole task (measured 621–1036 px) |

They are *not* interchangeable: `span` is a per-landmark statistic while the
stored box is a set-and-time envelope. An earlier revision of this note
calibrated the threshold on `span` and concluded 330–340 px; that number is
**wrong** for this gate and is superseded here.

The gate is applied **per clip**, so two session-level retentions are reported:

* **sessions-all** — every window of the session survives (strict).
* **sessions-any-clip** — at least one window survives, i.e. the session still
  contributes to training.

## Distribution (px at 1392×1040)

| | p50 | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| per-window sway (gate) | 24 | 49 | 96 | 147 | 286 | 560 |
| per-session max sway | 56 | 119 | 211 | 281 | 414 | 560 |
| per-window span (box side) | 65 | — | 171 | — | — | 861 |

Retention if a clip is dropped when `sway > max_sway_px`:

| `max_sway_px` | windows | sessions-all | sessions-any-clip |
|---|---|---|---|
| 10 | 17.9 % | 2.1 % | 30.9 % |
| 15 | 31.9 % | 6.9 % | 48.7 % |
| 20 | 42.4 % | 13.7 % | 61.1 % |
| 30 | 58.8 % | 26.9 % | 75.8 % |
| 40 | 68.9 % | 37.6 % | 83.0 % |
| 60 | 80.5 % | 52.5 % | 89.9 % |
| 80 | 86.8 % | 62.5 % | 93.2 % |
| 100 | 90.6 % | 70.3 % | 95.2 % |
| 150 | 95.1 % | 81.8 % | 97.3 % |
| 200 | 97.2 % | 88.5 % | 98.4 % |
| **220** | **97.8 %** | **90.7 %** | 98.7 % |
| 240 | 98.3 % | 92.6 % | 99.0 % |
| 300 | 99.2 % | 96.3 % | 99.5 % |

## Matching the TIR gate (`max_sway_px = 40`)

TIR's 40 px is fixed. To make the RGB2D gate reject *the same head motion*, the
number cannot be copied — it must be converted on the same sessions.
`tir_rgb2d_sway_ratio.py` pairs both tracks window-by-window (1375 sessions with
an `IRFeatures` **and** a `2DFeatures` file, 7005 paired 8 s windows):

| | p50 | p75 | p90 | p95 | p99 |
|---|---|---|---|---|---|
| TIR sway (726×480) | 7.5 | 14.7 | 32.8 | 77.3 | 423 |
| RGB2D sway (1392×1040) | 24.1 | 48.3 | 94.9 | 144 | 282 |
| ratio rgb2d / tir | 3.38 | — | — | — | — |

Two independent conversions:

* **retention-matched (pinned): `110 px`** — the RGB2D threshold that rejects the
  same *fraction* of clips TIR@40 rejects (TIR keeps 91.9 % at 40 px → 91.7 % of
  2D windows at 110 px). Robust, because it does not depend on the tail shape.
* **scale-matched: `128 px`** — `40 × median(rgb)/median(tir) = 40 × 3.21`. Upper
  bound; it is inflated by TIR's tracking-failure tail (TIR p99 = 423 px vs
  RGB 282 px).

The two agree within ~15 %, so **110 px** is the recommended value. Note it is
*not* the same objective as “keep > 90 % of sessions”: at 110 px only ~72 % of
sessions keep every clip, because a session has several windows and each is one
more chance to exceed the threshold.

## Verdict

1. **Pinned: `110 px`** — matches the fixed TIR gate (see the section above):
   same statistic, converted on the same sessions, aligned so both gates reject
   the same fraction of clips.
   Alternatives by objective: `220 px` keeps **> 90 % of sessions losing no
   clip** (90.7 % of sessions, 97.8 % of windows — the p90 of per-session max
   sway is 211 px); `80 px` keeps 93.2 % of sessions in the weaker “still has one
   clip” sense.
2. **The library default is now `110 px`** (`trimodal_dataset.DEFAULT_RGB2D_MAX_SWAY_PX`,
   mirroring the TIR module's measured `DEFAULT_MAX_SWAY_PX = 40`), and `0` still
   disables the gate. The default changed once the TIR-matched value was measured:
   an unconfigured rgb2d run must not be the only stream without a motion gate.
3. **A px threshold is modality-relative, not a shared constant.** TIR's
   `max_sway_px = 40` keeps **96.8 %** of TIR clips at 726×480; the *same* 40 px
   keeps **68.9 %** of 2D windows at 1392×1040. Same statistic, ~3× stricter in
   face-relative terms. Neither number may be copied to the other modality —
   convert with `tir_rgb2d_sway_ratio.py` instead.
4. **Design consequence (open item).** The stored box is a task-wide envelope
   (621–1036 px), and a 96 px rescale of it would be mostly background — but the
   loader crops the CLIP's hull out of the stored box first, so the model sees a
   tight box and the envelope costs **disk only** (measured 1.37 GB/session,
   `build_preflight.md`). If the 2D stream underperforms, the first thing to try
   is still a **shorter `clip_seconds` for the 2D cache** (a new cache key, so a
   rebuild), not a looser gate. Measured with `--clip_seconds 2` (same corpus):

   | window | sway p50 | span p50 | windows kept at 220 px |
   |---|---|---|---|
   | 8 s | 24 px | 65 px | 97.8 % |
   | 2 s | 9.5 px | 40 px | 99.5 % |

   A 2 s box is ~40 % narrower, so the face fills ~1.6× more of the 96 px crop.
   That is a `params.json` change (a new cache key), so it is deliberately not
   done here.
