# AU-Occurrence Probe — TIR-ROI + Resp Stage-2 encoder

> **PRIORITY: HIGHEST for AU-probing.** This file is the protocol of record for
> the AU-occurrence probe. Where any other document disagrees about how the AU
> probe is defined, scoped, or configured, **this file wins**.

**Status:** protocol (implemented — see `AU_OccurenceProbe_implementation.md`).
Authored 2026-10-13.
**Scope:** evaluate the quality of the **Stage-2 TIR-ROI + resp** encoder by
FACS AU *occurrence* decoding. Diagnostic only — it modifies no Stage-1/2/3 path.

**Reuses unchanged — do NOT re-specify (see the specs):**
- `core/au_probe.py` — `MultiModalMAEProbe`, `build_au_probe_model`,
  `load_au_probe_weights` (spec: `spec/02_model_architecture.md` §9).
- `engines/au_probe.py` — `train_one_epoch_au`, `evaluate_au`
  (spec: `spec/03_engine_and_loss.md` §5).

**Supersedes, for the TIR-ROI encoder only** (the RGB probe sections stay valid
for RGB encoders):
`README.md` → "AU-occurrence probe"; `plan/ImplementationPlan.md` §5;
`plan/simplifiedPlan.md` (the "method unchanged" line).

---

## 0. Conflicts resolved

| # | Existing docs say | Resolution here |
|---|---|---|
| 1 | Probe input = canonical **RGB** frames (`data/au_dataset.py`) | TIR-ROI encoder is fed the **thermal ROI crop**; `streams: tir`. Needs a new TIR-ROI + AU-label dataset. |
| 2 | Target AUs = BP4D **12-AU** default `{1,2,4,6,7,10,12,14,15,17,23,24}` | **Invalid here:** eyes/brows are outside the ROI. Use the ROI-valid set (§3). |
| 3 | Controls C0 (random) / ImageNet-MAE / `videomae:base` | Current code is **VideoMAE-only** and **C0 removed**; `--finetune` is a hard error if blank. C1 ∈ {`videomae:k400`, `videomae:ssv2`}. |
| 4 | Geometry 4 s / stride 1 / `num_frames` 100 | Must equal the probed checkpoint: **8 s / stride 2 / `num_frames` 100**, `input_size` 64 (local) or 112 (HPC). |
| 5 | `plan/simplifiedPlan.md`: "method unchanged" for the stride-2 lineage | The method **changes** (modality + AU set); the geometry contract is enforced as before (§4). |

---

## 1. Objective & claim

Did masked pre-training on a **thermal nose+mouth ROI** learn facial-action
semantics that are **linearly decodable** from frozen features — rather than only
the reconstruction shortcut?

This is a **secondary** diagnostic. AU decoding is a *biased witness*
(`plan/ImplementationPlan.md` §2): report it **alongside** the Stage-3 respiration
number, never instead of it.

---

## 2. The anatomical gate (what the encoder actually sees)

The encoder input is the **12-landmark nose+mouth box** defined by
`data/tir_resp_dataset.py::TARGET_LANDMARKS = (9,10,11,12,13,20,21,22,23,24,25,26)`
— nose-bridge sides, nostril wings, mouth corners, upper/lower lips, lip centres
— padded by `roi_padding`. Brows, both eyes and the broad cheek are **outside** the
crop. Therefore an AU is a *valid* target only if its visual evidence lies inside
this box; anything whose primary signature is the eyelid, brow or cheek measures
the encoder on information it was never shown. Do **not** duplicate the ROI
mechanics here — they live in `data/tir_resp_dataset.py` (`roi_box_from_landmarks`,
`parse_ir_features`, `missing_frame_mask`).

---

## 3. AU-set selection

### 3.1 Tier A — headline set (ROI-valid ∧ BP4D-12 standard subset)

Comma list `--au_list`: **`10,12,14,15,17,23,24`**

| AU | Action / muscle | Evidence in ROI? |
|---|---|---|
| 10 | upper lip raiser (levator labii sup.) | **yes, strong** |
| 12 | lip corner puller (zygomaticus major) | **yes, strong** |
| 14 | dimpler (buccinator) | **yes, strong** |
| 15 | lip corner depressor (DAO) | **yes** |
| 17 | chin raiser (mentalis) | yes (lower lip strong; chin at the padded edge) |
| 23 | lip tightener | **yes** |
| 24 | lip pressor | **yes** |

### 3.2 Tier B — borderline (report, exclude from headline)
`6` (cheek raiser — main signature is the **lower eyelid**, out of ROI),
`13` (cheek puffer — mouth component only; essentially absent in BP4D+ per
`runners/run_inspect_au.py`).

### 3.3 Tier C — FORBIDDEN
`1, 2, 4` (brows), `5, 7` (eyelids). **Never** include; the encoder has no pixels
for them and they read as chance.

### 3.4 Tier D — optional extended set (nose/mouth AUs not in BP4D-12)
`9` (nose wrinkler), `11` (nasolabial), `16` (lower lip depressor),
`18` (lip pucker), `20` (lip stretcher), `25` (lips part), `26` (jaw drop).
Add an AU **only** if it is (i) present in the `AU_OCC` header **and**
(ii) corpus presence rate ≥ ~0.05. Report the extended macro-F1 as **secondary**.

### 3.5 Selection procedure (run once, pin the result)
```bash
cd code
python runners/run_inspect_au.py --au_list 1,2,4,6,7,9,10,11,12,13,14,15,16,17,18,20,23,24,25,26
# read AU_summary.json: presence_rate, subset_missing_from_corpus, mean_active_per_frame
```
1. Keep an AU iff it is in the header **and** `presence_rate ≥ 0.05`.
2. Partition by §3.1–3.4. Headline = Tier A; secondary = Tier A ∪ (Tier D survivors).
3. **Pin the resulting comma-list in the config.** Never use `--au_freq_topk` for
   the headline: frequency selection would wrongly promote eye/brow AUs.

---

## 4. Input contract (MUST equal the probed checkpoint)

Feed **thermal-ROI clips only** (`streams: tir`). The resp stream is **not** fed
(physio would leak autonomic state into the AU label — same rule as the RGB probe).

| key | must equal the checkpoint | local | HPC |
|---|---|---|---|
| `input_size` | 64 / 112 | 64 | 112 |
| `roi_landmarks` | `''` (12-pt nose+mouth) | `''` | `''` |
| `roi_padding` | 0.2 | 0.2 | 0.2 |
| `clip_duration` / `fps` | 8.0 / 25.0 | same | same |
| `temporal_stride` | 2 | 2 | 2 |
| `tubelet` | 2,16,16 | same | same |
| ⇒ `num_frames` | 100 | 100 | 100 |

**Per-sample construction (for an AU-coded frame `f_c`, 1-based):**
1. Window = the **8 s span centred on `f_c`** (200 source thermal frames).
2. ROI box = min/max of the 12 landmarks over **all 200 source frames** (static
   per clip), padded 0.2, clamped — i.e. `roi_box_from_landmarks`.
3. Decimate by `temporal_stride` → 100 frames; resize to `input_size²`; `/255`.
4. Label = `AU_OCC` occurrence vector at the **centre time** (`f_c`).
5. Drop the window if any source frame is a landmark sentinel
   (`missing_frame_mask`), the AU row contains a `9`, or the window is out of range.

> **Blocker — fix before trusting any number.** `runners/run_au_probe.py` has no
> `--temporal_stride` flag, so `build_au_probe_model` computes
> `num_frames = clip_duration·fps = 200` against a stride-2 checkpoint (100).
> `_resize_t` then silently trims to the **first 100 frames**, while the dataset
> centres the label on frame 100 of 200 → **the label is no longer at the clip
> centre.** Thread `temporal_stride` through the runner **and** the dataset.

> **Prerequisite — AU↔thermal frame mapping.** `AU_OCC` frames are coded on the
> **visible** video (`jpg index = f − 1`, verified in `data/au_dataset.py`); the
> thermal video is a separate sensor (`data/alignment.py`). Adopt the 1:1
> assumption only after verifying per session that
> `n_thermal (== IRFeatures lines) == n_canonical_rgb_jpg == AU_csv span`.
> Sessions that disagree (precedent: `F003_T8`, `plan/refined_RGB_BP_study.md` §3.2)
> are excluded or re-aligned by a fitted offset — never probed silently.

**Reuse the ROI cache.** The offline cache (`analysis/tir_resp/hpc_precomputation.md`,
`data/roi_cache.py`) stores native pixels + landmarks, is hop-independent, and is
keyed on `roi_landmarks`/`roi_padding` — the same keys `stage2_hpc_tir_roi_resp.yaml`
uses. Build the probe dataset from that cache rather than re-decoding WMV.

---

## 5. Probe protocol

- **Read-out:** mean-pool over visible tokens → `Linear(D, n_AU)`; no CLS token.
- **Modes:** `--probe linear` = the formal diagnostic (frozen encoder, head only);
  `--probe ft` = secondary upper bound.
- **Split:** **subject-disjoint**, and the **same subject split as the Stage-3
  evaluation** so the diagnostic and the downstream number describe the same
  held-out subjects.
- **Loss / threshold:** `BCEWithLogitsLoss`; F1 threshold 0.5.
- **Controls** (identical protocol; only `--finetune` changes):
  - **C1** Stage-1 VideoMAE (`videomae:k400` / `videomae:ssv2`) — the 3-D tubelet
    patch embed transfers into `adapters.tir` via `fit_visual_patch_embed`.
  - **C2** Stage-2 TIR-ROI + resp checkpoint.
- **Metrics, always together:**
  1. **Headline** — macro-F1 @ 0.5 over **Tier A**.
  2. Per-AU F1 table (`_print_au_table`).
  3. **Trivial always-positive baseline per AU**, `F1 = 2p/(1+p)` from that AU's
     presence rate `p`. Not optional: AU10/12/14 fire on 50–66 % of coded frames,
     so a degenerate all-positive head already scores macro-F1 ≈ 0.7. Quote the
     probe F1 **minus** this baseline.
  4. Secondary: extended-set macro-F1; `probe: ft` macro-F1.

> **DDP caveat (from the AU-engine review).** `evaluate_au` does not all-reduce
> `tp/fp/fn`, and `model.set_train(True)` targets the DDP wrapper. Run the probe
> **single-process** until both are fixed.

---

## 6. Reporting & acceptance

- Report **{C1, C2} × {linear} × {Tier A}**, each with probe-F1, trivial-F1,
  Δ(headline − trivial), and the per-AU table.
- Judge a representation informative only if **C2 linear-F1 > trivial-F1 on a
  majority of Tier-A AUs** *and* ≥ C1 linear-F1.
- State the AU set, `input_size`, `temporal_stride` and the exact checkpoint path
  with every number (the geometry lineage is not interchangeable).
- Aggregate per subject (median/IQR + bootstrap CI over subjects), as in
  `design/Evaluation_Protocol.md`.

---

## 7. Confounds (state explicitly)

- **Thermal, false-colour, ~100 px, representation-limited.** The ROI is 8-bit
  palettised, near the saturated end of the palette (`analysis/tir_resp/FINDINGS.md`).
  Do **not** compare TIR-ROI F1 head-to-head with RGB-full-frame F1 as an encoder
  ablation.
- **Truncated face.** Five standard AUs are unobservable; the probe characterises
  lower-face semantics only.
- **Training distribution.** Stage 2 pre-trains on all subjects → the probe
  measures fidelity on the training distribution, not OOD generalisation.
- **Head motion / scene.** A static per-clip ROI sweeps background; keep
  `roi_padding` fixed and cite the global-motion control (`frame_argmax`) if a
  motion claim is made.
- **Class imbalance.** Mitigated only by the Tier-A set + the always-positive baseline.

---

## 8. Implementation map

| Piece | Status |
|---|---|
| `core/au_probe.py` (`streams=('tir',)`) | reuse as-is |
| `engines/au_probe.py` | reuse; add the trivial-baseline metric |
| `data/tir_resp_dataset.py`, `data/roi_cache.py` | reuse for ROI crop + sentinel rule |
| **NEW** `data/au_tir_roi_dataset.py` (or a `data_set: tir_roi` branch in `data/au_dataset.py`) | **required** — `AuOccurrenceDataset` reads canonical RGB only |
| **EDIT** `runners/run_au_probe.py` | add `--temporal_stride`, `--roi_landmarks`, `--roi_padding`, `--data_set`, `--raw_root`; thread stride into model + dataset |
| **NEW** `configs/finetune/au_tir_roi.yaml` | §8.1 |
| `runners/run_inspect_au.py` | reuse to pick the AU set (§3.5) |

### 8.1 Config sketch — `configs/finetune/au_tir_roi.yaml`
```yaml
data_set: tir_roi
raw_root: ../data/raw/BP4D            # Thermal/ + IRFeatures/ (+ Physiology/)
au_root: ../data/raw/BP4D/AUCoding/AU_OCC
streams: tir                          # thermal ROI only (no resp: would leak)
tir_channels: 3
roi_landmarks: ''                     # 12-point nose+mouth box  (= the checkpoint)
roi_padding: 0.2
input_size: 64                        # 112 for the HPC lineage
tubelet: 2,16,16
fps: 25.0
clip_duration: 8.0
temporal_stride: 2                    # -> num_frames 100  (REQUIRED plumbing)
num_frames: 0
au_list: 10,12,14,15,17,23,24         # Tier-A headline (NOT the default 12-AU set)
probe: linear
pool: mean
finetune: ../output/pretrain/stage2_local_tir_roi_resp/checkpoints/checkpoint-0039.pth
train_ratio: 0.8
epochs: 30
batch_size: 4
opt: adamw
lr: 1.0e-3
min_lr: 1.0e-4
warmup_epochs: 3
weight_decay: 0.05
clip_grad: 0.0
num_workers: 4
```

---

## 9. Run recipe
```bash
# 0) pick + pin the AU set (once)
python runners/run_inspect_au.py --au_list 1,2,4,6,7,9,10,11,12,14,15,16,17,18,20,23,24,25,26
# 1) verify the AU<->thermal frame mapping per session (§4) -> drop mismatches
# 2) C2 linear probe (single-process; DDP not yet safe)
python runners/run_au_probe.py -c configs/finetune/au_tir_roi.yaml
# 3) C1 control (same config, only the checkpoint changes)
python runners/run_au_probe.py -c configs/finetune/au_tir_roi.yaml --finetune videomae:k400
```

---

## 10. Technical Improvements & BDD Acceptance Criteria

### 10.1 Technical Improvements for Implementation
1. **Dataset Inheritance:** Instead of duplicating logic in a new `au_tir_roi_dataset.py`, subclass the existing `TirRoiRespFinetuneDataset` (which already handles the ROI cache, sentinel rules, and `temporal_stride` correctly). Override `__getitem__` to return the AU label for the center frame instead of the respiration waveform.
2. **Dynamic Thresholding:** Hardcoding the BCE threshold at `0.5` is suboptimal for highly imbalanced AUs. Improve the engine to tune the decision threshold per-AU on the validation split (e.g., searching `[0.1, 0.9]` to maximize F1), then apply those thresholds to the test split.
3. **DDP Fix Specification:** Resolve the DDP blocker by stacking `[tp, fp, fn]` into a single tensor, calling `torch.distributed.all_reduce()`, and replacing `model.set_train(True)` with `(model.module if hasattr(model, 'module') else model).set_train(True)`.

### 10.2 BDD Acceptance Criteria (Tests)

**Feature: Geometry and Temporal Stride Contract**
* **Scenario:** The probe prevents silent temporal mis-centering.
  * **Given** a Stage-2 checkpoint trained with `temporal_stride: 2`
  * **When** the AU probe is launched without specifying `--temporal_stride 2`
  * **Then** the runner must raise a geometry mismatch error during model initialization.
  * **And** the dataset must extract exactly 200 source frames, center the AU label at frame 100, and only then decimate to 100 frames.

**Feature: Anatomical AU Gating**
* **Scenario:** The probe prevents evaluation on unobservable AUs.
  * **Given** the TIR-ROI input is restricted to the 12-point nose+mouth box
  * **When** the user provides an `--au_list` containing Tier C AUs (e.g., AU1, AU2, AU4)
  * **Then** the runner should emit a strong warning or abort, stating that upper-face AUs are outside the anatomical crop.

**Feature: Trivial Baseline Reporting**
* **Scenario:** The probe contextualizes F1 scores against class imbalance.
  * **Given** the evaluation loop has finished processing the validation set
  * **When** the final metrics are printed and logged
  * **Then** the output must include the "always-positive" baseline F1 for each AU.
  * **And** the headline metric must report the $\Delta$ (Model F1 - Trivial F1).

**Feature: Distributed Evaluation (DDP) Safety**
* **Scenario:** The probe computes correct global metrics across multiple GPUs.
  * **Given** the probe is running via `torchrun` on multiple GPUs
  * **When** `evaluate_au` aggregates the predictions
  * **Then** `tp`, `fp`, and `fn` counts must be synchronized across all ranks before computing the macro-F1 score.
