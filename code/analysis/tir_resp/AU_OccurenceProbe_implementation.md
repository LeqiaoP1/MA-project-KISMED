# AU-Occurrence Probe — Implementation Record (TIR-ROI + Resp)

> Companion to the **protocol of record** `AU_OccurenceProbe.md`. That file
> defines *what* the probe must do; this file records *what was built* to satisfy
> it, where it lives, and the evidence that it works. On any conflict of intent,
> the protocol wins.

**Status:** implemented & unit-verified (single-process). Authored 2026-10-13.
**Scope:** diagnostic only — no Stage-1/2/3 code path is modified.

---

## 1. Deliverables

| File | Change | Contents |
|---|---|---|
| `code/data/au_tir_roi_dataset.py` | **NEW** | `AuTirRoiDataset` + pure helpers + `build_au_tir_roi_datasets` |
| `code/engines/au_probe.py` | **EDIT** | trivial-baseline + DDP-correct metrics + threshold tuning; `set_train` unwrap |
| `code/runners/run_au_probe.py` | **EDIT** | TIR-ROI branch, geometry flags, anatomical gate, baseline reporting, optional test split |
| `code/configs/finetune/au_tir_roi.yaml` | **NEW** | local 64 px / stride-2 lineage, Tier-A AU set |
| `code/tests/test_au_tir_roi_probe.py` | **NEW** | BDD acceptance tests (protocol §10.2) |

Reused unchanged: `core/au_probe.py` (`MultiModalMAEProbe`,
`build_au_probe_model`, `load_au_probe_weights`) and
`data/tir_resp_dataset.py::TirRoiRespPretrainDataset`.

---

## 2. Data layer — `data/au_tir_roi_dataset.py`

`AuTirRoiDataset` **subclasses** `TirRoiRespPretrainDataset`, so the ROI cache,
the `IRFeatures` sentinel gate, the static `roi_box_from_landmarks` box and the
`temporal_stride` decimation are reused **verbatim**. Only the label changes:

- each clip is labelled with the AU vector of its **centre frame**
  `frame_start + clip_frames // 2` (100 of the 200 source frames at 8 s / 25 fps);
- the respiration stream is present for geometry but **never returned** (no physio
  leakage): yields `(tir [3, num_frames, S, S] float32, au_label [n_au] float32)`;
- the resp-quality knobs `min_signal_spread` / `rail_touch_v` are forced `0.0`;
- only sessions that are BOTH AU-coded and TIR-usable survive;
  `self.entries` and `self.base.entries` are kept aliased so the base index stays
  aligned with `self.labels`.

Pure helpers (unit-tested without media):

| Symbol | Purpose |
|---|---|
| `au_probe_num_frames(clip_duration, fps, temporal_stride, tubelet_t)` | the encoder `num_frames` (stride-aware) |
| `centre_frame_index(clip_frames)` | label offset (`clip_frames // 2`) |
| `load_au_table(path, au_list)` | `AU_OCC` csv → `{frame0: label}`; drops `9`; errors on a missing target AU |
| `resolve_labels(entries, clip_frames, tables)` | keep only centre-frame-labelled clips |
| `discover_au_sessions(au_root)` | `<subject>_<task>.csv` discovery |
| `classify_au_set` / `check_au_gate` | Tier A/B/C gating (protocol §3) |
| `build_au_tir_roi_datasets(args, with_test=…)` | subject-disjoint `(train, val[, test])` |

Constants: `TIER_A_AU_LIST = (10,12,14,15,17,23,24)`,
`ROI_FORBIDDEN_AUS = (1,2,4,5,7)`, `ROI_BORDERLINE_AUS = (6,13)`,
`TIR_ROI_DATA_SETS = ('tir_roi','tir_roi_resp','tir-roi')`.

Frame-alignment prerequisite (protocol §4) is recorded per session in
`.frame_mismatch` (sessions whose AU coding extends past the thermal frame count)
and surfaced in `.warnings`.

---

## 3. Engine layer — `engines/au_probe.py`

`evaluate_au(data_loader, model, device, thresholds=None, tune_thresholds=False,
threshold_grid=DEFAULT_THRESHOLD_GRID)` returns:

| key | meaning |
|---|---|
| `f1`, `f1_per_au` | macro / per-AU F1 @ `thresholds` (default 0.5) — **headline** |
| `trivial_f1`, `trivial_f1_per_au` | always-positive baseline `2p/(1+p)` |
| `f1_minus_trivial` | headline − baseline (the honest number) |
| `presence_rate` | per-AU positive rate on the split |
| `oracle_f1`, `oracle_f1_per_au`, `oracle_thresholds` | per-AU-tuned **upper bound** (opt-in only) |
| `loss`, `n` | BCE loss, evaluated sample count |

- **DDP correct:** all predictions are `_gather_predictions`-concatenated across
  ranks *before* the F1 is computed ⇒ the metric is the **global** split, not one
  rank's shard.
- **Trivial baseline** is always reported; `tune_per_au_thresholds` is the only
  optional extra (`--tune_threshold`).
- `train_one_epoch_au` calls `_unwrap(model).set_train(True)` so a DDP-wrapped
  probe works.

Helpers exported: `trivial_f1`, `tune_per_au_thresholds`, `DEFAULT_THRESHOLD_GRID`.

---

## 4. Runner — `runners/run_au_probe.py`

New flags (all YAML-overridable):

```
--data_set {rgb|tir_roi}   --raw_root        --roi_landmarks  --roi_padding
--temporal_stride          --roi_cache       --roi_cache_lru  --strict_au_gate
--test_subjects            --tune_threshold
```

Flow when `data_set: tir_roi`:

1. resolve the AU set (`--au_list` / `--au_freq_topk`), then
   **`check_au_gate`** → strong warning, or hard error with `--strict_au_gate`;
2. build the probe model (`build_au_probe_model`, now stride-aware) and load the
   checkpoint;
3. `build_au_tir_roi_datasets(args, with_test=bool(--test_subjects))`;
4. train/eval with the trivial baseline printed every eval;
5. final report — if a test split exists and `--tune_threshold`, thresholds are
   tuned on **val** and applied to **test** (leakage-free, protocol §10.1);
   otherwise the @0.5 headline is reported on val.

The RGB path is unchanged (default `data_set: rgb`).

---

## 5. Config — `configs/finetune/au_tir_roi.yaml`

Mirrors the local Stage-2 TIR-ROI checkpoint
(`configs/pretrain/stage2_local_tir_roi_resp.yaml`): `streams: tir`,
`roi_landmarks: ''`, `roi_padding: 0.2`, `input_size: 64`, `clip_duration: 8.0`,
`temporal_stride: 2` (⇒ `num_frames 100`), enc 768/12/12,
`au_list: 10,12,14,15,17,23,24`. The **112 px HPC lineage is a separate config**
(change `input_size` + `finetune`). Every key is a valid runner option (the
unknown-key guard accepts it).

---

## 6. Tests — `tests/test_au_tir_roi_probe.py`

Maps 1:1 to the protocol's BDD scenarios (§10.2):

| Feature | Tests |
|---|---|
| Geometry & temporal-stride contract | `test_geometry_helper_matches_the_checkpoint_math`, `test_probe_model_uses_temporal_stride`, `test_runner_exposes_the_geometry_flags` |
| Anatomical AU gating | `test_classify_au_set_splits_the_tiers`, `test_au_gate_warns_on_forbidden_and_aborts_when_strict` |
| Centre-frame labelling | `test_load_au_table_*`, `test_resolve_labels_keeps_only_centre_frame_labelled_clips` |
| Trivial baseline / thresholding | `test_trivial_f1_matches_the_closed_form`, `test_evaluate_au_reports_probe_trivial_and_delta`, `test_tune_thresholds_recovers_a_hard_low_confidence_case`, `test_evaluate_au_oracle_upper_bound_is_opt_in` |
| DDP safety | `test_gather_is_a_noop_when_not_distributed`, `test_train_engine_unwraps_ddp_for_set_train` |
| Config wiring | `test_au_tir_roi_config_resolves_the_protocol` |

The suite needs **no raw media** (pure helpers + synthetic tensors).

---

## 7. Verification evidence (2026-10-13)

Run from `code/` (this workspace has no project venv/pytest; a shared torch
interpreter was used, and the same logic is mirrored by the pytest file):

- `python -m compileall` on all changed modules → **COMPILE_OK**.
- All **25** shipped configs in `configs/{pretrain,finetune}` parse through their
  runners → **0 failures** (the new `au_tir_roi.yaml` included).
- The BDD acceptance logic executed end-to-end → **ALL 31 CHECKS PASSED**,
  including: stride-2 probe ⇒ `num_frames 100`, `n_visual 800`; gate
  warn/abort; centre-frame label selection; `f1 100.00 / trivial 66.67 /
  delta +33.33` on a synthetic perfect predictor; threshold tuning; oracle
  opt-in; DDP no-op gather.

Not exercised here (needs the raw tree / a GPU): real media decode, the
`frame_mismatch` guard on live sessions, and multi-GPU DDP.

---

## 8. Run recipe

```bash
cd code
# pick + pin the AU set (once)
python runners/run_inspect_au.py --au_list 1,2,4,6,7,9,10,11,12,14,15,16,17,18,20,23,24,25,26

# C2 linear probe (single-process; DDP path exists but is unverified on-GPU)
python runners/run_au_probe.py -c configs/finetune/au_tir_roi.yaml \
    --max_entries 64 --max_sessions 4            # dev smoke: add caps

# C1 control (same config, only the checkpoint changes)
python runners/run_au_probe.py -c configs/finetune/au_tir_roi.yaml --finetune videomae:k400

# leakage-free threshold-tuning report
python runners/run_au_probe.py -c configs/finetune/au_tir_roi.yaml \
    --test_subjects F004 --tune_threshold
```

Report the probe macro-F1 **next to** its trivial baseline and Δ; the @0.5 number
is the headline, the tuned number is an upper bound only.

---

## 9. Known limitations / follow-ups

- **Resp file required:** `AuTirRoiDataset` inherits the base's requirement that
  each session has `IRFeatures` **and** `Resp_Volts.txt`; an AU-coded session
  without physiology is silently skipped (reported in `.warnings`).
- **Frame alignment** is *checked and warned*, not auto-corrected — sessions in
  `.frame_mismatch` should be excluded or offset-corrected manually
  (protocol §4).
- **HPC 112 px lineage** needs its own config (`input_size: 112` + the HPC
  checkpoint).
- **DDP** is code-correct but not yet run on multiple GPUs; the protocol still
  advises single-process until it is exercised.
- **Per-AU threshold tuning** requires a `--test_subjects` split to be
  leakage-free; without one it is reported only as a val-tuned upper bound.
