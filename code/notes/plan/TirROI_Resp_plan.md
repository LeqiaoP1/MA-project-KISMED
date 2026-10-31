# TIR-ROI → Respiration — plan of record

> **Scope.** The `tir_roi` / `tir_roi_resp` lineage: thermal face-ROI video in,
> respiration waveform out.
> **This is the only plan for this study.** *Why* a component exists is in
> `code/notes/design/`; *what it measured* is in `code/notes/eval/tir_resp/`; the
> build order for an individual change is the child plan in §9.
> **State.** Pipeline complete and measured over the whole corpus: 5 folds
> trained and scored, Stage 3 assembled and evaluated per subject (§5). The open
> question is a **decision**, not an implementation — whether a reconstruction
> that loses to a single-tone template is worth further compute (§6–§7).

---

## 1. Question

Can the respiration waveform of an unseen subject be reconstructed from the
thermal face ROI alone?

"Performance" is the **subject-level** reconstruction quality: clips are assembled
into continuous per-session waveforms and compared with the raw `Resp_Volts`
reference, one number per held-out subject
(`design/Evaluation_RespWaveform_protocol.md` §1–§2). Stages 1/2 are judged only
by whether they move that number.

---

## 2. Contracts (locked; changing one is a new lineage)

| item | value | owning design |
|---|---|---|
| clip / fps / stride | 8.0 s, source 25 fps, `temporal_stride 2` (12.5 fps effective) | `design/SolutionA_asymCrossMAE.md` §2 |
| tubelet / grid | `(2,16,16)`; `num_frames 100` → `G_t = 50` | `design/SolutionA_asymCrossMAE.md` §2.1 |
| 1-D grid | `fs 100`, `sig_kernel 16`, `seq_len 800` → `n_signal 50`; **160 ms per token** | same |
| alignment | `tubelet_t·stride/fps == sig_kernel/fs` (hard `ValueError` in `MultiModalMAE`) | same |
| ROI landmarks | 12-point nose+mouth (`9,10,11,12,13,20,21,22,23,24,25,26`) | `design/ROI_crop_and_rescaling.md` §3.1 |
| ROI box | `roi_padding 0.10`, `crop_rule='square'` (square **before** the single isotropic rescale), `max_sway_px 40` (drop, not absorb) | `design/ROI_crop_and_rescaling.md` §3.2/§3.4 |
| patch | `input_size 96` on HPC (6×6 grid), `64` local (4×4); patch 16 | `design/ROI_crop_and_rescaling.md` §4.5 |
| clip validity | any `(0,0)` landmark line inside the window drops the clip; `min_signal_spread 0.1 V`; `rail_touch_v 9.9 V` | `design/ROI_crop_and_rescaling.md` §3.4 |
| target space | **dataset-side** `physio_norm: session`; the model is pinned to `target_norm: none` (`run_pretrain.pin_roi_target_norm`) | `eval/tir_resp/physio_norm_impact.md` |
| visual mask | tube `0.90` | `design/SolutionC_EarlyTokenMixing_stage2.md`, `design/SolutionA_asymCrossMAE.md` §3 |
| 1-D mask | `0.50` **scattered** in every shipped config; `physio_mask: span` + `mask_span_s {resp: 4.0}` is the designed alternative (W2) | `design/SpanMask_PhysioSignals.md` §4 |
| Stage-2 loss | per-stream masked MSE, `λ_tir 1.0`, `λ_resp = signal_weight 0.5`; **MR-STFT 0.0** (it belongs to Stage 3) | `design/SpanMask_PhysioSignals.md` §5, §7 |
| Stage-3 | `streams: tir`, `head_hidden 0`, `alpha/beta/gamma = 1.0`, `fft_sizes 128,256,512`, `eval_band 0.1–0.6`, fixed-epoch (Option A), `clip_stride 2.0` | `design/Evaluation_RespWaveform_protocol.md` §3, `design/kfold_val_strategy_design_v2.md` §3.3 |
| split | subject-disjoint, `n_folds 5`, `split_role train/val/test`; Stage 2 uses `pretrain_split: fold` so the test fold is excluded | `design/kfold_val_strategy_design_v2.md` §2 |
| evaluation | Hann overlap-add assembly → per-subject median → median + bootstrap 95 % CI over subjects | `design/Evaluation_RespWaveform_protocol.md` §4–§6 |

The two lineages (`96 px` HPC, `64 px` local) are **not checkpoint-compatible**:
`positions.<stream>.pos_embed` has a different shape. Stage 3 must mirror its
Stage-2 run, and `run_waveform` aborts when the ROI contract keys recorded in the
checkpoint disagree.

---

## 3. Stages

| stage | objective | design | code | config |
|---|---|---|---|---|
| **1 — init** | VideoMAE-B (K400) → `enc_blocks`, `enc_norm`; `Conv3d(3,D,(2,16,16))` copied **verbatim** into the resolved visual adapter (`adapters.tir` here) | `design/SolutionA_vs_SolutionC.md` §2–§3 | `core/multimae.py::load_pretrained_encoder`, `fit_visual_patch_embed` | `pretrained_encoder: videomae:k400` |
| **2 — SSL, shipped (Solution C)** | visible `tir + resp` tokens in one ViT; each stream reconstructed by the shared self-attention decoder over its **own** visible tokens + mask tokens | `design/SolutionC_EarlyTokenMixing_stage2.md` | `core/multimae.py::MultiModalMAE` | `stage2_hpc_tir_roi_resp.yaml` |
| **2 — SSL, designed (Solution A)** | resp tokens **never enter the encoder**; reconstruction by a target-driven cross-attention head, so it can only succeed by reading the video (removes the interpolation shortcut structurally) | `design/SolutionA_asymCrossMAE.md`, `design/SolutionA_vs_SolutionC.md` | `resp_in_encoder: false`, `signal_decoder: resp=cross_attn`, `cross_attn_depth` | `stage2_local_tir_roi_crossmae.yaml` |
| **3 — waveform** | encoder duplicated per target; mean-pool per time step; `Linear(D,16)`; α·StdLoss + β·Pearson + γ·MR-STFT; no masking | `design/SolutionC_EarlyTokenMixing_stage3.md` | `core/waveform_model.py`, `core/waveform_losses.py` | `resp_tir_roi_hpc.yaml`, `resp_tir_roi_crossmae.yaml` |
| **eval** | per-clip → session assembly → subject medians → pooled k-fold | `design/Evaluation_RespWaveform_protocol.md` | `evaluation/{metrics,assemble}.py`, `runners/run_evaluate_{session,subject,kfold}.py` | — |

**Precomputation.** Decode is the wall clock. `data/roi_cache.py` stores the
production ROI crop (square hull, `CACHE_FORMAT_VERSION` in the key) and measured
**11.7×** epoch speed-up at equal batch (`eval/tir_resp/hpc_precomputation.md`,
`eval/batch_size_sweep.md`). Build it with `runners/run_build_roi_cache.py`.

---

## 4. Code map

| concern | symbol |
|---|---|
| dataset, ROI algebra, clip validity | `data/tir_resp_dataset.py` (`roi_box_from_landmarks`, `missing_frame_mask`, `BP4DPlusTIRRespDataset`, `TirRoiResp{Pretrain,Finetune}Dataset`) |
| cache | `data/roi_cache.py` (`hull_box`, `cache_key`, `find_cache`) |
| task rotation | `data/task_groups.py` (`SHIPPED_TASK_GROUPS`, `resolve_task_selection`) |
| folds | `utils/splits.py::subject_folds` |
| masking | `core/multimae.py::_tube_mask`, `_span_mask`, `make_masks`, `parse_fft_sizes` |
| losses | `core/criterion.py::MaskedMSELoss`, `core/waveform_losses.py::WaveformJointLoss`, `MultiResolutionSTFTLoss` |
| metrics | `evaluation/metrics.py` (band-limited Pearson, single-tone template, amplitude, rate, `hf_power_rel`, bootstrap CI) |
| assembly | `evaluation/assemble.py` (Hann OLA, `affine_calibrate`) |
| runners | `run_pretrain`, `run_waveform`, `run_evaluate_kfold`, `run_evaluate_subject`, `run_evaluate_session`, `run_survey_tir_resp`, `run_inspect_tir_resp` |
| tests | `test_tir_resp_sentinel`, `test_tir_resp_rail_filter`, `test_roi_square_rule`, `test_roi_motion_gate`, `test_roi_cache`, `test_kfold_splits`, `test_physio_norm`, `test_assemble_session`, `test_evaluation_*` |

---

## 5. Measured position

Full numbers and caveats: `eval/tir_resp/resp_5fold_cv.md` (headline) and
`eval/tir_resp/FINDINGS.md` (physical gate). One screen:

**Stage-3, pooled 5-fold, 138 subjects / 1260 sessions / 21 231 clips**

| metric | model | baseline |
|---|---|---|
| band-limited Pearson (0.1–0.6 Hz), per-subject median | **+0.307** [+0.252, +0.350] | single-tone template **+0.535**; subjects beating it **18 %** |
| sessions with band Pearson > 0.5 | 30.5 % | — |
| amplitude rel. error (oracle-calibrated) | **0.682**; pred/ref in-band ratio **0.334** | 0 is perfect |
| rate MAE, per subject | **4.64 BPM** | pooled 30 s windows: 5.67 BPM, **34.4 %** within 10 % |

* Ordering across tasks is stable: **T3 best** (+0.567 / 0.469 / 1.08 BPM) and
  `low` best as a group (+0.411) > `high` (+0.291) > `moderate` (+0.278).
* **Physical gate (independent of the model):** the face ROI's strongest
  reference is **head motion, not the belt**, and the whole-frame winners are
  off-face rig/body artifacts. The pipeline detects motion at 8/8 with
  `|r|` 0.67–0.98, so sensitivity is not the limitation — coupling is
  (`eval/tir_resp/FINDINGS.md`).
* **What the model has learned is closer to "one dominant frequency" than to
  morphology** — which is what the template baseline formalises.

---

## 6. Acceptance and the stop rule

A continuation must clear **all** of A1–A3 on the same 5-fold subject-disjoint
protocol; a change that only improves the reconstruction loss does not count.

| # | criterion | current | required |
|---|---|---|---|
| **A1** | information content: share of subjects whose band-limited Pearson beats the single-tone template | 18 % | **> 50 %** |
| **A2** | rate: per-subject MAE and pooled-window agreement | 4.64 BPM; 34.4 % within 10 % | **< 3 BPM**; **≥ 60 %** |
| **A3** | amplitude: in-band pred/ref ratio under a **deployable** calibration | 0.334 (oracle) | rel. error **< 0.5**, calibration computed without the reference |

**Stop rule.** If a structurally different objective (Solution A, §7 W1) on the
same input fails A1, the line is reported as a **controlled negative** with the
mechanism in `eval/tir_resp/FINDINGS.md`, and further compute is not invested. The input — not
the architecture — is then the limiting factor.

**Reopening condition.** Only a **new input** that first passes the pixel-level
gate (a session where the ROI carries respiration phase with a genuine collapse
at anti-phase) reopens the line. Candidates, in order: raw/radiometric thermal if
a release exists; a tight nostril ROI read non-linearly; the body/side geometry
known to work for visual respiration. A chest/body ROI and a whole-frame arm are
**out of scope** — their winners are the rig.

---

## 7. Open work

Ordered by expected information per unit compute.

| # | work | where | closed when |
|---|---|---|---|
| **W1** | **Solution A arm.** Train the cross-attention (resp excluded from the encoder) variant, then its Stage-3 twin, on the identical corpus and folds. This is the only lever that changes what the encoder *can* learn rather than reweighting the same shortcut. | `configs/pretrain/stage2_local_tir_roi_crossmae.yaml` → an HPC twin; `configs/finetune/resp_tir_roi_crossmae.yaml` | A1 measured against the shipped arm on the same folds |
| **W2** | **Span-mask ladder E1–E3** on the W1 geometry (`physio_mask: span`, 1.5 / 2 / 4 s). Scattered masking on a ~30–60× oversampled band-limited signal is locally solvable, which is why the 1-D stream currently contributes no encoder signal after the ramp. | `core/multimae.py::_span_mask`; configs | A1 moves; report the span geometry printed at build |
| **W3** | **Amplitude.** The output is ~3× under-scaled. The `std` ↔ `smoothl1` A/B moved band-Pearson only marginally and did not close the scale gap, so the remaining choice is a scale-aware loss term versus a deployable per-session calibration. | `core/waveform_losses.py`, `utils/optim_factory.py` | A3 met with a non-oracle calibration |
| **W4** | **Rate.** The rate claim needs a stable spectral peak, not just phase locking: harmonic tracking over overlapping 30 s windows, reported as BPM. | `evaluation/metrics.py` (`estimate_rate_bpm`, `rate_metrics`) | A2 met |
| **W5** | **Fold cohort confound.** `subject_folds` splits sorted ids into contiguous blocks, so fold membership is confounded with sex/recording cohort (`eval/tir_resp/resp_t3_spread_sources.md` §2). Pooled numbers are unaffected; per-fold spreads are not a variance estimate. | `utils/splits.py`, `design/kfold_val_strategy_design_v2.md` §2.2 | fold assignment is stratified/interleaved, and the pooled number is unchanged |
| **W6** | **Optional bounded retry** on an on-face nostril/lip tile, cleared by all three of: rank/Spearman gate, local-motion partialling, gradient/sign-flip test. | `code/notes/eval/tir_resp/` tools | a gate pass, or it is not pursued |
| **W7** | **Doc/spec sync** after any geometry change: `roi_geometry_ledger.py` must still exit 0, and the change must be hand-applied to `code/notes/spec/` (`scripts/project/write_specs.py` still contains stale text and overwrites hand edits — read its warning before running it). | `code/notes/spec/`, `scripts/project/write_specs.py` | ledger PASS + spec diff reviewed |

**Not planned:** chest/body ROI, whole-frame arm, a BP pivot on thermal (weaker
and task-concentrated), and re-tuning the Stage-2 reconstruction loss for its own
sake.

---

## 8. Run recipes

From `code/`, with `.venv` active and `source scripts/env_local.sh` (local) or
`source scripts/env_hpc.sh` (cluster).

```bash
# 0. ROI cache (once per geometry; makes epochs ~11.7x faster). The ROI flags
#    must mirror the Stage-2 config, or the cache keys will not match.
python runners/run_build_roi_cache.py --raw_root "$RAW_DATA_PATH" \
    --out_root "$WORK_SCRATCH/roi_cache" --roi_padding 0.10 \
    --crop_rule square --clip_seconds 8.0        # or scripts/hpc/submit_roi_cache.sbatch

# 1. Stage 2, one fold (the test fold is excluded by pretrain_split: fold)
python -u runners/run_pretrain.py -c configs/pretrain/stage2_hpc_tir_roi_resp.yaml \
    --n_folds 5 --fold 0 --pretrain_split fold --output_dir <run>/fold0

# 2. Stage 3, that fold
python -u runners/run_waveform.py -c configs/finetune/resp_tir_roi_hpc.yaml \
    --n_folds 5 --fold 0 --finetune <run>/fold0/checkpoints/checkpoint-<N>.pth \
    --output_dir <run3>/fold0

# 3. pooled evaluation over the five folds
python runners/run_evaluate_kfold.py --root <run3> --run_prefix resp_tir_roi_hpc_fold \
    --folds 0,1,2,3,4 --out <json>

# 4. gates
python notes/eval/tir_resp/roi_geometry_ledger.py --raw_root "$RAW_DATA_PATH"
python -m evaluation.metrics          # metric self-test
```

Queue through `scripts/hpc/submit_stage3_kfold.sh` (per-fold driver) or
`submit_waveform_tir_roi.sbatch` (sequential). Smoke runs need
`--warmup_epochs 0` when `--epochs 1`, and always `-u` / `PYTHONUNBUFFERED=1`
(piped stdout is block-buffered; an interrupted run otherwise leaves no epoch
lines).

---

## 9. Doc map

| topic | document |
|---|---|
| this plan | `code/notes/plan/TirROI_Resp_plan.md` |
| ROI square box / rescale / motion gate — build order | `code/notes/plan/ROI_crop_implementation_plan.md` |
| task-tier regrouping | `code/notes/plan/task_tiers_regroup_plan.md` |
| k-fold rollout | `code/notes/plan/subject_kfold_plan.md` |
| AMP rollout | `code/notes/plan/hpc_AMP_implementation.md` |
| ROI + rescale contract | `design/ROI_crop_and_rescaling.md` |
| span masking design + experiment ladder | `design/SpanMask_PhysioSignals.md` |
| asymmetric cross-MAE (Solution A) | `design/SolutionA_asymCrossMAE.md`, `design/SolutionA_vs_SolutionC.md`, `design/SolutionA_asymCrossMAE_diagram.md` |
| shared-encoder MAE (Solution C) | `design/SolutionC_EarlyTokenMixing_stage2.md`, `design/SolutionC_EarlyTokenMixing_stage3.md` |
| evaluation protocol | `design/Evaluation_RespWaveform_protocol.md` |
| folds | `design/kfold_val_strategy_design_v2.md` |
| task tiers rationale | `design/TIR_ROI_RESP_TasksDistortion.md` |
| headline numbers | `eval/tir_resp/resp_5fold_cv.md` |
| physical gate / verdict | `eval/tir_resp/FINDINGS.md` |
| data quality, rail rule, geometry ledger, motion confound | `eval/tir_resp/resp_data_quality.md`, `eval/tir_resp/resp_rail_touch_filter_results.md`, `eval/tir_resp/roi_geometry_ledger.md`, `eval/tir_resp/resp_t3_spread_sources.md` |
| throughput / cache | `eval/tir_resp/hpc_precomputation.md`, `eval/batch_size_sweep.md` |
| generated contract | `code/notes/spec/01_data_pipeline.md`, `code/notes/spec/02_model_architecture.md`, `code/notes/spec/03_engine_and_loss.md` |
