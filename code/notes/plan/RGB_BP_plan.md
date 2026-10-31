# RGB → Arterial BP — plan of record

> **Scope.** The visible-light lineage: RGB face video in, arterial blood-pressure
> waveform out (and respiration out of the same encoder, as the second branch).
> **This is the only plan for this study.** *Why* a component exists is in
> `code/notes/design/`; *what it measured* is in `code/notes/eval/`; the build
> order for an individual change is the child plan listed in §10.
> **State.** Front end, datasets, runners, metrics and configs are implemented;
> the external rPPG gate is measured (§5); **no Stage-2/3 run of this lineage
> exists yet** (§7 W2 is the first one).

---

## 1. The study

Two cells share one encoder:

| cell | input | target | config pair |
|---|---|---|---|
| **rgb → bp** (primary) | RGB face clip | `BP_mmHg.txt` waveform, 100 Hz | `stage2_local_rgb_roi_bp.yaml` → `bp_rgb_roi_local.yaml` (RAW ROI, 64 px); `stage2_multimodal.yaml` → `bp.yaml` (canonical, 224 px) |
| rgb → resp | RGB face clip | `Resp_Volts.txt` waveform, 100 Hz | `stage2_local_pretrained.yaml` → `resp_local.yaml` (canonical, 64 px). The RAW-ROI twin is **W6**; the 224 px `resp.yaml` reuses a Stage-2 run whose streams were `rgb,bp`, so `heads.resp` does not exist there and that head starts random. |

Stage 3 simulates **complete contact-sensor failure**: only the visual stream is
fed and the model must generate the waveform that is physically absent
(`design/SolutionC_EarlyTokenMixing_stage3.md`). Per-clip training plus offline
session assembly is the deployment shape, not a training stage.

**EDA is out of scope.** It is not observed by any camera (it is the electrical
conductance of skin, driven by sudomotor activity), the BP4D+ label is a
finger/palm contact measurement, and every remote estimate would be a surrogate
for a *palmar* process. `configs/finetune/eda.yaml` remains as a legacy branch
and is not part of this study.

---

## 2. Premise (what the measurement says is possible)

The premise is measured, not assumed: the BP4D+ target is a **1000 Hz finger-cuff
arterial trace** (not a cuff reading), and the visible-light face carries a
**skin-borne, motion-independent cardiac component** (rPPG).

Measured on this corpus — full detail and controls in
`code/notes/eval/rgb_bp/FINDINGS.md`:

| claim | measurement |
|---|---|
| the face carries a cardiac signal | **90 %** of sessions detect it (CHROM/POS + skin mask), vs **3–5 %** for a mismatched-subject control and 10 % for a face-free patch |
| it is blood, not head motion | survives regression of head pose, landmark centroid, inter-scale and frame mean (33/39) |
| it supports **rate** measurement | **no**: HR skill ≈ 0 at 20 s windows against **+0.305** for the BP positive control and **+0.278** for a synthetic pulse |
| the pixel budget is the lever | naive `G−R` gives 68 % detection, CHROM/POS + skin masking gives 90 % |

Consequences the plan inherits: (a) the front end is a **contract**, not a detail
(§3); (b) a waveform claim must clear the rate and amplitude baselines of §6, not
just a Pearson correlation; (c) a model that cannot track rate on 20 s windows
cannot be shown to track morphology.

---

## 3. Front-end contract

### 3.1 Two geometries, never interchangeable

`positions.<stream>.pos_embed` is `[1, G_t·Gh·Gw, D]`, so the spatial grid and the
temporal grid both enter the checkpoint. A 224 px / 14×14 model and a 64 px / 4×4
model are **separate lineages**; Stage 3 must mirror its Stage-2 run exactly.

| lineage | `data_set` | visual input | `input_size` | clip | `temporal_stride` | frames | `G_t` | `sig_kernel` | `seq_len` | configs |
|---|---|---|---|---|---|---|---|---|---|---|
| canonical, thesis scale | `bp4d+` | full frame, centre crop | 224 | 4.0 s | 1 | 100 | 50 | 8 | 400 | `stage2_multimodal.yaml` (800 ep), `bp.yaml`, `resp.yaml` |
| canonical, local | `bp4d+` | full frame, centre crop | 64 | 4.0 s | 1 | 100 | 50 | 8 | 400 | `stage2_local_pretrained*.yaml`, `bp_local.yaml`, `resp_local.yaml` |
| RAW face ROI, local | `rgb_roi` | 49-landmark face box, `roi_padding 0.2` | 64 | 5.0 s | 2 | 62 | 31 | 16 | 496 | `stage2_local_rgb_roi_bp.yaml`, `bp_rgb_roi_local.yaml` |

`fps 25.0` is the **source** rate in every lineage; decimation is
`temporal_stride`, never `fps`. Alignment is the hard identity enforced by
`MultiModalMAE._check_space_time_alignment`:
`tubelet_t·stride/fps == sig_kernel/fs`, and `num_frames/tubelet_t == seq_len/sig_kernel`.

### 3.2 Datasets

| `data_set` | class | tree | unusable rule |
|---|---|---|---|
| `bp4d+` | `PairedPretrainDataset` / `PairedSessionDataset` | canonical `data/processed/bp4d_canonical` | manifest-driven |
| `rgb_roi` | `RGBRoiPretrainDataset` / `RGBRoiFinetuneDataset` | RAW (`2D+3D/<S>/<T>/*.jpg` or `2D+3D/<S>.zip`, `2DFeatures/*.mat`, `Physiology/`) | a `(subject, task)` whose landmark frame count ≠ JPEG count is **unusable and excluded whole** (`F003_T8` locally), so a mis-aligned crop can never reach the label |

The RAW path accepts extracted directories **and** per-subject ZIPs through the
`zip+file://…!/prefix` source URI; an extracted directory always wins.

### 3.3 ROI doctrine

The contract of record is `design/ROI_crop_and_rescaling.md`: envelope → padding →
**square** → one isotropic scale → resize, with a clip-level **motion gate**
(`max_sway_px`) instead of absorbing head travel into a larger crop.

* **TIR** implements it (`crop_rule='square'`, `max_sway_px=40`, square hull cache).
* **RGB does not yet**: `RGBRoiDataset` still ships the legacy padded box
  (`crop_rule='box'` equivalent). The port is **W4**, and it must reuse the same
  constants and functions rather than re-derive them.

---

## 4. Stage map (design → code → config)

| stage | what it does | design | code | config / entry |
|---|---|---|---|---|
| **1 — init** | VideoMAE-B (K400) → encoder blocks, `enc_norm`, tubelet `Conv3d` copied **verbatim** (no boxcar), `positions`/physio adapters random | `design/SolutionC_EarlyTokenMixing_stage2.md` | `core/multimae.py::load_pretrained_encoder`, `fit_visual_patch_embed`, `canonicalise_vit_state_dict` | `pretrained_encoder: videomae:k400`; `--inflate_rgb_patch` |
| **2 — SSL** | joint tubelet + signal tokens in one ViT; 0.90 **tube** mask on video, 0.50 scattered (or `physio_mask: span`) on 1-D; per-stream linear heads; masked MSE, `λ_visual 1.0`, `λ_physio = signal_weight 0.5`; MR-STFT **off** | `design/SolutionC_EarlyTokenMixing_stage2.md`, `design/SpanMask_PhysioSignals.md` | `core/multimae.py::MultiModalMAE` | `runners/run_pretrain.py` |
| **3 — waveform** | encoder duplicated per target; `streams` = visual only; spatial mean-pool per time step; `Linear(D, sig_kernel)`; joint loss α·StdLoss + β·Pearson + γ·MR-STFT; no masking | `design/SolutionC_EarlyTokenMixing_stage3.md` | `core/waveform_model.py`, `core/waveform_losses.py` | `runners/run_waveform.py`, `configs/finetune/{bp,resp,bp_rgb_roi_local}.yaml` |
| **eval** | per-clip Tier 1/2; session assembly (Hann OLA) then Tier 1/2/3; one number per subject | `design/Evaluation_RespWaveform_protocol.md` | `evaluation/{metrics,clinical,assemble}.py`, `runners/run_evaluate_session.py`, `run_evaluate_subject.py`, `run_evaluate_kfold.py` | — |

**Transfer at Stage 3** (`core/waveform_model.py::load_stage2_encoder`): copies
`adapters.* / positions.* / enc_blocks.* / enc_norm.*`, maps
`heads.<target> → waveform_head` when `samples_per_token == sig_kernel` and
`head_hidden == 0`, and **raises** on an embed-dim mismatch. It also compares the
ROI contract keys recorded in the checkpoint (`roi_landmarks`, `roi_padding`,
`input_size`, …), because those change the pixels without changing any tensor
shape.

**Loss constants per branch** (`WaveformJointLoss`): BP band `1.0–2.5 Hz`,
`fft_sizes 64,128,256`; RESP band `0.1–0.6 Hz`, `fft_sizes 64,128,256`
(`128,256,512` in the HPC resp config). The MR-STFT term skips samples whose
target has (near-)zero norm — a clipped or railed window otherwise divides by ~0.

---

## 5. State

**Implemented and verified**

| item | evidence |
|---|---|
| canonical + RAW-ROI datasets, pretrain and finetune views | `data/paired_dataset.py`, `data/rgb_roi_dataset.py` |
| ZIP source (`zip+file://`) + unusable-pair rule | `tests/test_rgb_zip_source.py` |
| asymmetrical masking, span masking, per-modality FFT windows | `core/multimae.py`; `python -m core.multimae` self-tests |
| Stage-3 head transfer + ROI contract check | `core/waveform_model.py`; `tests/test_config_loading.py` |
| metric suite: time, band-limited Pearson, single-tone template, amplitude, rate, bootstrap CI | `evaluation/metrics.py` (`_self_test`) |
| session assembly + affine calibration | `evaluation/assemble.py`, `tests/test_assemble_session.py` |
| subject/k-fold evaluation | `runners/run_evaluate_{subject,kfold}.py`, `tests/test_kfold_splits.py` |
| AMP (`--amp_dtype {none,fp16,bf16}`) | `design/hpc_AMP.md`, `tests/test_amp.py` |

**Measured, not implemented**

* rPPG external gate (§2) — measurement only, no model involved.

**Not done**

* No Stage-2 or Stage-3 run of either RGB lineage exists in `output/`.
* No RGB ROI precomputation store; the ROI cache (`data/roi_cache.py`,
  `run_build_roi_cache.py`) is **TIR-only** (`tirroi_` keys), while RGB decode is
  the throughput bottleneck (`eval/batch_size_sweep.md`).
* RGB does not implement the square-box / motion-gate doctrine (§3.3).
* No BP-specific morphology metric and no Tier-3 validity guard beyond the
  `>= 30 s` window rule.

---

## 6. Acceptance gates (a run is only reportable if it clears these)

Evaluated on the **subject-disjoint** set, per subject after session assembly
(`design/Evaluation_RespWaveform_protocol.md` §4–§6):

| # | gate | baseline to beat | notes |
|---|---|---|---|
| G1 | the input carries the signal | the rPPG detection gate of §2 | a model cannot outperform its input; re-run the gate for any new ROI |
| G2 | trivial predictor | **constant-0 predictor** on the z-scored target | trivial floor; a model at or above it is not predicting amplitude at all |
| G3 | information content | **single-tone template** fitted at the reference's own peak frequency | the template is handed the answer's frequency, so it is a hard baseline; the model must beat it on a majority of subjects |
| G4 | bandwidth | rate MAE (BPM) on 30 s windows + % within 10 % | report BPM, not `dominant_freq_error_hz` (floored at one FFT bin) |
| G5 | the visual stream is used | **video-dependence probe**: replace the visual input with noise | if metrics do not degrade, the model is not using the modality |
| G6 | amplitude honesty | report `amp_rel_err` **oracle-calibrated** and mark it non-deployable | deployment needs a calibration the model can compute |

Tier 3 (HRV from NeuroKit2) is **BP-only** and is a short-term proxy: BP4D+
sessions are 21.9–64.6 s, so most yield a single 30 s window. Always report the
number of detected RR intervals and the fraction of windows where peak detection
succeeded; NaN-fill and warn below 30 s.

Two reporting rules apply to every gate: **stratify by task** (T1–T10 differ in
head motion and speech, and the motion confound is the suspected RGB failure
mode — pooling hides it), and **report the constant and template baselines next
to the model on the same rows**, never the model alone.

### 6.1 Study-level decisions the gates do not cover

These are open by design; they must be answered before the full-scale runs, and
they are answered on Stage-3 subject metrics — never on pre-text loss and never
on AU F1.

**Is Stage 1 (VideoMAE init) worth anything here?** Measure this ladder:
(1) random init → S3; (2) VideoMAE init → S3; (3) random init → S2 → S3;
(4) VideoMAE init → S2 → S3; optionally (5) no S2 at all (direct supervised).
If (3)/(4) match or beat (1)/(2), the init is dead weight and can be dropped
**with evidence** — itself a reportable result.

**Caveat on the AU probe as a witness.** The AU-occurrence probe
(`runners/run_au_probe.py`; see `code/notes/spec/02_model_architecture.md` §9) is a
*secondary* diagnostic here. AU decoding rewards object-semantic facial features
— exactly the axis an ImageNet/VideoMAE image objective helps with — so the init
looking better on the probe does **not** imply that it helps waveform recovery,
and
may mislead, because those objectives also induce invariance to colour and
illumination, the axis that carries the pulse.

**Is Stage 2 necessary?** The contact waveform is the label for every session,
so Stage 2 buys no labels — unlike MAE/VideoMAE. Its remaining justifications are
a representation prior, more pre-training subjects, and a multi-task gradient.
Decide with equal **total** wall-clock on subject-disjoint Stage-3 metrics:
(S3 only) vs (short S2 → S3 ≈ 200 epochs) vs (long S2 → S3 ≈ 800 epochs). If the
S3-only run matches the others within noise, collapse the pipeline to one
supervised stage and reframe the contribution as the cell matrix plus the
physical-basis audit.

---

## 7. Work plan

Ordered; each item states where it lands and what closes it.

| # | work | where | done when |
|---|---|---|---|
| **W1** | **RGB decode precomputation.** Extend the ROI cache (or a sibling store) to `rgb_roi`: key on the crop contract, resume-safe, parity-tested against a live decode. Decode is the epoch bottleneck; the TIR cache measured **11.7×** at equal batch. | `data/roi_cache.py`, `runners/run_build_roi_cache.py`, `tests/test_roi_cache.py` | a parity probe reproduces decoded tensors bit-exactly, and epoch time drops by the measured factor |
| **W2** | **Stage-2, `rgb_roi` + `bp`.** Smoke first (1 epoch, 2 clips, `--warmup_epochs 0`), then the HPC scale-up. | `configs/pretrain/stage2_local_rgb_roi_bp.yaml` | `mse_rgb` clearly below the 1.0 constant-predictor floor; `loss` equals the printed decomposition exactly; encoder-feature variation across clips is non-zero |
| **W3** | **Stage-3, `bp`.** Mirror the W2 geometry exactly (input size, clip, stride, `sig_kernel`, ROI keys), transfer `heads.bp`, then assemble sessions and score. | `configs/finetune/bp_rgb_roi_local.yaml` → an HPC twin | G1–G6 evaluated and reported per subject; artefacts (`preds.npy`, `targets.npy`, `entries.json`) written |
| **W4** | **ROI doctrine port to RGB.** Same square-box algebra, same isotropic rescale, same `max_sway_px` gate, same cache-hull rule — one doctrine, no per-modality exception. Threshold is modality-specific (`40 × 1392/726 ≈ 77 px`). | `data/rgb_roi_dataset.py`, `data/rgb_features.py`; extend `tests/test_roi_square_rule.py`, `test_roi_motion_gate.py` | the RGB dataset satisfies every invariant in `design/ROI_crop_and_rescaling.md` §5 |
| **W5** | **BP evaluation depth.** Morphology-sensitive score (pulse-shape correlation on aligned beats) plus the Tier-3 guards; declare in the protocol what "morphology" means before the first score is reported. | `evaluation/metrics.py`, `evaluation/clinical.py`, `design/Evaluation_RespWaveform_protocol.md` | a metric exists that ranks a deliberately distorted-but-in-phase waveform below a faithful one |
| **W6** | **RESP branch** on the W2 encoder (`Resp_Volts`), same protocol. | `configs/finetune/resp_local.yaml` + ROI twin | same gates; report against the TIR respiratory result, not against BP |
| **W7** | **Aliasing A/B**, only if G4/G5 show unexplained high-frequency energy: `tubelet_t 4` with `temporal_stride 1` (same 160 ms token, anti-aliased) at the cost of the verbatim VideoMAE patch-embed transfer. | `configs/pretrain/*` | a measured spectral difference, decided on subject metrics, not on reconstruction loss |

**Explicitly not planned:** a masking mixture at Stage 3 (the design is unmasked
and the code has no Stage-3 masking path); a 224 px TIR/RGB ROI run before W1
(the token count is 9 800 per visual stream and decode dominates); colour-jitter
augmentation (rPPG lives in the chromatic change).

---

## 8. Run recipes

From `code/`, with `.venv` active and `source scripts/env_local.sh`:

```bash
# Stage 2, RAW face ROI -> BP (smoke first; 1 epoch requires --warmup_epochs 0)
python -u runners/run_pretrain.py -c configs/pretrain/stage2_local_rgb_roi_bp.yaml \
    --epochs 1 --warmup_epochs 0 --max_entries 2 --num_workers 0 \
    --output_dir /tmp/rgb_roi_bp_smoke

# Stage 2, full local recipe
SMOKE=0 bash scripts/local/pretrain_rgb_roi_bp.sh

# Stage 3, BP branch from that checkpoint
python -u runners/run_waveform.py -c configs/finetune/bp_rgb_roi_local.yaml

# session-level scoring
python runners/run_evaluate_session.py --help
```

HPC submission goes through `scripts/hpc/submit_pretrain.sbatch` /
`submit_waveform.sbatch` / `submit_waveform_eval.sbatch`. Always run with `-u`
(or `PYTHONUNBUFFERED=1`): piped stdout is block-buffered, and an interrupted run
leaves a log with the config dump and no epoch lines.

---

## 9. Design gaps this plan depends on

These do not exist as design documents yet; the plan states the requirement and
names the decision, so the gap is visible instead of implicit.

| # | gap | why it blocks | where it lands |
|---|---|---|---|
| D1 | **BP-specific physical-basis note** — which waveform properties are claimed (rate, phase, amplitude, morphology) and at what bandwidth | §6/G5 is unanswerable without it; the rPPG gate covers detection, not morphology | new note in `code/notes/design/` |
| D2 | **Sliding-window CHROM/POS** as the reference extractor | the measured ceiling (skill +0.278) is estimator-bound, so a model result is compared against an under-powered baseline | new note in `code/notes/design/` |
| D3 | **Deployable amplitude calibration** | every amplitude number today is oracle-calibrated | new note in `code/notes/design/` |
| D4 | **RGB fold assignment** | `utils/splits.py::subject_folds` splits sorted ids into contiguous blocks, which confounds fold with sex/cohort (`eval/tir_resp/resp_t3_spread_sources.md` §2) | `design/kfold_val_strategy_design_v2.md` |

---

## 10. Doc map

| topic | document |
|---|---|
| this plan | `code/notes/plan/RGB_BP_plan.md` |
| local 64 px two-branch setup (commands, checkpoint, lineage lock) | `code/notes/plan/local_downstream_plan.md` |
| k-fold rollout | `code/notes/plan/subject_kfold_plan.md` |
| AMP rollout | `code/notes/plan/hpc_AMP_implementation.md` |
| Stage-2 / Stage-3 architecture | `design/SolutionC_EarlyTokenMixing_stage2.md`, `design/SolutionC_EarlyTokenMixing_stage3.md` |
| span masking for 1-D streams | `design/SpanMask_PhysioSignals.md` |
| ROI + rescale contract (RGB port pending, W4) | `design/ROI_crop_and_rescaling.md` |
| evaluation protocol | `design/Evaluation_RespWaveform_protocol.md` |
| folds | `design/kfold_val_strategy_design_v2.md` |
| rPPG evidence | `code/notes/eval/rgb_bp/FINDINGS.md` |
| throughput | `code/notes/eval/batch_size_sweep.md` |
| generated contract | `code/notes/spec/01_data_pipeline.md`, `code/notes/spec/02_model_architecture.md`, `code/notes/spec/03_engine_and_loss.md` |
