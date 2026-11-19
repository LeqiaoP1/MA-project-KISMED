# Evaluation Protocol — Respiration Waveform Reconstruction (tested subject)

> Status: **active** · authored 2026-10-12 · revised **2026-11-18** (the split and
> selection protocol is fixed by `ADR-001-subject-disjoint-split-and-selection.md`:
> `K = 5`, train/validation/test = 70/10/20 by subject, validation role mandatory in
> every stage; the earlier 80/0/20 allocation and the Option A/B wording are retired)
> · earlier revision 2026-11-01 (k-fold cross-validation implemented; reference must
> come from the respiration cache; **amplitude retired as a metric**)
> · scope: Stage-3 respiration output (the final task of the whole model).

## 1. Objective

The whole pipeline (Stage 1 VideoMAE init -> Stage 2 masked multimodal pretrain
-> Stage 3 waveform finetune) exists to answer one question:

> **Can the respiration waveform of an unseen subject be reconstructed from the
> thermal face ROI alone?**

"Performance of the entire model" is therefore the **subject-level respiration
reconstruction quality**, and Stages 1/2 are judged only by whether they move
that number (ablation), never on their own pretext losses.

## 2. Unit of evaluation = the tested subject

Clips are an implementation detail. The evaluation object is **one held-out
subject**: all their clips are assembled into continuous per-session waveforms
(`evaluation/assemble.py`, Hann overlap-add) and compared to the reference
waveform read from the respiration cache (§3.3). We then report **one number
per tested subject**, and aggregate across subjects.

## 3. Protocol

### 3.1 Unit of splitting — subject-disjoint, three explicit roles
- Train / val / **test** subjects are disjoint. `split_by: subject` is the only
  legal choice whenever `n_folds > 1` (a session split is rejected: two sessions
  of one subject must never straddle the boundary).
- The partition is computed **once, over the full usable subject list**
  (`protocol.FoldPlan` from `data.universe.corpus_subjects`: sorted ids +
  `np.array_split`, seeded within-stratum shuffle when `sex` is the stratifier),
  so
  fold membership cannot drift with `--subjects`, `--tasks`, `--max_sessions`
  or the rail/spread filters. Each fold writes ONE `partition.json` (fold,
  n_folds, seeds, roles, counts, universe + partition digests), which both stages
  read and must agree on by digest.
- **Model selection reads only the validation role** (`ADR-001`): every run has a
  val role, and `metrics_final.json` records the rule actually used —
  `selection: 'best_val' | 'fixed_epoch'` — with the selected epoch and criterion
  value, plus `eval_role: 'test'`. Absence of selection is therefore auditable off
  the artefact, and no run selects on the test role. (The legacy `n_folds = 0` path
  checkpoints and reports on the same split — selection leakage, off-protocol, never
  a headline.)
- `--test_subjects F001,F002,...` is now only a **manual override** on a legacy
  single-split run; it filters predictions *after* training and therefore cannot
  create a fold (see §3.2).

### 3.2 k-fold subject-disjoint cross-validation — IMPLEMENTED (2026-10-18)

- Mechanism and rollout: `code/notes/plan/ADR-001-subject-kfold-plan.md` (the
  requirement lives in
  `code/notes/design/ADR-001-subject-disjoint-split-and-selection.md`). Roles per fold from
  `protocol.build_plan` (implemented in `code/protocol/`):

  | role | share | subjects |
  |---|---|---|
  | `test` | 20 % | the block `F_k` |
  | `val` | 10 % | drawn from the four non-test blocks, frozen per fold |
  | `train` | 70 % | the four non-test blocks minus the val block |

  The realised counts are recorded per run; rounding may move a role by one
  subject.

- One run per fold: `--n_folds 5 --fold k`, submitted as a Slurm array
  (`FOLD=$SLURM_ARRAY_TASK_ID`), each fold writing its own run directory
  (`resp_tir_roi_hpc_fold<k>`). Stage 2 is fold-aware as well: its corpus excludes
  both the test block and that fold's validation block (`ADR-001`), so neither stage
  trains on a subject it validates or is tested on.
- `n_folds` is the protocol constant `5` (`ADR-001`). `n_folds = 0/1` remains the
  legacy single-split compatibility path: it has no `test` role and is
  off-protocol — it may not carry a headline claim.
- **Pooling** — `runners/run_evaluate_kfold.py` reads the per-fold
  `preds.npy` + `entries.json`, **asserts the folds form a subject-disjoint
  partition** (raises on any overlap or duplicated subject), scores every
  session, aggregates per subject → bootstrap CI, and writes one
  `kfold_<run_prefix>.json` with `protocol`, `per_fold`, `per_subject`,
  `per_task`, `per_group`, `overall` and `rate_overall`.
- `k = 5` is fixed for the thesis. Leave-one-subject-out or any other fold count
  is a separate study and needs its own ADR.
- **Read the per-fold column with care.** Folds are contiguous over sorted ids
  (`F001–F028`, `F029–F056`, …), so some folds are 100 % female / 100 % male and
  a per-fold difference is a *cohort* difference, not a fold effect. The pooled
  numbers are unaffected (open item, §9).

### 3.3 Reference and label must share ONE preprocessing — IMPLEMENTED (2026-10-31)

- **Rule: the reference is scored in exactly the space the training label lives
  in.** Both sides are read from the SAME respiration shard
  (`data/resp_cache.py`), produced by `data/resp_preprocess.py`: saturation
  events `|V| > 9.95 V` lasting ≥ 1 s, flatline (5 s window, p-p < 0.3 V), valid
  runs shorter than 8 s turned into holes, per-run DC removal, then
  `resample_poly` onto the 100 Hz grid. Amplitude is preserved; only the per-run
  offset is removed.
- Re-reading `Resp_Volts.txt` and resampling it with `np.interp`
  (`runners/run_evaluate_session.py::read_raw_series`) is a **different**
  preprocessing (raw whole-file volts, no anti-alias filter, no run split), so
  it must not produce a reported number: those differences would be charged to
  the model as reference error.
- Mechanism: `read_reference_from_cache(root, session, fs, cache_key)` returns
  `(t, y, mask, root)`; `y` is `NaN` exactly where the artefact says invalid and
  the mask is **ANDed with the clip-coverage mask** before scoring, so a
  reference hole is a hole, never a zero. Session records carry
  `reference_source: 'resp_cache'` + the resolved directory / `resp_cache_key`
  (`'raw_resp_volts'` marks the legacy route).
- Applied on all three respiration evaluators — `run_evaluate_session.py`,
  `run_evaluate_subject.py`, `run_evaluate_kfold.py` — via `--resp_cache
  <OUT_ROOT>` (the parent of the `<CACHE_KEY>/` directory), so **every reported
  number must pass it**. `--fs` must equal the cache's `fs_out` (100) and
  `--resp_cache_key` pins the directory name instead of resolving it. The flag
  is respiration-only (`run_evaluate_session.py` refuses it with
  `--waveform bp`).
- **Consequence for older numbers:** every measurement taken before 2026-10-31 —
  including the §5 template tie (0.593 vs 0.600) and the 5-fold pooled table —
  used the raw-`np.interp` reference and is on a different reference space. It
  must be recomputed on the artefact before being compared with a new run.

## 4. Metrics — morphological shape and rhythm only

The claim under test is **morphological shape and rhythm**, not level. Both
sides are shape-normalised before scoring: the prediction is **z-scored per
clip** (the dataset's only label normalisation,
`data.tir_resp_dataset.TARGET_NORM`) and the reference is the same artefact with
its per-run offset removed (§3.3). Every metric below is therefore
scale/offset-invariant by construction.

| Role | Metric | Rationale |
|---|---|---|
| **Headline** | **band-limited Pearson** (0.1–0.6 Hz), per session, median per subject | removes baseline drift; the "is the breathing rhythm reconstructed?" number |
| Reference | plain Pearson | same but drift-sensitive; shown for contrast |
| Rate | `dominant_freq_error_hz` (report in bpm) / `mae_bpm` (§4b) | breathing-rate accuracy |
| Shape — spectral | `psd_mae` (in-band, unit-sum) | where the in-band energy sits |
| Shape — morphology | template-subtracted residual correlation + per-breath shape | **required for a morphology claim** (not yet implemented, §9) |
| Artifact | `hf_power_rel` vs `target_hf_power_rel` | out-of-band (token-seam) energy |

### 4.1 Amplitude is NOT a metric (retired 2026-11-01)

The evaluation asks whether the **shape and the rhythm** of the breathing curve
are reconstructed. Amplitude is deliberately outside that claim:

- the prediction is z-scored per clip, so it carries no absolute scale at all.
  An amplitude number exists only after fitting a per-session affine map
  (`evaluation/assemble.py::affine_calibrate`) *against the reference*, i.e. it
  is **oracle-calibrated** and not deployable;
- a correctly scaled but wrongly shaped curve is not a better answer, and a
  shape-correct curve with the wrong depth still answers the question;
- amplitude stops being comparable across models as soon as the normalisation
  or the clip length changes.

Consequence: `amp_ratio` / `amp_rel_err` / `mae_affine` / `rmse_affine` stay in
the per-session / per-task / per-fold JSON blocks as **recorded diagnostics
only**. They are excluded from `overall`, from console reports and from figure
titles, and no acceptance criterion, headline or run-to-run comparison may quote
them.

### 4.2 Morphology is a distinct claim

Rate/phase tracking (high band-limited Pearson) does **not** imply waveform
morphology: a single tone at the reference's own frequency scores high on
Pearson and carries no inter-breath shape. Because shape is now the headline
claim, the morphology metric — template-subtracted residual correlation, and
per-breath shape after normalising each breath to unit depth — is **required**,
not optional. It is the one metric this protocol still lacks (§9).

## 4b. Future target — derived vital sign: respiration RATE (BPM)

When the training target changes from the waveform to a **summarized respiration
rate** (one averaged value per window), the evaluation switches to the two
literature-standard numbers, computed per window and pooled:

| Role | Metric | Definition |
|---|---|---|
| **Headline** | **MAE (BPM / breaths·min⁻¹)** | `mean(\|rate_pred - rate_ref\|)` in physical units — the universal reporting standard |
| Acceptance | **% within 10 %** | fraction of windows with `\|err\| ≤ 0.10 · rate_ref` (ICMI-2023 style; relative to the contact-sensor reading) |
| Supporting | RMSE (BPM), bias (BPM) | signed error / spread |

The rate is read from the band-limited FFT peak (`estimate_rate_bpm`,
0.1–0.6 Hz = 6–36 BPM, parabolic sub-bin). Because the metric is a rate, it is
**scale/offset-invariant by construction** — the voltage-space problem that
constrains the waveform metrics does not apply to it. Report the **window
length** with every number: a shorter window gives fewer cycles and a coarser
rate. Use the same window for prediction and reference.

Wired into `runners/run_evaluate_subject.py` and
`runners/run_evaluate_kfold.py` as an optional block (`--rate_window_s`, default
30 s; `--rate_tol`, default 0.10): per-session `rate` blocks feed the
per-subject rollup, and a pooled `rate_overall` block is the headline. Windows
are cut on the **scored** samples only, so a reference hole (§3.3) never forms
half a window.

## 5. Baselines a subject must beat

A reconstruction is only meaningful if it beats trivial competitors:

- **Constant / mean line** — Pearson 0 by construction (sanity floor).
- **Single-tone template** — a pure tone at the reference's own peak frequency
  (LS cos/sin fit; handed the true rate for free). This is the *best a
  rate-only model can do*; the model must score **above** it in a majority of
  sessions. (Evidence 2026-10-10, on the pre-artefact raw reference — see §3.3:
  the Stage-3 model's 0.593 median tied this baseline's 0.600, winning only
  49.1 %, i.e. currently NOT beaten.)
- **Off-face / global-motion control** — the frame-argmax finding (a chest/hair
  pixel drives the whole-frame gate). Any subject-level claim must survive
  regressing out the global/motion channel.

## 6. Aggregation & uncertainty

- **Per subject**: median of the subject's per-session metrics (robust to short,
  uninformative sessions; a 10 s session cannot show respiration).
- **Across subjects**: median + IQR, and a **bootstrap 95 % CI over subjects**
  (`bootstrap_ci`, `evaluation/metrics.py`) — each *subject* is one independent
  observation, clips within a subject are not.
- Report the **fraction of subjects whose headline beats the template baseline**.

## 7. Reporting artifacts

Per fold run: `partition.json` (the partition, §3.1), `selection.json`, and per
epoch the
`metrics.jsonl` / `metrics_final.json` carrying `selection` and `eval_role`.

Per run (restricted to the test subjects):

- `subject_metrics.json` — `overall` (with CI), `per_subject`, `per_task`,
  `per_group`, `sessions`.
- `subject_figures/subject_<S>.png` — reconstructed vs reference for that
  subject's **median-quality** session (whole session + zoom).
- A one-line console summary: median band-limited Pearson [CI], n subjects,
  beat-template fraction.

For the pooled cross-validation (headline of a k-fold study):

- `kfold_<run_prefix>.json` — `protocol`, `per_fold`, `per_subject`, `per_task`,
  `per_group`, `overall` (bootstrap CI over subjects), `rate_overall`.
- `session_metrics.json` per fold, so a fold can be inspected alone.

## 8. Implementation map

| Piece | Location |
|---|---|
| subject-fold partition, roles, record | `code/protocol/` (`build_plan`, `FoldPlan`, `save_plan`, `load_plan`) |
| fold-aware datasets / Stage-2 corpus | `data/tir_resp_dataset.py`, `data/rgb_roi_dataset.py`, `data/datasets.py` |
| fold CLI (`--n_folds` fixed at 5, `--fold`, `--split_role`) | `runners/_common.py::add_split_args` |
| selection record (`best_val` or `fixed_epoch`) + test-role eval | `runners/run_waveform.py` |
| band-pass + band-limited Pearson + template + bootstrap | `evaluation/metrics.py` |
| session assembly (Hann OLA), affine calibration (diagnostics only) | `evaluation/assemble.py` |
| subject-level runner | `runners/run_evaluate_subject.py` |
| pooled k-fold runner | `runners/run_evaluate_kfold.py` |
| reference reader (`read_reference_from_cache`) / smoother | `runners/run_evaluate_session.py` |
| reference preprocessing (the artefact) | `data/resp_preprocess.py`, `data/resp_cache.py` |
| tests | `tests/test_evaluation_subject.py`, `tests/test_evaluation_breakdown.py`, `tests/test_kfold_splits.py`, `tests/test_resp_eval_reference.py`; `tests/test_evaluation_amplitude.py` covers the retired diagnostic helper only |

Commands (from `code/`), with `RESP_CACHE` = the cache OUT ROOT:

```bash
# per fold (one Slurm array element each), then pool
python runners/run_waveform.py -c configs/finetune/resp_tir_roi_hpc.yaml \
    --n_folds 5 --fold 0 --output_dir OUT/fold0
python runners/run_evaluate_kfold.py --root OUT --run_prefix resp_tir_roi_hpc_fold \
    --folds 0,1,2,3,4 --resp_cache "$RESP_CACHE" --fs 100 --band 0.1,0.6

# single held-out set (legacy override)
python runners/run_evaluate_subject.py --pred_dir OUT_DIR --waveform resp \
    --fs 100 --band 0.1,0.6 --resp_cache "$RESP_CACHE" \
    --test_subjects F001,F002,F003
```

## 9. Non-goals / open items

- **The morphology metric does not exist yet (required, §4.2).**
  `evaluation/metrics.py` has no residual/per-breath shape metric, so today
  "right rate, wrong shape" cannot be separated from "right shape". Until it
  is implemented, no morphology claim may be made.
- Amplitude is out of scope **by decision** (§4.1). The oracle-calibrated
  `amp_ratio` / `amp_rel_err` / `mae_affine` / `rmse_affine` fields are
  diagnostics: they must not be quoted as results or compared between runs.
- **Fold membership is contiguous** over sorted subject ids, so gender (and
  anything else correlated with id order) is confounded with the fold index. The
  pooled number is unaffected, but a per-fold *effect* claim needs a round-robin
  or stratified partition — not implemented.
- The validation role is **mandatory** in every stage (`ADR-001`); a run that
  selects on it must report `selection: 'best_val'` with the selected epoch. The
  retired no-val variant (former "Option A") is off-protocol.
- Clinical HRV (Tier 3) is BP-only and out of scope for respiration.
- Statistical test *between* two models (e.g. std vs smoothl1) is a paired
  subject-level comparison — a follow-up, not part of the single-model report.
- Evaluation without the respiration cache (§3.3) is a legacy compatibility
  route only; it is not a protocol variant and must not produce a reported
  number.
