# Respiration Stage-3: pooled 5-fold CV, `96respgrid` lineage

> Measured 2026-11-01 → 2026-11-06 · protocol
> `code/notes/design/Evaluation_RespWaveform_protocol.md` §3.2 ·
> partition design `code/notes/design/kfold_val_strategy_design_v2.md`
>
> Run dirs `$WORK_SCRATCH/{pretrain/stage2_hpc_tir_roi_resp_96respgrid,finetune/resp_tir_roi_hpc_96respgrid}_fold{0..4}`
> (Lichtenberg scratch; `$PROJ_DIR/output` is a symlink to it) ·
> artifact `$WORK_SCRATCH/eval/resp_kfold_96respgrid/kfold_resp_96respgrid.json` ·
> freeze record `$WORK_SCRATCH/eval/resp_kfold_96respgrid/experiment_freeze.json`
>
> **This page is a DIFFERENT LINEAGE from `resp_5fold_cv.md` (+0.307).** Four
> things changed together: the respiration label contract (offline cache grid
> instead of per-file `Resp_Volts.txt` volts with the rail/flat gates),
> the Stage-3 loss (point-wise Huber instead of amplitude-only `L_time`), the ROI
> geometry (96 px / padding 0.10 / `square` instead of 112 px / 0.20 / `minmax`),
> and the subject partition (seeded sex-stratified instead of contiguous blocks).
> Never quote a number from one page as if it came from the other. The section
> *What this comparison does NOT attribute* separates what the comparison can and
> cannot support.

## What was measured

Stage 2 (masked pre-training, 5 folds) → Stage 3 (TIR-ROI → respiration, 5 folds)
→ single-process eval dump per fold → session assembly → pooled k-fold. Every
subject is predicted exactly once, by the encoder and head that never saw it.

| item | value |
|---|---|
| corpus | **133 subjects / 1 191 sessions / 19 769 clips** |
| fold test blocks | 28 / 28 / 28 / 28 / 27 subjects |
| partition check | 139-subject universe, disjoint, every subject in exactly one fold |
| epochs | 150 Stage-2, 150 Stage-3 (Option A: `use_small_val: false`, fixed epochs) |
| band | 0.1–0.6 Hz, per-subject median of session `pearson_band` |
| CI | bootstrap 95 % over subjects |
| headline metric | `pearson_band`. `amp_ratio` / `amp_rel_err` / `mae_affine` are oracle-calibrated diagnostics, excluded from `overall` |

The corpus total is the **same number the pre-flight per-fold dataset build
predicted** (`resp_preprocess_impact.md` §2, build #3), so nothing drifted across
five Stage-2 runs, five Stage-3 runs and five re-dumps.

## Contract frozen before launch

Recorded in `experiment_freeze.json`; see also `code/notes/plan/kfold_run_checklist.md`.

| key | value |
|---|---|
| `input_size` / `roi_padding` / `crop_rule` | 96 / 0.10 / `square` (σ = 1.01, see `roi_geometry_ledger.md`) |
| `max_sway_px` | 40 |
| `clip_duration` / `fps` / `clip_stride` | 8.0 s / 25 / 2.0 s |
| `tubelet` / `temporal_stride` / `sig_kernel` | 2,16,16 / 2 / 16 |
| `physio_norm` | zscore (per clip, done by the dataset) |
| label artefact | `respgrid_src1000_fs100_sat9.95-1s_flat0.3-5s_run8s_v2_24852055a5` |
| `rail_touch_v` / `min_signal_spread` | **0 / 0** — the cache's per-sample mask is the only validity rule (design D6) |
| pixel artefact | `tirroi_nose_mouth_pad0.10_square_window-hull_clip8s_src726x480_native_v3_fa9f91bdd5` |
| split | `n_folds: 5`, `split_seed: 42`, `split_stratify: sex` (identical in both stages) |
| `task_groups` | v2: `low=T2,T3,T5 / moderate=T1,T4 / high=T6..T10` |
| Stage-3 loss | `alpha/beta/gamma = 1/1/1`, `huber_beta 1.0`, `fft_sizes 128,256,512` |
| GPUS | 4 in every fold (global batch 64 → same LR schedule), `amp_dtype: none` |

## Runs

Stage 2 (one job per fold, 150 epochs):

| fold | job | gpu | wall |
|---|---|---|---|
| 0 | 55835559 | rtx6000 | 2:45:45 |
| 1 | 55835572 | rtx6000 | 2:28:15 |
| 2 | 55835573 | rtx6000 | 2:46:39 |
| 3 | 55879207 | a100 | 7:44:02 |
| 4 | 55906299 | rtx6000 | 2:48:51 |

Stage 3 (one job per fold, 150 epochs): folds 0–3 on rtx6000, 5:17–5:46 each
(jobs 55934802–55934805); fold 4 on a100, 12:57 (job 55981196).
Eval dumps (single process, 1 GPU, ~1–3 min): 55981186/55981188–90, 56017539.

Hardware is heterogeneous because the `acc` partition was saturated for days
(see §8); `GPUS=4` and `amp_dtype: none` are identical in all five folds, so the
global batch, the LR schedule and the numerics class are unchanged — the difference
is wall clock and cross-vendor kernel order only.

## Headline result

```
pooled 5-fold, band 0.1-0.6 Hz
133 subjects / 1191 sessions / 19769 clips

PEARSON  per-subject median : +0.542  [95% CI +0.493, +0.576]
         template baseline  : +0.537   (beaten by 52 % of subjects)
RATE     MAE                : 3.75 BPM [95% CI 2.81, 4.34]
         pooled windows     : MAE 5.06 BPM, within_10pct 45 % (n = 1610)
```

| fold | subjects | sessions | `pearson_band` | `mae_bpm` | `within_10pct` |
|---|---|---|---|---|---|
| 0 | 27 | 216 | **0.582** | 2.32 | 0.50 |
| 1 | 27 | 258 | 0.542 | 2.85 | 0.00 |
| 2 | 25 | 218 | 0.574 | 3.50 | 0.00 |
| 3 | 28 | 256 | **0.414** | 4.26 | 0.00 |
| 4 | 26 | 243 | 0.572 | 3.87 | 0.00 |

**Verdict: statistically at parity with a single-tone template, not better.**
The CI upper bound (+0.576) exceeds the template (+0.537) but the point estimate
does not, and only 52 % of subjects beat it. Any claim must be phrased against the
baseline, and no per-fold ordering may be read as model quality: with a stratified
partition a fold is a *sample*, but 25–28 subjects still describes who is held out.

Fold 3 is the weak fold (0.414 / 4.26 BPM vs 0.542–0.582 elsewhere). It is the
candidate for a per-subject error analysis; the `per_subject` and `sessions` blocks
of the artifact carry everything needed, no GPU required.

## Comparison with the old 112 px lineage (same reference)

The recorded old number (+0.307, `resp_5fold_cv.md`) was scored against raw
`Resp_Volts.txt`. To remove the reference as an explanation, the **old dumps were
re-scored against the same respiration-cache artefact** as the new labels
(design C8):

| | old 112 px, cache reference | new 96 px |
|---|---|---|
| artifact | `kfold_old112_respcache_ref.json` | `kfold_resp_96respgrid.json` |
| pooled `pearson_band` | **+0.313** [+0.257, +0.376] | **+0.542** [+0.493, +0.576] |
| template baseline | +0.543 | +0.537 |
| subjects beating template | 19 % | 52 % |
| scored population | 134 subj / 1 173 sess / 20 302 clips | 133 / 1 191 / 19 769 |

Re-scoring moved the old lineage only +0.307 → +0.313, so the gap is not a
reference artefact.

Paired over the **132 subjects scored by both** (same reference, same session
protocol; the partition rule changed, so the fold labels are not comparable
across lineages and the subject is the only valid pairing unit):

| statistic | value |
|---|---|
| median `pearson_band`, old → new | +0.320 → **+0.546** |
| paired Δ (new − old) | median **+0.175**, mean +0.159 |
| improved / worsened | **104 / 28** of 132 (79 % / 21 %) |
| improved by > +0.20 | 57 |
| worsened by > −0.20 | 2 |
| Wilcoxon signed-rank | p = 4.2e-14 |
| Δ of the model-minus-template margin | median +0.181 (p = 9.9e-15) |

Distribution of the per-subject Δ (nearest-rank): p5 −0.157, p10 −0.096,
p25 +0.027, p50 +0.177, p75 +0.277, p90 +0.400, p95 +0.463, min −0.331,
max +0.664.

The gain is **broad, not a few outliers**: the cohort's total positive gain is
24.0 points and the top 5 subjects supply 3.0 of it (13 %, i.e. 4 % of subjects);
every fold has a positive median Δ (+0.092 / +0.212 / +0.186 / +0.133 / +0.177);
both sexes improve (F +0.202 on 65/77, M +0.113 on 39/55).

Against the template, the cohort splits:

```
was worse than template, now better : 44
was better before,     now worse    :  1
better in both lineages             : 25
worse in both                       : 62
```

So the pipeline converted 44 "below baseline" subjects into "above baseline" ones
(one went the other way), and **47 % of subjects are still below the template**.

## What this comparison does NOT attribute

The two lineages differ in five ways at once:

| change | old | new | note |
|---|---|---|---|
| Stage-3 loss | amplitude-only `L_time`, \|std(pred) − std(target)\| | point-wise Huber `huber_beta 1.0` | the only term that sees shape/phase, i.e. precisely what `pearson_band` measures. **Most plausible main driver**; the old dumps predate it (`configs/finetune/resp_tir_roi_hpc.yaml` header) |
| label / corpus contract | raw volts + `rail_touch_v 9.9` + `min_signal_spread 0.1` | cache grid (sat 9.95 V ≥ 1 s, flat 0.3 V / 5 s, runs ≥ 8 s), gates forced OFF | different targets *and* different clip population: 21 231 → 19 769 clips, 138 → 133 subjects |
| geometry block | 112 px, pad 0.20, `minmax` | 96 px, pad 0.10, `square` | **density-neutral**: σ = s/side = 1.01 at both `(0.20, 112)` and `(0.10, 96)` (`roi_geometry_ledger.md`) |
| partition | contiguous blocks (fold ≡ sex block) | seeded (42) sex-stratified | changes which model scores a given subject; handled by pairing on subject |
| `task_groups` | v1 | v2 | post-hoc label metadata, no effect on training or scoring |

**`input_size` is not supported as the cause.** At equal density the 96 px grid
keeps the face at ~61 grid points but drops background occupancy 70 % → 59 % of
the patch and the token count 49 → 36 per frame (2 450 → 1 800 per clip). It buys
no resolution and costs none; a broad, near-uniform +0.18 shift across 79 % of
subjects is what a loss/label change looks like, not a geometry change (which
would act on the clip subset that is up- or down-sampled). Measured, the token
saving is a speed win only:

| | old 112 px | new 96 px |
|---|---|---|
| Stage-2 s/iteration | 0.3098 (268 steps, 81 s/epoch) | 0.2634 (255 steps, 67 s/epoch) ≈ 15 % faster |
| Stage-3 s/epoch (rtx6000) | ~3:28 (112 px, 200-epoch run) | 2:08 |

To attribute the gain, one variable must be isolated: retrain Stage 2 + Stage 3 at
`112 / 0.20` (σ still 1.01) under the new loss and label contract — ≈ 15 h + 18 h
of GPU for five folds — or re-run the new geometry with the amplitude-only loss.
Until then the correct headline is "the `96respgrid` lineage", never "96 px beats
112 px".

## Traps hit (and their guards)

| trap | symptom | outcome |
|---|---|---|
| Stage-3 loaded the **wrong lineage** | all four folds died in ~10 s: `1 of the Stage-2 tensors have the wrong shape (e.g. ['positions.tir.pos_embed'])` | `submit_stage3_kfold.sh` preflighted `STAGE2_NAME` but never forwarded it, so the job fell back to `stage2_hpc_tir_roi_resp` (112 px). Fixed: the driver forwards `STAGE2_DIR`/`STAGE2_RUN_NAME` per fold; pinned by `tests/test_stage3_driver_env.py`. The geometry guard in `core/waveform_model.load_stage2_encoder` caught it loudly — no half-loaded encoder survived |
| **rank-0 DDP shard** used as the dump | `preds` 848 vs 3391 entries (exactly ¼) in every fold | re-dumped at world size 1 (`--eval_only`, `submit_waveform_eval.sbatch`), after backing up to `<fold>/shard_backup/` and restoring `fold_manifest.json` (the eval-only pass wipes `roles.train`) |
| `--eval_only` clobbered the manifest | `roles.train = []` | backup + restore, verified train 111/111/111/111/112 |
| manual `--fold k` on a sweep job | job ends `FAILED` after 150 epochs (`nothing to do: … already at epoch 150`) | the sbatch appends `"$@"` after its own per-fold args, so the flag leaked into every iteration. Drive it with `FOLDS` or `submit_stage3_kfold.sh`; the extra iterations all resolved to the same fold dir (resume guard printed `fold contract OK … fold 4/5`), so the trained artefacts are intact |
| k-fold report header | "pooled 5-fold" printed for a 4-fold run | `run_evaluate_kfold.py` now prints the number actually pooled |
| `--resp_cache` omitted from the report | would score against raw `Resp_Volts.txt`, i.e. a different preprocessing than the labels | both `run_evaluate_session.py` and `run_evaluate_kfold.py` are called with it; `NEXT_STEPS.md` §3 records why |

Scheduling notes worth keeping: `acc_long` has **no rtx6000** (submission rejected),
h100 is only 8 GPUs and was dominated by a higher-priority user (ETA slipped to
Nov 19), and a
≤30-minute job in `acc_short` gets a completely different (and much better)
priority class than the same job in `acc` — the four eval dumps started in seconds
there after waiting until Nov 7 in `acc`. A site-wide `DRAIN+REBOOT_REQUESTED`
window blocks every GPU job regardless of free GPUs shown by `scontrol`.

## Limits

- **One seed.** Stage-2 init is the shared VideoMAE file and the partition is
  seeded, but there is no repeat run, so the ±CI covers subjects, not training
  variance.
- **Option A** (no validation split, fixed 150 epochs) means no early stopping and
  no model selection; the last checkpoint is the model by construction.
- **The template is a weak baseline.** Parity with a single-tone fit means "the
  learned model adds nothing measurable yet", not "the task is solved".
- Rate error is large (3.75 BPM median, i.e. ~25 % of a 15 BPM breathing rate) and
  `within_10pct` is 0 in four of five folds.
- Amplitude is diagnostic only: `amp_ratio_median` ≈ 0.55 in every fold says the
  prediction's spread is systematically about half the target's, which matters if
  the deliverable is ever a calibrated waveform rather than a shape.

## Reproduce

```
# from code/, venv active, scripts/env_hpc.sh sourced
python runners/run_evaluate_kfold.py --root "$WORK_SCRATCH/finetune" \
    --run_prefix resp_tir_roi_hpc_96respgrid_fold --folds 0,1,2,3,4 \
    --waveform resp --fs 100 --resp_cache "$WORK_SCRATCH/resp_cache" \
    --out "$WORK_SCRATCH/eval/resp_kfold_96respgrid/kfold_resp_96respgrid.json"

# old 112 px dumps on the same reference (the §6 comparison)
python runners/run_evaluate_kfold.py --root "$WORK_SCRATCH/finetune" \
    --run_prefix resp_tir_roi_hpc_fold --folds 0,1,2,3,4 \
    --waveform resp --fs 100 --resp_cache "$WORK_SCRATCH/resp_cache" \
    --out "$WORK_SCRATCH/eval/resp_kfold_96respgrid/kfold_old112_respcache_ref.json"
```

Session-level assembly (per fold, writes `session_metrics.json` + figures) and the
launch order for a fresh sweep are in
`$WORK_SCRATCH/eval/resp_kfold_96respgrid/NEXT_STEPS.md`.
