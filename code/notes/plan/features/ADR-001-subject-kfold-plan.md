# ADR-001 — Subject K-Fold Implementation Plan

> Specification: `code/notes/design/ADR-001-subject-disjoint-split-and-selection.md`.
> Where code, configs, scripts or notes disagree with it they are **deleted or
> overwritten**; no compatibility path survives on the reporting pipeline.
> Runbook:
> `code/notes/plan/experiments/kfold_run_checklist.md`. Metric definitions:
> `code/notes/design/Evaluation_RespWaveform_protocol.md`.

## 1. Decisions

| # | Decision |
|---|---|
| D1 | `code/protocol/` is the **only** place a role is computed. `code/utils/splits.py` is deleted. |
| D2 | `K = 5` and `70/10/20` are module constants. **No `--n_folds` flag, no `n_folds` config key** (ADR §2.1). |
| D3 | Datasets take a `FoldPlan` + role name, never a seed, fold index or ratio. |
| D4 | One write-once `<exp_root>/fold<k>/partition.json`, created by `runners/run_partition.py`, by Stage 2, or by a **from-scratch** Stage 3; every other consumer reads it and must match the digest. |
| D5 | One `SelectionTracker` for both stages; `val_select=fixed_epoch` by default. |
| D6 | A masked forward without `val_kept_<stream>` refuses to start (I6). |
| D7 | `train_ratio`/`val_subject`/`split_by`/`split_role`/`use_small_val`/`pretrain_split` are deleted from the protocol path and the CLI. |
| D8 | Training never builds the test role; `--eval_role test` scores it once and refuses a repeat (I5). |
| D9 | Val pass = `eval()`, `inference_mode()`, FP32, `shuffle=False`, `drop_last=False`, `num_workers=0`; masks take `mask_seed = base + batch_idx` with a constant `base`, so every epoch sees identical masks while batches do not share one pattern (Criterion 2, I6). |
| D10 | `eval_freq ≥ 3` (val ≈ 10/70 epoch against the 5% budget); `0` only with `fixed_epoch`. |
| D11 | Every artefact carries `compliant`; the pooler accepts only five compliant folds of one partition. |
| D12 | New root `output/protocol_v1/`; no tool reads the old `output/finetune|pretrain` runs. |

## 2. `code/protocol/`

```
constants.py  K, ROLE_SHARE, ROLES, SCHEMA_VERSION, MIN_EVAL_FREQ
partition.py  split_universe()
plan.py       FoldPlan: build / validate / save / load / digest
selection.py  SelectionTracker
records.py    eval / selection / final writers (atomic, rank-0)
guards.py     assert_roles_disjoint, LeakGuard
```

`build_plan(universe, fold, split_seed, split_stratify, val_seed=None)`
(`val_seed` defaults to `split_seed`):

1. sorted unique universe; require `|universe| ≥ K`.
2. `blocks = split_universe(...)`; `test = blocks[fold]` — sorted ids, optional
   `sex` stratum shuffle + proportional interleave, `np.array_split`, block sizes
   differ by ≤ 1, never `GroupKFold` (it balances samples, not subjects).
3. `pool = sorted(universe − test)` — sorted, never a set: `Generator.choice`
   over a set's iteration order is hash-seed dependent. `n_val =
   clamp(round(.10·N), 1, |pool|−1)`, drawn with `rng = default_rng(val_seed +
   fold)`: `split_stratify='none'` → `rng.choice(pool, n_val, replace=False)`;
   `'sex'` → largest-remainder allocation of `n_val` over the pool's strata
   (strata consumed in sorted order), `rng.choice` within each, then sorted. The
   val draw is stratified for the same reason the blocks are: an unstratified 14
   of 111 can be one sex, and `train = pool − val` inherits that skew.
4. `train = sorted(pool − val)`.
5. `digest = sha256(canonical_json({n_folds, split_seed, split_stratify,
   val_seed, roles}))`.

Raises `ProtocolError`, never warns: role overlap; union ≠ universe; empty
`train`/`val`/`test`; `fold ∉ [0, K)`; `n_folds ≠ K`; `test` ≠ a recomputed
`split_universe()[fold]`; `|realised − ROLE_SHARE| > 0.02` (Criterion 1, reported
instead of raised under `--dev_smoke`). `save_plan` is `O_CREAT|O_EXCL`; an
existing record with another digest is an error. `--dev_smoke` writes
`partition.dev.json` so a dev run cannot masquerade as a protocol record.

| Artefact | Fields |
|---|---|
| `partition.json` | `schema, fold, n_folds, split_seed, split_stratify, val_seed, universe{size,digest}, roles{train,val,test}, counts, share, digest, dev_smoke, writer{stage,argv,git}` |
| `metrics.jsonl` line | `epoch, value_kind, criterion_value, components, counts{clips,subjects}, seed{val_seed,mask_base}, pinned, role_digest` |
| `selection.json` | `rule, eval_freq, selected_epoch, criterion, direction, criterion_value, n_evals, dev_smoke` |

`best.pth` is written **only** by `SelectionTracker`, i.e. only from a val record.

## 3. Layers

**Data**
- `data/universe.py::corpus_subjects(args)` — whole raw tree, availability
  predicate only (video + `IRFeatures` + `Resp_Volts`); independent of `--tasks`,
  `--subjects`, caps and cleaning thresholds (I2). The only definition of "the
  universe".
- `build_dataset(args, plan, role)` / `build_pretraining_dataset(args, plan, role)`;
  `role ∈ ROLES`.
- ROI datasets: delete `train_ratio, split_by, val_subject, split_role, n_folds,
  fold, use_small_val` (tir L1932-1938, rgb L1174-1180) and the
  `resolve_split_keys` blocks (tir L1997-2099, rgb L1235-1285). Subjects are
  `plan.roles[role] ∩ --subjects`, then sessions are filtered by `--tasks`.
- Empty `val`/`test` = a real empty view with a recorded reason; empty `train` is
  fatal; an empty role is never forwarded as `subjects=[]`.
- `bp`/`au` (`paired_dataset.py:232`, `au_dataset.py:317`) keep their ratio cuts,
  get no `partition.json`, hence no thesis number.

**Engines**
- `evaluate_waveform_loss(...)`, `evaluate_pretrain(...)` — the D9 pass returning
  `val_loss`, per-component losses, metric values, `n_clips`, `n_subjects`
  (+ `mse_<stream>`, `kept_<stream>` for Stage 2).
- `MultiModalMAE.forward(x, mask_generator=None, kept_override=None)`; `None`
  reproduces today's training bits (golden test). Only `core/` change.
- `LeakGuard` is checked **inside the training step, before `optimizer.step()`**,
  so the abort precedes any weight update (Criterion 3);
  `assert_roles_disjoint` remains the build-time static check.

**Runners**

| Runner | Builds | Refuses |
|---|---|---|
| `run_partition.py --exp_root R --fold k` | none — writes `partition.json` only | an existing record with another digest |
| `run_pretrain.py --exp_root R --fold k` | train, val (creates the record if absent) | digest clash; masked stream without `val_kept_*`; empty train |
| `run_waveform.py --exp_root R --fold k` | train, val | digest clash; leak; `best_val` without val; a missing record **unless** `--finetune` is absent (a from-scratch baseline creates it) |
| `run_waveform.py … --eval_role test` | test only | `stage3/test/metrics.json` already present (the sentinel; `--overwrite_test` records the supersede); training in the same process |

Deleted: `run_finetune.py` fold logic and `best.pth` (L119-145, L178-188);
`run_visualize_pretrain.py` re-derivation (L338-380); `run_eval_recon.py` manifest
guessing (L299-305); `check_resume_fold` (L55) → digest-based resume;
`add_split_args` (L73-110).

**CLI** — `add_protocol_args` declares exactly the ADR §2.3 interface:
`--exp_root`, `--fold`, `--split_seed 42`, `--split_stratify sex`,
`--val_seed` (default `= --split_seed`, so the draw adds no free parameter; the
resolved value is what the record stores), `--eval_freq 5`,
`--val_select fixed_epoch|best_val`, `--val_criterion joint_loss|pearson_band`
(`joint_loss` → minimised, `pearson_band` → maximised, recorded as `direction`),
`--val_kept_<stream>`, `--dev_smoke`. Omitting `--n_folds` entirely is stricter
than ADR §2.3 on purpose: a value that must never change should not be
expressible.
Removed: `--n_folds`, `--use_small_val`, `--split_role`, `--train_ratio`,
`--val_subject`, `--split_by`, `--pretrain_split`, `--test_subjects`; a YAML still
carrying one fails to load (that is the migration alarm).

```
<R>/fold<k>/partition.json                       # write-once, both stages
<R>/fold<k>/stage2/{checkpoints,metrics.jsonl,selection.json}
<R>/fold<k>/stage3/{checkpoints,metrics.jsonl,selection.json,test/}
<R>/report/kfold.json                            # pooler only
```

## 4. Phases

| P | Work | Gate |
|---|---|---|
| P0 | `output/protocol_v1/`; banner the two recorded sweeps as off-protocol (no new index file) | nothing writes to the new root without `--exp_root` |
| P1 | `code/protocol/` (§2) | `tests/test_protocol.py` green, no GPU |
| P2 | `data/universe.py`, dataset + engine signatures, `core/multimae.py` args | 139 subjects; same fold under different `--tasks`/`--subjects`/caps → same roles; golden-bit forward |
| P3 | `run_waveform.py` (train+val, `--eval_role test`), `run_partition.py`, `_common.py` | selection record written; test scored once; from-scratch baseline creates its own record; `best.pth` only under `best_val` |
| P4 | `run_pretrain.py` val role, pinned masks | Stage 2 val loss reproducible twice; same digest as Stage 3 |
| P5 | four HPC configs, one `submit_sweep.sh` | `grep -r 'n_folds\|use_small_val\|split_role\|val_subject\|train_ratio\|pretrain_split' code/configs code/scripts` empty |
| P6 | `run_evaluate_kfold.py` gate | pooling refuses dev-smoke, single-fold or mixed-partition input |
| P7 | spec 01/04/05, GLOSSARY, checklist, `TirROI_Resp_plan.md` (`code/notes/plan/experiments/`), delete retired tests | docs use the ADR vocabulary only |
| P8 | pilots: determinism (two val passes byte-identical) and cost (`eval_freq × cost ≤ 5%`, VRAM `< 5%`) | measured numbers recorded below |
| P9 | Stage 2 ×5, Stage 3 ×5, test ×5, pool; then the Criterion 5 study | headline sweep unchanged by the study |

### First end-to-end smoke (2026-11-19, fold 0, 1 epoch, 2 sessions, T4)

`run_partition → run_pretrain → run_waveform → run_waveform --eval_role test →
run_evaluate_kfold --allow_partial` completes, and the gate prints
`protocol gate OK: 1 fold(s), universe ab2828422f51, rule=fixed_epoch [PARTIAL]`.
It found four defects that no unit test could:

1. **`build_pretraining_dataset` overwrote `args.subjects`** with the role's
   list, so building `val` right after `train` intersected it against train and
   the val role came out EMPTY. It now builds each role from a shallow copy.
2. **`run_waveform` referenced `plan` before it was created** (the pairing guard
   runs during checkpoint loading, the plan was built in the data section).
   The plan is now created at the top of `main`.
3. **`--raw_root` was read but never declared** by the Stage-3 parser.
4. **`make_eval_record` crashed** on `spectral_metrics`' non-numeric fields
   (`psd_band_bins=None`); the record now keeps only scalar numbers.

The smoke also confirmed two refusals behave as designed: a second
`--eval_role test` is rejected (sentinel), and `eval_freq: 1` in a config is
rejected at tracker construction (Criterion 4).

Criterion 5 study (licenses `best_val` only): two arms differing in one Stage-3
knob (`huber_beta ∈ {0.5, 1.0}`), folds `{0, 1}`, Stage-2 encoder reused → four
runs. The arm ranked higher by the final val criterion must also rank higher on
the test headline in both comparisons, with a paired bootstrap CI
(`evaluation/metrics.py::bootstrap_ci`) excluding zero. Otherwise `val_select`
stays `fixed_epoch`.

## 5. Tests, deletions, budget

| Clause | Test (`code/tests/test_protocol.py`) |
|---|---|
| I1 | `test_roles_are_pairwise_disjoint` |
| I2 | `test_partition_is_a_pure_function`, `test_two_constructions_are_byte_identical`, `test_val_draw_is_stratified` (val honours the `sex` allocation) and `test_hashseed_does_not_change_the_record` (same record under a different `PYTHONHASHSEED`) |
| I3 / Crit. 1 | `test_roles_partition_the_universe`, `test_share_within_two_points` |
| I4 | `test_both_stages_share_one_record`, `test_partition_utility_matches_stage_2` |
| I5 | `test_training_never_builds_the_test_role` |
| I6 | `test_val_pass_is_deterministic`, `test_missing_val_kept_is_refused`, `test_mask_seed_varies_per_batch_not_per_epoch` |
| I7 | `test_fixed_epoch_writes_no_best_checkpoint`, `test_best_val_reads_only_the_val_role`, `test_criterion_direction` |
| I8 | `test_pooled_report_uses_subject_level_ci` |
| I9 / Crit. 5 | `test_val_role_is_fixed_across_arms` |
| Crit. 3 | `test_leak_guard_aborts`, `test_digest_mismatch_refuses_resume_and_pairing` |
| D2 | `test_no_fold_count_is_expressible` |
| — | `test_training_forward_bits_unchanged` |

`tests/test_kfold_splits.py` (23 tests) and the fold cases in
`test_config_loading.py`, `test_visualize_pretrain.py` are deleted with their code.

**Void, kept as historical records only:** `output/finetune/resp_tir_roi_hpc*`,
`output/pretrain/stage2_hpc_tir_roi_resp*`, `eval/tir_resp/resp_5fold_cv.md`,
`eval/tir_resp/resp_5fold_cv_96respgrid.md` — 80/0/20 with no val role, no
recorded selection rule, a `L_time` predating the Huber term, and a partition
built from filtered subjects.

**Expected per fold** (139-subject universe, verified against `build_plan`):
val 14 (8 F / 6 M), test 28/28/28/28/27, train 97/97/97/97/98 — realised shares
0.698/0.101/0.201, all within Criterion 1's ±2 pp. **Budget:** ≈3-3.5 h per Stage-3 fold plus minutes per test pass
⇒ ≈18 h GPU for the five-fold Stage 3; Stage 2 comparable at its shipped epoch
budget.

**Pilot results (measured 2026-11-19, Tesla T4, `scripts/local/pilot_protocol.py`):**
both criteria pass at the default `eval_freq = 5`.

| | determinism (Crit. 2) | val pass | train pass | wall clock (Crit. 4) | peak VRAM |
|---|---|---|---|---|---|
| Stage 2 | **bit-identical** | 0.037 s | 1.568 s | **0.47 %** | −2.25 % (no increase) |
| Stage 3 | **bit-identical** | 0.073 s | 0.641 s | **2.29 %** | −0.15 % (no increase) |

Two things the pilot changed:

1. `evaluate_waveform_loss` computed `criterion(p, t)` **and** the three terms,
   running the multi-resolution STFT twice per batch; it now sums the weighted
   terms once. Without that, a val pass cost ~1.8× a training step.
2. The first version of the pilot drew **new random batches per pass**, so it
   compared different data and reported "not deterministic". A determinism check
   must reuse one batch list — the engine was right, the harness was wrong.

The absolute seconds are a T4 on a login node; only the RATIOS are the measured
quantities, and they are what Criterion 4 constrains. Re-run this script on the
cluster before quoting a per-fold wall-clock budget.

**Done when** every fold artefact of both stages says `compliant: true`;
`best.pth` exists iff `val_select=best_val`; the pooler cannot be fed a mixed or
dev sweep; each ADR §3 criterion has a test or a recorded measurement.

---

## 6. Status — P0–P8 closed 2026-11-19

The protocol is implemented and enforced **by construction**: roles are computed
only in `code/protocol/`, datasets receive a `FoldPlan` + role, and every
consumer refuses an off-protocol input. Verified state at closure:

- `python -m pytest -q` → **657 passed** (~19 s, CPU).
- **Criterion 1** — realised 97/14/28 per fold, shares 0.698/0.101/0.201, all
  five folds, byte-identical under a different `PYTHONHASHSEED`.
- **Criterion 2 / 4** — measured in the pilot below; bit-identical val passes,
  no peak-VRAM increase.
- **Criterion 3** — `LeakGuard` inside the training step, before `optimizer.step()`,
  plus digest guards on resume and on Stage-2/Stage-3 pairing, with tests.
- **P5 gate** — `code/configs` and `code/scripts` carry no fold-count or
  split-policy key on the protocol path. The only remaining matches are the
  `bp`/`au` carve-out of ADR §3 (`configs/finetune/{au_local,bp_rgb_roi_local}.yaml`,
  which keep their ratio cuts and get no record) and the paired-lineage text
  inside `scripts/project/write_specs.py`, which AGENTS.md forbids running.
  `scripts/hpc/submit_sweep.sh/.sbatch` is the only Stage-2/Stage-3 path, and the
  eleven pre-ADR launchers are deleted with their tests and doc references.

### Open — execution, not implementation

| Item | What it needs |
|---|---|
| **P9 real sweep** | five folds, ≈18-20 h GPU: `EXP_ROOT=… sbatch scripts/hpc/submit_sweep.sbatch`, then `run_evaluate_kfold.py --exp_root …`. Run one fold first and read its artefacts before the other four. |
| **Criterion 5** | the two-arm study (folds {0, 1}, `huber_beta ∈ {0.5, 1.0}`, Stage-2 reused, paired bootstrap CI excluding zero). Until it passes, `val_select` stays `fixed_epoch` and the headline does not depend on it. |
| **Pooler licence** | `gate_test_artefacts` enforces that the five folds agree on ONE selection rule, not that the rule is licensed. A veto would be *stricter than the ADR*, so it needs one sentence added to Criterion 5 first. |

**No thesis number exists until P9 runs.** Closing this plan does not assert any
result — it asserts that the pipeline can only produce a number one way.
