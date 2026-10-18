# Subject K-Fold Plan — cyclic rotation splits for Stage 2 and Stage 3

> **Status: IMPLEMENTED (2026-10-18).** All seven phases landed; §8 records the
> verification run and the four deviations from this plan. The six BDD
> scenarios are pinned by `code/tests/test_kfold_splits.py`.
>
> **Relationship to the other docs.** The *requirement* lives in
> `code/notes/design/kfold_val_strategy_design_v2.md`. This file is the *plan*:
> where the code changes go, in what order, and what must be verified. The
> evaluation protocol that consumes the result is
> `code/notes/design/Evaluation_RespWaveform_protocol.md` §3.2, which links here.
> Where the two disagree, the design document wins on intent and this file wins
> on code placement.

---

## 0. Scope decisions (2026-10-18)

| # | decision | value |
|---|---|---|
| 1 | split roles | explicit `split_role: train \| val \| test` — three disjoint dataset views, not a two-view workaround |
| 2 | stage coverage | **both** Stage 2 (the pretrain corpus excludes the test fold) and Stage 3 |
| 3 | fold count | explicit `n_folds` config/CLI value; `n_folds=0` keeps today's behaviour; 5 on HPC, 4 locally |
| 4 | fold algorithm | sorted unique subject ids + `np.array_split`, **not** `sklearn.model_selection.GroupKFold` (see §3 trap 4) |
| 5 | test reporting | the fold owns the held-out set; `--test_subjects` stays a manual override only |
| 6 | Option A selection | fixed-epoch checkpointing, no val-based selection, recorded in the metrics JSON |

**Explicitly out of scope:** AU-probe fold wiring (`runners/run_au_probe.py`
gets the new flags but no fold behaviour), any change to `core/`, and the
224 px vs 64 px geometry question. This plan only moves *who is in which split*.

---

## 1. Current state (verified 2026-10-18)

The k-fold protocol is **not implemented**. `n_folds`, `fold`, `val_frac` and
`use_small_val` appear in zero files under `code/data`, `code/runners`,
`code/configs`, `code/tests`, `code/scripts`. The documentation is ahead of the
code in three places:

- `code/notes/spec/01_data_pipeline.md` §51-52 calls cyclic rotation "the
  authoritative evaluation protocol".
- `code/notes/spec/04_configs_and_execution.md` §118 already lists
  `n_folds / fold / val_frac` as config keys — no runner defines them, so a YAML
  carrying them is rejected at load.
- `code/notes/spec/05_risks_and_compliance.md` §161 tells the reader to "always
  use the new GroupK-Fold Cyclic Rotation".

What exists instead is a ratio cut over **sorted** keys, duplicated in three
places:

| File | Symbol | Cut |
|---|---|---|
| `code/data/tir_resp_dataset.py` | `TirRoiRespFinetuneDataset.__init__` (~1460-1500) | `keys[:n_keep]` on sorted subjects/sessions, `val_subject` override |
| `code/data/rgb_roi_dataset.py` | `RgbRoiRespirationFinetuneDataset.__init__` (~931-960) | identical duplicate |
| `code/data/paired_dataset.py` | `PairedSessionDataset` | same ratio logic, `test_mode` switch |

This is why every config comment says a `0.8` ratio "can only ever produce ONE
fold, always F004": nothing rotates, and `train_ratio=1.0` is the only way to
keep everyone.

Stage 2 is additionally **hard-blocked** from splitting: `code/data/datasets.py`
lines 106-112 raise unless `pretrain_split == 'all'`.

---

## 2. Target behaviour

For a corpus of `N` usable subjects and `n_folds = k`, one `GroupKFold`-style
partition is computed once, over sorted subject ids, into folds
`F_0 … F_(k-1)`. Fold `j` of the run then assigns:

| Role | Subjects | Stage 2 | Stage 3 Option A | Stage 3 Option B |
|---|---|---|---|---|
| `test` | `F_j` | excluded from pretraining | reported | reported |
| `val` | empty | empty | `~10%` of the non-test pool | `~10%` of the non-test pool |
| `train` | all folds `!= j` | the pretrain corpus | all folds `!= j` | non-test pool minus val |

For `N=140, k=5` this gives 28 / 0 / 112 (Option A) and 28 / 14 / 98 (Option B),
matching design scenarios 1-3. For the 4-subject local corpus, `k=4` gives
1 / 0 / 3 (design scenario 6).

`n_folds = 0` or `1` is the **legacy path**: `train` and `val` behave exactly as
today (`train_ratio` / `split_by` / `val_subject`), and there is no `test` role.
This is the compatibility guarantee — every existing config and smoke script
must produce byte-identical split keys after this change.

---

## 3. Traps verified in the code

Each of these produces a silently wrong result rather than an error.

1. **Empty subject list means "all subjects".**
   `code/data/tir_resp_dataset.py:263` computes
   `want_subj = {str(s) for s in subjects} if subjects else None`.
   An empty `val` role forwarded as `subjects=[]` therefore loads the **whole
   corpus** into the validation set. The empty role must be short-circuited
   before the base dataset is constructed, never passed down.

2. **An empty loader is a hard failure.**
   `code/runners/_common.py:167` `check_loader_not_empty` raises `SystemExit` on
   zero batches, and `code/runners/run_waveform.py:582` builds the val dataset
   unconditionally. Option A must not construct a val loader at all.

3. **Unknown YAML keys are rejected.**
   `code/runners/_common.py::_check_config_keys` fails the run when a config key
   is not an option of the target parser. Five runners call
   `parse_args_with_config` — `run_waveform.py`, `run_finetune.py`,
   `run_pretrain.py`, `run_visualize.py`, `run_au_probe.py` — so the new flags
   must be registered in all five via one shared helper.

4. **`GroupKFold` is the wrong primitive.**
   It balances by **sample counts per group**, so a subject that gains or loses
   clips changes fold membership. Folds are defined over subjects, so
   `np.array_split` on the sorted unique ids is exact and reproducible, and it
   gives the even 28-subject folds the acceptance criteria assert.

5. **Global seeding belongs to `init_env`.**
   `AGENTS.md` §7 centralises seeding in `runners/_common.py::init_env`. The
   Option-B val draw must use `np.random.default_rng(val_seed + fold)`, never
   `np.random.seed`, or it perturbs every rank's global state.

6. **Checkpoint selection lives in the runner, not the engine.**
   `code/runners/run_waveform.py` (~712-736) writes `best.pth` when
   `pearson > best_pearson` on the **val** loader. With Option A there is no val
   set, so this either crashes or selects on the test fold.

7. **The partition can drift between stages.**
   The partition derives from the usable-subject list, so `--subjects`,
   `--tasks`, `--max_sessions` and the rail/spread filters can change the
   universe between the Stage-2 and Stage-3 runs of the same fold, silently
   re-shuffling membership. The partition must be computed on the **full usable
   subject list**, and written to `fold_manifest.json` in the run directory so a
   mismatch is visible after the fact.

---

## 4. Implementation phases

Phases 2, 3 and 4 can partly overlap once phase 1 lands. Phase 5 blocks the
first Option-A production run. Phase 7 gates any commit.

### Phase 1 — split core

New `code/utils/splits.py`, pure numpy, no torch, no sklearn:

- `subject_folds(subjects: Sequence[str], n_folds: int) -> List[List[str]]` —
  sorted unique ids, `np.array_split` into `n_folds` disjoint folds.
- `get_stage_splits(subjects, fold, n_folds=5, stage=3, use_small_val=False,
  val_seed=42) -> Dict[str, List[str]]` with keys `train`, `val`, `test`.
  - `n_folds <= 1`: `train = test = sorted(all)`, `val = []`.
  - `stage` must be 2 or 3; `stage == 2` forces `use_small_val = False`.
  - `test = F_fold`; `train = union(F_i, i != fold)` when val is unused.
  - Option B draws `max(1, round(0.10 * n_total))` subjects from the non-test
    pool using `default_rng(val_seed + fold)`, capped at `len(pool) - 1`, and
    raises when the corpus cannot support a val role.
  - Raise on `not 0 <= fold < n_folds` and on `n_folds < 0`.
- `write_fold_manifest(path, fold, n_folds, stage, roles)` — JSON record of the
  partition, referenced by trap 7.

Type hints on every argument and return, and a docstring stating that each
returned list holds **subject ids**, not sample or clip indices.

### Phase 2 — dataset role plumbing

- `code/data/tir_resp_dataset.py::TirRoiRespFinetuneDataset.__init__`: add
  `split_role: Optional[str] = None`, `n_folds: int = 0`, `fold: int = 0`,
  `use_small_val: bool = False`. Resolve
  `self.split_role = split_role or ('train' if is_train else 'val')`. When
  `n_folds > 1`, take `self.split_keys` from `get_stage_splits(...)[role]` and
  skip the ratio branch entirely. Empty `val` is allowed only on this path;
  empty `train`/`test` still raise.
- `code/data/rgb_roi_dataset.py::RgbRoiRespirationFinetuneDataset.__init__`:
  the same change. The split block is currently duplicated in both classes —
  factor it into one helper in `code/utils/splits.py` rather than adding a third
  copy.
- `code/data/datasets.py::build_dataset`: add `split_role: Optional[str] = None`
  (default derives from `is_train`, so existing call sites are untouched) and
  forward it to `build_paired_dataset`, `build_tir_roi_finetune_dataset` and
  `build_rgb_roi_finetune_dataset`.
- Reject `split_by='session'` whenever `n_folds > 1`: a session split can put
  two sessions of one subject on both sides, which breaks design scenario 5 even
  though clip overlap itself is handled correctly.

### Phase 3 — Stage 2 fold-awareness

- `code/data/datasets.py::build_pretraining_dataset`: replace the
  `pretrain_split != 'all'` raise with a two-value guard — `all` (today) and
  `fold` (new). On `fold`, compute `get_stage_splits(stage=2, ...)['train']` and
  pass it as the builders' `subjects=`.
- Confirm the three pretrain builders honour `subjects`
  (`TirRoiRespPretrainDataset`, `RgbRoiRespPretrainDataset`,
  `PairedPretrainDataset`) — all three reach `discover_sessions(subjects=...)`.

### Phase 4 — CLI

- New `add_split_args(parser)` in `code/runners/_common.py` adding `--n_folds`
  (0 = off), `--fold`, `--use_small_val` and `--split_role`. Call it from all
  five config-parsing runners.
- `code/runners/run_waveform.py`: build the val dataset only when `n_folds == 0`
  or `use_small_val`; always build `train`; when a test fold exists, evaluate it
  once at the end through `build_dataset(..., split_role='test')`.

### Phase 5 — Option A checkpoint policy

- `code/runners/run_waveform.py` (~712-736): when `n_folds > 1 and not
  use_small_val`, do not write `best.pth` from val Pearson. Keep the fixed epoch
  schedule, save the final weights, and record `selection: 'fixed_epoch'` in
  `metrics_final.json` so the absence of model selection is auditable.

### Phase 6 — configs and HPC rollout

- Add `n_folds` / `fold` / `use_small_val` to the HPC configs only
  (`code/configs/finetune/resp_tir_roi_hpc.yaml`,
  `code/configs/finetune/resp_tir_roi_hpc_smoothl1.yaml`). Leave the
  `*_local.yaml` files on `val_subject`: four subjects are already
  leave-one-subject-out.
- `code/scripts/hpc/submit_waveform_tir_roi.sbatch`: add an array submission
  that exports `FOLD=$SLURM_ARRAY_TASK_ID` and a per-fold run name, in the same
  style as the existing `VAL_SUBJECT` override. sbatch does not expand variables
  inside `#SBATCH` lines, so the array bound stays a literal with a comment.

### Phase 7 — tests and docs

- New `code/tests/test_kfold_splits.py` covering design scenarios 1-6, plus a
  regression test that `n_folds=0` with `val_subject` reproduces today's split
  keys exactly.
- Regenerate the spec with `python code/scripts/project/write_specs.py` and
  review the diff; the generator overwrites hand-edited content.
- No new index or status files (AGENTS.md §5).

---

## 5. Acceptance criteria

Design §6 scenarios, mapped to this plan:

| # | Scenario | Covered by |
|---|---|---|
| 1 | Stage 2, fold 0, 140 subjects → 28 test / 0 val / 112 train | `get_stage_splits(stage=2)`, phase 1 test |
| 2 | Stage 3 Option A, fold 2 → 28 / 0 / 112 | `get_stage_splits(stage=3, use_small_val=False)` |
| 3 | Stage 3 Option B, fold 4 → 28 / 14 / 98, mutually disjoint | `get_stage_splits(use_small_val=True)` |
| 4 | Cyclic rotation determinism across runs | sorted-id partition; no RNG on this path |
| 5 | No clip from one subject in two roles | partition is over subjects, before clip generation; `split_by='session'` rejected |
| 6 | `n_folds=4` on 4 subjects → 1 test / 3 train, modulo 4 | phase 1 test |

---

## 6. Verification

1. `python -m compileall code`
2. `cd code && python -m pytest` — the new file plus `test_config_loading.py`
   must both stay green.
3. Local smoke: `source scripts/env_local.sh`, then a `tir_roi_resp` run with
   `--n_folds 4 --fold 0`; confirm the printed split keys hold one test subject
   and three train subjects, and that the sets are disjoint.
4. Regression: the same smoke with `--n_folds 0 --val_subject F004` prints the
   same keys as before the change.
5. Stage 2: `--pretrain_split fold --n_folds 4 --fold 0` must not read any
   session of the fold-0 subject.
6. Inspect `fold_manifest.json` across two folds and confirm the test sets
   partition the corpus with no overlap.
7. `python code/scripts/project/write_specs.py`, then `git diff code/notes/spec`.

---

## 7. Open items

1. Option A assumes "last epoch, no selection". Confirm this is acceptable, since
   it removes the `best_pearson` safety net that every Stage-3 run currently has.
2. Whether the duplicated split block is refactored into one shared helper now
   (recommended, ~40 lines x2) or copied again — AGENTS.md discourages new
   duplication.

---

## 8. Implementation record (2026-10-18)

### What landed

| Phase | File | Symbol |
|---|---|---|
| 1 | `code/utils/splits.py` (new) | `subject_folds`, `get_stage_splits`, `write_fold_manifest`, `resolve_split_keys` |
| 2 | `code/data/tir_resp_dataset.py`, `code/data/rgb_roi_dataset.py` | the duplicated split block is now one call to `resolve_split_keys`; both classes take `split_role` / `n_folds` / `fold` / `use_small_val` |
| 2 | `code/data/datasets.py` | `build_dataset(..., split_role=...)` forwards to both ROI builders |
| 3 | `code/data/datasets.py` | `build_pretraining_dataset` accepts `pretrain_split='fold'` and excludes the test fold |
| 4 | `code/runners/_common.py` | `add_split_args`, called from `run_waveform`, `run_finetune`, `run_pretrain`, `run_visualize`, `run_au_probe` |
| 5 | `code/runners/run_waveform.py` | val loader skipped under Option A; `selection` + `eval_role` recorded in `metrics_final.json` |
| 6 | two HPC configs + `submit_waveform_tir_roi.sbatch` | `n_folds: 5`, `--array` support via `FOLD=$SLURM_ARRAY_TASK_ID` |
| 7 | `code/tests/test_kfold_splits.py` (new) | design scenarios 1-6 + legacy-key regression |

### Deviations from this plan (all deliberate)

1. **An empty role is represented, not raised.** The plan allowed an empty `val`;
   the implementation extends that to any SCORED role: a held-out subject whose
   clips are all filtered out (locally F004) yields a real empty dataset with an
   `empty_reason`, and the runner records `skipped: true` instead of crashing.
   An empty `train` role still fails — there is nothing to learn from. The base
   dataset can never be handed `subjects=[]` (that means "every subject").
2. **`--split_role` is an override, not a second switch.** One process needs three
   views, so runners pick the role; the flag only forces a single-view run.
3. **`write_fold_manifest` is wired, not optional.** It is written by Stage 2 (in
   `build_pretraining_dataset`) and Stage 3 (in `run_waveform`), because trap 7
   can only be caught by comparing the two files after the fact.
4. **The spec was hand-edited, not regenerated.** `write_specs.py` produced a
   net **-313 lines** and deleted the k-fold protocol, `physio_norm` and
   `rail_touch_v` sections. Only `01_data_pipeline.md` §split-policy,
   `04_configs_and_execution.md` (the key table) and `05_risks_and_compliance.md`
   §4.6 were edited by hand. Re-running the generator will regress them again.

### Verification run

Dataset-level:

| Check | Result |
|---|---|
| `python -m compileall code` | clean |
| `python -m pytest` (venv) | 328 passed (7 new) |
| Local corpus, `n_folds=4, fold=0` | test `['F001']`, train `['F002','F003','F004']`, 10 vs 30 clips |
| Local corpus, Option B `fold=1` | test `['F002']`, val `['F003']`, train `['F001','F004']` |
| Legacy `n_folds=0` (ratio and `val_subject`) | identical keys to the pre-change code |
| Stage 2 `build_pretraining_dataset(fold)` | corpus has no `F001` session; manifest written |
| Config parsing | HPC configs → `n_folds=5`; local configs → `n_folds=0` |

End-to-end runner execution (local corpus, 4 subjects, 64 px, 1 epoch):

| Command | Result |
|---|---|
| `run_pretrain.py --pretrain_split fold --n_folds 4 --fold 0` | 1 epoch; `fold 0/4 -> 3 train, 1 held out ['F001']`; manifest written |
| `run_pretrain.py --pretrain_split fold --n_folds 0` | falls back to the whole corpus, warning printed, 1 epoch |
| `run_waveform.py --n_folds 4 --fold 0` (Option A) | no val loader; `metrics_final.json` has `selection='fixed_epoch'`, `eval_role='test'`, 4 test clips scored |
| `run_waveform.py --n_folds 0` (legacy) | val = `['F004']`, `selection='best_pearson'`, `eval_role='val'` |

### Empty folds are supported (data, not code)

With the shipped cleaning filters (`rail_touch_v: 9.900`, `min_signal_spread: 0.1`),
**F004's respiration is rail-saturated and yields zero usable clips**. A fold
whose scored subject is F004 is therefore a *valid fold with nothing to score*,
and the framework now says so instead of failing:

| fold | test subject | test view | result |
|---|---|---|---|
| 0 | F001 | 7 clips | scored normally |
| 1 | F002 | 10 clips | scored normally |
| 2 | F003 | 10 clips | scored normally |
| 3 | F004 | **empty** | `skipped: true`, `n_clips: 0`, reason recorded, exit 0 |

Contract:

- `utils/splits.py::resolve_split_keys` returns `[]` for an empty `val`/`test`
  role (an empty `train` still raises).
- Both Stage-3 datasets short-circuit an empty subject list into an EMPTY view
  and pass `allow_empty=(role != 'train')` to the base, so a scored subject whose
  clips are all filtered also becomes an empty view.
- `run_waveform` degrades an empty **val** to no-val (loud warning) and, for an
  empty **test**, writes `metrics_final.json` with `skipped: true`,
  `n_clips: 0`, `skip_reason` and the fold identity, then exits 0 — so a sweep
  over many folds continues instead of dying on one unusable subject.
- An empty **train** is a hard `SystemExit`: there is nothing to train on.

The legacy `n_folds = 0` default (`train_ratio 0.8`, val = F004) hits the same
rail property; it is visible as the same empty-view path, not a k-fold regression.

### Per-fold artefact ownership (audited 2026-10-18)

Every stage writes into `--output_dir`. Fold isolation therefore depends on the
output dir being fold-specific, and on nothing else being shared:

| Artefact | Owner | Fold-safe because |
|---|---|---|
| `<out>/checkpoints/checkpoint-NNNN.pth`, `latest_checkpoint.txt` | Stage 2 / 3 | per-fold `OUT_DIR`; `auto_resume_model` follows that pointer only |
| `<out>/fold_manifest.json` | Stage 2 and 3 | one file per fold; the pair must agree |
| `<out>/metrics.jsonl`, `metrics_final.json`, `training_curves.png` | Stage 3 | per-fold `OUT_DIR` |
| `<out>/preds.npy`, `targets.npy`, `entries.json`, `predictions_final.png` | Stage 3 | per-fold `OUT_DIR` (`--save_preds`) |
| `<out>/subject_metrics.json`, `subject_figures/` | `run_evaluate_subject` | per-fold `--pred_dir` |
| ROI cache (shared) | corpus | key hashes the **unfiltered** corpus, deliberately: one cache for all folds, no rebuild, no per-fold copy. Fold membership is enforced by the dataset's `subjects=` filter, not by the cache |
| `fold_manifest.json` of Stage 2 vs 3 | both | **was** write-only; now verified, see below |

Two collisions were found and fixed in the same change:

1. **`STAGE2_DIR` was not fold-scoped.** Under `--array=0-4` every fold resolved
   the same "newest checkpoint" directory, so fold k could be fine-tuned from a
   Stage-2 encoder that had already seen fold k's test subjects — the exact leak
   the protocol exists to prevent. It now defaults to
   `pretrain/stage2_hpc_tir_roi_resp_fold$FOLD`.
2. **`submit_pretrain.sbatch` had no fold concept.** Two folds sharing one output
   dir would collide on `checkpoint-NNNN.pth` and `latest_checkpoint.txt`, and
   `auto_resume_model` would silently continue the *other* fold. The script now
   supports `--array`, appends `_fold$FOLD` to the run name, and passes
   `--n_folds/--fold/--pretrain_split fold`.

In addition, `runners/run_waveform.py::check_stage2_fold_contract` now compares
the fold recorded in the checkpoint's `args` against the fold being run:

| Checkpoint records | Result |
|---|---|
| fold 0, run is fold 0 | `fold contract OK` |
| fold 0, run is fold 1 | **`SystemExit`** — refuses to run a leaked pair |
| no fold (`pretrain_split=all`) | loud warning (legitimate baseline, invalid CV number) |

The verdict is stored as `fold_contract` in `metrics_final.json`.

### Stage 1 (backbone init) isolation

Every fold must start Stage 2 from the **same pristine VideoMAE ViT-Base**, and
no fold may inherit another fold's weights. Three properties make that true:

| Property | Mechanism |
|---|---|
| Stage-1 source is shared and read-only | `models/pretrained.py::resolve_encoder_weights` only *reads* the cached file; nothing in training writes to `INITIAL_MODELS_DIR`. Checkpoints go to `<out>/checkpoints/` |
| Concurrent folds cannot corrupt the shared file | the download temp is now `…<kind>.<pid>.part`; `os.replace` is the only contended step and it is atomic. A shared `.part` name previously let two array tasks interleave writes into one file and install a corrupt checkpoint for all later folds |
| A fold can only resume its own state | `utils/checkpoint.py::check_resume_fold` compares the fold recorded in the checkpoint's `args` with the running fold: same -> OK, different -> `SystemExit`, checkpoint written without folds -> warning |

`run_pretrain.py` also prints a provenance line per fold:

```
[stage1] init from <repo>/models/initial/videomae_base_patch16_224.pth
         (spec 'videomae:k400', 148 encoder tensors, 376.9 MB, mtime …)
```

so a fold sweep can be checked by eye: identical path, size and mtime means all
folds started from the same baseline.

Verified locally: fold 0 and fold 1 each re-initialise from that file, and
pointing fold 1 at fold 0's `--output_dir` is refused with
`[resume] FOLD MISMATCH: … fold 0 of 4, but this run is fold 1 of 4`.

### HPC rollout: folds run SEQUENTIALLY

Both job scripts run the whole sweep inside **one** allocation. The 4 GPUs are
used *within* a fold (DDP); fold `k+1` starts only after fold `k` exits.

| Aspect | Behaviour |
|---|---|
| submission | `sbatch scripts/hpc/<job>.sbatch` → folds `0..N_FOLDS-1`, in order |
| subset | `FOLDS="0 1"` (still sequential); `FOLDS=3` for a single fold |
| `--array` | **refused** (exit 2) — an array task would run the whole sweep, and the tasks would run in parallel |
| per fold | own `OUT_DIR` (`…_fold<k>`), own Stage-2 checkpoint dir, own `MASTER_PORT` offset (avoids a TIME_WAIT rebind) |
| failure | a fold that cannot start is reported, the sweep continues, and the job exits 1 listing the failed folds |
| resume | each fold has its own `latest_checkpoint.txt`, so a requeued job continues where it stopped |

**Resource note.** A full Stage-3 sweep is 5 × ~7-9 h (~45 h), which exceeds the
`acc` 1-day cap, so the default partition in both scripts is now `acc_long`
(`--time=5-00:00:00` for Stage 3, `2-00:00:00` for Stage 2, ~4.5 h per fold). A
single fold can be submitted back to the short queue with
`sbatch --partition=acc --time=24:00:00 …`.

Verified with a stubbed `srun` (no cluster needed): folds execute 0→4 in order,
each with a distinct output dir and port; removing one fold's Stage-2 checkpoint
produces `ERROR: fold 2: no Stage-2 checkpoint resolved`, the remaining folds
still run, and the job exits 1 with `these folds failed: 2`.

**Not implemented:** a pooling script that walks the per-fold run dirs and
aggregates `subject_metrics.json`. Until it exists, a pooler must skip run dirs
whose `metrics_final.json` has `skipped: true` (those folds have no
`preds.npy`/`entries.json`, so `run_evaluate_subject.py` cannot read them).



