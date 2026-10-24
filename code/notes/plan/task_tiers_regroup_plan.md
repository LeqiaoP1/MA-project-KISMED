# Task Distortion-Tier Regrouping v2 (TIR ROI RESP)

Status: **IMPLEMENTED** 2026-10-24. Date: 2026-10-24.

## Outcome (2026-10-24)

All phases landed; `python -m pytest -q` -> **371 pass** (was 345; the new
`tests/test_task_group_configs.py` adds 26, including the four BDD scenarios).

- **Phase 1**: `SHIPPED_TASK_GROUPS` in `data/task_groups.py`; the 8 configs,
  the survey default, the gate-probe default and the gate sbatch literal synced.
- **Phase 2**: `--task_groups` on the three eval runners. Each output JSON now
  carries `task_group_source`, `task_groups` and `task_levels`; a task in no
  group prints a WARNING (it would otherwise be skipped silently).
- **Phase 3**: `check_stage2_roi_contract` notes a `task_groups` mismatch ONLY
  when a group is selected; `test_config_loading.py` compares it parsed.
- **Phase 4** (re-derived from the same dumps, no re-training; v1 artifacts
  kept):

  | scope | low | moderate | high |
  |---|---|---|---|
  | k-fold, per-subject band-Pearson (n) | +0.411 (387) | +0.278 (239) | +0.291 (634) |
  | session, std arm (n) | +0.573 (82) | +0.413 (51) | +0.435 (140) |
  | session, smoothl1 arm (n) | +0.578 (82) | +0.494 (51) | +0.437 (140) |
  | subject, std arm (n) | +0.663 (82) | +0.464 (51) | +0.546 (140) |

  New files: `session_metrics_regrouped.json` (both arms),
  `subject_metrics_regrouped.json`, `kfold_resp_tir_roi_hpc_regrouped.json`.
  Every per-group `n` sums to the scored session count (273 / 1260).
- **Phase 5**: spec template + generated spec, `resp_data_quality.md`
  convention sentence, `code/README.md`, `TirROI_Resp_Guide.md`,
  `slides_spec.yaml`, plus dated v2 sections in `resp_5fold_cv.md` and
  `finetuned1_subjectReport.md`.
- **Figures**: `resp_kfold_task_group_medians_v2.svg` (new generator
  `notes/eval/tir_resp/plot_task_group_medians.py`, reads the regrouped JSON;
  task rows in T1..T10 order, the `low`-tier tasks in bold, no caption below the
  axes) and `resp_T3_waveforms_v2.svg` (same probe with `--no-footers`: no
  annotation and no caption, so the figure ends at the axis labels). The version
  lives in the FILENAME only -- `--note` defaults to empty, so no figure carries
  a tier annotation. The unsuffixed SVGs stay as the v1 record.
- **Scatter**: `resp_kfold_task_scatter_v2.svg` from the new
  `notes/eval/tir_resp/plot_task_scatter.py` (per-task Pearson vs rate MAE,
  colour = oracle amplitude error). Per-task only, so it carries no tier text at
  all and no caption below the axes.

Deviations: (a) the eval figures were written to each run's existing `figures/`
directory instead of a `figures_regrouped/` copy -- the plots do not depend on
the grouping; (b) `presentation/build_deck.py` was NOT run because
`python-pptx` is not installed in `.env` (the YAML is updated; rebuild later);
(c) `notebooks/TirROI_Resp_Pipeline.ipynb` stored outputs are left untouched as
an execution log.

Rationale of record: `code/notes/design/TIR_ROI_RESP_TasksDistortion.md`.

Replaces the shipped tiers

```text
low      = [T2, T3]
moderate = [T4, T7, T8, T10]
high     = [T1, T5, T6, T9]
```

with

```text
low      = [T2, T3, T5]              # passive viewing + short yes/no answers
moderate = [T1, T4]                  # laughter / facial expression + startle
high     = [T6, T7, T8, T9, T10]     # singing, threat, pain, anger, disgust
```

Delta: **T5 high -> low**, **T1 high -> moderate**, **T10 moderate -> high**.
The new mapping partitions T1..T10 (3 / 2 / 5) with every task in exactly one tier.

## Decisions

- **Tiers stay labels by default.** Every `tir_roi*` config keeps `task_set: ''`, so the
  corpus is unchanged and no existing checkpoint is invalidated. Tier-restricted
  training is already plumbed (`--task_set`, plus `TASK_SET` in
  `scripts/hpc/submit_pretrain.sbatch` and `submit_waveform_tir_roi.sbatch`); this
  plan only documents and tests that path.
- **Existing dumps are re-labelled, not rewritten.** `entry['task_level']` is pure
  post-hoc metadata written by `runners/run_waveform.py`, so adding a
  `--task_groups` override to the three eval runners re-derives every per-group
  table from the same predictions with **no retraining and no GPU**.
- **The mapping is enforced as a partition** even though `data/task_groups.py`
  supports many-to-many, because `runners/run_survey_tir_resp.py::_aggregate` and
  `notes/eval/tir_resp/gate_model_input.py::group_of` assume one group per task
  (first match / no de-duplication).
- **v1 result tables are preserved verbatim**; v2 numbers are appended as dated
  sections so the historical measurements stay traceable.
- The mapping is **never defined in code today** -- it is passed in from YAML/CLI.
  Only three defaults hardcode it, and all three must be synced.

## Phase 1 - Canonical definition (blocks phases 3, 4, 5)

1. **Add a single shipped-definition constant** to `code/data/task_groups.py`:

   ```python
   SHIPPED_TASK_GROUPS = 'low=T2|T3|T5;moderate=T1|T4;high=T6|T7|T8|T9|T10'
   ```

   Export it in `__all__` and point its docstring at the design doc above.
   Resolution stays config-driven -- the constant only supplies defaults and a
   test anchor, it is never applied implicitly.

2. **Edit the 8 `task_groups:` blocks**, keeping `task_set: ''` and rewriting the
   inline tier comments so they explain the NEW criterion:
   - `configs/pretrain/stage2_local_tir_roi_resp.yaml`
   - `configs/pretrain/stage2_local_tir_roi_crossmae.yaml`
   - `configs/pretrain/stage2_hpc_tir_roi_resp.yaml`
   - `configs/finetune/resp_tir_roi_local.yaml`
   - `configs/finetune/resp_tir_roi_local_matched.yaml`
   - `configs/finetune/resp_tir_roi_crossmae.yaml`
   - `configs/finetune/resp_tir_roi_hpc.yaml`
   - `configs/finetune/resp_tir_roi_hpc_smoothl1.yaml`

   The `# MUST mirror ...` comments stay valid, but they are only comments -- see
   phase 3 for the check that finally enforces them.

3. **Sync the three hardcoded defaults.**
   - `runners/run_survey_tir_resp.py:47-48` -> `DEFAULT_TASK_GROUPS = tgrp.SHIPPED_TASK_GROUPS`
     (the module already imports `data.tir_resp_dataset`).
   - `notes/eval/tir_resp/gate_model_input.py:192-195` -> same constant (the script
     already bootstraps `sys.path` for `data.*` imports).
   - `scripts/hpc/gate_model_input.sbatch:58` keeps a literal
     (`low=T2,T3,T5;moderate=T1,T4;high=T6,T7,T8,T9,T10`) because bash cannot import
     Python; it is pinned by the test in phase 3.

## Phase 2 - Relabel path for existing prediction dumps (parallel with phase 1)

4. **Add `--task_groups` to the three eval runners**
   (`runners/run_evaluate_session.py`, `run_evaluate_subject.py`,
   `run_evaluate_kfold.py`). Semantics:
   - default `''` = current behaviour, i.e. read `task_level` from `entries.json`;
   - when non-empty: `parse_task_groups(...)` then invert with `levels_by_task(...)`
     and use the result in place of the recorded label.

   Hook points (the only places `task_level` is read):
   - `run_evaluate_session.py:399` (`levels_of[name] = ...`)
   - `run_evaluate_subject.py:308` (`levels_of[name] = ...`)
   - `run_evaluate_kfold.py:130` (inside `_score_fold`, so `aggregate_kfold` is
     untouched -- it already aggregates `rec['task_level']`)

   Also required:
   - **Warn once** about tasks that fall in no group, because
     `evaluation.metrics.breakdown_by_label` silently skips unmapped sessions and
     the per-group `n` would quietly shrink.
   - **Record provenance** in the output JSON: the `task_groups` definition used,
     a `task_group_source: 'cli' | 'entries'` flag, and the resolved
     `task_levels` map. Without this a re-derived file is indistinguishable from
     an original one.
   - Do **not** give the flag a default mapping. A silent default would be a
     fourth copy of the tiers and would relabel dumps by accident.

## Phase 3 - Drift guards (depends on phase 1)

5. **`runners/run_waveform.py::check_stage2_roi_contract`** (~lines 186-197): add
   `task_groups` to the corpus-level note loop, compared **parsed**
   (`parse_task_groups` on both sides) rather than as strings. Skip the
   comparison when *both* sides have empty `task_set` and empty `tasks`: with no
   group selected the mapping is label-only, so a mismatch cannot change the
   corpus. This keeps existing k-fold Stage-3 re-runs from printing a false note
   while still catching the real hazard (same `task_set`, different definitions).

6. **`tests/test_config_loading.py:150`** (`test_stage2_and_stage3_agree_on_the_roi_contract`):
   extend the compared keys from `task_set` to `task_set` + parsed `task_groups`
   for every Stage-2/Stage-3 sibling pair.

7. **New `tests/test_task_group_configs.py`**:
   - every shipped config's `task_groups` parses equal to `SHIPPED_TASK_GROUPS`;
   - the mapping is a partition of T1..T10 (each task exactly once, none missing);
   - `run_survey_tir_resp.DEFAULT_TASK_GROUPS` and the gate-probe `--groups`
     default parse to the same mapping;
   - `scripts/hpc/gate_model_input.sbatch`'s `TASK_GROUPS=` literal matches
     (regex on the file -- no execution);
   - tier-restricted resolution: `task_set='low'` -> `('T2','T3','T5')`,
     `'low,moderate'` -> `('T1','T2','T3','T4','T5')`, `''` -> `None`;
   - `TaskSelection.levels_of` correct for all ten tasks.

## Phase 4 - Re-derive measurements (depends on phase 2)

8. Re-run the evaluators locally (CPU, minutes) against the dumps already under
   `output/finetune/`, writing to **new** paths so the v1 artifacts referenced by
   the existing docs stay intact:

   - k-fold (all subjects):
     `--out ../output/eval/resp_kfold/kfold_resp_tir_roi_hpc_regrouped.json`
   - session level, both arms (`resp_tir_roi_hpc`, `resp_tir_roi_hpc_smoothl1`):
     `--out .../session_metrics_regrouped.json --fig_dir .../figures_regrouped`
   - subject level:
     `--out .../subject_metrics_regrouped.json`

   Empty `--out` / `--fig_dir` would overwrite the originals -- always pass them.

9. Record the v2 numbers: append a dated "grouping v2" section to
   `code/notes/eval/tir_resp/resp_5fold_cv.md` and
   `code/notes/eval/tir_resp/finetuned1_subjectReport.md` with the tier ->
   `pearson_band` median, n and the delta versus v1, explicitly stating that the
   numbers are re-aggregations of the same dumps (no model change).

## Phase 5 - Docs (depends on phases 1 and 4)

10. Update the current-state statements (historical tables stay untouched):
    - `code/scripts/project/write_specs.py:258` **and**
      `code/notes/spec/01_data_pipeline.md:96` -- both, otherwise the next spec
      regeneration silently reverts the .md.
    - `code/notes/eval/tir_resp/resp_data_quality.md:44-45` -- the "convention:"
      sentence only.
    - `code/README.md` task-groups section -- add the tier definitions and the
      `TASK_SET=low` / `--task_set low,moderate` corpus-restriction path, with the
      "must mirror between Stage 2 and Stage 3" caveat.
    - `notebooks/TirROI_Resp_Guide.md:46-49, 169` -- the Group A/B/C definition.

11. `presentation/slides_spec.yaml`: definitions at lines 156 and 263 -> v2;
    the per-group result rows at 145-147 -> the re-derived numbers if phase 4
    covers `resp_tir_roi_hpc`, otherwise label the table "v1 tiers". Rebuild with
    `presentation/build_deck.py`.

## Files

- `code/data/task_groups.py` - `parse_task_groups`, `levels_by_task`, `levels_of`;
  new `SHIPPED_TASK_GROUPS`.
- `code/configs/pretrain/*tir_roi*.yaml`, `code/configs/finetune/*tir_roi*.yaml` -
  the 8 `task_groups:` blocks.
- `code/runners/run_waveform.py` - `ROI_CONTRACT_KEYS`, `check_stage2_roi_contract`.
- `code/runners/run_evaluate_session.py`, `run_evaluate_subject.py`,
  `run_evaluate_kfold.py` - the three `task_level` hooks and `get_args`.
- `code/runners/run_survey_tir_resp.py` - `DEFAULT_TASK_GROUPS`.
- `code/evaluation/metrics.py::breakdown_by_label` - skips unmapped sessions
  (the reason the phase-2 warning matters).
- `code/notes/design/TIR_ROI_RESP_TasksDistortion.md` - the rationale of record.
- `code/notes/eval/tir_resp/{resp_5fold_cv.md, finetuned1_subjectReport.md,
  resp_data_quality.md}` - v2 sections and the convention sentence.

## Verification

1. `cd code && ../.env/bin/python -m pytest -q` -- baseline is 345 passing; expect
   the new test file and no regressions.
2. `python data/task_groups.py` (module self-test) and `python -m compileall` over
   the four touched runners.
3. Dataset check: `runners/run_inspect_tir_resp.py ... --task_set low` ->
   every clip's `task_level == ['low']` and `clips_per_level` covers 3 tasks;
   `--task_set low,moderate` -> 5 tasks.
4. Contract check: re-running an existing Stage-3 config (which records the OLD
   `task_groups` in its Stage-2 checkpoint) with `task_set: ''` prints **no** new
   `task_groups` note; `TASK_SET=low` makes it fire.
5. Re-derived JSON sanity: `per_group` keys are exactly `{low, moderate, high}`
   and their `n` sums to the scored session count -- proves no session lost its
   label.

## Open questions

1. How to supply the flag during re-derivation: (A) a literal string on the CLI,
   documented once in `code/README.md`; (B) derive it from the constant
   (`$(python -c 'from data.task_groups import SHIPPED_TASK_GROUPS; print(...)')`);
   (C) add `--task_groups_config <yaml>`. Recommend A for the documented form plus
   B in the recorded commands -- the eval runners have no YAML loader, and C adds
   config plumbing for a single string.
2. `notebooks/TirROI_Resp_Pipeline.ipynb` has old-group stdout in its stored cell
   outputs. Recommend leaving it as an execution log and regenerating only when
   the notebook is next executed.
3. Scheduling a `task_set: low` Stage-2 arm is **out of scope**: the path is
   documented and tested, but any arm needs the GPU queue.

## BDD Test Cases

**Scenario 1: Parsing the shipped task groups**
* **Given** the system uses `SHIPPED_TASK_GROUPS`
* **When** the task groups are parsed by `parse_task_groups`
* **Then** Task 5 is assigned to the "low" tier
* **And** Task 1 is assigned to the "moderate" tier
* **And** Task 10 is assigned to the "high" tier
* **And** every task from T1 to T10 is assigned exactly once

**Scenario 2: Evaluating predictions with overridden task groups**
* **Given** an existing prediction dump where Task 5 was originally labeled "high"
* **When** `run_evaluate_session.py` is executed with `--task_groups "low=T2|T3|T5;moderate=T1|T4;high=T6|T7|T8|T9|T10"`
* **Then** the evaluation metrics for Task 5 are aggregated under the "low" tier
* **And** the output JSON records `task_group_source: 'cli'`
* **And** the output JSON records the resolved `task_levels` map

**Scenario 3: Stage-2 and Stage-3 contract validation (Mismatch)**
* **Given** a Stage-2 checkpoint trained with `task_set='low'`
* **When** a Stage-3 finetuning run is launched with `task_set='low,moderate'`
* **Then** `check_stage2_roi_contract` detects the mismatch
* **And** a warning note is printed about the corpus restriction difference

**Scenario 4: Stage-2 and Stage-3 contract validation (No restriction)**
* **Given** a Stage-2 checkpoint trained with `task_set=''` (all tasks)
* **When** a Stage-3 finetuning run is launched with `task_set=''` and a different `task_groups` definition
* **Then** `check_stage2_roi_contract` skips the comparison
* **And** no warning note is printed, as the corpus itself is unchanged

