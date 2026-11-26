# K-fold run checklist — pre-flight, launch, and verdict gates

> **Status: runbook (2026-11-19).** Executable order of operations for a full
> `Stage-2 -> Stage-3 -> test -> pooled k-fold` run. It defines no intent.
> Authority: `code/notes/spec/` (what it must be) ->
> `code/notes/design/ADR-001-subject-disjoint-split-and-selection.md` and
> `code/notes/design/ADR-009-resp-waveform-evaluation-protocol.md` (intent) ->
> `code/notes/plan/experiments/TirROI_Resp_plan.md` (code placement) ->
> `code/notes/eval/archive/tir_resp/resp_5fold_cv.md` (measured facts).
> Where this file disagrees with any of them, they win.
>
> Baseline: `configs/pretrain/stage2_hpc_tir_roi_resp.yaml` (96 px) +
> `configs/finetune/resp_tir_roi_hpc.yaml`, `K = 5`, roles 70/10/20 read from
> `<exp_root>/fold<k>/partition.json`, `val_select = fixed_epoch`. There is no
> Option A/B and no fold-count flag (ADR-001 D2/D5). Copy-paste commands assume
> the working directory is `code/` unless stated otherwise.

---

## 0. Freeze the experiment (before a GPU is spent)

- [ ] State the claim in one sentence plus the headline metric. For respiration
      the headline is `pearson_band` (0.1-0.6 Hz) and `mae_bpm` /
      `within_10pct`. **Amplitude is NOT a metric** — `amp_ratio` /
      `amp_rel_err` / `mae_affine` are oracle-calibrated diagnostics, excluded
      from `overall` and from the console summary by
      `runners/run_evaluate_kfold.py`.
- [ ] Record `git rev-parse HEAD`, the two config paths, `fold`, the partition
      digest, and the fold -> subject map you EXPECT (read the record, never a
      re-derivation).
- [ ] There is **one** protocol (ADR-001): no Option A/B choice. `val_select`
      defaults to `fixed_epoch`; `best_val` requires the Criterion-5 study.
- [ ] `K = 5` is a module constant (`protocol/constants.py`), not a flag. Folds
      come from `protocol/partition.py::split_universe` over the unique usable
      subject ids, so a subject gaining or losing clips cannot shift a boundary.
      **Partition (who is held out)** is over the subjects that own at least one
      session with both IRFeatures and Resp_Volts: **139** on the current corpus
      (M049 has none), i.e. `139 = 5*27 + 4` -> **28/28/28/28/27**.
      **Partition rule is pinned in the config pair** (2026-11-01):
      `split_seed: 42` + `split_stratify: sex` -> shuffle within each sex
      stratum, then interleave proportionally, so every fold is roughly
      **16-17 F / 11-12 M** instead of a contiguous sex block
      (`F001-F028` / `F029-F056` / `F057-M002` / `M003-M030` / `M031-M058` was
      the OLD rule, and it makes fold == cohort). Stage 2 and Stage 3 must carry
      the SAME pair, otherwise fold k's encoder saw fold k's test subjects —
      the contract guard compares `fold`/`n_folds` only.
      **Scored set** (after clip drops) is smaller — 138 subjects / 1260 sessions
      / 21 231 clips, which is why `code/notes/eval/archive/tir_resp/resp_5fold_cv.md`
      shows 27/28/28/28/27 per fold. That file measured the LEGACY contiguous
      partition (and the 112/0.2 geometry), so a stratified sweep is a different
      experiment: never mix folds across the two rules, and state which one a
      number came from. Re-derive the universe for a changed corpus: adding one
      usable subject re-splits EVERY fold.
- [ ] Pre-register the confound: the partition is stratified (`sex`), so each
      fold is a near-representative sample instead of a sex block — but 14 val
      and ~28 test subjects per fold still leave a per-fold spread that
      describes cohorts as much as model quality. Never claim "fold k's model is
      better".
- [ ] Decide new lineage vs re-run. A new lineage also needs the YAML, a
      `code/tests/` entry, and (only if the architecture changed)
      `python scripts/project/write_specs.py` — read its warning first, it
      overwrites hand-edited spec text.

## 1. Environment

- [ ] Activate the repo venv. It is **`.env/`, not `.venv/`** (AGENTS.md is
      stale): `source ~/MA-project-KISMED/.env/bin/activate`.
- [ ] Local profile `source scripts/env_local.sh`; HPC profile
      `scripts/env_hpc.sh` (`VENV`, `RAW_DATA_PATH=$WORK_PROJ/test`,
      `OUTPUT_DIR`).
- [ ] `mkdir -p logs` in `code/` — sbatch opens `--output` before the job starts.
- [ ] `unset SQUEUE_USERS` before reading `squeue`; it is exported in the login
      environment and silently hides every other user's jobs.
- [ ] Disk budget on `$WORK_SCRATCH`, never `$HOME` (60 GB quota): ROI cache
      ~95 GB + Stage-2 checkpoints ~18 GB/fold (15 x 1.2 GB at
      `save_ckpt_freq: 10`) + Stage-3 checkpoints + dumps/figures. Probe with
      `df -Pm`; `df -P --output` aborts the whole job under `set -euo pipefail`.

## 2. Code gate (~5 min, catches drift early)

- [ ] `python -m compileall -q .`
- [ ] `python -m pytest -q` -> baseline **657 passed**, ~19 s, CPU. Judge only
      NEW failures.
- [ ] `python -m core.waveform_model` (module form; running the file as a script
      dies on relative imports).
- [ ] Edited a config? `tests/test_config_loading.py` globs every YAML
      automatically and unknown keys are a HARD error at load.

## 3. Data and caches (CPU only — never `acc`)

**Verified on Lichtenberg, 2026-11-01** — both artefacts resolved through the
REAL dataset builders (`build_pretraining_dataset` for Stage 2, `build_dataset`
with `split_role='test'` for Stage 3), i.e. `find_cache` passed and a sample
came back as `tir (3,100,96,96)` + `resp (1,800)`:

| artefact | resolved key | completeness |
|---|---|---|
| `$WORK_SCRATCH/tir_roi_cache` | `tirroi_nose_mouth_pad0.10_square_window-hull_clip8s_src726x480_native_v3_fa9f91bdd5` | 1385 shards + 15 `missing_ir_features` skips, 0 missing, 78.4 GB |
| `$WORK_SCRATCH/resp_cache` | `respgrid_src1000_fs100_sat9.95-1s_flat0.3-5s_run8s_v2_24852055a5` | 1399 shards + 1 `missing_resp_volts` skip, 0 missing, 42 MB |

The two other `resp_cache/` directories (`...flat0.5-5s...`, `...sat9.95-2s_flat0.05-5s...`)
are complete but UNREACHABLE: the loaders pin the DEFAULT validity knobs, so
only the key above can ever be selected. `tir_roi_cache_smoke/` holds a v2
pad0.20 12-shard smoke cache and no sbatch points at it.

- [ ] Corpus count with the shipped filters (`rail_touch_v: 9.9`,
      `min_signal_spread: 0.1`, `clip_stride: 2.0`): a collapse or a jump of
      >5 % means a filter or geometry change slipped in.
- [ ] Re-derive the SUBJECT UNIVERSE the folds partition (it is not the raw
      subject count):

      python -c "from data.tir_resp_dataset import discover_sessions as d; s=d('$RAW_DATA_PATH'); print(len(s), len({x['subject'] for x in s if x['ir_file'] and x['resp_file']}))"

      Discovery keys on the THERMAL VIDEO (`Thermal/<S>/<S>_<T>.<ext>`), then a
      session must also have `IRFeatures/<S>_<T>.txt` and
      `Physiology/<S>/<T>/Resp_Volts.txt`. Measured: 1400 sessions / 140
      subjects -> **1384 / 139**; 15 sessions lack IRFeatures (all 10 `M049`
      tasks, `F016_T2..T4`, `F054_T10`), 1 lacks Resp_Volts (`F082_T1`), and
      `M049` is the single subject that drops out. Clip-level drops (rail,
      flatline, sentinel, too-short) happen AFTER the split and never move a
      boundary. Note the asymmetry: Stage 2 applies `--subjects` AFTER the
      partition (as a train-pool intersection), Stage 3 applies it DURING
      discovery (`TirRoiRespFinetuneDataset`), so a non-empty `--subjects`
      changes Stage 3's folds but not Stage 2's. Both configs ship `''`.
- [ ] Thermal ROI cache exists AND matches the contract key (`roi_landmarks` /
      `roi_padding` / `crop_rule` / `clip_seconds` / `fps` / corpus). A key
      mismatch ABORTS rather than reading wrong pixels:

      python -u runners/run_build_roi_cache.py --status --out_root "$WORK_SCRATCH/tir_roi_cache"
      python -u runners/run_build_roi_cache.py --index  --out_root "$WORK_SCRATCH/tir_roi_cache"
      python -u runners/run_build_roi_cache.py --check  --out_root "$WORK_SCRATCH/tir_roi_cache"

      Build only if absent: `CACHE_ROI_PADDING=0.10 sbatch scripts/hpc/submit_roi_cache.sbatch`.
      Judge completion by `--status` (`os.stat`), NOT by `ls | wc -l` — `/work`
      shows fresh dirents before `stat` resolves them.
- [ ] Resp cache (optional): `runners/run_build_resp_cache.py --init --build`,
      then `--status` / `--check`. When active it forces
      `--min_signal_spread 0 --rail_touch_v 0` and its label space is
      run-mean-removed volts, so it must be ON or OFF in BOTH stages.
- [ ] Stage-1 VideoMAE weights pre-fetched on a LOGIN node (compute nodes are
      offline): `bash scripts/hpc/download_weights.sh` ->
      `models/initial/videomae_base_patch16_224.pth` (~377 MB). The submit script
      hard-errors on an empty dir.
- [ ] `data/` is read-only for agents: no writes into the raw or canonical tree.

## 4. Contract freeze (the #1 silent-invalidation source)

Stage 2 (`stage2_hpc_tir_roi_resp.yaml`) and Stage 3 (`resp_tir_roi_hpc.yaml`)
must agree on:

- [ ] `input_size` (96), `roi_padding` (0.10), `crop_rule` (square),
      `roi_landmarks` (''), `max_sway_px` (40). These are
      `ROI_CONTRACT_KEYS` in `runners/run_waveform.py`: a mismatch ABORTS.
- [ ] Reusing an existing Stage-2 encoder? Its checkpoint must record the SAME
      geometry. The folds 0-4 on disk record `input_size 112`, `roi_padding
      0.2`, no `crop_rule` (the old `minmax`) and `resp_cache None`, so they are
      a DIFFERENT lineage from the current 96 / 0.10 / square config: Stage 3
      aborts on the ROI contract. A fresh sweep is therefore also NOT comparable
      to `code/notes/eval/archive/tir_resp/resp_5fold_cv.md` (+0.307), which measured
      that 112/0.2 lineage with NO respiration cache. Re-evaluating an old fold
      must pass `RESP_CACHE=''`, or the reference is read from a different
      preprocessing than the labels (design C8).
- [ ] `clip_duration` (8.0), `fps` (25.0), `tubelet` (2,16,16),
      `temporal_stride` (2), `sig_kernel` (16), `clip_stride` (2.0),
      `physio_norm` (session). `clip_stride` / `task_set` /
      `min_signal_spread` only WARN — compare them by hand.
- [ ] `task_groups` identical in both files and equal to
      `data/task_groups.py::SHIPPED_TASK_GROUPS` (v2:
      `low=T2,T3,T5 / moderate=T1,T4 / high=T6..T10`); `task_set: ''`.
- [ ] Stage 3 keeps `finetune: ''` so the submit script resolves the PER-FOLD
      encoder. A hardcoded whole-corpus checkpoint only WARNS in
      `check_stage2_fold_contract` — it yields a healthy-looking leaked CV
      number.
- [ ] Stage 2: confirm in `stage2/metrics.jsonl` that the val curve exists and
      that the fold's TEST subjects are absent from the corpus (the role counts
      in `partition.json` are the authority; the builders cannot widen them).
- [ ] Keep `GPUS=4` (global batch 64). GPU count changes the global batch, hence
      the LR, and folds stop being comparable.
- [ ] Leave `--max_sessions` / `--max_clips` / `--max_entries` unset for real
      runs. (The partition is computed over the FULL usable subject list, so caps
      never shift fold membership — they just make the number meaningless.)
- [ ] AMP stays `none` unless measured: `AMP_DTYPE=fp16|bf16` first on ONE fold
      (`FOLDS="0"` through `scripts/hpc/submit_sweep.sh`); resuming with a
      different mode only prints a warning.

## 5. Cheap gates before spending GPU-days

- [ ] Local data smoke: `bash scripts/local/waveform_local.sh` (or
      `inspect_tir_resp.sh`) with `--max_entries 64`. Expect
      `clip=(3,100,96,96)`, `waveform=(800,)`, the cache line, the
      `rail_touch_v`/`min_signal_spread` cleaning line.
- [ ] **1-epoch trap:** `--epochs 1` aborts because
      `cosine_scheduler` asserts `len(schedule) == epochs * niter_per_ep` and
      `warmup_epochs >= epochs`. Pass `--warmup_epochs 0` for a 1-epoch bench.
- [ ] Per-fold Stage-2 -> Stage-3 compatibility, CPU only:

      PYTHONPATH=. python notes/eval/archive/tir_resp/stage3_compat_probe.py \
          --finetune "$WORK_SCRATCH/pretrain/stage2_hpc_tir_roi_resp_fold<k>/checkpoints/checkpoint-0149.pth"

      Exit 0 = the runner's own verdict. Exit 1 = at least one BLOCKER.
      `PYTHONPATH=.` is required (it imports `core.*`).
- [ ] Partition dry run (CPU, seconds) — the record must be written and re-read
      with the same digest:

      python runners/run_partition.py --exp_root /tmp/proto_dry --fold 0 \
          --data_set tir_roi --raw_root "$RAW_DATA_PATH" --dry_run

## 6. Launch order

The sweep runs all four steps of a fold — partition → Stage 2 → Stage 3 → test —
inside ONE allocation, folds SEQUENTIALLY (`scripts/hpc/submit_sweep.sh`). There
is deliberately no `--array`: parallel folds would share `MASTER_PORT` and a
reader could pool a half-written sweep.

1. [ ] **First fold only** (proves the whole chain before spending the rest):

       EXP_ROOT=$WORK_SCRATCH/protocol_v1 FOLDS="0" \
           bash scripts/hpc/submit_sweep.sh

       Then gate on the four artefacts of fold 0 (§7) BEFORE submitting the
       rest. `partition.json` is write-once: a resumed sweep reuses it when the
       digest matches and aborts otherwise.

2. [ ] **Full sweep** (one allocation, ~18-20 h GPU):

       EXP_ROOT=$WORK_SCRATCH/protocol_v1 sbatch scripts/hpc/submit_sweep.sbatch
       # resource override:
       #   sbatch --partition=acc --time=24:00:00 scripts/hpc/submit_sweep.sbatch

       Env knobs: `FOLDS`, `CFG2`, `CFG3`, `RAW_ROOT`, `SPLIT_SEED`,
       `SPLIT_STRATIFY`, `DATA_SET2`, `DATA_SET3`, `PY`.

3. [ ] **Pool** (refuses a dev-smoke record, a partial sweep, mixed seeds or
       mixed partitions):

       python runners/run_evaluate_kfold.py --exp_root "$WORK_SCRATCH/protocol_v1"

       Writes `<EXP_ROOT>/report/kfold.json`. `--allow_partial` is for smokes
       only — never for a number that reaches the thesis.

## 7. Verification gates (run before reading any number)

- [ ] `<EXP_ROOT>/fold<k>/partition.json` agrees across folds on `split_seed`
      and `split_stratify`, while each fold's `digest` differs — five distinct
      test blocks of ONE partition.
- [ ] `<EXP_ROOT>/fold<k>/stage3/test/metrics.json`: `eval_role == 'test'`, not
      `skipped`, `n_clips > 0`, and `partition_digest` equals the fold's record.
      A `skipped` artefact means the fold has nothing to score and cannot enter
      the pooled number.
- [ ] `stage3/test/{preds,targets}.npy` and `entries.json` come from the SAME
      process: `len(entries.json) == preds.shape[0]`. A shard-sized `preds`
      means you are holding rank 0's DDP slice.
- [ ] `stage3/selection.json` exists and every fold records the SAME `rule`
      (the pooler refuses a mix). `fixed_epoch` is the only licensed value until
      Criterion 5 exists.
- [ ] `stage3/metrics.jsonl` shows the val curve at the expected cadence
      (`eval_freq: 5`), and Stage 2 is not flat at the constant-predictor level
      (`mse_resp` ~ the target variance ⇒ the head learned nothing).
- [ ] Partition audit: the union of the five test roles is the whole usable
      corpus and they are pairwise disjoint — the pooler asserts both.
- [ ] Fold k's own `session_metrics.json` reproduces the pooled per-fold row
      (the two come from different scoring paths; a mismatch means one dump is
      stale).
- [ ] The `template_pearson_band` single-tone baseline is printed NEXT TO the
      model number. Last measured: model +0.307 vs template +0.535 — the model
      loses to the baseline, so any claim must be phrased against it.
- [ ] Report the per-fold table AND the pooled bootstrap 95 % CI over subjects,
      plus the cohort confound from step 0.

## 8. Traps that silently invalidate a K-fold number

| Trap | Symptom | Guard |
|---|---|---|
| Whole-corpus Stage-2 encoder used | Plausible number, slightly better | `finetune: ''` + the sweep's `partition_digest` pairing |
| Stage-3 on another fold's ckpt | Loads perfectly, wrong result | `check_stage2_fold_contract` (digest) |
| `--eval_only` clobbered the manifest | `roles.train = []` | backup + restore (step 6.3/6.4) |
| Rank-0 shard scored as the dump | `preds` = `entries`/4 | row-count gate (step 7) |
| Two folds sharing one `OUT_DIR` | Fold k resumes fold j | driver pins `OUT_DIR` per fold |
| `clip_stride` / `input_size` drift | Different windows scored | warn-only → compare configs |
| `RESP_CACHE` on in one stage only | Different label space | same switch both stages |
| Mixed Option A/B or `n_folds` | Folds not comparable | freeze in step 0 |
| `GPUS` changed mid-sweep | Global batch/LR changed | pin `GPUS=4` |
