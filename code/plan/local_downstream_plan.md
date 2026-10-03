# Local downstream plan — Stage-3 waveform reconstruction (BP and RESP)

> **Status: APPROVED (2026-09-23).** Authoritative setup record for the **local 64 px
> downstream track**: two Stage-3 fine-tuning runs, one per target waveform (`bp`,
> `resp`), executed on the local dev GPU over the 11 canonical sessions.
>
> **Relationship to the other docs.** `simplifiedPlan.md` stays authoritative for the
> **thesis-scale 224 px lineage** (clip 8.0 s, `temporal_stride 2`, `sig_kernel 16`,
> `input_size 224`, `seq_len 800`). This file covers the **separate 64 px lineage**
> that the existing Stage-2 checkpoint belongs to; it supersedes nothing in
> `simplifiedPlan.md` / `ImplementationPlan.md`. The two lineages are **not
> checkpoint-compatible** (`positions.<s>.pos_embed` shape), so nothing here may be
> silently mixed with the 224 px configs.

---

## 0. Scope decisions (2026-09-23)

| # | decision | value |
|---|---|---|
| 1 | geometry | **local 64 px track** — mirror the existing Stage-2 checkpoint exactly |
| 2 | initialization | **pretrained-init only** → exactly **2 runs** (`bp`, `resp`) |
| 3 | code scope | configs + launcher **plus the cheap-correctness set**: Stage-2 `heads.<target>` → `waveform_head` transfer, subject-disjoint split, LR schedule + `--resume` in `run_waveform.py` |
| 4 | evaluation | per-clip Tier-1/2 **and** session-level assembly + Tier-1/2/3 (`evaluation/assemble.py`, new `runners/run_evaluate_session.py`, Tier-3 ≥ 30 s guard, `neurokit2` install) |

**Explicitly out of scope** (the 224 px thesis-scale track): `simplifiedPlan.md` §2/§10
geometry (224 px, clip 8 s, stride 2, `sig_kernel` 16, `seq_len` 800) and §11 items
1, 2, 3, 4, 5, 6, 7, 9, 10 (npy frame store, SDPA, query-based decoder, mask-ratio 1.0
and per-step ratio, 1-D span masking, per-session z-score target, zero-init modality
embeddings + weight-decay groups, xavier conv init, Stage-3 masking mixture). Also out
of scope: HPC submission wiring (`env_hpc.sh` placeholders; the cluster tree has no
`2D+3D/` RGB folder that `prepare_bp4d.py` requires), the TIR stream and the EDA
branch.

---

## 1. Why this geometry (verified facts)

**Source of truth:** `configs/pretrain/stage2_local_pretrained.yaml` — `project_multimae_base`
(768-d / 12-layer / 12-head), `streams: rgb,bp,resp`, `tir_channels: 3`,
`tubelet: 2,16,16`, `clip_duration: 4.0`, `fps: 25.0`, `temporal_stride: 1`,
`input_size: 64`, `fs: 100.0`, `sig_kernel: 8`, `pos_init: sincos3d`, `epochs: 40`,
`batch_size: 2`, `max_sessions: 11`, `max_entries: 32`. `stage2_local_scratch.yaml` is
its byte-identical twin except `pretrained_encoder` / `inflate_rgb_patch` /
`output_dir`.

**Checkpoint to reuse (both branches):**
`output/pretrain/stage2_local_pretrained/checkpoints/checkpoint-0039.pth`
(newest of `0000/0010/0020/0030/0039`; `latest_checkpoint.txt` points at it; ≈1.23 GB,
base geometry).

**Resulting Stage-3 geometry:** `num_frames = 100` → `grid_t = 50`, `grid = 4×4`,
`n_visual = 800`, `output_len = 400`, `samples_per_token = 800/50 = 8 == sig_kernel`.
`sec_per_visual_token = tubelet_t·temporal_stride/fps = 2/25 = 80 ms` and
`sec_per_signal_token = sig_kernel/fs = 8/100 = 80 ms`, `grid_t == n_signal == 50`,
`100 % 2 == 0`, `400 % 8 == 0` — i.e. every condition of
`MultiModalMAE._check_space_time_alignment` passes.

**Why not 224 px (yet).** Three independent blockers, all verified:

1. `configs/finetune/{bp,resp,eda}.yaml` point `finetune:` at
   `../output/pretrain/stage2_multimodal/checkpoints/checkpoint-0799.pth`, which does
   not exist (no `output/pretrain/stage2_multimodal/` at all). Because the spec looks
   like a path, `models/pretrained.py` raises `FileNotFoundError` before training.
2. Their `input_size: 224` does not match the only local checkpoints (`input_size: 64`)
   → `positions.*.pos_embed` shape mismatch → `load_stage2_encoder` raises
   (`core/waveform_model.py`, shape guard near :365-366, raise :376-382).
3. Even with a matching checkpoint, `MultiModalWaveformRegressor.forward` tokenizes the
   **full** clip and `core/blocks.py::Attention` materialises the score matrix
   (`:117`), so 9 800 tokens costs ~4.6 GB per sample per decoder block in fp32 and the
   encoder path ~0.5 GB/batch-16 per block — unusable at `batch_size: 16` and marginal
   at batch 1 without SDPA. That is `simplifiedPlan.md` §11 item 2.

**`resp` needs no extra Stage-2 work.** The local pair was trained with
`streams: rgb,bp,resp`, so `heads.resp` exists next to `heads.bp`; the "there is NO
Stage-2 encoder for the RESP target" header in `configs/finetune/resp.yaml` is stale.

**What this track is for.** With 11 sessions / 4 subjects and a 40-epoch,
32-clip, batch-2 encoder, these two runs are **engineering validation** of the
downstream path (head transfer, masking-free sensor-failure protocol, joint
time+Pearson+MR-STFT loss, subject-disjoint split, session assembly, Tiered
evaluation) — *not* thesis results.

---

## 2. Phase 0 — configs and launcher — **DONE 2026-09-23**

### 2.1 New configs

`configs/finetune/bp_local.yaml` and `configs/finetune/resp_local.yaml`, cloned from
the Section 1 geometry. Per-branch differences are **only** `target`, `eval_band` and
`output_dir`; everything else must stay identical or the two branches stop being
comparable.

| key | value | note |
|---|---|---|
| `target` | `bp` / `resp` | the only branch selector |
| `model` | `project_multimae_base` | 768/12/12, same as Stage 2 |
| `streams` | `rgb` | sensor-failure protocol: visual only |
| `use_tir` | `false` | TIR adapter untrained / out of scope |
| `finetune` | `../output/pretrain/stage2_local_pretrained/checkpoints/checkpoint-0039.pth` | the Stage-2 encoder |
| `head_hidden` | `0` | single `Linear` head → enables the shape-exact head transfer |
| `clip_duration` | `4.0` | mirrors Stage 2 |
| `fps` | `25.0` | source rate (the dataset decimates, not the config) |
| `tubelet` | `2,16,16` | |
| `temporal_stride` | `1` | |
| `input_size` | `64` | **must** equal the Stage-2 value |
| `sig_kernel` | `8` | 2·1/25 == 8/100 |
| `seq_len` | `0` | → 400 samples |
| `fs` | `100` | |
| `signal_norm` | `zscore` | raw mmHg has mean ≈101 / std ≈11 |
| `data_set` / `data_path` | `bp4d+` / `../data/processed/bp4d_canonical` | |
| `train_ratio` | `0.8` | subject-disjoint (Phase 1.2) |
| `split_by` | `subject` | **new key** — inert until Phase 1.2 lands |
| `clip_stride` | `1.0` | overlapping windows, ~4× more clips on a small corpus |
| `alpha` / `beta` / `gamma` | `1.0` / `1.0` / `1.0` | L1 + (−Pearson) + MR-STFT |
| `fft_sizes` | `64,128,256` | 1.56 / 0.78 / 0.39 Hz at 100 Hz |
| `eval_band` | `1.0,2.5` / `0.16,0.4` | cardiac vs breathing |
| `epochs` | `40` | matches the local Stage-2 scale |
| `batch_size` | `4` (fallback 2) | see §6 item 1 |
| `update_freq` / `save_ckpt_freq` / `eval_freq` | `1` / `10` / `1` | |
| `opt` / `lr` / `min_lr` | `adamw` / `1e-4` / `1e-6` | |
| `warmup_epochs` | `2` | **dead until Phase 1.3 lands** |
| `weight_decay` / `clip_grad` | `0.05` / `0.0` | `0.0` = **no clipping** — but see §9 "Phase 3" for the bug that made this value zero every gradient |
| `num_workers` | `4` | |
| `output_dir` | `../output/finetune/bp_local` / `resp_local` | mind the `../` |
| `log_wandb` / `wandb_project` | `false` / `thesis-project` | wandb flags are unwired |

> Caveat: `parse_args_with_config` feeds YAML through `parser.set_defaults(**cfg)`, so
> a YAML key that has no matching argparse option is **silently absorbed and ignored**.
> `split_by` is read from `args` via `getattr` by the dataset builders, which is why it
> works from YAML but not from the command line. (`clip_stride`, `train_ratio`,
> `roi_padding` and `val_subject` DID get real flags in the 2026-09-26 TIR-ROI work,
> so they work from both.)
> `train_mask_ratios` / `eval_mask_ratio` would be silently ignored — do not add them
> (Stage-3 masking is out of scope).

### 2.2 Repairs to the existing templates (no behaviour change for this track)

- `configs/finetune/resp.yaml` and `eda.yaml`: `output_dir` is bare
  (`output/finetune/resp`) and therefore resolves to `code/output/...` instead of
  `<repo>/output/...`; make it `../output/finetune/...`.
- `configs/finetune/resp.yaml`: drop the stale "no Stage-2 encoder for RESP" header —
  the local checkpoint carries `heads.resp`.
- `configs/finetune/{bp,resp,eda}.yaml`: add a header note that `finetune:` points at
  the intended (not yet produced) HPC 224 px artifact and that these are the
  thesis-scale templates, distinct from the `*_local.yaml` pair.

### 2.3 Launcher

`code/scripts/local/waveform_local.sh`, in the style of `au_smoke.sh` /
`inspect_smoke.sh`: source `scripts/env_local.sh`, `TARGET=${TARGET:-bp}`, then run
`runners/run_waveform.py -c configs/finetune/${TARGET}_local.yaml "$@"`.

Optional: add two `.vscode/launch.json` entries for the two runs, and fix the stale
`configs/pretrain/stage2_local.yaml` reference already in that file (renamed to
`stage2_local_scratch.yaml`).

---

## 3. Phase 1 — code fixes — **DONE 2026-09-23** (must land before the runs)

### 3.1 Stage-2 head → Stage-3 head transfer

`core/waveform_model.py::load_stage2_encoder` (def :301) copies only keys starting with
`enc_blocks.` / `enc_norm.` / `adapters.` / `positions.` (whitelist :359-361), so
`heads.<signal>` is skipped and `waveform_head` always starts random.
At this geometry the shapes are **exactly** equal: Stage-2 `heads.bp` is
`Linear(768, sig_kernel=8)` and Stage-3 `waveform_head` is
`Linear(768, samples_per_token=8)`.

Change: thread a `target` argument into the loader (call site
`runners/run_waveform.py:173`); when `head_hidden == 0`, `samples_per_token ==
sig_kernel`, and the state carries `heads.<target>.weight` / `.bias`, copy them into
`waveform_head.*`, count them in the loaded tally and log it. For the `head_hidden > 0`
MLP case (`:149-152`) copy only what has a counterpart and say so. Also add the missing
`samples_per_token == sig_kernel` assertion in `build_waveform_model` (:253-298).

### 3.2 Subject-disjoint split

`data/paired_dataset.py` splits sessions with `sessions[:n_train]` / `sessions[n_train:]`
(:211-217), i.e. **per session**, which lets one subject straddle train and val — the
Stage-3 protocol requires subject-disjoint (see the plan record in
`ImplementationPlan.md` §4).

Change: add `split_by: str = 'session'` to `PairedSessionDataset.__init__`
(:140-155) and group at the split block with a module-level `_subject_of(session)`
helper (precedent: `data/au_dataset.py:65-67`; do **not** import `au_dataset`, it pulls
heavier deps). Wire the value from args in `data/datasets.py`
(`getattr(args, 'split_by', 'session')`). With 4 subjects and `train_ratio 0.8`:
**train = F001-F003 (9 sessions), val = F004 (2 sessions)** — publish that val set in
the run log/README, because a single-subject val set is a real limitation of this
corpus.

### 3.3 LR schedule and `--resume` in `run_waveform.py`

Today `runners/run_waveform.py::main` (:223-235) builds no scheduler: `--warmup_epochs`,
`--min_lr` and `--resume` are parsed but dead, and the effective LR is a constant
`args.lr`. A crashed 40-epoch run cannot be resumed.

Change: build `steps_per_epoch` and `cosine_scheduler(args.lr, args.min_lr, args.epochs,
steps_per_epoch, warmup_epochs=args.warmup_epochs)` following
`runners/run_pretrain.py:227-238`; pass `lr_schedule_values` into the training call;
then `auto_resume_model(args, model_without_ddp, optimizer, loss_scaler)` and start the
loop at `args.start_epoch`, following `runners/run_finetune.py:113-117`.
`engines/finetune.py::train_one_epoch` (:16-24) needs a new `start_steps=None` parameter
with `sched_idx = step if start_steps is None else start_steps + step`, mirroring
`engines/pretrain.py:40-41`; the default keeps `run_finetune.py` bit-identical.
Order note: `--finetune` loads the Stage-2 encoder first, then the resume checkpoint
overwrites it — correct, but worth a log line.

---

## 4. Phase 2 — session-level evaluation

### 4.1 `evaluation/assemble.py` (new)

`assemble_session(preds, entries, fs, clip_duration, window='hann')` — place each
predicted window at its start offset on the session timeline, accumulate
window·prediction and accumulate window, then divide by the accumulated weight
(Hann overlap-add with weight-sum normalisation; never crop `D - stride` per clip).

**Offset convention — correction to `simplifiedPlan.md` §9:** the 1-D target is *not*
decimated by `temporal_stride` (only the video is), so a clip that starts at `t_start`
occupies signal samples `round(t_start·fs) … + seq_len`. Derive offsets with the same
helper the dataset uses (`data/alignment.py::sample_indices` :26-30 /
`slice_1d` :33-48) so the assembly can never desync from training.

### 4.2 Prediction dump

Add `--save_preds` to `runners/run_waveform.py`, writing `preds.npy`, `targets.npy` and
`entries.json` (`{session, t_start}` in loader order) for the val loader. The val loader
is `shuffle=False, drop_last=False` (:206-207), so row *i* ↔ `entries[i]`; assert
`len(preds) == len(entries)` and document that the dump is exact only for a
single-process run (a DDP sampler would shard it).

### 4.3 `runners/run_evaluate_session.py` (new)

Load the dump, assemble per session, build the reference from that session's
`signals.csv` (`meta['signals_file']`), and report two clearly labelled levels:
(i) **per-clip macro** Tier-1/2 (the training/validation signal) and (ii)
**session-level post-stitch** Tier-1/2 + Tier-3 on the assembled waveform.
Write JSON in the same shape as `runners/run_evaluate.py` (:88-90).

Scale handling: with per-clip `signal_norm: zscore`, each prediction is defined only up
to a per-clip affine map, so the assembled trace is a patchwork of scales. Report
affine-invariant quantities session-level (Pearson, PSD shape, RR-interval metrics) as
primary, and optionally apply one per-session LSQ affine calibration before quoting
MAE/RMSE (see §6 item 2).

### 4.4 Tier-3 guard

`evaluation/clinical.py::extract_hrv_metrics` (def :14) silently returns an all-NaN dict
below `2*fs` samples (:38). Raise the guard to **30 s** with a warning, compute metrics
on 30-60 s sub-windows and aggregate (median/IQR), and add `n_rr` (detected RR-interval
count) plus the fraction of sub-windows where peak detection succeeded to the returned
dict. `neurokit2` is **not installed** in `.venv`; install it before Tier-3 can run.

**Expectation setting:** the local sessions are 9.2-64.6 s (RGB duration), so Tier-3
will be NaN for most of them even after this change. Locally Tier-3 is a smoke check;
it becomes meaningful only on the 224 px / full-corpus track.

---

## 5. Phase 3 — running the two branches

```bash
# from the repo root, after Phase 0-1 (smoke first, one branch at a time)
cd code
source scripts/env_local.sh

# 1) smoke each branch (random-init then real checkpoint)
TARGET=bp   scripts/local/waveform_local.sh --epochs 1 --warmup_epochs 0 \
    --batch_size 2 --max_sessions 2 --max_entries 2
TARGET=resp scripts/local/waveform_local.sh --epochs 1 --warmup_epochs 0 \
    --batch_size 2 --max_sessions 2 --max_entries 2

# 2) the two headline local runs (sequential: one GPU)
TARGET=bp   scripts/local/waveform_local.sh
TARGET=resp scripts/local/waveform_local.sh

# 3) session-level evaluation of each branch
TARGET=bp   scripts/local/waveform_local.sh --resume --save_preds --epochs <best>
TARGET=resp scripts/local/waveform_local.sh --resume --save_preds --epochs <best>
.venv/bin/python runners/run_evaluate_session.py \
    --pred_dir ../output/finetune/bp_local  --waveform bp  --fs 100 --tier 1,2,3 \
    --out ../output/finetune/bp_local/session_metrics.json
```

Expected artifacts per branch: `<output_dir>/checkpoints/checkpoint-NNNN.pth` +
`latest_checkpoint.txt`, `<output_dir>/best.pth` (best val Pearson), the prediction dump,
and the session-level JSON.

Smoke acceptance: a finite loss, a `[epoch e] <target>: {...}` Tier-1/2 line, and a
`[stage3] loaded …` line whose count includes the transferred head tensor.

---

## 6. Verification

1. `yaml.safe_load` all 12 configs; `python -m py_compile` every touched module
   (YAML is not covered by editor diagnostics — see the trap recorded in repo memory).
2. Build both configs through `build_waveform_model(args)` and assert `n_visual == 800`,
   `output_len == 400`, `samples_per_token == 8`, `grid_t == 50`.
3. Load the real checkpoint with `target='bp'`: assert `waveform_head.weight` is
   elementwise equal to the checkpoint's `heads.bp.weight`, and that `target='resp'`
   selects `heads.resp` (the two tensors differ); print loaded / skipped / mismatch
   counts.
4. Split check: build train and val datasets with `split_by: subject` and assert no
   subject appears on both sides (expect F001-F003 vs F004).
5. 1-epoch smoke per branch (random init, then real checkpoint): finite loss + Tier-1/2
   line; `--target` selects the right CSV column.
6. LR/resume: the printed `lr` now changes across warmup/cosine (it is constant 1e-4
   today); re-running with `--resume` starts at checkpoint epoch + 1.
7. Assembly unit test: two synthetic overlapping windows containing a known ramp →
   the assembled trace reproduces the ramp, the weight-sum normalisation holds, and the
   offsets equal `alignment.sample_indices` values.
8. Tier-3: 10 s synthetic → NaN + warning; 40 s synthetic (after `neurokit2` install) →
   metrics + `n_rr` + peak-success fraction.
9. End-to-end: each branch's session JSON contains both the per-clip and the
   session-level blocks.

---

## 7. Relevant files

| file | role / what changes |
|---|---|
| `configs/finetune/bp_local.yaml`, `resp_local.yaml` | **new** — the two runs |
| `configs/finetune/resp_tir_roi_local.yaml` | **new (2026-09-26)** — the TIR-ROI branch, see §8 |
| `configs/finetune/bp.yaml`, `resp.yaml`, `eda.yaml` | template repairs only (§2.2) |
| `data/tir_resp_dataset.py` | **new (2026-09-26)** — `TirRoiRespFinetuneDataset` (Stage-3 view), see §8 |

---

## 8. Third local branch — TIR-ROI -> RESP (added 2026-09-26)

The two branches above take **RGB** as the visual input. The 2026-09-26 Stage-2
run changed the input modality: `configs/pretrain/stage2_local_tir_roi_resp.yaml`
pre-trained on the **thermal ROI crop** (raw tree, 40 sessions, 8 s clips,
`temporal_stride 2`, `sig_kernel 16`). Its downstream branch is a THIRD local
run, not a variant of the `resp_local` one:

| | `resp_local.yaml` (RGB) | `resp_tir_roi_local.yaml` (new) |
|---|---|---|
| input | RGB jpg sequence, full frame | **thermal ROI crop** (`BP4DPlusTIRRespDataset`) |
| `data_set` / `data_path` | `bp4d+` / `data/processed/bp4d_canonical` | `tir_roi_resp` / `data/raw/BP4D` |
| checkpoint | `stage2_local_pretrained` (**missing on disk**) | `stage2_local_tir_roi_resp/checkpoint-0039.pth` |
| geometry | 4 s / stride 1 / `sig_kernel` 8 / `seq_len` 400 | 8 s / stride 2 / `sig_kernel` 16 / `seq_len` 800 |
| corpus | 11 sessions, 320/66 clips | 40 sessions, 588/168 clips (`clip_stride` 2.0) |
| `clip_grad` | `0.0` (off) | **`5.0`** (on -- this branch carries the `gamma=1.0` MR-STFT term) |
| loss / optimizer | identical (`alpha=beta=gamma=1.0`, `fft_sizes 64,128,256`) | identical |

The geometry row is NOT interchangeable: only the 8 s / stride-2 / `sig_kernel`-16
combination makes the weight transfer exact (150 tensors, and `heads.resp` ->
`waveform_head` verbatim). Launch it with
`python -u runners/run_waveform.py -c configs/finetune/resp_tir_roi_local.yaml`;
for the leave-one-subject-out sweep add `--val_subject F001|F002|F003|F004` and a
per-fold `--output_dir`. Full record: `TirROI_Resp_plan.md` §10.
| `configs/pretrain/stage2_local_pretrained.yaml` | geometry source of truth (`stage2_local_scratch.yaml` = twin) |
| `scripts/local/waveform_local.sh` | **new** launcher (`TARGET` switch) |
| `core/waveform_model.py` | `MultiModalWaveformRegressor` (:75-155), `build_waveform_model` (:253-298), `load_stage2_encoder` (:301-390) → head transfer |
| `data/paired_dataset.py` | ctors (:140-155, :389-402), split (:211-217), `entries` (:219-275), `_signal_at` (:291-309) → `split_by` |
| `data/datasets.py` | arg plumbing for `split_by` |
| `runners/run_waveform.py` | main (:140-252) → scheduler, resume, `--save_preds` |
| `runners/run_pretrain.py` (:227-238), `runners/run_finetune.py` (:113-117) | patterns to mirror (schedule, resume) |
| `engines/finetune.py` (:16-24, unpack :35) | add `start_steps` |
| `utils/checkpoint.py` (:15-67), `utils/lr_sched.py::cosine_scheduler` (:22) | reuse as-is |
| `evaluation/assemble.py` | **new** session assembly |
| `evaluation/clinical.py` (:14, guard :38) | Tier-3 ≥ 30 s guard, `n_rr`, success fraction |
| `runners/run_evaluate_session.py` | **new** session-level evaluation runner |
| `runners/run_evaluate.py` (:37-90) | unchanged; JSON shape reused |
| `runners/run_inspect_data.py` (:119-126, :257-281) | reference pattern for walking `ds.entries` |
| `data/alignment.py` | `sample_indices` (:26-30), `slice_1d` (:33-48), `plan_clip` (:106-135) |

---

## 8. Open items and risks

1. **Throughput vs batch size.** UPDATED from the real 2026-09-23 run: batch 4 with
   `num_workers: 4` gives **~0.78 s/iteration** and **62 s per epoch** (80 steps plus the
   validation pass), i.e. ~45-60 min for the full 40-epoch BP run on the local GPU. The
   earlier ~4 s/iteration figure came from a 0-worker smoke and was data-loader bound, so
   the batch/epoch trade-off (options A/B/C) is no longer a real constraint at this scale.
2. **Stitched-waveform scale convention.** Option A *(default)*: report
   affine-invariant session metrics (Pearson / PSD shape / RR) and keep MAE/RMSE
   per-clip. Option B: chain-wise affine calibration between overlapping clips.
   Option C: switch `signal_norm` to `none` for the final epochs so absolute amplitude
   is learned — needs a target-scale decision first.
3. **From-scratch control.** The init A/B ("is Stage 2 necessary" at the downstream
   level) needs a third/fourth run off `stage2_local_scratch/checkpoints/checkpoint-0039.pth`.
   Deferred until the two headline runs are green.
4. **Val set is one subject (F004, 2 sessions).** Best-Pearson checkpoint selection on
   that set is noisy; state it as a limitation, and prefer the final-epoch checkpoint if
   the val curve is unstable.
5. **`clip_stride: 1.0`** raises the clip count ~4× on this small corpus; if epochs
   become too slow, either drop it or reduce `epochs`.
6. **Known code caveats not addressed here:** Tier-2 `nperseg=min(256, len)` gives
   ~0.39 Hz resolution, so the RESP band (0.16-0.4 Hz) is near-degenerate at 400-sample
   windows; `evaluate_waveforms` uses `batch[:2]` and returns only macro aggregates;
   `--log_wandb` is unwired in `run_waveform.py`.

---

## 9. Implementation status log — 2026-09-23

**Phase 0 — DONE**

* `configs/finetune/bp_local.yaml`, `configs/finetune/resp_local.yaml` — twin configs; the
  only intended differences are `target`, `eval_band` and `output_dir`.
* `scripts/local/waveform_local.sh` — `TARGET=bp|resp`, resolves the repo `.venv`
  automatically (`$PYTHON` wins), passes `"$@"` through to `runners/run_waveform.py`.

**Phase 1 — DONE**

* `core/waveform_model.py`:
  * `load_stage2_encoder(model, path, target=None)` now transfers `heads.<target>` into
    `waveform_head.*` when the shapes line up (`samples_per_token == sig_kernel`,
    `head_hidden == 0`); otherwise it says so explicitly instead of silently leaving the
    head random.
  * `build_waveform_model` raises when `samples_per_token != sig_kernel`. Nothing checked
    this before, so a right `output_len` with a wrong `sig_kernel` trained happily while
    prediction and target drifted apart.
* `data/paired_dataset.py`: new `split_by` (`'session'` | `'subject'`), implemented with a
  local `_subject_of` (duplicated from `data.au_dataset` on purpose — that import pulls
  pandas into the Stage-3 path). An empty split now raises instead of silently training on
  nothing. `data/paired_dataset.py::build_paired_dataset` plumbs the value from `args`, and
  `run_waveform` gained a real `--split_by` flag so a smoke run can override the config.
* `runners/run_waveform.py`: step-level warmup + cosine schedule (previously
  `--warmup_epochs` / `--min_lr` were parsed but dead and the LR stayed constant at
  `args.lr`), `auto_resume_model(...)`, `--split_by`, a guard for an already-complete run,
  and `--target` threaded into the loader.
* `engines/finetune.py::train_one_epoch(..., start_steps=None)`: the LR/WD schedule index is
  now global (`start_steps + step`) instead of restarting every epoch. `run_finetune.py` is
  unaffected (default `None` → previous behaviour).

**Two bugs found while wiring this (both fixed)**

1. `utils/checkpoint.py::load_model` called `torch.load(...)` without `weights_only=False`.
   Under torch >= 2.6 the default is `True`, so resume died with a `_pickle.UnpicklingError`
   (the checkpoint stores the `args` namespace, which pickles numpy scalars). Resume could
   never have worked on this environment.
2. The Stage-2 checkpoint on disk **still uses the pre-rename head key** — `heads.bvp.*`
   (its stored `args.streams` is `rgb,bvp,resp`, and `adapters.bvp.*` / `positions.bvp.*`
   likewise). `load_stage2_encoder` therefore accepts the legacy name for `bp` as an
   explicit, reported fallback rather than starting the head random.

**Verified locally (2026-09-23, `.venv`, torch 2.11)**

* geometry from both configs: `grid_t` 50, `n_visual` 800, `output_len` 400,
  `samples_per_token` 8 == `sig_kernel` 8.
* `load_stage2_encoder(model, checkpoint-0039.pth, target='bp')`: 150 encoder tensors
  loaded, 0 shape-mismatched, `head transferred: waveform_head.weight,
  waveform_head.bias (from heads.bvp)`; both tensors elementwise equal to the checkpoint's
  `heads.bvp.*`, and `positions.rgb.pos_embed` equal as well.
* subject-disjoint split: train = F001-F003 (320 clips / 9 sessions), val = F004
  (66 clips / 2 sessions).
* `TARGET=bp scripts/local/waveform_local.sh --epochs 1 --warmup_epochs 0 --batch_size 2
  --num_workers 0 --max_entries 2 --output_dir /tmp/bp_smoke` → loss 4.9399, Tier-1
  (`mae`/`rmse`/`pearson`) and Tier-2 (`psd_mae`/`dominant_freq_error_hz`) printed for `bp`,
  `checkpoints/checkpoint-0000.pth` + `best.pth` written.

**Phase 3 — first run is VOID: `clip_grad: 0.0` zeroed every gradient (fixed)**

The first 40-epoch `RGB → bp` run completed and *looked* healthy, but learned nothing:

| signal | epoch 0 | epoch 39 |
| --- | --- | --- |
| train loss | 4.8836 | 4.8789 |
| val `mae` | — | 0.9307 (frozen) |
| val `pearson` | 0.0004 | ~0.0002 |
| `psd_mae` | 0.1678 | 0.2252 |
| `dominant_freq_error_hz` | 1.10 | 1.10 (constant) |

Root cause, proven by direct experiment: `utils/native_scaler.py`
`NativeScalerWithGradNormCount.__call__` tested `if clip_grad is not None:` and then handed
`clip_grad` straight to `torch.nn.utils.clip_grad_norm_`. With the config's `0.0` that call
scales **every** gradient by `clip_coef = 0.0 / (total_norm + eps) = 0`:

```python
clip_grad_norm_(max_norm=0.0) -> total_norm 1.732 ; grad now [0.0, 0.0, 0.0]
clip_grad_norm_(max_norm=1.0) -> grad now [0.577, 0.577, 0.577]
```

The returned `grad_norm` still looked plausible because it is measured *before* the scaling,
so the log gave no hint. Two collateral findings:

* This is **pre-existing and repo-wide**, not a Stage-3 issue. The AMP branches
  `engines/pretrain.py:71` and `engines/finetune.py:72` both pass `clip_grad=max_norm`, and
  `runners/run_pretrain.py` defaults `--clip_grad` to `0.0` — so every AMP Stage-2/Stage-3 run
  at the default was frozen too. This is the most likely explanation for the historical
  "Stage-2 loss trends down but slowly" observation (3.371 → 3.336 over 20 epochs).
* Only the AMP branches are affected; the non-AMP branch in the same class guards
  `max_norm > 0`, which is why a CPU/no-scaling run *did* learn.

Fix (central, so every caller is corrected at once):

```python
if clip_grad is not None and clip_grad > 0:   # 0.0 now genuinely means "off"
```

Acceptance test — `--epochs 5 --lr 1e-3 --warmup_epochs 0 --max_entries 16` on 16 clips:
val `mae` 1.652 → 1.781 → 1.329 → 1.143 → **1.037**, versus the frozen 0.9307 above. The
optimiser now moves the weights.

The void run's directory was moved to `output/finetune/bp_local_BROKEN_clipgrad0/` (its
checkpoints are bit-identical copies of the Stage-2 encoder, since nothing was ever
updated), and the real run was relaunched into a fresh `output/finetune/bp_local/`.

> Launching trap: `… | tee ../output/finetune/bp_local/train.log` fails with "No such file
> or directory" when that directory does not exist yet. `tee` exits, the pipe closes, and
> the training process dies on `SIGPIPE` **with no visible traceback** (its stderr goes to
> the same dead pipe). Always `mkdir -p` the output directory before relaunching.

**Phase 3 (continued) — the corrected run trains, but does not learn the task**

The relaunched 40-epoch run finished with every expected artefact
(`metrics.jsonl`, `training_curves.png`, `metrics_final.json`, `predictions_final.png`,
`preds.npy` / `targets.npy` / `entries.json`, `checkpoints/`, `best.pth`, plus the
session-level `session_metrics.json` + `figures/session_*.png`). The optimisation is now
real — but the task is not solved.

| quantity | epoch 0 | epoch 39 |
| --- | --- | --- |
| train loss (noisy, 3.7-6.8 range) | 4.36 | 3.77 |
| val `pearson` | 0.0006 | 0.0118 (best **0.0203** at epoch 25) |
| val `mae` | 0.916 | 0.962 |
| `psd_mae` | 0.314 | 0.101 (plateau after epoch ~15) |
| `dominant_freq_error_hz` | 0.710 | 0.071 |
| LR (warmup → cosine) | 5e-5 | 1e-6 |

The LR curve confirms the Phase-1.3 schedule fix works, and the loss genuinely trends down,
so the `clip_grad` fix is doing its job. The problem is that the *val* metrics never
improve. Four independent diagnostics explain why.

**1. The prediction is the same waveform for every clip.**

| check | value |
| --- | --- |
| across-clip correlation of **predictions** | **0.9998** |
| across-clip correlation of **targets** | −0.028 |
| std across clips at each time index | 0.0053 (vs output std 0.393 → **1.3 %**) |

**2. The encoder hands the head a near-constant feature.**

`waveform_head` is a single `Linear(768, 8)` applied per time step, so `h` is the only
thing that can carry input information. Pooled over the 16 val clips:

| check | value |
| --- | --- |
| `h` std across clips | 0.00086 (vs `h` std overall 0.556 → **0.15 %**) |
| &#124;h(clip0) − h(clip1)&#124; | 0.00114 |
| &#124;h(clip0) − h(**zero input**)&#124; | 0.0348 (**31×** larger) |
| &#124;pred(clip0) − pred(clip1)&#124; | 0.0082 |
| &#124;pred(clip0) − pred(**zero input**)&#124; | 0.192 (**23×** larger) |

So the network is not "dead" — it responds strongly to whether there is an image at all. It
has collapsed to a **coarse global scene statistic** and discards the per-clip variation.

**3. The model is worse than predicting a constant.** The target is z-scored per clip, so
predicting 0 is the natural baseline: `mean|target| = 0.792` versus the model's
`mean|pred − target| = 0.962`. The model adds variance without adding signal, i.e. it
optimises L1/STFT magnitude by *shrinking* (prediction std 0.33-0.39 against a target
std of 1.0) — the classic collapse-to-the-conditional-mean failure.

**4. The information IS in the input, so this is a model/objective problem, not a data
problem.** A one-line baseline — the frame-wise mean of the green channel, no learning —
already recovers the pulse:

| input path (same 10 s window of F004_T1) | Welch peak | corr with reference `bp` |
| --- | --- | --- |
| **training input** (`target_size=64`, `IMREAD_REDUCED_8` decode) | 1.30 Hz | **−0.306** |
| full-quality decode then downscale to 64 | 1.30 Hz | −0.294 |
| native resolution, full-frame mean | 1.30 Hz | −0.294 |
| reference `bp` itself | 1.17 Hz | — |

Against the per-clip z-scored targets on 16 val clips, the same trivial baseline scores
`mean r = −0.19` (consistently negative on every clip — the physiologically expected
anti-phase between the pressure wave and green-channel intensity), versus the trained
model's `+0.017`. Two conclusions follow:

* the **reduced-decode path is exonerated** — a suspected culprit that turned out to make no
  difference (`−0.306` vs `−0.294`);
* a **102 M-parameter network performs 10× worse than averaging green pixels**, so the fault
  lies in the training setup, not in the data or the input pipeline.

Leading explanations, in order of likelihood: (a) the loss lets the model satisfy L1 and
MR-STFT magnitude by shrinking towards a constant, and the scale-invariant Pearson term is
too weak to prevent it; (b) the Stage-2 encoder contributes little — it was itself trained
under the `clip_grad` bug and its objective was RGB-dominated masked reconstruction, which
does not reward preserving sub-1 % intensity modulation; (c) 320 clips against 102 M
parameters gives the model an easier degenerate optimum than the intended one.

Diagnostic probes used above are throwaway scripts (`/tmp/bp_input_probe.py`,
`/tmp/bp_model_probe2.py`); re-run them from `code/` after any objective change to check
whether the encoder has started to encode per-clip variation.

* resume: re-running with `--epochs 3` auto-resumes (`start_epoch=1`) and the LR now moves
  (epoch 1 `lr` 7.5e-5, epoch 2 `lr` 2.6e-5) instead of the old constant 1e-4; an
  already-complete run exits with a clear message.

**Phase 2 — DONE 2026-09-23 (evaluation artefacts + session assembly)**

* `evaluation/report.py` (new): `save_json`, `append_jsonl`, `plot_training_curves`,
  `plot_waveform_panel`, `plot_session_waveform` — matplotlib is imported lazily and forced
  to the headless `Agg` backend, so the training path never needs a display; everything is
  written as PNG/JSON.
* `engines/waveform.py`: new `predict_waveforms(...)` returning the raw `[N, T]` arrays in
  loader order; `evaluate_waveforms` now reuses it and keeps its metric keys.
* `runners/run_waveform.py`: every evaluated epoch appends `<output_dir>/metrics.jsonl` and
  refreshes `<output_dir>/training_curves.png`; the end of a run writes
  `metrics_final.json` + `predictions_final.png`; `--save_preds` (enabled in both local
  configs) dumps `preds.npy` / `targets.npy` / `entries.json` for the session evaluator.
* `evaluation/assemble.py` (new): Hann overlap-add with weight-sum normalisation, offsets
  derived as `round(t_start * fs)` from the dataset entries, plus `affine_calibrate` for the
  amplitude metrics.
* `runners/run_evaluate_session.py` (new): consumes the dump, rebuilds each session, and
  writes `<pred_dir>/session_metrics.json` + `figures/session_<name>.png` with per-clip
  macro Tier-1/2, session-level Tier-1/2 (raw **and** affine-calibrated) and Tier-3 over
  30 s windows.
* `evaluation/clinical.py`: Tier-3 guard raised from 2 s to **30 s** (and it now prints the
  reason it skips), with `n_rr` and `peak_success` added to the result.

**Verified**: a 2-epoch RGB→BP smoke produced `metrics.jsonl`, `training_curves.png`,
`metrics_final.json`, `predictions_final.png`, `preds.npy`/`targets.npy`/`entries.json`,
`checkpoints/checkpoint-000{0,1}.pth`, `best.pth`, and — from
`runners/run_evaluate_session.py` — `session_metrics.json` +
`figures/session_F004_T1.png`. Both figures were opened and look correct. Scale check on
that untrained 2-epoch model: session MAE 101 (raw mmHg reference) vs 7.71 after the
per-session affine calibration, Pearson ~0.0003 — i.e. the artefact pipeline is sound and
the numbers are honest about the (un)trained state.

**Phase 3 — started 2026-09-23, BP branch ONLY** (user decision: no RESP run for now).`TARGET=bp scripts/local/waveform_local.sh` with the config defaults (40 epochs, batch 4,
`num_workers` 4); the log is teed to `output/finetune/bp_local/train.log`. **Measured
throughput**: ~0.78 s/iteration -> epoch 0 took **62 s** including the validation pass, so
the whole 40-epoch run is ~45-60 min (the earlier ~8 h estimate came from a 0-worker smoke
that was data-loader bound and is superseded). The RESP branch (`TARGET=resp …`) is NOT
started and stays ready to launch. Phase 4 (README / simplifiedPlan status + memory) is the
remaining step.
