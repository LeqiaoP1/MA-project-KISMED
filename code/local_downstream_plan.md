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

## 2. Phase 0 — configs and launcher (no code changes)

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
| `weight_decay` / `clip_grad` | `0.05` / `0.0` | |
| `num_workers` | `4` | |
| `output_dir` | `../output/finetune/bp_local` / `resp_local` | mind the `../` |
| `log_wandb` / `wandb_project` | `false` / `thesis-project` | wandb flags are unwired |

> Caveat: `parse_args_with_config` feeds YAML through `parser.set_defaults(**cfg)`, so
> a YAML key that has no matching argparse option is **silently absorbed and ignored**.
> `clip_stride` and `split_by` are read from `args` via `getattr` by the dataset
> builders, which is why they work from YAML but not from the command line.
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

## 3. Phase 1 — three surgical code fixes (must land before the runs)

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
| `configs/finetune/bp.yaml`, `resp.yaml`, `eda.yaml` | template repairs only (§2.2) |
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

1. **Throughput vs batch size.** Measured ~4 s/iteration at batch 2 / input 64 on the
   local GPU (RTX 5060 Ti, 16 GB) — ~12 min per epoch at ~175 iterations, i.e. ~8 h for
   40 epochs per branch. Option A: batch 4, 40 epochs *(default)*. Option B: batch 8 to
   roughly halve wall-clock if VRAM allows. Option C: cap `max_clips` to shorten
   epochs while prototyping.
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
