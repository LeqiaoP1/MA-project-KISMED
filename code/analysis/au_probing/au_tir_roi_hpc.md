# AU-occurrence probe — TIR-ROI + Resp Stage-2 encoder (HPC, 112 px)

> **Verdict: NEGATIVE.** The frozen Stage-2 thermal-ROI encoder does **not**
> carry linearly decodable facial-action information. The linear head scores
> **macro-F1 33.04 against a trivial always-positive baseline of 44.59
> (Δ = −11.55)**, beats that baseline on only **2 of 7** Tier-A AUs, and the
> per-AU F1 tracks the AU's **presence rate at r = +0.984** — the signature of a
> head that has learned the class marginal and essentially nothing else.
>
> This **fails** the protocol's acceptance criterion
> (`AU_OccurenceProbe.md` §6: *"informative only if C2 linear-F1 > trivial-F1 on
> a MAJORITY of Tier-A AUs **and** ≥ C1"* — majority not met, and C1 was not run).

Protocol of record: `au_probing/AU_OccurenceProbe.md`.
Implementation notes: `au_probing/AU_OccurenceProbe_implementation.md`.

---

## 1. What ran

| | |
|---|---|
| Job | **55450134** `au_probe_tir`, partition `acc_short`, node `gaoc0002` |
| State | **COMPLETED, ExitCode 0:0**, elapsed **17:34**, **empty `.err`** |
| Window | 2026-10-15T20:59:42 → 21:17:16 |
| Config | `configs/finetune/au_tir_roi_hpc.yaml` |
| Runner | `runners/run_au_probe.py` (via `scripts/hpc/submit_au_probe.sbatch`) |
| Encoder | `stage2_hpc_tir_roi_resp/checkpoints/checkpoint-0149.pth` |
| Out dir | `$WORK_SCRATCH/finetune/au_tir_roi_hpc` |
| Artifacts | `best_au.pth` (352 MB), `checkpoints/`, `train.log` |

Job history around it: **55450025 FAILED** after 13 s
(`ValueError: MultiModalMAEProbe: stream "tir" missing from input; got keys
['rgb']`, `core/au_probe.py:167` — a config↔dataset wiring mismatch, unrelated to
the earlier 55427926 arity bug), then 55450027 (smoke, COMPLETED 1:26),
55450131 (CANCELLED, superseded), then **55450134** (the run reported here).

## 2. Input contract — VERIFIED (the documented blocker was fixed)

`AU_OccurenceProbe.md` §4 flagged a **blocker**: *"`runners/run_au_probe.py` has
no `--temporal_stride` flag, so `build_au_probe_model` computes
`num_frames = clip_duration·fps = 200` against a stride-2 checkpoint (100).
`_resize_t` then silently trims to the first 100 frames while the label sits at
the clip centre — the label is no longer at the clip centre."*

**That blocker is resolved on this branch**, so the numbers below are not the
known-broken configuration:

| key | required (= checkpoint) | this run | ok |
|---|---|---|---|
| `input_size` | 112 | 112 | ✅ |
| `roi_landmarks` | `''` (12-pt nose+mouth) | `''` | ✅ |
| `roi_padding` | 0.2 | 0.2 | ✅ |
| `clip_duration` / `fps` | 8.0 / 25.0 | 8.0 / 25.0 | ✅ |
| `temporal_stride` | 2 | 2 (`--temporal_stride` now exists) | ✅ |
| ⇒ `num_frames` | 100 | 100 (`num_frames: 0` ⇒ `round(8·25/2)`) | ✅ |
| encoder tensors loaded | — | **150, 0 shape-mismatched, 4 skipped** | ✅ |
| ROI cache | same keys as Stage 2 | `..._nose_mouth_pad0.20_minmax_src726x480_native_v2_...` | ✅ |

The run log confirms the cache was attached and printed
`src 726x480, landmarks nose_mouth (12 pts), padding 0.2, format v2`.

## 3. Protocol actually used

- **Linear probe** — encoder frozen, **head only, 5,383 trainable params**.
- Pooling `mean`; loss `BCEWithLogitsLoss`; **decision threshold fixed at 0.5**;
  `tune_threshold: false`.
- AU set `10,12,14,15,17,23,24` (Tier-A headline set, ROI-valid).
- **Subject-disjoint** split: **112 train subjects → 631 samples**,
  **27 val subjects → 138 samples**. Selected by `train_ratio: 0.8` over the
  AU-coded subject pool, **not** by an explicit subject list, and the printed
  counts are POST-filtering (a subject with no surviving clip disappears — the
  val list jumps `M048 → M050`, so `M049` was dropped).
- 30 epochs, batch 16, `lr 1e-3`, `warmup_epochs 3` (117 steps),
  `weight_decay 0.05`, FP32 (`amp_dtype none`).
- **Single process** (world size 1): 39 iterations/epoch = ⌈631/16⌉ with
  `drop_last`, so this is not a shard-limited metric. `evaluate_au` all-gathers
  `tp/fp/fn` when DDP is live, so the metric is global anyway.

## 4. Result

Validation curve over the full 30 epochs — it **peaks at epoch 3 and never
improves again**:

```
ep0  f1=26.50   ep3  f1=34.33  (best)   ep10  f1=32.73   ep20  f1=32.19
ep1  f1=33.67   ep5  f1=32.97            ep15  f1=32.52   ep25  f1=32.89
ep2  f1=34.32   ep9  f1=34.20            ep19  f1=34.23   ep29  f1=33.04
```

Train loss falls 0.557 → 0.437 by ~epoch 10 and is then **flat** (0.4369 at
epoch 29). So the head converges within ~3 epochs and the remaining 27 epochs buy
nothing — the binding constraint is not optimisation budget.

Per-AU, at threshold 0.5 (trivial = always-positive, `F1 = 2p/(1+p)`):

| AU | probe F1 | trivial F1 | Δ | presence *p* | verdict |
|---|---|---|---|---|---|
| 10 | 81.08 | 78.41 | **+2.67** | 0.645 | beats trivial |
| 12 | 76.07 | 74.55 | **+1.52** | 0.594 | beats trivial |
| 14 | 74.12 | 78.95 | −4.83 | 0.652 | below trivial |
| 15 | **0.00** | 19.61 | −19.61 | 0.109 | collapsed (all-negative) |
| 17 | **0.00** | 25.32 | −25.32 | 0.145 | collapsed (all-negative) |
| 23 | **0.00** | 29.63 | −29.63 | 0.174 | collapsed (all-negative) |
| 24 | **0.00** | 5.63 | −5.63 | 0.029 | collapsed (all-negative) |
| **macro** | **33.04** | **44.59** | **−11.55** | | |

`presence_rate` is **derived, not measured** — the runner does not persist it
(see §7.4). It is recovered exactly by inverting the engine's own baseline
(`engines/au_probe.py::trivial_f1`, `p = t / (2 − t)`). Best val macro-F1 over the
run was **34.33** (epoch 3).

## 5. Diagnosis — the head learned the marginal, not the action

Two independent observations point the same way.

**(a) The zero-F1 AUs are exactly the rare ones.** All four AUs scoring 0.00 have
*p* ≤ 0.174, while the three that score non-zero have *p* ≥ 0.594. An F1 of
exactly 0.00 with a non-zero presence rate means the head **never predicted
positive** for that AU: an uncalibrated BCE logit for a class with prior
*p* < 0.5 sits below the 0.5 threshold, so the decision collapses to
all-negative. Those four are precisely the AUs whose trivial baseline *does* get
credit (19.6 / 25.3 / 29.6) — which is why the macro ends up far **below**
trivial rather than merely at it.

**(b) probe-F1 is almost a pure function of presence rate: r = +0.984.** Across
the 7 AUs, knowing only how often the AU occurs predicts the probe's F1. An
always-positive predictor would give *exactly* the trivial F1; the probe is a
slightly perturbed version of that (AU10 +2.7, AU12 +1.5, AU14 −4.8), i.e. the
head reproduces the marginal and adds ~nothing.

**So the honest reading is not "the probe is broken" but "the probe works and the
representation is empty":** on a frozen TIR-ROI encoder, a linear head's best
available strategy is to predict the class prior, and the two AUs that clear
trivial by 1.5–2.7 points are within the range a marginal-fit head can produce by
noise. This is consistent with the protocol's own §7 confounds — a 112 px,
8-bit **palettised false-colour** crop of a **truncated face** (no brows, no
eyelids, no cheek), and a Stage-2 objective that never had to encode facial
action.

## 6. Against the acceptance criterion

`AU_OccurenceProbe.md` §6:

> Judge a representation informative only if **C2 linear-F1 > trivial-F1 on a
> majority of Tier-A AUs** *and* **≥ C1 linear-F1**.

| clause | outcome |
|---|---|
| C2 > trivial on a **majority** of Tier-A AUs | ❌ **2/7** — not a majority |
| C2 ≥ C1 (Stage-1 VideoMAE control) | ⚠️ **not run** — no C1 arm exists yet |

The first clause fails outright. The second is untested, so the probe currently
cannot be called informative on either half of the criterion.

## 7. What is NOT established (read before quoting)

1. **Risk of probe-capacity confound is low; risk of threshold confound is
   real.** The head is 5,383 params and converged in 3 epochs, so this is not an
   under-fitting story — but a **single hard threshold at 0.5** was used. The
   collapse of the rare AUs is at least partly a *calibration* phenomenon, and
   the protocol already anticipates this (`oracle_f1` per-AU tuning is defined as
   an upper bound and is deliberately **not** a headline). See §8.1.
2. **~100 px, false-colour, truncated face.** Per protocol §7 this must not be
   read as an encoder ablation against RGB full-frame numbers.
3. **Scale.** 631 train / 138 val samples from 112/27 subjects, 30 epochs. Small.
4. **The presence rates are derived, not logged.** The runner prints only F1
   values; `evaluate_au` computes `presence_rate` but nothing persists it, so
   there is **no report JSON** in the run directory. That is a reporting gap
   worth fixing (it is what forced the inversion in §4).
5. **`best_au.pth` is the epoch-3 checkpoint**, i.e. the best *val* F1 (34.33),
   not the last — quote 33.04 or 34.33 explicitly, they are different epochs.
6. **No test split.** `test_subjects` was unset, so the reported split did double
   duty (model selection and report). Leakage-free threshold tuning (protocol
   §10.1) is therefore not usable on this run.

## 8. Decisive next experiments (cheapest first)

1. **Flip `tune_threshold: true`** (already plumbed: `--tune_threshold`,
   `runners/run_au_probe.py:144`; config key `tune_threshold: false`). The
   oracle per-AU threshold is the direct test of "calibration or representation":
   * if AU15/17/23/24 rise to ≈ their trivial F1 → the head holds only the
     marginal, confirming §5 and closing the question;
   * if they rise **well above** trivial → there is linear AU signal that the 0.5
     threshold was hiding, and the whole verdict changes.
   This can also be run **offline on `best_au.pth`** — no retraining.
2. **Class imbalance**: add `pos_weight` to `BCEWithLogitsLoss`, or report
   average-precision / AUROC alongside F1@0.5. F1 at a fixed threshold is a poor
   instrument when *p* spans 0.03–0.65.
3. **Run the C1 control** (Stage-1 VideoMAE, identical protocol, only
   `--finetune` changes) — required by §6 and currently missing.
4. **Persist the report** (`presence_rate`, `trivial_f1_per_au`,
   `oracle_*`, thresholds) to a JSON next to `best_au.pth`, so the next reader
   does not have to invert the baseline.
5. Only **then** consider a `probe: ft` (fine-tuned) upper bound — and treat it
   as an upper bound, never a headline.

## 9. Reproduce

```bash
cd code
# the run itself (already done; resubmitting re-runs it)
sbatch scripts/hpc/submit_au_probe.sbatch

# the tables in this document, from the run's own log (CPU-only, no model)
python analysis/au_probing/au_probe_report.py \
    --log /work/scratch/$USER/finetune/au_tir_roi_hpc/train.log \
    --name "au_tir_roi_hpc (job 55450134)" \
    --out analysis/au_probing/au_tir_roi_hpc.json
```

`au_probe_report.py` parses the per-epoch validation curve and the per-AU table
out of the log, derives the presence rates, and writes the machine-readable JSON
that this document quotes.

## 10. Provenance

| datum | source |
|---|---|
| job state / exit code / elapsed / node | `sacct -u $USER` |
| input contract, split sizes, encoder load | `code/logs/au_probe_tir_55450134.out` |
| per-epoch curve, per-AU F1, best F1 | `$WORK_SCRATCH/finetune/au_tir_roi_hpc/train.log` |
| presence rates, deltas, correlation | **derived** by `au_probe_report.py` (§4/§5) |
| acceptance criterion | `au_probing/AU_OccurenceProbe.md` §6 |
| baseline definition | `engines/au_probe.py::trivial_f1` |

_Compiled 2026-10-16 on branch `au_probing`, from job 55450134 (2026-10-15)._
