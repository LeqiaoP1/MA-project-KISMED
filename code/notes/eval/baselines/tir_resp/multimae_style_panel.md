# MultiMAE-Figure-6-style panel for the two-stream ADR-008 checkpoint (2026-11-26)

**Figure**: [`artifacts/multimae_style_kept_density.png`](artifacts/multimae_style_kept_density.png)
(2 rows × 9 columns: the original clip, followed by three constant-budget allocations and one
remote-sensing column, each shown as `[masked input | prediction]`).
**Script**: `code/scripts/project/figure_multimae_style.py` (selects the allocation preset from the
checkpoint's stream set; the two-stream preset was added for this arm).
**Checkpoint**: `/work/scratch/ne95ocyg/pretrain/adr008_2stream_fold0/fold0/stage2/checkpoints/checkpoint-0149.pth`
(fold 0, epoch 149 = all 150, ADR-008 composite objective).
**Clip**: subject `F064`, session `F064_T6`, `t_start = 12.0 s` — drawn with `--seed 0` from the
**1,742 held-out clips** of the fold-0 test role (`--partition …/fold0/partition.json`, 213 sessions,
27 subjects), never seen in training or validation.

## 1. What the figure shows

Rows show the two streams (`tir`, `resp`). Columns show the original clip followed by four masking
conditions, each rendered as `[masked input | prediction]`. The first three columns spend the **same
200 visible tokens** (`n_budget`), shifting budget between the visual stream and the waveform; the
fourth is a **budget-exempt** remote-sensing probe with the entire TIR clip visible.

| Column | Kept tokens (`tir` / `resp`) | TIR density | Resp density | Mode / realised split |
| :--- | :--- | :--- | :--- | :--- |
| resp-max | 100 / 100 | 4 of 36 tubes (11.1 %) | 100 of 250 (40 %) | Mode 2, `S_core=1 S_periph=3` |
| mid | 125 / 75 | 5 of 36 tubes (13.9 %) | 75 of 250 (30 %) | Mode 2, `S_core=1 S_periph=4` |
| dense visual | 200 / 0 | 8 of 36 tubes (22.2 %) | 0 (virtual sensing) | Mode 1, `S_core=1 S_periph=7` |
| **remote sensing** | **900 / 0** | **36 of 36 tubes (100 %)** | 0 (virtual sensing) | Mode 1, `S_core=4 S_periph=32`, **budget-exempt** |

The last column reflects the **deployment condition** — the whole TIR clip is encoded (36 tubes × 25
tokens = 900 tokens) and no respiration token reaches the encoder, matching what a contactless system
receives in the field. Because it deliberately leaves the 200-token constant budget, it is built via
an explicit exempt plan (`plan_from_counts(…, exempt_budget=True)`, `BudgetPlan.budget_exempt`)
rather than drawn from the training policy; see §2.

## 2. Why this allocation sweep

The constant-budget range is fixed by the ADR-007 policy at this geometry (`core/budget_masking.py`,
`[mask]` banner in any run log):

- respiration is capped at `physio_cap_ratio 0.40` × 250 tokens = **100 tokens**, leaving a 100-token
  residual, so the sparsest expressible visual column is **4 tubes** (`video_min_tubes 4`);
- the visual stream is capped at `video_cap_ratio 0.25` × 36 tubes → **8 tubes** (22.2 %), which
  consumes the entire 200-token budget and gives Mode 1 (respiration fully hidden);
- `core/multimae.py` asserts that every policy-drawn encoder input has length `n_budget`, so a
  “fewer visible tokens overall” sweep like MultiMAE Figure 6 is not expressible — what varies across
  the first three columns is the split, not the total.

Those three columns therefore span the endpoints and midpoint of the architecture's budgeted axis:
**how much of the fixed 200-token budget the waveform keeps**. The fourth column answers the question
the budgeted sweep cannot: how the frozen checkpoint behaves when the visual stream is *not* rationed
at all.

- **Why it needed an explicit code path.** A fully visible TIR clip has 36 tubes × 25 tokens =
  **900 tokens**, which cannot be expressed inside a 200-token budget. `BudgetPlan.budget_exempt`
  (default `False`) marks an evaluation-only plan that declares its own encoder length, and
  `MultiModalMAE.forward` sizes its length assertion from it. The training sampler can never set this
  flag, so every training step still sees exactly `n_budget` tokens (ADR-002 / ADR-007 §2.6
  unchanged), enforced by `code/tests/test_plan_budget_exempt.py`.
- **How full coverage is reached.** The core draw takes the full anatomical core
  (`S_core = |A_core| = 4`) and the periphery draw takes its complement (`S_periph = 32`), so
  stratified placement covers every tube and the TIR mask is all zeros.
- **What stays validated.** Whole tubes only, every stream count within its own unit count, and a
  legal stratified split. Only the constant-budget constraints (`n_budget` sum, visual floor/cap, and
  `S_core` band) are relaxed.

## 3. What the panel says

- **TIR reconstructions sharpen monotonically with visible tube count.** In the prediction panels,
  kept patches are overlaid verbatim (`reconstruct_visual`) while masked patches are reconstructed.
  At 4 tubes the masked nostril region is blurred out and crisp mouth detail appears only where a
  mouth-edge tube was kept; at 5 tubes the right nostril is kept while the masked mouth remains a
  diffuse block; at 8 tubes (Mode 1) both nostrils are visible and the surrounding facial, philtrum,
  and lip structure is substantially sharper. This follows the same density ordering seen on the
  three-stream panel, without an `rgb2d` stream.
- **Respiration is reconstructed almost perfectly when its own tokens are visible** (§4:
  `r_masked ≈ 0.999` across the test role in both inpainting columns). This is the
  *local-interpolation* regime rather than cross-modal sensing: with 30–40 % of a 32 ms token grid
  kept in scattered positions, masked gaps are narrow and sit immediately between visible tokens. In
  the `resp` prediction panels, black dots mark the kept ground-truth tokens and the red curve traces
  the masked-token reconstructions cleanly across the gaps (the vertical red spikes down to zero
  occur at visible positions, where `MultiModalMAE.forward` leaves unmasked prediction slots
  zero-filled).
- **With respiration fully hidden (`dense visual`, 200/0), the waveform does not collapse to a flat
  line**, though it is imprecise on this single clip: it produces a broad low-frequency oscillation
  over the 8 s window rather than tracking the flat mid-clip plateau accurately. This column is one
  mask draw on one clip; the full test-set statistics for the same allocation are subject-median
  `r_masked = +0.326` and `Var(ŷ) = 0.472` (§4).
- **The remote-sensing column (`900/0`) recovers more cross-modal structure from the full TIR clip.**
  Its TIR row has nothing masked, so the right-hand panel is the unmodified frame (labelled
  `no masked token`), while its RESP row is a pure cross-modal prediction from all 36 tubes. On this
  clip it tracks the second-half rise and double-peaked crest more closely than the 200-token column
  (at the cost of visibly more high-frequency jitter), showing that the 200-token budget cap — not
  the TIR modality alone — was limiting the budgeted probe.

## 4. The same conditions measured on the whole held-out test role

Zero-shot Stage-2 evaluation only (no Stage-3 fine-tuning): 1,742 clips × 5 mask draws, `--seed 0`,
with identical placement seeds across arms so comparisons are paired within clip. All three
hidden-respiration rows (`TIR-200`, `TIR-900`, `TIR-900-rolled`) keep 0 respiration tokens; because a
zero-count signal draw consumes no random numbers (`core/mask_placement.place_signal` returns an
empty index set), those three rows score the exact same 100 %-masked respiration positions.

| Column | Kept tokens (`tir` / `resp`) | `r_masked` mean (median, sd) | subject-median `r` [95 % CI] | MSE (z²) mean | `Var(ŷ)` (subj. median) | peak err (subj. median) |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| resp-max | 100 / 100 | +0.9984 (+0.9989, 0.0021) | **+0.9989** [+0.9986, +0.9990] | 0.0037 | 0.9997 | 0.00 bpm |
| mid | 125 / 75 | +0.9981 (+0.9987, 0.0024) | **+0.9986** [+0.9983, +0.9987] | 0.0043 | 0.9897 | 0.00 bpm |
| dense visual — **`TIR-200`** | 200 / 0 | +0.3400 (+0.3623, 0.3070) | **+0.3256** [+0.2444, +0.4061] | 0.9794 | 0.4716 | 4.92 bpm |
| remote sensing — **`TIR-900`** | **900 / 0** | **+0.3952** (+0.4565, 0.3698) | **+0.3819** [+0.2731, +0.5349] | 1.0036 | **0.6518** | **3.66 bpm** |
| remote sensing rolled 16 frames — **`TIR-900-rolled`** | 900 / 0 | +0.1081 (+0.1093, 0.3697) | +0.0823 [+0.0379, +0.1527] | 1.4636 | 0.6277 | 5.49 bpm |

**Nomenclature for the three hidden-respiration contrasts below** — each name encodes *how many TIR
tokens the encoder receives* and *whether the TIR clip is time-aligned with the respiration window*:

| Name | Encoder input | Plan / flags | Report |
| :--- | :--- | :--- | :--- |
| **`TIR-200`** ("budgeted", "200-token") | **200** TIR tokens (8 of 36 tubes of the **intact** clip); 0 respiration tokens | constant budget, Mode 1 (`--kept_tir 1.0 --resp_probe hidden`, cap 8 tubes) | `output/eval/adr008_pretext/mode1_fold0.json` |
| **`TIR-900`** ("full frame", "real 900") | **900** TIR tokens (all 36 tubes of the **intact** clip); 0 respiration tokens | budget-**exempt** (`--exempt_budget --kept_tir 1.0 --resp_probe hidden`) | `output/eval/adr008_recon/remote_sensing_fold0.json` |
| **`TIR-900-rolled`** ("rolled") | the same **900** tokens, but the TIR clip is circularly shifted 16 frames (2.56 s) **before** patching, breaking TIR↔respiration temporal alignment; 0 respiration tokens | budget-exempt plus `--ablate_stream tir --ablate_mode roll` (plan and masks untouched) | `output/eval/adr008_recon/remote_sensing_tir_rolled_fold0.json` |

All three arms score the **same** masked respiration samples on the **same** 1,742 clips; they differ
only in what the TIR stream supplies.

- **Reports.** Trimmed test-set summaries are archived alongside this note in [`artifacts/`](artifacts/):
  [`respmax_fold0_summary.json`](artifacts/respmax_fold0_summary.json),
  [`mid_fold0_summary.json`](artifacts/mid_fold0_summary.json),
  [`mode1_pretext_fold0_summary.json`](artifacts/mode1_pretext_fold0_summary.json) (the pre-existing
  `tir=200 / resp=0` Mode-1 report, not re-run),
  [`remote_sensing_fold0_summary.json`](artifacts/remote_sensing_fold0_summary.json), and
  [`remote_sensing_tir_rolled_fold0_summary.json`](artifacts/remote_sensing_tir_rolled_fold0_summary.json).
  Full per-clip reports remain under `output/eval/adr008_recon/` (and `output/eval/adr008_pretext/`).
- **Paired contrast `resp-max` vs `mid`** (trading 25 respiration tokens for 1 TIR tube):
  Δ(`mid` − `resp-max`) `r_masked` = **−0.0004 ± 0.0000**, t = −21.7, 1,537 of 1,742 clips worse,
  205 better (`output/eval/adr008_recon/respmax_vs_mid.json`). Statistically clean but practically
  irrelevant: both columns sit on the interpolation ceiling, showing only that 75 visible tokens
  already pin the waveform.
- **The informative drop lies between columns 2 and 3:** once respiration tokens are removed,
  subject-median `r_masked` drops from +0.999 to +0.326 and `Var(ŷ)` from 0.99 to 0.47. The two
  inpainting columns are not evidence of cross-modal sensing; the Mode-1 column is.
- **Does the full TIR frame help? Yes — it yields the strongest zero-shot respiration result of this
  arm.** Paired within clip against `TIR-200` (identical weights, clips, and fully masked
  respiration; only visual evidence changes):

  | Quantity | `TIR-200` | `TIR-900` | Paired Δ (`TIR-900` − `TIR-200`) | t | Clips better / worse |
  | :--- | ---: | ---: | :--- | ---: | :--- |
  | `r_masked` (mean) | +0.3400 | +0.3952 | **+0.0552 ± 0.0066** | +8.40 | 1,023 / 719 |
  | `Var(ŷ)` (mean) | 0.4934 | 0.6610 | **+0.1676 ± 0.0050** | +33.85 | 1,347 / 395 |
  | peak error (mean) | 5.79 bpm | 5.43 bpm | **−0.37 ± 0.12 bpm** | −2.98 | 1,019 / 695 |
  | MSE (mean) | 0.9794 | 1.0036 | +0.0242 ± 0.0101 | +2.38 | 841 / 901 |

  Shape, amplitude, and rate all improve. Mean MSE rises slightly, whereas median MSE improves
  (0.963 → 0.915 z²): the full frame recovers more signal amplitude (subject-median `Var(ŷ)`
  0.47 → 0.65 of the target's 1.0), which helps correlation and typical clips while increasing
  squared-error overshoot when the predicted envelope is out of phase. Subject-median `r` climbs from
  +0.326 to +0.382, and subject-median spectral peak error falls from 4.92 to 3.66 bpm — still inside
  the ADR-008 §3.2 *failure* band (`> 3.0 bpm`), so the rate target remains missed, only by less.
  Giving the checkpoint its deployment-shaped input therefore recovers part of the gap imposed by
  the 200-token training budget.
- **The rolled-TIR control confirms a content effect, not a token-count effect.** The final row
  repeats `TIR-900` with the TIR tensor circularly shifted 16 frames along time
  (`--ablate_stream tir --ablate_mode roll`, `torch.roll(..., shifts=16, dims=T)`) — identical plan,
  masks, token count (900), and frames, with only TIR↔respiration alignment broken. Each clip has 50
  frames at 0.16 s spacing (`temporal_stride 4` at `fps 25`), so 16 frames is **2.56 s** (nearly a
  third, ~32 %, of the 8 s window):

  | Contrast | Paired Δ `r_masked` | t | Clips better / worse |
  | :--- | :--- | ---: | :--- |
  | `TIR-900` − `TIR-900-rolled` | **+0.2872 ± 0.0100** | +28.7 | 1,307 / 435 |
  | `TIR-200` − `TIR-900-rolled` | **+0.2320 ± 0.0100** | +23.2 | 1,214 / 528 |

  `TIR-900-rolled` performs *worse* than `TIR-200`, showing that extra tokens are not a benign
  resource: without valid temporal correspondence they add noise rather than evidence. The +0.055
  gain of `TIR-900` over `TIR-200` is therefore driven by aligned thermal content, and the deployed
  ceiling is governed by what the encoder can extract from the face rather than merely by sequence
  length.
- **Variance retention alone is not evidence of sensing.** `TIR-900-rolled` retains subject-median
  `Var(ŷ) = 0.63` — comfortably inside the ADR-008 §3.2 success band (`>= 0.60`) — while its
  subject-median correlation collapses to **+0.082** (below the `< +0.10` failure threshold;
  per-clip mean `+0.108`). The two diagnostics must be read together: an amplitude-honest but
  phase-unaligned waveform passes variance retention while failing correlation.

## 5. Honest limits

- One clip, one seed (`--seed 0`), and one checkpoint (fold 0, epoch 149). The three budgeted
  columns draw random anatomy-aware placements; the remote-sensing column is deterministic (every TIR
  tube kept, every respiration token masked), so its panel is not a random policy draw.
- On the TIR row, each left panel blanks masked patches and each right panel overlays the true kept
  patches (`reconstruct_visual`), following MultiMAE Figure 6.
- On the `resp` row, each right panel plots the raw `preds['resp']` tensor (red) together with black
  dots at the kept ground-truth tokens. Because `MultiModalMAE.forward` only populates masked slots
  and leaves visible slots at zero, only the masked gaps are reconstructions; §4's `r_masked` scores
  those masked positions only and is the honest metric.
- The tiny Δ between the two inpainting columns (`resp-max` vs `mid`, paired within clip under the
  same placement seed) is a property of *this* token geometry, not of masking budgets in general. At
  75 kept tokens over 8 s, visible tokens average one every ~107 ms (~9.4 Hz effective sampling of a
  signal band-limited to 0.6 Hz, Nyquist 1.2 Hz), and scattered placement keeps gaps to a few tokens
  — so reconstruction there is local interpolation, not cross-modal inference.
- **The remote-sensing column is out-of-regime by construction.** The encoder was trained on exactly
  200 visible tokens; `TIR-900` asks it to attend over 4.5× as many (900). Its score is therefore a
  zero-shot probe of the frozen checkpoint under deployment-shaped input rather than a training-regime
  result. Properly training in this regime would require a variable- or large-budget pre-training
  run, which does not exist.
- Not in this note: the gating ADR-008 §3.3 criterion (five-fold Stage-3 band-limited Pearson vs. the
  matched pure-MSE ADR-007 encoder). Nothing here supports a Go/No-Go claim.

## 6. Reproduce

From `code/` (activate the project environment first: `source scripts/env_hpc.sh` and the venv):

```bash
CKPT=/work/scratch/ne95ocyg/pretrain/adr008_2stream_fold0/fold0/stage2/checkpoints/checkpoint-0149.pth

# Figure
python -m scripts.project.figure_multimae_style \
  --checkpoint "$CKPT" \
  --partition /work/scratch/ne95ocyg/pretrain/adr008_2stream_fold0/fold0/partition.json \
  --seed 0 --device cuda \
  --out notes/eval/baselines/tir_resp/artifacts/multimae_style_kept_density.png

# Two inpainting columns on the held-out test role (~2.5 min per mask draw on one T4)
for arm in "respmax 0.111 0.40" "mid 0.139 0.30"; do
  set -- $arm
  python -u runners/run_eval_recon.py --checkpoint "$CKPT" --split test \
    --kept_tir "$2" --kept_resp "$3" --resp_probe random \
    --repeats 5 --batch_size 16 --preload --pretext_metrics \
    --seed 0 --device cuda --label "adr008_fold0_$1" \
    --out ../output/eval/adr008_recon/$1_fold0.json
done

# Remote-sensing column on the held-out test role (budget-exempt: full TIR clip)
python -u runners/run_eval_recon.py --checkpoint "$CKPT" --split test \
  --kept_tir 1.0 --resp_probe hidden --exempt_budget \
  --repeats 5 --batch_size 16 --preload --pretext_metrics \
  --seed 0 --device cuda --label adr008_fold0_remote_sensing \
  --out ../output/eval/adr008_recon/remote_sensing_fold0.json

# Paired contrast: resp-max vs mid
python -m scripts.project.compare_recon_arms \
  --a ../output/eval/adr008_recon/respmax_fold0.json \
  --b ../output/eval/adr008_recon/mid_fold0.json \
  --out ../output/eval/adr008_recon/respmax_vs_mid.json

# Remote-sensing contrast (200-token Mode 1 vs full frame; swap --metric mse|var_hat|peak_err_bpm
# for the remaining rows of §4's contrast table)
python -m scripts.project.compare_recon_arms \
  --a ../output/eval/adr008_pretext/mode1_fold0.json \
  --b ../output/eval/adr008_recon/remote_sensing_fold0.json \
  --metric r_masked \
  --out ../output/eval/adr008_recon/mode1_vs_remote_sensing.json

# Content control: identical plan and masks, TIR clip rolled 16 frames
python -u runners/run_eval_recon.py --checkpoint "$CKPT" --split test \
  --kept_tir 1.0 --resp_probe hidden --exempt_budget \
  --ablate_stream tir --ablate_mode roll \
  --repeats 5 --batch_size 16 --preload --pretext_metrics \
  --seed 0 --device cuda --label adr008_fold0_remote_sensing_tir_rolled \
  --out ../output/eval/adr008_recon/remote_sensing_tir_rolled_fold0.json
```

`--kept_tir 0.111` / `0.139` are requests: the realised counts are 4 / 5 tubes (100 / 125 tokens),
recorded in each report's `masking` string, because the fraction path rounds to whole tubes and
clamps into `[video_min_tubes, video_cap]`.
