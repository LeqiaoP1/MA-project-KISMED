# RESP reconstruction (virtual sensing) on held-out subjects — arms A/B (2026-11-22)

- **Checkpoint**: `checkpoint-0149.pth` (epoch 149, scored the fold-0 partition)
- **Split**: `--split test` of the fold-0 record (`dc80d46c…`), restricted to
  `--subjects F001,F003,F005` — three of the 28 subject-disjoint test subjects,
  which the checkpoint never trained or validated on.
- **Clips**: 199 from 24 sessions (≈110 clips/subject after the BP `usable` gate).
- **Probe**: `--resp_probe span --span_seconds 4 --kept_resp 0` ⇒ RESP entirely
  hidden as ONE contiguous 125-token (4 s) block, so it cannot be interpolated
  from neighbouring kept tokens. The pinned plan is **mode 1** — exactly the
  virtual-sensing condition. `repeats 5`, `seed 0`, so the two arms saw the same
  mask draws (paired).
- **Verdict**: **no measurable RESP reconstruction in either arm**, and no
  measurable difference between them.

## 1. What was actually run

Both arms keep the ADR-002 constant budget (350 visible tokens) and pin BP to
zero (new `--kept_bp 0.0`, plan item P0-E), so the budget is spent only on the
two visual streams. The requested fractions are converted to whole held-out
modules and the residual is absorbed by the apportioner, so the *realised*
counts are what the report records:

| Arm | Requested (`tir`, `rgb2d`) | Realised kept tokens | Sum |
|---|---|---|---|
| **A — TIR-max** | 1.0, 0.0 | `tir 225` (9 tubes, cap), `rgb2d 125` (5) | 350 |
| **B — RGB2D-max** | 0.0, 1.0 | `tir 125` (5), `rgb2d 225` (9, cap) | 350 |

`bp 0` and `resp 0` in both. A 100 %-kept visual (36 tubes = 900 tokens) is not
expressible: `core/multimae.py` asserts the encoder input has length `n_budget`,
so 225 tokens per visual is the maximum this checkpoint can be probed with.

## 2. Results (z-scored signal, masked positions only)

| Metric | Arm A (TIR-max) | Arm B (RGB2D-max) |
|---|---|---|
| `r_masked` mean | **+0.0062** | **+0.0140** |
| `r_masked` median | +0.0173 | +0.0215 |
| `r_masked` sd | 0.2110 | 0.2123 |
| `r_full` mean | +0.0062 | +0.0140 |
| `mse` (z²) | 1.0254 | 1.0218 |
| `rmse` (z) | 1.0114 | 1.0096 |
| `mae` (z) | 0.8534 | 0.8526 |
| per-clip `r > 0.3` | 8.0 % | 8.5 % |
| per-clip `r < −0.3` | 8.5 % | 6.5 % |

Paired comparison over the same 199 clips:

$$\Delta r = r_B - r_A = +0.0078 \pm 0.0102\ \text{(SE)},\quad t = +0.76$$

No difference. The per-clip tails are **symmetric** (±0.3 at roughly equal
rates), which is the signature of noise rather than of systematic tracking: a
model that recovered breathing would show a positive tail only.

## 3. Interpretation

- `mse ≈ 1.02` in z units means the decoder predicts approximately the clip mean
  over the masked block: there is essentially no waveform recovered.
- `r_masked ≈ 0` in both arms, with a paired difference two orders of magnitude
  below the per-clip spread.
- This is **consistent with the protocol's own expectation** (§0.3, §5.2): the
  per-stream decoder conditions only on the encoder latents plus that stream's
  own `[MASK]` tokens, so a fully hidden RESP has no cross-modal path that the
  four-stream arm was trained to exploit. It also matches the earlier tri-modal
  reference (ViT-Base, `hidden` probe: `r = +0.0055`, `notes/eval/dataset_inspection/rgb2d/resp_recon_zeroshot.md`).
- Moving 100 tokens of budget from RGB2D to TIR (or back) does not change the
  outcome, so within this checkpoint and this probe the *allocation* between the
  two visual streams is not the limiting factor.

## 4. What this does not establish

- **Not** a statement about a fully visible face: both arms split 350 tokens, and
  225 per visual is the architectural maximum for this checkpoint.
- **Not** a modality conclusion: only the budget split was varied, not the
  presence/absence of a stream (that needs the matched 3-stream control, §5.4,
  which does not exist yet).
- Small sample: 3 of 28 test subjects, 199 clips, no cross-subject CI reported.
- The `span` probe differs from training placement (`scattered`), so part of the
  failure could be placement shift rather than absence of cross-modal signal;
  `--resp_probe random` (the training condition) is the reference that separates
  these, and it was not run.

## 5. Code changes made to run this

- `runners/run_eval_recon.py`: `--kept_bp` (default 1.0 = legacy behaviour; 0.0
  removes BP from the budget — plan item **P0-E**) and `--subjects` (refuses a
  subject outside the evaluated role, since the test block is subject-disjoint).
- Test suite: `800 passed` before and after.
- Reports: `/tmp/armA_tir_max.json`, `/tmp/armB_rgb2d_max.json` (per-clip rows
  included; copy them into `EXP_ROOT` if they are to be kept).
