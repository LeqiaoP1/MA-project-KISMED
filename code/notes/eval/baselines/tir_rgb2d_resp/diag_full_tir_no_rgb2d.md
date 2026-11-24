# Off-policy diagnostic — full TIR input, no RGB2D (2026-11-24)

**Status: diagnostic, not a policy arm.** These runs deliberately break the
ADR-005 constant-budget contract. They are recorded because the question “what do
the trained weights do with the whole TIR stream and no RGB2D?” cannot be
answered by any valid arm of the policy.

- **Checkpoint**: `checkpoint-0149.pth` (fold 0, epoch 149)
- **Split**: fold-0 `test` — the same 1,594 clips / 201 sessions / 27 subjects as
  `resp_recon_arms_ab.md`
- **Probe**: `run_eval_recon.py --override_counts … --preload`; RESP fully
  masked. One draw: `diag-900` keeps all 36 TIR tubes, so its kept **set** is the
  whole grid and placement cannot vary; `diag-350` keeps 14 of 36 tubes and is
  therefore **one random placement draw** (repeats were not run — see §4).
- **Contract bypass**: `--override_counts` builds the `BudgetPlan` directly and
  calls the model with `strict_budget=False`. The normal strict path still
  refuses any input that is not 350 tokens (verified).

## 1. What was fed

| Run | TIR | RGB2D | RESP | Encoder length | In contract? |
|---|---:|---:|---:|---:|---|
| **A — reference arm** | 225 (9/36 tubes) | 125 (5/36) | 0 | 350 | **yes** |
| **diag-900** | **900 (36/36 tubes)** | **0** | 0 | **900** | no |
| **diag-350** | 350 (14/36 tubes) | 0 | 0 | 350 | no |

`diag-350` exists to separate two shifts at once: it keeps the trained sequence
length but removes RGB2D and concentrates the budget in TIR; `diag-900` changes
both. TIR at 14 tubes is 39 % of the stream, above the 25 % training cap.

## 2. Results (masked positions, RESP fully hidden)

| Run | `r_masked` mean | median | subject median [95 % CI] | MSE (z²) | `r`>0.3 / <−0.3 |
|---|---:|---:|---|---:|---:|
| **A — mixed 350** | **+0.1091** | +0.1126 | +0.1129 [+0.0775, +0.1363] | 1.0408 | 25.8 / 8.1 % |
| **diag-900 — all TIR** | **−0.0026** | −0.0037 | −0.0032 [−0.0302, +0.0235] | 1.0165 | 12.0 / 13.2 % |
| **diag-350 — all TIR** | **−0.0076** | −0.0103 | −0.0116 [−0.0379, +0.0203] | 1.0116 | 11.8 / 13.2 % |

Paired over the same clips:

| Contrast | Δ | s.e. | $t$ | clips worse / better |
|---|---:|---:|---:|---:|
| diag-900 − A | **−0.1118** | 0.0092 | **−12.20** | 990 / 604 |
| diag-350 − A | **−0.1167** | 0.0095 | **−12.34** | 976 / 618 |
| diag-900 − diag-350 | −0.0049 | 0.0032 | −1.56 | 835 / 759 |

## 3. Reading

1. **The trained model collapses to the noise floor on a TIR-only input.** Both
   all-TIR variants give `r ≈ 0`, symmetric tails (about 12 % above +0.3 and
   13 % below −0.3) and a subject CI that straddles zero. The MSE is slightly
   *lower* than arm A (1.012–1.017 vs 1.041): the decoder shrinks toward the
   clip mean, which minimises MSE while carrying no phase.
2. **The collapse is not a 900-token artefact.** At the trained 350-token length
   the result is statistically the same as at 900 (Δ = −0.005, t = −1.56). What
   differs from arm A is the **composition** of the input: no RGB2D tokens and
   14 TIR tubes instead of 9 + 5 mixed.
3. **This is consistent with the content ablations, not contradictory.** In
   `resp_recon_arms_ab.md` §5, corrupting RGB2D's *content* while keeping its
   tokens changed nothing; here **removing the tokens** (and changing the TIR
   allocation) collapses the reconstruction. The trained pathway depends on the
   structure of the mixed 350-token input, not on RGB2D's respiration content.
4. **Two changes are confounded and cannot be separated within this policy.**
   diag-350 has no RGB2D *and* TIR at 39 % visible (above the 25 % cap). With
   RESP hidden, staying at TIR ≤ 9 tubes forces RGB2D to hold at least 125
   tokens — which is exactly arm A. A clean “no RGB2D, same TIR allocation”
   probe is not expressible; that is the policy arithmetic already stated in
   `resp_recon_arms_ab.md` §1.

## 4. What this does not establish

- It is **not** a 2-stream model. It is the 3-stream weights under an input they
  never saw. A 2-stream `tir+resp` arm requires a single-visual masking policy
  and a new training run.
- It does **not** show that TIR lacks the respiration signal. It shows that the
  trained pathway is **allocation-specific**: the same TIR pixels at a different
  token count/composition do not elicit it.
- It does **not** separate “no RGB2D tokens” from “TIR above its trained cap”.
- Zero-shot, fold 0, one checkpoint, one seed; no band-limited metric and no
  Stage-3 fine-tuning.
- `diag-350` uses **one** random placement draw of 14 of 36 TIR tubes (the
  per-clip values carry that sampling noise, sd ≈ 0.25). The 27 subject medians
  are symmetric around zero (12 positive, 15 negative, range −0.21 to +0.29),
  which is the signature of noise; the arm was not averaged over repeats.
- The 900-token encoder/decoder prefix is far outside the trained 350-token
  regime; attention and LayerNorm statistics are not expected to transfer.

## 5. Reproduce

```bash
python -u runners/run_eval_recon.py \
  --checkpoint <EXP_ROOT>/fold0/stage2/checkpoints/checkpoint-0149.pth \
  --partition <EXP_ROOT>/fold0/partition.json --split test \
  --override_counts 'tir=900,rgb2d=0,resp=0' \
  --batch_size 8 --repeats 1 --seed 0 --device cuda --preload \
  --out notes/eval/baselines/tir_rgb2d_resp/artifacts/diag_full_tir_no_rgb2d.json

# length-matched variant
python -u runners/run_eval_recon.py … --override_counts 'tir=350,rgb2d=0,resp=0' …
```
