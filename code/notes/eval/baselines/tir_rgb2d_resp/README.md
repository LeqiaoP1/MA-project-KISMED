# Three-stream fold-0 baseline — evaluation record (2026-11-24)

This folder evaluates the ADR-005 three-stream Stage-2 checkpoint
(`tir`, `rgb2d`, `resp`; fold 0, epoch 149) without fine-tuning. It answers the
two questions that motivated the evaluation:

1. **Is respiration reconstructed obviously and reliably?**
2. **Does RGB2D help the reconstruction?**

## 0. Run identity

| Item | Value |
|---|---|
| Checkpoint | `/work/scratch/ne95ocyg/pretrain/adr005_3stream_fold0/fold0/stage2/checkpoints/checkpoint-0149.pth` |
| Model | `project_multimae_base`, 105,581,600 params, VideoMAE k400 → TIR adapter |
| Geometry | 8 s, 50 frames, tubelet `2×16×16`, 96 px, 1000 Hz, 350-token budget |
| Curriculum | `p = 0.10` epochs 0–49, `p = 0.50` epochs 50–149 |
| Split | fold-0 `test` role: 1,594 clips / 201 sessions / **27** subjects (`F027` has no usable clip) |

## 1. Is respiration reconstruction obvious and reliable?

**It depends on whether the sensor is present.**

| Condition | `r_masked` | MSE (z²) |
|---|---:|---:|
| Respiration **visible** (125 of 250 tokens; 210 clips) | **+0.996** (100 % of clips > 0.9) | 0.009 |
| Respiration **hidden** — TIR-max (1,594 clips) | **+0.109**, subject median +0.113 [+0.078, +0.136] | 1.041 |
| Respiration **hidden** — RGB2D-max (1,594 clips) | **+0.080**, subject median +0.092 [+0.071, +0.108] | 1.059 |

- **Inpainting: yes.** With some respiration tokens visible, the masked gaps are
  reconstructed almost perfectly.
- **Virtual sensing: real but weak.** Both hidden-respiration arms have
  subject-level CIs that exclude zero, and the per-clip tails are asymmetric
  (about one clip in four reaches `r > 0.3`, one in twelve `r < −0.3`). But the
  effect is ≈ 1 % of variance, the amplitude is heavily shrunk (`mse ≈ 1`), and
  it is heterogeneous across subjects. It is a lead, not a usable waveform.

Details: `resp_recon_arms_ab.md` §§2–3.

## 2. Does RGB2D help?

**No — not for this target, at inference.**

**Budget allocation (A vs B).** Both arms contain both visual streams; only the
split differs, because a zero-token visual is not expressible (2-tube floor, and
one stream cannot hold the 350-token budget under the 25 % cap):

| Arm | TIR | RGB2D | resp | `r_masked` |
|---|---:|---:|---:|---:|
| A — TIR-max | 225 (9 tubes) | 125 (5) | 0 | **+0.109** |
| B — RGB2D-max | 125 (5) | 225 (9) | 0 | +0.080 |

Paired Δ (B − A) = **−0.0292 ± 0.0044**, t = −6.62; 876 of 1,594 clips worse,
19 of 27 subjects worse. Moving budget from TIR to RGB2D **hurts**.

**Content ablation (same plan and masks, corrupt only the input).**

| Contrast | Δ | s.e. | t |
|---|---:|---:|---:|
| RGB2D rolled (temporal alignment destroyed) | −0.0012 | 0.0026 | −0.47 |
| RGB2D zeroed (content removed) | −0.0028 | 0.0064 | −0.45 |
| **TIR rolled** (positive control) | **−0.0539** | 0.0054 | **−9.96** |

RGB2D carries 9 of the 14 visual tubes yet contributes nothing measurable; TIR's
5 tubes carry the effect. Details: `resp_recon_arms_ab.md` §§4–5.

**What this does not say.** The RGB2D tokenizer is random-initialised (ADR-005
I2), so “RGB2D is not used by this model” is not “RGB2D carries no information”.
A training-time presence/absence test needs a matched 2-stream arm, which does
not exist.

## 3. Contents

| File | What it is |
|---|---|
| `fold0_pretrain_150ep.md` | The run: config, 13-link chain, 150-epoch curves, artefacts, gates |
| `kept_ratio_distribution.md` | Exact mask-policy replay: kept ratios, mode split, curriculum, Criterion-1 checks |
| `kept_ratio_hist.png`, `kept_ratio_by_mode.png` | Figures for the replay |
| `multimae_style_panel.md`, `multimae_style_kept_density.png` | Qualitative single-clip reconstruction at three allocations |
| `resp_recon_arms_ab.md` | Quantitative held-out respiration probe (the two questions above) |
| `hpc_throughput.md` | Measured execution shape of the 4×A100 chain |
| `policy_replay.json` | Machine-readable replay record |
| `artifacts/` | Per-clip reports: reference, A, B, and the three ablations; paired contrasts |

## 4. Reproduce

From `code/`:

```bash
# mask-policy replay + figures
python -m scripts.project.mask_policy_report \
  --checkpoint <EXP_ROOT>/fold0/stage2/checkpoints/checkpoint-0149.pth \
  --logs 'logs/adr001_pretrain_571*.out' \
  --out-dir notes/eval/baselines/tir_rgb2d_resp

# qualitative panel
python -m scripts.project.figure_multimae_style \
  --checkpoint <EXP_ROOT>/fold0/stage2/checkpoints/checkpoint-0149.pth \
  --partition <EXP_ROOT>/fold0/partition.json --seed 0 --device cuda \
  --out notes/eval/baselines/tir_rgb2d_resp/multimae_style_kept_density.png

# virtual-sensing arm A (swap the kept_* flags and ablation for the other arms)
python -u runners/run_eval_recon.py \
  --checkpoint <EXP_ROOT>/fold0/stage2/checkpoints/checkpoint-0149.pth \
  --partition <EXP_ROOT>/fold0/partition.json --split test \
  --kept_tir 1.0 --kept_rgb2d 0.0 --kept_resp 0.0 --resp_probe hidden \
  --repeats 5 --batch_size 16 --seed 0 --device cuda --preload \
  --out notes/eval/baselines/tir_rgb2d_resp/artifacts/armA_tir_max_resp_hidden.json

# paired contrast of two reports
python -m scripts.project.compare_recon_arms \
  --a notes/eval/baselines/tir_rgb2d_resp/artifacts/armA_tir_max_resp_hidden.json \
  --b notes/eval/baselines/tir_rgb2d_resp/artifacts/armB_rgb2d_max_resp_hidden.json \
  --out notes/eval/baselines/tir_rgb2d_resp/artifacts/armA_vs_armB.json
```

## 5. Limits

- Zero-shot Stage-2 only: no Stage-3 fine-tuning, no band-limited (0.1–0.6 Hz)
  subject-level headline, no rate metric.
- Fold 0 only; one checkpoint; one seed; 27 of 28 held-out subjects.
- The target is a per-clip z-score; `r_masked` is on masked positions only.
- Subject CIs use the median over clips (the per-session step is not applied).
- The RGB2D tokenizer is random-initialised; TIR is the only stream that
  inherited the VideoMAE kernel.
