# MultiMAE-Figure-6-style panel for the three-stream checkpoint (2026-11-24)

**Figure**: `multimae_style_kept_density.png` (in this folder).
**Script**: `scripts/project/figure_multimae_style.py` (now selects the allocation
preset from the checkpoint's own stream set).
**Checkpoint**: `checkpoint-0149.pth` (fold-0 selection, epoch 149).
**Clip**: subject `F053`, session `F053_T9`, `t_start = 20 s` — chosen with
`--seed 0` from the **1,594 held-out clips** of the fold-0 test subjects
(`--partition …/fold0/partition.json`), i.e. never seen in training or validation.

## 1. What the figure shows

Rows: the three streams (`tir`, `rgb2d`, `resp`). Columns: the original clip,
then for each allocation `[masked input | prediction]`.

| Column | Kept tokens (`tir`/`rgb2d`/`resp`) | Visual density | Resp density | Resolved mode |
|---|---|---|---|---|
| resp-max | 125 / 100 / 125 | 5 + 4 of 36 tubes (13.9 % + 11.1 %) | 125 of 250 (50 %) | 2 |
| mid | 150 / 150 / 50 | 6 + 6 of 36 tubes (16.7 %) | 50 of 250 (20 %) | 2 |
| dense visual | 175 / 175 / 0 | 7 + 7 of 36 tubes (19.4 %) | 0 (virtual sensing) | 1 |

## 2. Why this allocation sweep

The panel holds the constant budget at 350 and moves it between the visuals and
respiration. The achievable range is fixed by the three-stream policy:

- Respiration caps at 125 tokens, so at least 225 tokens must be visual. The
  sparsest expressible visual column is therefore **4 + 5 tubes** (225 tokens).
- The densest visual column is the Mode-1 split 7 + 7 = 350, i.e. respiration
  fully hidden. A 33 %-of-tokens visual column (the paper's 1/3 point) is not
  expressible: `core/multimae.py` asserts the encoder input has exactly
  `n_budget` tokens.

## 3. What the panel says

- **The visuals sharpen with visible density.** At 4–5 tubes the RGB2D
  prediction is a blurred face without texture; at 6–7 tubes the face, the head
  pose and the thermal nostrils sharpen. Reconstruction rises monotonically with
  the number of visible tubes.
- **Respiration tracks its own visible tokens tightly.** At 125 kept tokens
  (resp-max) the red prediction follows the black kept dots almost exactly; at 50
  kept (mid) it follows the slow waveform and interpolates across the longer
  gaps. This is the local-interpolation regime.
- **With respiration fully hidden**, the prediction is noisy and amplitude-shrunk
  but follows the target's two-cycle envelope over the full 8 s. The masked-only
  statistic for this condition over the whole test set is in
  `resp_recon_arms_ab.md`; this column is one draw on one clip.
- **The TIR dense column is the deployed condition.** The right-most pair is
  exactly the sensor-free setting the model is meant to serve: no respiration
  token reaches the encoder.

## 4. Honest limits

- One clip, one seed, one checkpoint (fold 0, epoch 149). The right-most column
  is a single draw of the placement RNG.
- The left panel of each pair blanks the masked patches; the prediction panel
  overlays the true values where tokens were kept (`reconstruct_visual`), as in
  the paper.
- The `resp` row's "prediction" panel overlays the whole waveform, including the
  positions that were **visible**; only the blank gaps are reconstructions.
  The masked-only number in `resp_recon_arms_ab.md` is the honest one.
- Reproduce with:

```bash
python -m scripts.project.figure_multimae_style \
  --checkpoint /work/scratch/ne95ocyg/pretrain/adr005_3stream_fold0/fold0/stage2/checkpoints/checkpoint-0149.pth \
  --partition /work/scratch/ne95ocyg/pretrain/adr005_3stream_fold0/fold0/partition.json \
  --seed 0 --device cuda \
  --out code/notes/eval/baselines/tir_rgb2d_resp/multimae_style_kept_density.png
```
