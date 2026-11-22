# MultiMAE-Figure-6-style panel for the four-stream checkpoint (2026-11-22)

**Figure**: `multimae_style_kept_density.png` (in this folder).
**Script**: `scripts/project/figure_multimae_style.py`.
**Checkpoint**: `checkpoint-0149.pth` (fold-0 selection, epoch 149).
**Clip**: subject `M012`, session `M012_T3`, `t_start = 8 s` — chosen with
`--seed 0` from the **1,561 held-out clips** of the fold-0 test subjects
(`--partition …/fold0/partition.json`), i.e. never seen in training or validation.

## 1. What the figure shows

Rows: the four streams (`tir`, `rgb2d`, `bp`, `resp`). Columns: the original clip,
then for each allocation `[masked input | prediction]`.

| Column | Kept tokens (`tir`/`rgb2d`/`bp`/`resp`) | Visual density | Physio density | Mode |
|---|---|---|---|---|
| sparse | 75 / 75 / 100 / 100 | 3 of 36 tubes (8.3 %) | 100 of 250 (40 %) | 2 |
| mid | 150 / 150 / 25 / 25 | 6 of 36 tubes (16.7 %) | 25 of 250 (10 %) | 2 |
| dense visual | 175 / 175 / 0 / 0 | 7 of 36 tubes (19.4 %) | 0 (virtual sensing) | 1 |

## 2. Why the sweep is an ALLOCATION, not an absolute density

MultiMAE's Figure 6 varies the *number* of visible patches (1/12, 1/6, 1/3).
That is not expressible here: `core/multimae.py` asserts the encoder input has
exactly `n_budget` tokens, so the total visible count is fixed at 350 and only
its distribution across streams can vary. The three columns therefore hold the
budget constant and move it between the visuals and the physiologies. The
achievable visual range is also bounded by the policy: 2–9 of 36 tubes, i.e.
5.6 %–25 %, so the paper's 33 % column has no counterpart.

## 3. What the panel says

- **Visuals behave like the paper's figure**: at 3 tubes the RGB2D prediction is
  a blurred face with no texture; at 6 and 7 tubes the face and the thermal
  pattern sharpen and the head pose is recovered. Reconstruction quality rises
  monotonically with the number of visible tubes, which is the qualitative claim
  of MultiMAE's Figure 6 reproduced on our geometry.
- **Physiology does not follow the visual density at all.** `resp` is recovered
  when 100 of its own 250 tokens are visible (the red curve tracks the black kept
  dots), degrades at 25 kept, and is **noise at 0 kept** — even though the visuals
  are at their densest (7 tubes each). `bp` is essentially flat at every density.
  The predictor for a 1-D stream therefore rides on *its own visible tokens*
  (local interpolation); the visual streams contribute almost nothing.
- That is the same finding as the quantitative probe in
  `resp_recon_arms_ab.md` (`r_masked ≈ 0` with RESP fully hidden, at the maximum
  visual density this checkpoint supports), now visible qualitatively in one
  picture: the "dense visual / physio 0" column is exactly the virtual-sensing
  condition, and the RESP prediction is a noise floor.

## 4. Honest limits

- One clip, one seed, one checkpoint (fold 0, epoch 149). The right-most column is
  a single draw of the placement RNG.
- The left panel of each pair blanks the masked patches; the prediction panel
  overlays the true values where tokens were kept (`reconstruct_visual`), as in
  the paper.
- `bp` at every density sits near a constant: the four-stream arm never learned a
  BP waveform predictor either (see `mse_bp` ≈ 2.0 in `fold0_pretrain_150ep.md`).
- Reproduce with:
  `python -m scripts.project.figure_multimae_style --checkpoint <ckpt> --partition <partition.json> --seed 0`
