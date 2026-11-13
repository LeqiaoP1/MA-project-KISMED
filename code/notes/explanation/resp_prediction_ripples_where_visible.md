# Why does the RESP prediction look smooth where the signal is MASKED and rippled where it is VISIBLE?

*Plain-English explanation of the `ammm_case*` Stage-2 reconstruction figures
(`runners/run_visualize_pretrain.py`). Everything below was **measured** on the
existing fold-0 checkpoint (`stage2_ammm_hpc_tir_roi_resp_fold0/checkpoints/
checkpoint-0299.pth`) on 2026-11-13 — nothing was retrained. Numbers quoted are
for the 50 %-masked setting of `ammm_case4_half_masked.png`, on the three pinned
held-out clips F053_T3@54 s, M012_T1@8 s, M058_T3@40 s.*

## The short version

The red "prediction" curve is **not** an inpainting output at the visible
positions — it is the decoder's answer at *every* token, and the training loss
multiplies the visible ones by **exactly zero**. So the model was never asked what
to output there; the part of the curve that ripples is a **by-product of a dense
computation**, while the smooth part is the task it was actually trained on.

## 1. What the panel plots, and what "masked" means here

- `mask == 1` = **hidden** (removed from the encoder, scored by the loss).
  `mask == 0` = **visible / kept** (given to the encoder, never scored).
- The "masked input" panel shows the clip *after* masking, i.e. only the kept
  samples (blue); the hidden samples are blank (`NaN`).
- The red curve is `preds['resp']`, i.e. `heads['resp'](dec)` over **all 50
  tokens**, masked and visible alike (`core/multimae.py::forward`).

For the 50 % case (`kept_resp = 0.50`): 25 tokens (400 samples) visible,
25 tokens (400 samples) hidden; token = 16 samples = 0.16 s.

## 2. The inference pass in order (shapes for one clip)

| step | what happens | shape |
|---|---|---|
| 1 | dataset clip, already per-clip z-scored (`target_norm: none`) | `resp [1, 800]` |
| 2 | mask plan pinned (`ammm.plan_step`, `kept_resp=0.50`) → counts | 25 / 25 tokens |
| 3 | tokenise **all** 50 (`Conv1d(1→768, k=16, s=16)` + pos) | `[50, 768]` |
| 4 | gather the **visible 25** (`ids_keep`); hidden embeddings discarded | `[25, 768]` |
| 5 | shared encoder over `900 visible TIR + 25 visible RESP = 925` tokens, 12 blocks | `[925, 768]` |
| 6 | decoder input = 25 **encoder outputs** + 25 learned `mask_token`, un-shuffled to time order, `+ pos_embed`, 2 blocks | `[50, 768]` |
| 7 | `Linear(768→16)` per position → waveform in z-space | `[50, 16]` |
| 8 | `MaskedMSELoss` over `mask == 1` only | scalar |

Two facts follow directly: the hidden samples never reach the encoder in any form,
and an output vector still exists at every position because the head is a dense
linear layer with no copy/skip path.

## 3. Why the visible band is a *by-product* from the model's perspective

1. **The computation is dense.** The head is applied to all 50 positions; the model
   has no mechanism to "not answer" at a visible position.
2. **The loss multiplies those positions by zero.** `(loss·mask).sum()/(mask.sum()
   + 1e-6)` ⇒ exactly zero gradient. Over the whole run every position *was*
   masked sometimes, so the head was fitted at each position **in the mask-token
   regime only**; at inference a visible position presents it with an *encoder
   feature* instead — a regime it was never scored in. That extrapolation is the
   ripple.
3. **Visible positions are trained as context, not as outputs.** Gradient from the
   hidden queries flows back through attention into the visible tokens' decoder
   states, so the model does have an incentive to make them useful *keys/values* —
   just not to make `head(state)` equal the signal there. Measured consequence:
   level and trend at visible positions are right (token-mean sd 0.89–1.03), the
   fine texture is not (4× too rough).
4. **A classic autoencoder would not show this**, because it scores all positions
   or has an identity path. This is a masked-prediction objective, not
   reconstruction of everything.

## 4. Measurements (50 % masked, same 3 held-out clips)

| clip | group | MSE per token | intra-token sd of pred | target's own |
|---|---|---|---|---|
| F053_T3 | hidden | 5.8e-4 | 0.073 | 0.068 |
| F053_T3 | visible | 9.1e-2 | 0.292 | 0.063 |
| M012_T1 | hidden | 1.04e-3 | 0.044 | 0.039 |
| M012_T1 | visible | 1.25e-1 | 0.281 | 0.057 |
| M058_T3 | hidden | 1.56e-3 | 0.102 | 0.098 |
| M058_T3 | visible | 1.01e-1 | 0.297 | 0.093 |

**Mean: 1.06e-3 (hidden) vs 1.06e-1 (visible) — 100× worse where the signal was
given**, with a within-token spread of 0.29 against a true 0.07.

One token (16 samples, z-space), clip 0:

```
HIDDEN   tgt 0.8 0.8 0.7 0.7 0.7 0.6 0.6 0.6 0.5 0.5 0.5 0.4 0.4 0.4 0.4 0.3
HIDDEN   prd 0.8 0.8 0.8 0.7 0.7 0.6 0.6 0.6 0.6 0.5 0.5 0.5 0.4 0.4 0.4 0.3
VISIBLE  tgt -0.5 -0.6 -0.6 -0.6 -0.6 -0.6 -0.7 -0.7 -0.7 -0.7 -0.7 -0.8 -0.8 -0.8 -0.8 -0.9
VISIBLE  prd -0.9 -1.1 -0.6 -1.0 -0.9 -0.6 -0.4 -0.3 -0.9 -0.8 -0.4 -0.8 -1.1 -1.0 -0.7 -1.5
```

The degenerate configurations confirm the mechanism: `ammm_case1` (RESP fully
visible) and `ammm_case3` (nothing masked) have `mse_resp = 0.0` **exactly**, yet
the red curve still tracks at r = +0.96 — the shared decoder reproduces visible
tokens as a side effect of learning to reconstruct hidden ones.

## 5. What this does not mean

- **Not a bug, not a misalignment, not a plotting artefact.** The RESP panel plots
  the raw z-space tensor; no inverse transform is involved (unlike the TIR panel,
  which inverts `cube-z → clip-z → raw`), and the dots are the true input.
- **Not caused by the thermal stream.** The same jitter appears with TIR fully
  visible (intra-token sd 0.273 with TIR hidden vs 0.298 with TIR visible).
- **Not evidence about respiratory quality.** Read the hidden positions only; the
  trained prediction there is smooth and, at 50 % masked, ~18× better (paired
  median over 40 mask draws) than linear interpolation through the visible
  samples. The recipe for that baseline is in section 7 of this file: replace
  `preds` by `np.interp` over the kept samples and score the same masked tokens.

## 6. What would remove it

- Score it: add a reconstruction/consistency term at `mask == 0`, e.g.
  `‖pred − target‖²` over visible tokens (changes the objective — a different
  experiment, not a bug fix).
- Add a copy path: residual skip from input tokens to the head.
- Cheapest, and what MAE/VideoMAE figures do: **plot only the hidden positions**.
  In `runners/run_visualize_pretrain.py` this is a one-line change in the RESP
  panel (`prd[vis] = nan` alongside the existing `masked_y[~vis] = nan`).

## 7. Reproduce

The figure itself (from `code/`):

```bash
../.env/bin/python -u runners/run_visualize_pretrain.py \
  --checkpoint ../output/pretrain/stage2_ammm_hpc_tir_roi_resp_fold0/checkpoints/checkpoint-0299.pth \
  --sessions F053_T3@54,M012_T1@8,M058_T3@40 --split test \
  --mask_mode kept --kept_tir 0.50 --kept_resp 0.50 --seed 0 \
  --out ../output/pretrain/stage2_ammm_hpc_tir_roi_resp_fold0/ammm_case4_half_masked.png
```

The hidden/visible split (CPU, ~2 min; same dataset/selection as the runner):

```python
# from code/
import json, torch
from core.multimae import build_pretraining_model
from data import build_pretraining_dataset
from runners.run_visualize_pretrain import _pin_session_indices

ROOT = '../output/pretrain/stage2_ammm_hpc_tir_roi_resp_fold0/'
ck = torch.load(ROOT + 'checkpoints/checkpoint-0299.pth', map_location='cpu',
                weights_only=False)
a = ck['args']; a.pretrain_split, a.subjects, a.max_sessions = 'all', '', None
test = set(json.load(open(ROOT + 'fold_manifest.json'))['roles']['test'])
ds = build_pretraining_dataset(a)
pool = [i for i in range(len(ds)) if str(ds.entries[i].get('subject', '')) in test]
idx = _pin_session_indices(pool, {i: str(ds.entries[i].get('session', '')) for i in pool},
                           ['F053_T3@54', 'M012_T1@8', 'M058_T3@40'], 0,
                           starts={i: float(ds.entries[i].get('t_start', float('nan'))) for i in pool})
batch = {s: torch.stack([ds[i][s] for i in idx]).float() for s in ds[idx[0]]}
m = build_pretraining_model(a); m.load_state_dict(ck['model'], strict=False); m.eval()
m.masking, m.ammm_caps, m.ammm_min_kept = 'ammm', {'tir': 0.5, 'resp': 1.0}, 0
m.ammm_overrides = {'tir': 0.50, 'resp': 0.50}
torch.manual_seed(7)
with torch.no_grad():
    o = m(batch)
p, msk = o['preds']['resp'], o['masks']['resp'].bool()
t = batch['resp'].reshape(p.shape)
for grp, sel in (('hidden', msk), ('visible', ~msk)):
    print(grp, 'MSE/token %.2e' % float(((p - t) ** 2).mean(-1)[sel].mean()),
          'intra-token sd %.3f' % float(p[sel].std(-1).mean()))
```

See also `code/notes/explanation/resp_t3_why_some_predictions_are_better.md` for
what the *trained* part of these curves is actually worth.
