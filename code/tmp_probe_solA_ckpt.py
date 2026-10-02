"""Decisive readout for the Solution A (asymmetric cross-MAE) Stage-2 checkpoint.

Reads the TRAINED model's own resp prediction (the cross-attention decoder IS the
trained TIR->resp regressor) and reports the metrics that can see phase, not the
masked MSE that once ranked a collapsed model better:

  * MAE vs the per-clip constant-0 predictor (targets are clip-z-scored)
  * phase-locking CONTRAST  |r(0)| - mean|r(|shift| >= 1.2 s)|
    (genuine locking collapses at anti-phase; the shared-slow-trend artefact
     has a floor and does NOT collapse)
  * cross-clip correlation of the predictions
  * pred std vs target std
"""
import sys
import numpy as np
import torch

sys.path.insert(0, '.')

CFG = 'configs/pretrain/stage2_local_tir_roi_crossmae.yaml'
CKPT = ('../output/pretrain/stage2_local_tir_roi_crossmae/'
        'checkpoints/checkpoint-0039.pth')
N_CLIPS = int(sys.argv[1]) if len(sys.argv) > 1 else 48
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 7

sys.argv = ['probe', '-c', CFG]
from runners import run_pretrain as rp                      # noqa: E402
from core.multimae import build_pretraining_model           # noqa: E402
from data import build_pretraining_dataset                  # noqa: E402

args = rp.get_args()
model = build_pretraining_model(args)
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
state = ckpt.get('model', ckpt)
missing, unexpected = model.load_state_dict(state, strict=False)
print(f'[ckpt] loaded  missing={len(missing)} unexpected={len(unexpected)}')
if missing:
    print('       MISSING:', missing[:5])
model.eval()

ds = build_pretraining_dataset(args)
idx = list(range(0, len(ds), STRIDE))[:N_CLIPS]
print(f'[data] {len(ds)} clips total; probing {len(idx)} (stride {STRIDE})')

preds, tgts = [], []
with torch.no_grad():
    for k in idx:
        s = ds[k]
        x = {'tir': s['tir'].unsqueeze(0),
             'resp': s['resp'].unsqueeze(0)}
        out = model(x)
        p = out['preds']['resp'].reshape(1, -1)[0].numpy()
        y = np.asarray(s['resp'], dtype=np.float64).reshape(-1)
        y = (y - y.mean()) / (y.std() + 1e-8)        # target_norm: clip
        preds.append(p)
        tgts.append(y)
P = np.stack(preds)
Y = np.stack(tgts)


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 0 else 0.0


r0 = np.array([corr(P[i], Y[i]) for i in range(len(P))])
mae = float(np.abs(P - Y).mean())
mae0 = float(np.abs(Y).mean())                       # constant-0 predictor

# phase-locking contrast: |r| as a function of time shift
shifts = list(range(-400, 401, 20))                  # +-4 s in 0.2 s steps
absr = {}
for sh in shifts:
    vals = []
    for i in range(len(P)):
        a = P[i]
        b = Y[i]
        if sh > 0:
            a, b = a[sh:], b[:-sh]
        elif sh < 0:
            a, b = a[:sh], b[-sh:]
        vals.append(abs(corr(a, b)))
    absr[sh] = float(np.mean(vals))
anti = float(np.mean([v for sh, v in absr.items() if abs(sh) >= 120]))
contrast = absr[0] - anti

# a fixed waveform broadcast to every clip (the "prior" baseline)
fixed = P.mean(axis=0)
r_fixed = float(np.mean([abs(corr(fixed, Y[i])) for i in range(len(P))]))
# cross-clip redundancy of the predictions
sub = P[:min(len(P), 24)]
cp = [abs(corr(sub[i], sub[j])) for i in range(len(sub))
      for j in range(i + 1, len(sub))]
# ceiling: best single sinusoid fitted to each target
fs = 100.0
t = np.arange(Y.shape[1]) / fs
ceil = []
for y in Y:
    best = 0.0
    for f in np.arange(0.1, 0.7, 0.01):
        s = np.sin(2 * np.pi * f * t)
        c = np.cos(2 * np.pi * f * t)
        A = np.stack([s, c], 1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        fit = A @ coef
        best = max(best, abs(corr(fit, y)))
    ceil.append(best)

print('\n================ Solution A Stage-2 checkpoint readout ================')
print(f'clips probed                 : {len(P)}')
print(f'MAE(model) vs MAE(constant-0): {mae:.4f}  vs  {mae0:.4f}   '
      f'-> {"BETTER" if mae < mae0 else "WORSE"} than predicting nothing')
print(f'pred std / target std        : {P.std():.4f} / {Y.std():.4f}')
print(f'|r(0)|                       : {absr[0]:.4f}')
print(f'|r(anti-phase >=1.2 s)|      : {anti:.4f}   (artefact floor)')
print(f'PHASE-LOCKING CONTRAST       : {contrast:+.4f}   '
      f'-> {"REAL phase locking" if contrast > 0.15 else "NO phase information"}')
print(f'|r| of ONE FIXED waveform    : {r_fixed:.4f}   (broadcast to every clip)')
print(f'cross-clip corr of preds     : {float(np.mean(cp)):+.4f}')
print(f'best single sinusoid vs target: {float(np.mean(ceil)):.4f}  '
      f'(per-target ceiling)')
print(f'median per-clip r(0)         : {float(np.median(r0)):+.4f}')
print('======================================================================')
