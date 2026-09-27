"""Is the skin/CHROM cardiac signal BLOOD or MOTION?

Face-specific and control-validated (90% vs 5% mismatched). The remaining
question is mechanism: CHROM/POS are designed to cancel motion, but a face ROI
still moves. Project out head pose (3) + landmark centroid (2) + inter-ocular
scale + the full-frame mean, then re-test.

  survives -> the signal is blood volume (skin colour), i.e. genuine rPPG
  dies     -> it was head/camera motion dressed up as a pulse

Also re-checks the ballistocardiogram route directly (pose/landmarks vs BP).
"""
import os
import sys

sys.path.insert(0, '/mnt/f/Studium/KISMed/github/MA-project-KISMED/code')
os.chdir('/mnt/f/Studium/KISMed/github/MA-project-KISMED/code')

import numpy as np
from data.rgb_features import parse_2d_features, find_2d_features
from data.tir_resp_dataset import _load_1d

ROOT = '../data/raw/BP4D'
FPS, RESP_FS = 25.0, 1000.0
BAND = (0.9, 3.0)
NSUR = 200
LAGS = np.arange(-50, 51, 2)
CACHE = '/tmp/rppg_skin_v1.npz'


def bandpass(x, lo=BAND[0], hi=BAND[1], axis=0):
    X = np.fft.rfft(x, axis=axis)
    f = np.fft.rfftfreq(x.shape[axis], d=1.0 / FPS)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, n=x.shape[axis], axis=axis)


def zs(x, axis=0):
    return (x - x.mean(axis=axis, keepdims=True)) / (x.std(axis=axis, keepdims=True) + 1e-12)


def stat(b, C, lags=LAGS):
    one = b.ndim == 1
    if one:
        b = b[None, :]
    n = C.shape[0]
    best = np.zeros((b.shape[0], C.shape[1]), np.float32)
    for L in lags:
        np.maximum(best, np.abs(np.roll(b, L, axis=1) @ C) / n, out=best)
    return best[0] if one else best


def methods(rgb):
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    Rn = R / (R.mean() + 1e-9); Gn = G / (G.mean() + 1e-9); Bn = B / (B.mean() + 1e-9)
    r_, g_, b_ = bandpass(Rn), bandpass(Gn), bandpass(Bn)
    X = 3 * r_ - 2 * g_
    Y = 1.5 * r_ + g_ - 1.5 * b_
    S1 = g_ - b_
    S2 = -2 * r_ + g_ + b_
    return {'GR': g_ - r_,
            'CHROM': X - (X.std() / (Y.std() + 1e-12)) * Y,
            'POS': S1 + (S1.std() / (S2.std() + 1e-12)) * S2}


def regress_out(y, X):
    A = np.column_stack([np.ones(len(y))] + [X[:, i] for i in range(X.shape[1])])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


z = np.load(CACHE, allow_pickle=True)
data = {k: (z[f'{k}||S'], z[f'{k}||frac'])
        for k in sorted({k.rsplit('||', 1)[0] for k in z.files})}

BP, NUIS = {}, {}
for name, (S, fr) in data.items():
    subj, task = name.split('_')
    f = f'{ROOT}/Physiology/{subj}/{task}/BP_mmHg.txt'
    p2 = find_2d_features(ROOT, subj, task)
    if not os.path.exists(f) or p2 is None:
        continue
    n = S.shape[0]
    bp = _load_1d(f)
    if bp.shape[0] < int((n - 1) * RESP_FS / FPS) + 1:
        continue
    BP[name] = zs(bandpass(np.interp(np.arange(n) * (RESP_FS / FPS),
                                     np.arange(bp.shape[0]), bp.astype(np.float64))))
    pts, pose, frame, _ = parse_2d_features(p2)
    m = min(n, len(frame))
    P = np.where(np.isfinite(pts[:m]).all(axis=(1, 2))[:, None, None], pts[:m], np.nan)
    cen = np.nanmean(P, axis=1)
    iod = np.linalg.norm(P[:, 19:25].mean(1) - P[:, 25:31].mean(1), axis=1)
    for arr in (cen, iod[:, None]):
        bad = ~np.isfinite(arr).all(1)
        if bad.any():
            good = np.flatnonzero(~bad)
            for c in range(arr.shape[1]):
                arr[bad, c] = np.interp(np.flatnonzero(bad), good, arr[good, c])
    N = np.column_stack([np.nan_to_num(pose[:m]), cen, np.log(np.maximum(iod, 1e-6)),
                         S[:m, 9:12].mean(1)])
    NUIS[name] = zs(bandpass(N), axis=0)
    BP[name] = BP[name][:m]

rng = np.random.default_rng(61)
SURN = {}
for name, b in BP.items():
    Xt = np.fft.rfft(b); amag = np.abs(Xt)
    s = np.empty((NSUR, len(b)), np.float32)
    for i in range(NSUR):
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, len(Xt))); ph[0] = 1.0
        s[i] = np.fft.irfft(amag * ph, n=len(b))
    SURN[name] = s

print('=== face_skin signal: RAW vs MOTION-REGRESSED ===')
print('%-28s %10s %10s' % ('signal', 'rate', 'median|r|'))
for meth in ('CHROM', 'POS'):
    for mode in ('raw', 'regressed'):
        o, p, drop = [], [], 0
        for name in BP:
            S, fr = data[name]
            m = len(BP[name])
            y = methods(S[:m, 3:6])[meth]
            if not np.isfinite(y).all() or y.std() < 1e-12:
                drop += 1
                continue
            if mode == 'regressed':
                y = regress_out(zs(y), NUIS[name])
                y = y / (y.std() + 1e-12)
            C = zs(y)[:, None].astype(np.float32)
            obs = float(stat(BP[name].astype(np.float32), C)[0])
            nul = stat(SURN[name], C).ravel()
            o.append(obs); p.append(float(np.mean(nul >= obs)))
        p = np.array(p)
        print('%-28s %10s %10.1f  (dropped %d)' %
              (f'face_skin/{meth} {mode}', '%d/%d' % (int((p < .05).sum()), len(p)),
               np.median(o), drop))

print('\n=== ballistocardiogram route: pose/landmark MOTION vs BP ===')
rows = []
for name in BP:
    S, fr = data[name]
    m = len(BP[name])
    C = np.ascontiguousarray(NUIS[name][:, :6], dtype=np.float32)
    obs = float(stat(BP[name].astype(np.float32), C).max())
    nul = stat(SURN[name], C).max(axis=1)
    rows.append((obs, float(np.mean(nul >= obs))))
rows = np.array(rows)
print('  n=%d  p<0.05: %d (chance %.1f)  median|r| %.3f' %
      (len(rows), int((rows[:, 1] < .05).sum()), 0.05 * len(rows), np.median(rows[:, 0])))
print("""
READ: if face_skin/CHROM survives motion regression at ~the same rate, the signal
is blood volume (skin colour) -> genuine rPPG. If it collapses to chance, it was
head/camera motion. The motion route alone was already at chance earlier.
""")
