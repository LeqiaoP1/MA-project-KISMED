"""Verify the skin/CHROM results: NaN handling + the mismatched-subject control.

Found a hazard in my own gate: when `y` contains NaN, `y.std() < 1e-12` is False
(NaN comparisons are False), so the session is NOT skipped. The statistic then
returns NaN and `np.mean(nul >= NaN)` is 0.0 -- i.e. a NaN session is silently
counted as SIGNIFICANT. The face_skin rows printed `median|r| = nan`, so this
needs checking before the 35/40 can be quoted.

This script:
  1. counts NaN sessions per ROI block;
  2. re-runs the gate with NaN sessions SKIPPED (and reports how many were dropped);
  3. adds the MISMATCHED-SUBJECT control for the new signals, which is the only
     control that cannot be gamed by ROI placement.
"""
import os
import sys

sys.path.insert(0, '/mnt/f/Studium/KISMed/github/MA-project-KISMED/code')
os.chdir('/mnt/f/Studium/KISMed/github/MA-project-KISMED/code')

import numpy as np
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


z = np.load(CACHE, allow_pickle=True)
data = {k: (z[f'{k}||S'], z[f'{k}||frac'])
        for k in sorted({k.rsplit('||', 1)[0] for k in z.files})}
print('sessions: %d' % len(data))

BLOCKS = {'face_plain': slice(0, 3), 'face_skin': slice(3, 6),
          'bg': slice(6, 9), 'full': slice(9, 12)}
print('\n=== NaN audit (per ROI block) ===')
print('%-12s %12s %12s' % ('block', 'sessions w/NaN', 'mean NaN frames'))
for nm, cols in BLOCKS.items():
    bad = 0
    fracs = []
    for k, (S, fr) in data.items():
        nanrows = ~np.isfinite(S[:, cols]).all(1)
        if nanrows.any():
            bad += 1
            fracs.append(nanrows.mean())
    print('%-12s %12d %12.4f' % (nm, bad, np.mean(fracs) if fracs else 0.0))

BP = {}
for name, (S, fr) in data.items():
    subj, task = name.split('_')
    f = f'{ROOT}/Physiology/{subj}/{task}/BP_mmHg.txt'
    if not os.path.exists(f):
        continue
    n = S.shape[0]
    bp = _load_1d(f)
    if bp.shape[0] < int((n - 1) * RESP_FS / FPS) + 1:
        continue
    BP[name] = zs(bandpass(np.interp(np.arange(n) * (RESP_FS / FPS),
                                     np.arange(bp.shape[0]), bp.astype(np.float64))))
rng = np.random.default_rng(53)
SURN = {}
for name, b in BP.items():
    Xt = np.fft.rfft(b); amag = np.abs(Xt)
    s = np.empty((NSUR, len(b)), np.float32)
    for i in range(NSUR):
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, len(Xt))); ph[0] = 1.0
        s[i] = np.fft.irfft(amag * ph, n=len(b))
    SURN[name] = s

print('\n=== GATE, NaN sessions SKIPPED, matched pairing ===')
print('%-24s %10s %10s %10s' % ('signal / method', 'rate', 'median|r|', 'dropped'))
gate = {}
for nm, cols in BLOCKS.items():
    for meth in ('GR', 'CHROM', 'POS'):
        o, p, drop = [], [], 0
        for name in BP:
            y = methods(data[name][0][:, cols])[meth]
            if not np.isfinite(y).all() or y.std() < 1e-12:
                drop += 1
                continue
            C = zs(y)[:, None].astype(np.float32)
            obs = float(stat(BP[name].astype(np.float32), C)[0])
            nul = stat(SURN[name], C).ravel()
            o.append(obs); p.append(float(np.mean(nul >= obs)))
        p = np.array(p)
        gate[(nm, meth)] = (o, p)
        print('%-24s %10s %10.1f %10d' %
              (f'{nm} / {meth}', '%d/%d' % (int((p < .05).sum()), len(p)),
               np.median(o) if o else np.nan, drop))

print('\n=== MISMATCHED-SUBJECT CONTROL (must fire at ~5%) ===')
names = list(BP)
for nm, meth in (('face_plain', 'GR'), ('face_skin', 'CHROM'), ('face_skin', 'POS'),
                 ('face_plain', 'CHROM'), ('bg', 'CHROM')):
    mm = []
    for i, n1 in enumerate(names):
        y1 = methods(data[n1][0][:, BLOCKS[nm]])[meth]
        if not np.isfinite(y1).all():
            continue
        for j in (1, 7, 13, 19):
            n2 = names[(i + j) % len(names)]
            if n2 == n1:
                continue
            b2 = BP[n2]
            m = min(len(y1), len(b2))
            if m < 400:
                continue
            yy = y1[:m]
            if yy.std() < 1e-12:
                continue
            C = zs(yy)[:, None].astype(np.float32)
            obs = float(stat(b2[:m].astype(np.float32), C)[0])
            Xt = np.fft.rfft(b2[:m]); amag = np.abs(Xt)
            s = np.empty((NSUR, m), np.float32)
            for t in range(NSUR):
                ph = np.exp(1j * rng.uniform(0, 2 * np.pi, len(Xt))); ph[0] = 1.0
                s[t] = np.fft.irfft(amag * ph, n=m)
            mm.append(float(np.mean(stat(s, C).ravel() >= obs)))
    mm = np.array(mm)
    print('  %-24s n=%-4d  p<0.05: %3d (%.0f%%)  median|r| %.3f' %
          (f'{nm} / {meth}', len(mm), int((mm < .05).sum()),
           100 * np.mean(mm < .05), np.nanmedian(
               [g for g in gate.get((nm, meth), ([np.nan],))[0]] or [np.nan])))
print("""
READ: the matched rate must clearly exceed the mismatched rate for the SAME
signal. If the NaN audit shows dropped>0, compare against the dropped count.
""")
