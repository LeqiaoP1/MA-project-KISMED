"""HR comparison with ONE consistent truth -> rows are directly comparable.

Earlier runs used different truths (noisy instantaneous peak-HR for the face vs
the smoothed shipped rate for the synthetic), so their "skill" numbers could not
be ranked against each other. This uses the SAME smoothed reference for every
signal:

  SYNTH  a clean pulse whose instantaneous frequency IS the truth  = CEILING
  BP     the arterial pressure waveform                            = POS CONTROL
  faceGR / faceG / noseGR / bgGR / fullGR                          = candidates

Only if faceGR approaches the SYNTH row is the face carrying a usable pulse.
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
WINS = (20, 30, 40, 60)
SESSIONS = [f'{s}_T{t}' for s in ('F001', 'F002', 'F003', 'F004') for t in range(1, 11)]
CACHE = '/tmp/rgb_roi_means_v3.npz'


def bandpass(x, lo=BAND[0], hi=BAND[1], fs=FPS, axis=0):
    X = np.fft.rfft(x, axis=axis)
    f = np.fft.rfftfreq(x.shape[axis], d=1.0 / fs)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, n=x.shape[axis], axis=axis)


def win_hr_sub(x, win_s, hop_s):
    x = np.nan_to_num(np.asarray(x, float))
    x = x - x.mean()
    win, hop = int(win_s * FPS), int(hop_s * FPS)
    n = len(x)
    if n < win:
        return np.array([], int), np.array([])
    w = np.hanning(win)
    f = np.fft.rfftfreq(win, d=1.0 / FPS)
    df = f[1] - f[0]
    sel = np.flatnonzero((f >= BAND[0]) & (f <= BAND[1]))
    out, ctr = [], []
    for a in range(0, n - win + 1, hop):
        seg = x[a:a + win]
        if seg.std() < 1e-12:
            continue
        P = np.abs(np.fft.rfft(seg * w)) ** 2
        k = sel[np.argmax(P[sel])]
        if 0 < k < len(P) - 1:
            a1, b1, c1 = P[k - 1], P[k], P[k + 1]
            den = a1 - 2 * b1 + c1
            delta = float(np.clip(0.5 * (a1 - c1) / den, -0.5, 0.5)) if abs(den) > 1e-30 else 0.0
        else:
            delta = 0.0
        out.append((k + delta) * df * 60.0)
        ctr.append(a + win // 2)
    return np.array(ctr, int), np.array(out)


z = np.load(CACHE, allow_pickle=True)
tmp = {}
for key in z.files:
    nm, fld = key.rsplit('||', 1)
    tmp.setdefault(nm, {})[fld] = z[key]
data = {k: (v['S'], [str(x) for x in v['keys']]) for k, v in tmp.items()}

SIG = {}
for name, (S, keys) in data.items():
    subj, task = name.split('_')
    bp_f = f'{ROOT}/Physiology/{subj}/{task}/BP_mmHg.txt'
    pr_f = f'{ROOT}/Physiology/{subj}/{task}/Pulse Rate_BPM.txt'
    if not (os.path.exists(bp_f) and os.path.exists(pr_f)):
        continue
    n = S[0].shape[0]
    bp = _load_1d(bp_f).astype(np.float64)
    if bp.shape[0] < int((n - 1) * RESP_FS / FPS) + 1:
        continue
    pr = _load_1d(pr_f)
    fhz = np.interp(np.arange(n) * (RESP_FS / FPS), np.arange(pr.shape[0]),
                    pr.astype(np.float64)) / 60.0
    k = int(2.0 * FPS)
    truth = np.convolve(fhz, np.ones(k) / k, mode='same') * 60.0
    fhz_c = np.clip(np.convolve(fhz, np.ones(k) / k, mode='same'),
                    BAND[0] * 1.05, BAND[1] * 0.95)
    phase = 2 * np.pi * np.cumsum(fhz_c) / FPS
    syn = (np.sin(phase) + 0.35 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
           + 0.30 * bandpass(np.random.default_rng(5).standard_normal(n)))
    idx = {kk: i for i, kk in enumerate(keys)}
    gr = {}
    for kk in keys:
        g = S[idx[kk]][:, 1] / (S[idx[kk]][:, 1].mean() + 1e-9)
        r = S[idx[kk]][:, 0] / (S[idx[kk]][:, 0].mean() + 1e-9)
        gr[kk] = g - r
    SIG[name] = dict(
        n=n, truth=truth, syn=syn, bp=bandpass(np.interp(
            np.arange(n) * (RESP_FS / FPS), np.arange(bp.shape[0]), bp)),
        faceGR=gr['face_all'], faceG=S[idx['face_all']][:, 1],
        noseGR=gr['nose_mouth'],
        bgGR=np.nanmean([gr[k] for k in ('bg_left', 'bg_right', 'bg_topleft',
                                         'bg_topright')], axis=0),
        fullGR=gr['full_frame'])

print('%-13s %5s | %8s %8s %8s %6s | %8s' %
      ('signal', 'win', 'corr', 'MAE', 'baseMAE', 'n', 'skill'))
print('-' * 68)
for win_s in WINS:
    rows = {}
    for key in ('syn', 'bp', 'faceGR', 'faceG', 'noseGR', 'bgGR', 'fullGR'):
        R = []
        for name, d in SIG.items():
            if d['n'] < int(win_s * FPS) + 20:
                continue
            ctr, _ = win_hr_sub(d['bp'], win_s, max(5, win_s // 4))
            if len(ctr) < 3:
                continue
            _, est = win_hr_sub(d[key], win_s, max(5, win_s // 4))
            if len(est) != len(ctr):
                m = min(len(est), len(ctr))
                est, tw = est[:m], d['truth'][ctr[:m]]
            else:
                tw = d['truth'][ctr]
            if len(est) < 3 or est.std() < 1e-9:
                continue
            R.append((float(np.corrcoef(est, tw)[0, 1]),
                      float(np.mean(np.abs(est - tw))),
                      float(np.mean(np.abs(tw.mean() - tw)))))
        rows[key] = np.array(R) if R else None
    for key, lab in (('syn', 'SYNTH(CEIL)'), ('bp', 'BP(POSCTRL)'),
                     ('faceGR', 'face GR'), ('faceG', 'face G'),
                     ('noseGR', 'nose GR'), ('bgGR', 'bg GR'), ('fullGR', 'full GR')):
        A = rows[key]
        if A is None or len(A) < 3:
            continue
        mae, base = A[:, 1].mean(), A[:, 2].mean()
        print('%-13s %5d | %+8.3f %8.2f %8.2f %6d | %+8.3f' %
              (lab, win_s, A[:, 0].mean(), mae, base, len(A),
               1 - mae / base if base > 0 else np.nan))
    print()
print("""READ: compare each row against SYNTH(CEIL) at the same window. Face GR is
only carrying a usable pulse if it approaches SYNTH. bp/baseMAE are the same for
every row within a window, so skill is directly comparable down a column.""")
