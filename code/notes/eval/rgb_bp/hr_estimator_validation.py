"""Is the HR ESTIMATOR capable at all?  Validate on a synthetic pulse.

The v1/v3 HR tests are inconclusive because the BP positive control scored
skill ~0.01-0.13 -- the arterial pressure waveform itself came out no better
than a constant. Either the estimator is too weak, or the truth/BP are noisy.

This isolates it. A synthetic pulse is built with harmonics whose instantaneous
frequency IS the measured HR trajectory, so the answer is known exactly:
  - estimator recovers it well  -> estimator is fine, so any BP/face failure is real
  - estimator fails on it too   -> the estimator is the binding constraint and the
                                  BP/face HR numbers mean nothing either way

Also diagnoses the peak-detection ground truth, which correlates only +0.408 with
the shipped Pulse Rate_BPM -- suspiciously low for a systolic-peak method.
"""
import os
import sys

sys.path.insert(0, '/mnt/f/Studium/KISMed/github/MA-project-KISMED/code')
os.chdir('/mnt/f/Studium/KISMed/github/MA-project-KISMED/code')

import numpy as np
from scipy.signal import find_peaks
from data.tir_resp_dataset import _load_1d

ROOT = '../data/raw/BP4D'
FPS, RESP_FS = 25.0, 1000.0
BAND = (0.9, 3.0)
WINS = (20, 30, 40, 60)
SESSIONS = [f'{s}_T{t}' for s in ('F001', 'F002', 'F003', 'F004') for t in range(1, 11)]


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


# ---------------------------------------------------------------- diagnostics
print('=== ground-truth diagnostics ===')
pk_stats = []
for name in SESSIONS:
    subj, task = name.split('_')
    f = f'{ROOT}/Physiology/{subj}/{task}/BP_mmHg.txt'
    if not os.path.exists(f):
        continue
    x = _load_1d(f).astype(np.float64)
    x = x - np.median(x)
    pk, _ = find_peaks(x, distance=int(0.30 * RESP_FS), prominence=5.0)
    pr = _load_1d(f'{ROOT}/Physiology/{subj}/{task}/Pulse Rate_BPM.txt')
    dur = x.shape[0] / RESP_FS
    expected = pr.mean() / 60.0 * dur
    pk_stats.append((name, dur, len(pk), expected, len(pk) / expected if expected else np.nan))
S = np.array([[p[2], p[3], p[4]] for p in pk_stats])
print('  sessions=%d  detected peaks %.0f vs expected %.0f  ratio %.3f (sd %.3f)'
      % (len(pk_stats), S[:, 0].mean(), S[:, 1].mean(), S[:, 2].mean(), S[:, 2].std()))
print('  ratio far from 1.0 => peak detection is missing or inventing beats')

# ------------------------------------------------- synthetic estimator check
print('\n=== ESTIMATOR VALIDATION on a synthetic pulse (known rate) ===')
print('  target = the shipped Pulse Rate_BPM trajectory (smoothed)')
print('%-8s %6s | %8s %8s %8s %8s %8s' %
      ('window', 'n', 'corr', 'MAE', 'baseMAE', 'sd_est', 'sd_truth'))
for win_s in WINS:
    R = []
    for name in SESSIONS:
        subj, task = name.split('_')
        f = f'{ROOT}/Physiology/{subj}/{task}/BP_mmHg.txt'
        if not os.path.exists(f):
            continue
        bp = _load_1d(f).astype(np.float64)
        n = int(bp.shape[0] * FPS / RESP_FS)
        if n < int(win_s * FPS) + 20:
            continue
        pr = _load_1d(f'{ROOT}/Physiology/{subj}/{task}/Pulse Rate_BPM.txt')
        # instantaneous frequency trajectory on the video grid, lightly smoothed
        fhz = np.interp(np.arange(n) * (RESP_FS / FPS), np.arange(pr.shape[0]),
                        pr.astype(np.float64)) / 60.0
        k = int(2.0 * FPS)
        fhz = np.convolve(fhz, np.ones(k) / k, mode='same')
        fhz = np.clip(fhz, BAND[0] * 1.05, BAND[1] * 0.95)
        phase = 2 * np.pi * np.cumsum(fhz) / FPS
        syn = (np.sin(phase) + 0.35 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
               + 0.30 * bandpass(np.random.default_rng(5).standard_normal(n)))
        ctr, est = win_hr_sub(syn, win_s, max(5, win_s // 4))
        if len(ctr) < 3:
            continue
        truth = fhz[ctr] * 60.0
        R.append((float(np.corrcoef(est, truth)[0, 1]),
                  float(np.mean(np.abs(est - truth))),
                  float(np.mean(np.abs(truth.mean() - truth))),
                  float(est.std()), float(truth.std())))
    A = np.array(R)
    if len(A) < 3:
        print('%-8d %6d  (too few sessions)' % (win_s, len(A)))
        continue
    print('%-8d %6d | %+8.3f %8.2f %8.2f %8.2f %8.2f   skill %+.3f' %
          (win_s, len(A), A[:, 0].mean(), A[:, 1].mean(), A[:, 2].mean(),
           A[:, 3].mean(), A[:, 4].mean(), 1 - A[:, 1].mean() / A[:, 2].mean()))
print("""
READ: if the synthetic pulse is recovered with low MAE and high corr, the
estimator is capable and the BP/face failures are REAL. If the synthetic also
fails, the estimator is the binding constraint and no HR conclusion is possible.
""")
