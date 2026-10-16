"""CARDIAC GATE v3 -- strictly face-free controls + a mismatched-session control.

Fixes the v2 bug: background boxes were the SAME SIZE as the face box (690x690)
parked in the frame corners, which in a 1040x1392 frame overlap each other AND
cover the face's left/right halves. They were never background, which is why
"best background" fired 24/40 vs the face's 27/40.

v3 controls:
  bg_left / bg_right : strips strictly OUTSIDE the landmark bbox (x0-40 / x1+40),
                       so they cannot contain face skin. Brightness recorded.
  bg_topleft / bg_topright : small corner boxes well above the face.
  MISMATCHED-SESSION control: correlate each session's face signal with a
  DIFFERENT session's BP waveform. Any real physiology is destroyed by pairing
  the wrong subject; only a generic statistical artifact survives. This control
  cannot be gamed by box placement.
"""
import os
import sys

sys.path.insert(0, '/mnt/f/Studium/KISMed/github/MA-project-KISMED/code')
os.chdir('/mnt/f/Studium/KISMed/github/MA-project-KISMED/code')

import cv2
import numpy as np
from data.rgb_features import parse_2d_features, find_2d_features
from data.tir_resp_dataset import _load_1d

ROOT = '../data/raw/BP4D'
FPS, RESP_FS = 25.0, 1000.0
BAND = (0.9, 3.0)
NSUR, SCALE = 200, 4
LAGS = np.arange(-50, 51, 2)
WIN_HR, HOP_HR = int(20 * FPS), int(5 * FPS)
W, H = 1040.0, 1392.0
SESSIONS = [f'{s}_T{t}' for s in ('F001', 'F002', 'F003', 'F004') for t in range(1, 11)]
CACHE = '/tmp/rgb_roi_means_v3.npz'
ROI_PTS = {
    'face_all':   tuple(range(1, 50)),
    'nose_mouth': (13, 14) + tuple(range(15, 20)) + tuple(range(32, 50)),
}


def bandpass(x, lo=BAND[0], hi=BAND[1], axis=0):
    X = np.fft.rfft(x, axis=axis)
    f = np.fft.rfftfreq(x.shape[axis], d=1.0 / FPS)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, n=x.shape[axis], axis=axis)


def zs(x, axis=0):
    return (x - x.mean(axis=axis, keepdims=True)) / (x.std(axis=axis, keepdims=True) + 1e-12)


def stat(b, C, lags=LAGS):
    one = b.ndim == 1
    if b.ndim == 1:
        b = b[None, :]
    n = C.shape[0]
    best = np.zeros((b.shape[0], C.shape[1]), np.float32)
    for L in lags:
        np.maximum(best, np.abs(np.roll(b, L, axis=1) @ C) / n, out=best)
    return best[0] if one else best


def win_hr(x, win=WIN_HR, hop=HOP_HR):
    x = np.nan_to_num(np.asarray(x, float))
    x = x - x.mean()
    n = len(x)
    if n < win:
        return np.array([], int), np.array([])
    w = np.hanning(win)
    f = np.fft.rfftfreq(win, d=1.0 / FPS)
    sel = (f >= BAND[0]) & (f <= BAND[1])
    fs_ = f[sel]
    r, c = [], []
    for a in range(0, n - win + 1, hop):
        seg = x[a:a + win]
        if seg.std() < 1e-12:
            continue
        P = np.abs(np.fft.rfft(seg * w)) ** 2
        r.append(float(fs_[np.argmax(P[sel])]) * 60.0)
        c.append(a + win // 2)
    return np.array(c, int), np.array(r)


def boxes_for(med):
    b = {}
    for key, idxs in ROI_PTS.items():
        q = med[[i - 1 for i in idxs]]
        x0, x1 = q[:, 0].min(), q[:, 0].max()
        y0, y1 = q[:, 1].min(), q[:, 1].max()
        ex, ey = 0.2 * (x1 - x0), 0.2 * (y1 - y0)
        b[key] = (x0 - ex, x1 + ex, y0 - ey, y1 + ey)
    x0, x1, y0, y1 = b['face_all']
    # STRICTLY outside the landmark bbox -> cannot contain face skin
    b['bg_left'] = (0.0, max(1.0, x0 - 40.0), y0, y1)
    b['bg_right'] = (min(W - 1.0, x1 + 40.0), W, y0, y1)
    b['bg_topleft'] = (0.0, 180.0, 0.0, 180.0)
    b['bg_topright'] = (W - 180.0, W, 0.0, 180.0)
    b['full_frame'] = (0.0, W, 0.0, H)
    return b


def one_session(name):
    subj, task = name.split('_')
    p2 = find_2d_features(ROOT, subj, task)
    d = f'{ROOT}/2D+3D/{subj}/{task}'
    if p2 is None or not os.path.isdir(d):
        return name, None, None
    pts, pose, frame, _ = parse_2d_features(p2)
    files = sorted(f for f in os.listdir(d) if f.endswith('.jpg'))
    n = min(len(files), len(frame))
    keep = np.isfinite(pts[:n]).all(axis=(1, 2))
    if keep.sum() < 10:
        return name, None, None
    boxes = boxes_for(np.nanmedian(pts[:n][keep], axis=0))
    keys = list(boxes)
    sig = {k: np.zeros((n, 3)) for k in keys}
    for i in range(n):
        im = cv2.imread(os.path.join(d, files[i]), cv2.IMREAD_REDUCED_COLOR_4)
        if im is None:
            continue
        h, w = im.shape[:2]
        sx, sy = w / W, h / H
        for k, (a, bx, c, e) in boxes.items():
            xa, xb = int(max(0, a * sx)), int(min(w, bx * sx))
            ya, yb = int(max(0, c * sy)), int(min(h, e * sy))
            sig[k][i] = (im[ya:yb, xa:xb].reshape(-1, 3).mean(0)
                         if (xb > xa and yb > ya) else np.nan)
    return name, np.stack([sig[k] for k in keys]), keys


if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    tmp = {}
    for key in z.files:
        nm, fld = key.rsplit('||', 1)
        tmp.setdefault(nm, {})[fld] = z[key]
    data = {k: (v['S'], [str(x) for x in v['keys']]) for k, v in tmp.items()}
    print(f'loaded cache: {len(data)} sessions')
else:
    from multiprocessing import Pool
    import multiprocessing as mp
    nproc = min(12, mp.cpu_count())
    print(f'decoding 45867 jpgs on {nproc} processes...', flush=True)
    data = {}
    with Pool(nproc) as pool:
        for i, (nm, S, keys) in enumerate(pool.imap_unordered(one_session, SESSIONS), 1):
            if S is not None:
                data[nm] = (S, list(keys))
            print(f'  [{i}/{len(SESSIONS)}] {nm}', flush=True)
    payload = {}
    for k, (S, keys) in data.items():
        payload[f'{k}||S'] = S
        payload[f'{k}||keys'] = np.array(keys, dtype=object)
    np.savez_compressed(CACHE, **payload)
    print('cache saved')

# ---------------------------------------------------------------- analysis
rng = np.random.default_rng(37)
BP, GR, BRIGHT, HRROWS = {}, {}, {}, []
for name, (S, keys) in data.items():
    subj, task = name.split('_')
    bp_f = f'{ROOT}/Physiology/{subj}/{task}/BP_mmHg.txt'
    if not os.path.exists(bp_f):
        continue
    n = S[0].shape[0]
    bp = _load_1d(bp_f)
    if bp.shape[0] < int((n - 1) * RESP_FS / FPS) + 1:
        continue
    b = zs(bandpass(np.interp(np.arange(n) * (RESP_FS / FPS),
                              np.arange(bp.shape[0]), bp.astype(np.float64))))
    BP[name] = b
    idx = {k: i for i, k in enumerate(keys)}
    GR[name] = {k: (S[idx[k]][:, 1] / (S[idx[k]][:, 1].mean() + 1e-9)
                    - S[idx[k]][:, 0] / (S[idx[k]][:, 0].mean() + 1e-9))
                for k in keys}
    BRIGHT[name] = {k: float(np.nanmean(S[idx[k]])) for k in keys}
    pr_f = f'{ROOT}/Physiology/{subj}/{task}/Pulse Rate_BPM.txt'
    if os.path.exists(pr_f):
        pr = _load_1d(pr_f)
        HRROWS.append((name, np.interp(np.arange(n) * (RESP_FS / FPS),
                                       np.arange(pr.shape[0]), pr.astype(np.float64)),
                       b, GR[name]['face_all']))

SURN = {}
for name, b in BP.items():
    Xt = np.fft.rfft(b)
    amag = np.abs(Xt)
    s = np.empty((NSUR, len(b)), np.float32)
    for i in range(NSUR):
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, len(Xt)))
        ph[0] = 1.0
        s[i] = np.fft.irfft(amag * ph, n=len(b))
    SURN[name] = s


def gate(name, y):
    bb, sur = BP[name], SURN[name]
    y = bandpass(np.nan_to_num(y))
    if y.std() < 1e-12:
        return np.nan, 1.0
    C = zs(y)[:, None].astype(np.float32)
    obs = float(stat(bb.astype(np.float32), C)[0])
    nul = stat(sur, C).ravel()
    return obs, float(np.mean(nul >= obs))


names = list(BP)
print('\n=== GATE, n=%d sessions. |r| vs BP at p<0.05 (chance %.1f) ===' %
      (len(names), 0.05 * len(names)))
print('%-13s %10s %12s %12s %12s %12s' %
      ('ROI', 'bright', 'sig', 'rate', 'median|r|', 'vs-face-wins'))
for k in ('face_all', 'nose_mouth', 'bg_left', 'bg_right', 'bg_topleft',
          'bg_topright', 'full_frame'):
    o, p, br = [], [], []
    for name in names:
        if k not in GR[name]:
            continue
        a, b_ = gate(name, GR[name][k])
        o.append(a); p.append(b_); br.append(BRIGHT[name][k])
    p = np.array(p)
    print('%-13s %10.1f %12d %12s %12.3f' %
          (k, np.nanmean(br), int((p < .05).sum()), '%d/%d' % (int((p < .05).sum()), len(p)),
           np.nanmedian(o)))

# mismatched-session control on the face signal
print('\n=== MISMATCHED-SESSION CONTROL (face signal vs ANOTHER session\'s BP) ===')
mm = []
for i, name in enumerate(names):
    for j in (1, 7, 13, 19):
        other = names[(i + j) % len(names)]
        if other == name:
            continue
        y = GR[name]['face_all']; b2 = BP[other]
        m = min(len(y), len(b2))
        if m < 300:
            continue
        yy = bandpass(np.nan_to_num(y[:m]))
        if yy.std() < 1e-12:
            continue
        C = zs(yy)[:, None].astype(np.float32)
        obs = float(stat(b2[:m].astype(np.float32), C)[0])
        Xt = np.fft.rfft(b2[:m]); amag = np.abs(Xt)
        s = np.empty((NSUR, m), np.float32)
        for t in range(NSUR):
            ph = np.exp(1j * rng.uniform(0, 2 * np.pi, len(Xt))); ph[0] = 1.0
            s[t] = np.fft.irfft(amag * ph, n=m)
        nul = stat(s, C).ravel()
        mm.append((obs, float(np.mean(nul >= obs))))
mm = np.array(mm)
print('  n_pairs=%d   p<0.05: %d (%.0f%%)  median |r| %.3f' %
      (len(mm), int((mm[:, 1] < .05).sum()), 100 * np.mean(mm[:, 1] < .05),
       np.nanmedian(mm[:, 0])))
print('  -> a valid control must fire at ~5%%. Much more means the statistic is')
print('     anti-conservative and the matched face numbers are inflated.')

# HR estimation
print('\n=== HEART-RATE ESTIMATION (face chrominance vs Pulse Rate_BPM) ===')
rows = []
for name, prt, b, y in HRROWS:
    ctr, est = win_hr(y)
    if len(ctr) < 3:
        continue
    ref = prt[ctr]
    mae = float(np.mean(np.abs(est - ref)))
    bmae = float(np.mean(np.abs(ref.mean() - ref)))
    corr = float(np.corrcoef(est, ref)[0, 1]) if est.std() > 0 else np.nan
    _, hr_bp = win_hr(b)
    maeb = float(np.mean(np.abs(hr_bp - ref))) if len(hr_bp) == len(ref) else np.nan
    rows.append((name, corr, mae, bmae, maeb))
A = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
print('  n=%d sessions' % len(rows))
print('  mean corr(est, ref)      : %+.3f (median %+.3f)' %
      (np.nanmean(A[:, 0]), np.nanmedian(A[:, 0])))
print('  mean MAE face            : %.2f BPM' % np.nanmean(A[:, 1]))
print('  mean MAE constant base   : %.2f BPM' % np.nanmean(A[:, 2]))
print('  mean MAE BP (POS CTRL)   : %.2f BPM' % np.nanmean(A[:, 3]))
print('  face better than base    : %d/%d sessions' %
      (int(np.nansum(A[:, 1] < A[:, 2])), len(rows)))
