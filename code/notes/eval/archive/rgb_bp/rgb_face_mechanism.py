"""Can a FACE-ONLY RGB ROI relate to respiration at all?  Decompose by mechanism.

Pathways from respiration to a face-only visible-light ROI:
  M  MECHANICAL  chest/shoulder motion -> neck -> head translation + pose
  C  COLOUR      perfusion / respiratory-sinus-arrhythmia / venous return
                 -> skin COLOUR change (rPPG-like), no motion required
  L  LIGHTING    global illumination / auto-exposure / JPEG -> camera artifact
                 (NOT face-locked; caught by the background + full-frame controls)

Design: an ROI MEAN is a 1-channel signal, so it is far more sensitive to a
common-mode effect than per-pixel tests (averaging beats noise down by sqrt(N)
while preserving a shared signal). For each ROI we test the raw band-passed mean
and the mean with nuisance regressors (pose 3 + centroid 2 + log inter-ocular
scale + full-frame mean) projected out. If a signal lives through nuisance
regression it is NOT mechanical -> it is pathway C (or an uncontrolled artifact).
The BACKGROUND ROI is the negative control: if it fires too, the effect is
pathway L and nothing to do with the face.

Statistic: max |r| over lags in +-2 s (stride 2), null = 200 phase-scrambled
belts, symmetric (same lag set) for observed and null.
"""
import os
import sys

sys.path.insert(0, '/mnt/f/Studium/KISMed/github/MA-project-KISMED/code')
os.chdir('/mnt/f/Studium/KISMed/github/MA-project-KISMED/code')

import cv2
import numpy as np
from data.rgb_features import parse_2d_features, find_2d_features
from data.tir_resp_dataset import find_resp_file, _load_1d

ROOT = '../data/raw/BP4D'
FPS, RESP_FS = 25.0, 1000.0
BAND = (0.1, 0.5)
NSUR, SCALE = 200, 4          # IMREAD_REDUCED_GRAYSCALE_4 -> 348 x 260
LAGS = np.arange(-50, 51, 2)
PATCH = 16                    # per-pixel test grid
SESSIONS = ['F001_T1', 'F001_T2', 'F001_T6', 'F001_T7', 'F001_T8', 'F002_T1',
            'F002_T2', 'F003_T1', 'F003_T2', 'F004_T1', 'F004_T2']
# ROI landmark sets, 1-indexed (see data.rgb_features.GROUP_RANGES)
ROI_PTS = {
    'face_all':   tuple(range(1, 50)),
    'nose_mouth': (13, 14) + tuple(range(15, 20)) + tuple(range(32, 50)),
    'nose_only':  tuple(range(15, 20)),
    'mouth_only': tuple(range(32, 50)),
}


def bandpass(x, fs=FPS, lo=BAND[0], hi=BAND[1], axis=0):
    X = np.fft.rfft(x, axis=axis)
    f = np.fft.rfftfreq(x.shape[axis], d=1.0 / fs)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, n=x.shape[axis], axis=axis)


def zs(x, axis=0):
    return (x - x.mean(axis, keepdims=True)) / (x.std(axis, keepdims=True) + 1e-12)


def maxabs_r(b, y, lags=LAGS):
    """max |r| over lags.  b: [n] or [NS,n];  y: [n]."""
    one = b.ndim == 1
    if one:
        b = b[None, :]
    n = len(y)
    y = (y - y.mean()) / (y.std() + 1e-12)
    out = np.zeros(b.shape[0], np.float32)
    for L in lags:
        r = np.abs(np.roll(b, L, axis=1) @ y) / n
        np.maximum(out, r, out=out)
    return out[0] if one else out


def best_lag(b, y, lags=LAGS):
    n = len(y)
    y = (y - y.mean()) / (y.std() + 1e-12)
    r = np.array([float(np.abs(np.roll(b, L) @ y) / n) for L in lags])
    return int(lags[int(np.argmax(r))]), float(r.max())


def regress_out(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


rows_mean, rows_pix = [], []
for name in SESSIONS:
    subj, task = name.split('_')
    p2 = find_2d_features(ROOT, subj, task)
    rf = find_resp_file(ROOT, subj, task)
    d = f'{ROOT}/2D+3D/{subj}/{task}'
    if p2 is None or rf is None or not os.path.isdir(d):
        print(f'{name}: skipped (missing input)')
        continue
    pts, pose, frame, valid = parse_2d_features(p2)
    files = sorted(f for f in os.listdir(d) if f.endswith('.jpg'))
    n = min(len(files), len(frame))

    resp = _load_1d(rf)
    belt = zs(bandpass(np.interp(np.arange(n) * (RESP_FS / FPS),
                                 np.arange(resp.shape[0]),
                                 resp.astype(np.float64)), FPS))

    # static boxes from the median landmark bbox over the WHOLE sequence
    keep = np.isfinite(pts[:n]).all(axis=(1, 2))
    med = np.nanmedian(pts[:n][keep], axis=0)          # [49,2] median frame
    x0s, x1s = [], []
    boxes = {}
    for key, idxs in ROI_PTS.items():
        q = med[[i - 1 for i in idxs]]
        bx0, bx1 = q[:, 0].min(), q[:, 0].max()
        by0, by1 = q[:, 1].min(), q[:, 1].max()
        ex, ey = 0.2 * (bx1 - bx0), 0.2 * (by1 - by0)
        boxes[key] = (bx0 - ex, bx1 + ex, by0 - ey, by1 + ey)
    # background control: same box SIZE as the face, parked at the top-left
    fx0, fx1, fy0, fy1 = boxes['face_all']
    fw, fh = fx1 - fx0, fy1 - fy0
    H_full, W_full = 1392.0, 1040.0
    boxes['background'] = (2.0, 2.0 + fw, 2.0, 2.0 + fh)
    # "neck": directly below the face, the closest thing to chest in frame
    boxes['neck'] = (fx0, fx1, fy1, min(H_full, fy1 + 0.35 * fh))

    sigs = {k: np.empty(n, np.float64) for k in boxes}
    sigs['full_frame'] = np.empty(n, np.float64)
    pix = {k: np.empty((n, PATCH * PATCH), np.float32)
           for k in ('face_all', 'background', 'nose_mouth')}
    for i in range(n):
        im = cv2.imread(os.path.join(d, files[i]), cv2.IMREAD_REDUCED_GRAYSCALE_4)
        h, w = im.shape
        sx, sy = w / W_full, h / H_full
        sigs['full_frame'][i] = im.mean()
        for k, (a, b, c, e) in boxes.items():
            xa, xb = int(max(0, a * sx)), int(min(w, b * sx))
            ya, yb = int(max(0, c * sy)), int(min(h, e * sy))
            if xb <= xa or yb <= ya:
                sigs[k][i] = np.nan
                continue
            crop = im[ya:yb, xa:xb]
            sigs[k][i] = crop.mean()
            if k in pix:
                pix[k][i] = cv2.resize(crop, (PATCH, PATCH),
                                       interpolation=cv2.INTER_AREA).ravel()

    # nuisance: pose + landmark centroid + log inter-ocular scale + frame mean
    kp = np.isfinite(pts[:n]).all(axis=(1, 2))
    Pp = np.where(kp[:, None, None], pts[:n], np.nan)
    cen = np.nanmean(Pp, axis=1)
    cen = np.where(np.isfinite(cen).all(1, keepdims=True), cen, np.nan)
    iod = np.linalg.norm(Pp[:, 19:25].mean(1) - Pp[:, 25:31].mean(1), axis=1)
    for arr in (cen, iod[:, None]):
        bad = ~np.isfinite(arr).all(1)
        if bad.any():
            good = np.flatnonzero(~bad)
            for c in range(arr.shape[1]):
                arr[bad, c] = np.interp(np.flatnonzero(bad), good, arr[good, c])
    NUI = np.column_stack([pose[:n], cen, np.log(np.maximum(iod, 1e-6)),
                           sigs['full_frame']])
    NUI = zs(bandpass(np.nan_to_num(NUI), FPS), axis=0)

    # phase-scrambled belts: same |amplitude spectrum|, random phases, DC kept
    rng = np.random.default_rng(1000)
    Xt = np.fft.rfft(belt)
    amag = np.abs(Xt)
    sur = np.empty((NSUR, n), np.float32)
    for s in range(NSUR):
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, len(Xt)))
        ph[0] = 1.0
        sur[s] = np.fft.irfft(amag * ph, n=n)

    print(f'\n=== {name}  (n={n}) ===')
    hdr = '%-11s %8s %6s %6s %6s | %8s %6s %6s' % (
        'ROI', '|r|', 'null95', 'p', 'lag', '|r|reg', 'null95', 'p')
    print(hdr); print('-' * len(hdr))
    for k in ('full_frame', 'background', 'neck', 'face_all', 'nose_mouth',
              'nose_only', 'mouth_only'):
        y = bandpass(np.nan_to_num(zs(sigs[k])))
        lg, _ = best_lag(belt, y)
        obs = float(maxabs_r(belt, y))
        nul = maxabs_r(sur, y)
        p = float(np.mean(nul >= obs))
        n95 = float(np.percentile(nul, 95))
        yr = bandpass(regress_out(zs(sigs[k]), NUI))
        obs_r = float(maxabs_r(belt, yr))
        nul_r = maxabs_r(sur, yr)
        p_r = float(np.mean(nul_r >= obs_r))
        n95_r = float(np.percentile(nul_r, 95))
        print('%-11s %8.3f %6.3f %6.3f %6d | %8.3f %6.3f %6.3f' %
              (k, obs, n95, p, lg, obs_r, n95_r, p_r))
        rows_mean.append((name, k, obs, p, obs_r, p_r))

    for k in ('face_all', 'background', 'nose_mouth'):
        P = np.ascontiguousarray(zs(bandpass(pix[k]), axis=0), dtype=np.float32)
        bm = np.zeros((1, n), np.float32)
        bm[0] = belt
        # per-pixel: max over pixels AND lags, identically for obs and null
        best = np.zeros((NSUR, P.shape[1]), np.float32)
        ob = np.zeros(P.shape[1], np.float32)
        for L in LAGS:
            ob = np.maximum(ob, np.abs(np.roll(bm, L, axis=1) @ P)[0] / n)
            best = np.maximum(best, np.abs(np.roll(sur, L, axis=1) @ P) / n)
        o = float(ob.max())
        nu = best.max(axis=1)
        rows_pix.append((name, k, o, float(np.mean(nu >= o))))

print('\n' + '=' * 78)
print('ROI-MEAN SUMMARY across %d sessions (p<0.05 count, and the '
      'MOTION-IMMUNE count)' % len(SESSIONS))
print('%-11s %10s %10s %10s   %s' % ('ROI', 'raw p<.05', 'reg p<.05', 'both<.05',
                                     'chance=%.1f of %d' % (0.05 * len(SESSIONS), len(SESSIONS))))
for k in ('full_frame', 'background', 'neck', 'face_all', 'nose_mouth',
          'nose_only', 'mouth_only'):
    sel = [r for r in rows_mean if r[1] == k]
    a = sum(1 for r in sel if r[3] < 0.05)
    b = sum(1 for r in sel if r[5] < 0.05)
    c = sum(1 for r in sel if r[3] < 0.05 and r[5] < 0.05)
    print('%-11s %10d %10d %10d' % (k, a, b, c))
print('\nPER-PIXEL (max over %dx%d pixels and lags)' % (PATCH, PATCH))
for k in ('face_all', 'background', 'nose_mouth'):
    sel = [r for r in rows_pix if r[1] == k]
    a = sum(1 for r in sel if r[3] < 0.05)
    print('%-11s p<0.05: %d/%d   median |r| %.3f' %
          (k, a, len(sel), float(np.median([r[2] for r in sel])) if sel else np.nan))
print("""
READ: `background` and `full_frame` are CONTROLS. If they fire as often as
`face_all`, the effect is camera/illumination (pathway L), not the face.
Compare `raw` vs `reg`: a signal that dies under nuisance regression was
MECHANICAL head motion (pathway M); one that survives is colour/perfusion (C).
""")
