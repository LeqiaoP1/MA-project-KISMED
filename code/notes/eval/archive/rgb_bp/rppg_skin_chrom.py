"""rPPG done properly: SKIN MASK + CHROM + POS.

v3 used a plain box mean over the whole face (eyes, mouth, brows, and any hair or
background inside the box all included) and a naive G_norm - R_norm chrominance
difference. Published rPPG relies on two things that were missing:

  1. SKIN MASKING -- eyes/mouth/brows carry motion, not blood; hair, background
     and shadow are not skin. Two stages:
       a. landmark exclusion: per-frame boxes over the brows (1-10), both eyes
          (20-25, 26-31) and the mouth (32-49), tracked with the head;
       b. skin-colour gate in YCrCb (133<=Cr<=173, 77<=Cb<=127) plus a luminance
          floor, which drops hair/background/shadow.
  2. CHROM (de Haan & Jeanne 2013) and POS (Wang et al. 2017) instead of GR.

Comparisons are all within this one run so they are directly comparable:
  face_plain (v3 behaviour) / face_skin   x   GR / CHROM / POS   +  bg + full
Signals stored per frame so the |r| gate and the HR test both run off one decode.
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
W, H = 1040.0, 1392.0
WINS = (20, 30, 40)
SESSIONS = [f'{s}_T{t}' for s in ('F001', 'F002', 'F003', 'F004') for t in range(1, 11)]
CACHE = '/tmp/rppg_skin_v1.npz'
# landmark groups (1-indexed) to EXCLUDE from the skin mask
EXCL = {'brows': tuple(range(1, 11)), 'eyes': tuple(range(20, 32)),
        'mouth': tuple(range(32, 50))}


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
    """rgb (n,3) in **RGB** order -> dict of band-limited pulse candidates."""
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    Rn = R / (R.mean() + 1e-9); Gn = G / (G.mean() + 1e-9); Bn = B / (B.mean() + 1e-9)
    r_, g_, b_ = bandpass(Rn), bandpass(Gn), bandpass(Bn)
    out = {}
    out['GR'] = g_ - r_
    X = 3 * r_ - 2 * g_                       # CHROM
    Y = 1.5 * r_ + g_ - 1.5 * b_
    out['CHROM'] = X - (X.std() / (Y.std() + 1e-12)) * Y
    S1 = g_ - b_                              # POS
    S2 = -2 * r_ + g_ + b_
    out['POS'] = S1 + (S1.std() / (S2.std() + 1e-12)) * S2
    return out


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


def one_session(name):
    subj, task = name.split('_')
    p2 = find_2d_features(ROOT, subj, task)
    d = f'{ROOT}/2D+3D/{subj}/{task}'
    if p2 is None or not os.path.isdir(d):
        return name, None
    pts, pose, frame, _ = parse_2d_features(p2)
    files = sorted(f for f in os.listdir(d) if f.endswith('.jpg'))
    n = min(len(files), len(frame))
    keep = np.isfinite(pts[:n]).all(axis=(1, 2))
    if keep.sum() < 10:
        return name, None
    med = np.nanmedian(pts[:n][keep], axis=0)
    q = med
    x0f, x1f = q[:, 0].min(), q[:, 0].max()
    y0f, y1f = q[:, 1].min(), q[:, 1].max()
    ex, ey = 0.2 * (x1f - x0f), 0.2 * (y1f - y0f)
    box = (x0f - ex, x1f + ex, y0f - ey, y1f + ey)          # static face box
    bx0, bx1, by0, by1 = box
    fw, fh = bx1 - bx0, by1 - by0
    bgbox = (0.0, 180.0, 0.0, 180.0)                        # face-free corner

    S = np.zeros((n, 12), np.float64)   # face_plain(3) face_skin(3) bg(3) full(3)
    skinfrac = np.zeros(n)
    # one array PER group: the groups have 10 / 12 / 18 points and cannot be
    # stacked. Keep NaN (do NOT nan_to_num) so lost landmarks can be skipped
    # instead of collapsing their box onto the origin.
    excl_groups = [(nm, np.asarray(pts[:n][:, [i - 1 for i in idx], :], dtype=np.float64))
                   for nm, idx in EXCL.items()]
    for i in range(n):
        im = cv2.imread(os.path.join(d, files[i]), cv2.IMREAD_REDUCED_COLOR_4)
        if im is None:
            continue
        h, w = im.shape[:2]
        sx, sy = w / W, h / H
        # cv2 reads BGR -- reverse to RGB at STORAGE so `methods` is correct.
        S[i, 9:12] = im.reshape(-1, 3).mean(0)[::-1]                 # full frame
        ca = (int(max(0, bgbox[0] * sx)), int(min(w, bgbox[1] * sx)),
              int(max(0, bgbox[2] * sy)), int(min(h, bgbox[3] * sy)))
        S[i, 6:9] = im[ca[2]:ca[3], ca[0]:ca[1]].reshape(-1, 3).mean(0)[::-1]  # bg
        xa, xb = int(max(0, bx0 * sx)), int(min(w, bx1 * sx))
        ya, yb = int(max(0, by0 * sy)), int(min(h, by1 * sy))
        if xb <= xa or yb <= ya:
            continue
        crop = im[ya:yb, xa:xb]
        S[i, 0:3] = crop.reshape(-1, 3).mean(0)[::-1]                # face_plain
        ch, cw = crop.shape[:2]
        mask = np.ones((ch, cw), bool)
        for nm, pa in excl_groups:
            p = pa[i]
            if not np.isfinite(p).all():
                continue
            pad = 0.15 if nm == 'mouth' else 0.25
            ex0, ex1 = p[:, 0].min(), p[:, 0].max()
            ey0, ey1 = p[:, 1].min(), p[:, 1].max()
            px, py = pad * (ex1 - ex0), pad * (ey1 - ey0)
            a0 = int((ex0 - px - bx0) * sx); a1 = int((ex1 + px - bx0) * sx)
            b0 = int((ey0 - py - by0) * sy); b1 = int((ey1 + py - by0) * sy)
            a0, a1 = max(0, a0), min(cw, a1); b0, b1 = max(0, b0), min(ch, b1)
            if a1 > a0 and b1 > b0:
                mask[b0:b1, a0:a1] = False
        ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        skin = ((ycc[:, :, 1] >= 133) & (ycc[:, :, 1] <= 173)
                & (ycc[:, :, 2] >= 77) & (ycc[:, :, 2] <= 127)
                & (ycc[:, :, 0] >= 40))
        m = mask & skin
        skinfrac[i] = m.mean()
        if m.sum() >= 10:
            S[i, 3:6] = crop[m].mean(0)[::-1]                        # face_skin
        else:
            S[i, 3:6] = np.nan
    return name, (S, skinfrac)


if os.environ.get('SMOKE'):
    # single-session smoke test: catches shape/mask bugs in ~40 s instead of a
    # 5-minute parallel run that dies at the end
    _nm, _res = one_session(os.environ['SMOKE'])
    _S, _frac = _res
    print('smoke OK for %s' % _nm)
    print('  S shape %s, all-finite rows %d/%d' %
          (_S.shape, int(np.isfinite(_S).all(1).sum()), _S.shape[0]))
    print('  skin fraction mean %.3f min %.3f max %.3f' %
          (float(np.nanmean(_frac)), float(np.nanmin(_frac)), float(np.nanmax(_frac))))
    print('  face_plain mean RGB %s' % np.round(_S[:, 0:3].mean(0), 2))
    print('  face_skin  mean RGB %s' % np.round(np.nanmean(_S[:, 3:6], 0), 2))
    print('  bg         mean RGB %s' % np.round(_S[:, 6:9].mean(0), 2))
    print('  (R > G > B is expected for skin; B-dominant would mean the order is wrong)')
    sys.exit(0)

if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    data = {k: (z[f'{k}||S'], z[f'{k}||frac']) for k in
            sorted({k.rsplit('||', 1)[0] for k in z.files})}
    print(f'loaded cache: {len(data)} sessions')
else:
    from multiprocessing import Pool
    import multiprocessing as mp
    print(f'decoding with skin masking on {min(12, mp.cpu_count())} processes...', flush=True)
    data = {}
    with Pool(min(12, mp.cpu_count())) as pool:
        for i, (nm, res) in enumerate(pool.imap_unordered(one_session, SESSIONS), 1):
            if res is not None:
                data[nm] = res
            print(f'  [{i}/{len(SESSIONS)}] {nm}', flush=True)
    payload = {}
    for k, (S, fr) in data.items():
        payload[f'{k}||S'] = S
        payload[f'{k}||frac'] = fr
    np.savez_compressed(CACHE, **payload)
    print('cache saved')

means = np.array([np.nanmean(v[1]) for v in data.values()])
print('mean skin-pixel fraction inside the face box: %.3f (min %.3f max %.3f)'
      % (means.mean(), means.min(), means.max()))

# ------------------------------------------------------------------ |r| gate
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
rng = np.random.default_rng(43)
SURN = {}
for name, b in BP.items():
    Xt = np.fft.rfft(b); amag = np.abs(Xt)
    s = np.empty((NSUR, len(b)), np.float32)
    for i in range(NSUR):
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, len(Xt))); ph[0] = 1.0
        s[i] = np.fft.irfft(amag * ph, n=len(b))
    SURN[name] = s

print('\n=== |r| GATE vs BP, p<0.05 (chance 2.0 of 40), band %.1f-%.1f Hz ===' % BAND)
print('%-22s %8s %8s %8s' % ('signal / method', 'rate', 'median|r|', 'vs face_skin GR'))
res = {}
for roi, cols in (('face_plain', slice(0, 3)), ('face_skin', slice(3, 6)),
                  ('bg', slice(6, 9)), ('full', slice(9, 12))):
    for meth in ('GR', 'CHROM', 'POS'):
        o, p = [], []
        for name in BP:
            y = methods(data[name][0][:, cols])[meth]
            if y.std() < 1e-12:
                continue
            C = zs(y)[:, None].astype(np.float32)
            obs = float(stat(BP[name].astype(np.float32), C)[0])
            nul = stat(SURN[name], C).ravel()
            o.append(obs); p.append(float(np.mean(nul >= obs)))
        if not p:
            continue
        p = np.array(p)
        res[(roi, meth)] = (int((p < .05).sum()), len(p), float(np.median(o)))
        print('%-22s %8s %8.3f   %d/%d' % (f'{roi} / {meth}',
              '%d/%d' % (int((p < .05).sum()), len(p)), np.median(o),
              int((p < .05).sum()), len(p)))
print('  (chance is 2/40; the mismatched-subject control sat at 4% earlier)')

# --------------------------------------------------------------- HR estimate
print('\n=== HR ESTIMATION: skill = 1 - MAE/baselineMAE (<=0 is worse than a constant) ===')
SIG = {}
for name, (S, fr) in data.items():
    subj, task = name.split('_')
    f = f'{ROOT}/Physiology/{subj}/{task}/BP_mmHg.txt'
    pf = f'{ROOT}/Physiology/{subj}/{task}/Pulse Rate_BPM.txt'
    if not (os.path.exists(f) and os.path.exists(pf)):
        continue
    n = S.shape[0]
    bp = _load_1d(f).astype(np.float64)
    if bp.shape[0] < int((n - 1) * RESP_FS / FPS) + 1:
        continue
    pr = _load_1d(pf)
    fhz = np.interp(np.arange(n) * (RESP_FS / FPS), np.arange(pr.shape[0]),
                    pr.astype(np.float64)) / 60.0
    k = int(2.0 * FPS)
    truth = np.convolve(fhz, np.ones(k) / k, mode='same') * 60.0
    fhz_c = np.clip(np.convolve(fhz, np.ones(k) / k, mode='same'), BAND[0] * 1.05, BAND[1] * 0.95)
    phase = 2 * np.pi * np.cumsum(fhz_c) / FPS
    syn = (np.sin(phase) + 0.35 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
           + 0.30 * bandpass(np.random.default_rng(5).standard_normal(n)))
    SIG[name] = dict(n=n, truth=truth, syn=syn,
                     bp=bandpass(np.interp(np.arange(n) * (RESP_FS / FPS),
                                           np.arange(bp.shape[0]), bp)),
                     **{f'{roi}_{m}': methods(S[:, cols])[m]
                        for roi, cols in (('face', slice(0, 3)), ('skin', slice(3, 6)),
                                          ('bg', slice(6, 9)))
                        for m in ('GR', 'CHROM', 'POS')})

for win_s in WINS:
    R = {}
    for key in ('syn', 'bp', 'face_GR', 'face_CHROM', 'face_POS', 'skin_GR',
                'skin_CHROM', 'skin_POS', 'bg_GR', 'bg_CHROM', 'bg_POS'):
        rows = []
        for name, d in SIG.items():
            if d['n'] < int(win_s * FPS) + 20:
                continue
            ctr, _ = win_hr_sub(d['bp'], win_s, max(5, win_s // 4))
            if len(ctr) < 3:
                continue
            _, est = win_hr_sub(d[key], win_s, max(5, win_s // 4))
            m = min(len(est), len(ctr))
            if m < 3 or est[:m].std() < 1e-9:
                continue
            tw = d['truth'][ctr[:m]]
            rows.append((float(np.corrcoef(est[:m], tw)[0, 1]),
                         float(np.mean(np.abs(est[:m] - tw))),
                         float(np.mean(np.abs(tw.mean() - tw)))))
        R[key] = np.array(rows) if len(rows) >= 3 else None
    print('\n-- window %d s --' % win_s)
    for key, lab in (('syn', 'SYNTH (ceiling)'), ('bp', 'BP (positive ctrl)'),
                     ('face_GR', 'face_plain / GR'), ('face_CHROM', 'face_plain / CHROM'),
                     ('face_POS', 'face_plain / POS'), ('skin_GR', 'face_skin / GR'),
                     ('skin_CHROM', 'face_skin / CHROM'), ('skin_POS', 'face_skin / POS'),
                     ('bg_GR', 'bg / GR'), ('bg_CHROM', 'bg / CHROM'), ('bg_POS', 'bg / POS')):
        A = R[key]
        if A is None:
            continue
        mae, base = A[:, 1].mean(), A[:, 2].mean()
        print('  %-22s corr %+0.3f  MAE %6.2f  base %6.2f  skill %+0.3f  n=%d' %
              (lab, A[:, 0].mean(), mae, base, 1 - mae / base if base > 0 else np.nan, len(A)))
print("""
READ: compare every row against SYNTH and BP at the same window. Skin masking
and CHROM/POS only help if face_skin/CHROM or face_skin/POS move clearly toward
those two rows. bg must stay far away, or the metric is not skin-specific.
""")
