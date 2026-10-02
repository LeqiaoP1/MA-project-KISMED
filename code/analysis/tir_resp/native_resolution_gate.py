"""Native-resolution vs 64x64 positive-control gate for the TIR -> RESP line.

WHY THIS EXISTS
---------------
The recorded gate (``POSITIVE-CONTROL-GATE-result``, 2026-09-26) found
respiration undetectable in the thermal ROI -- 0/8 sessions, observed best |r|
BELOW the null median -- while the SAME pipeline detects head motion 8/8 at
p < 0.005. That closed the line (``TirROI_Resp_plan.md`` §12).

But that gate ran at the MODEL's input resolution: the recorded statistic was
"max over 4096 ROI pixels" = 64x64. Measured on this corpus, the dataset's own
ROI box is ~72x93 native, i.e. the 64x64 patch is a ~1.6x DOWNSAMPLE (1.28x per
axis). So "the resize averaged away a LOCALISED signal" is a real, untested
alternative to "there is no signal". (224x224 would be a 7.5x UPSAMPLE of the
same box, so it cannot be the answer; 96/112 is the native ceiling.)

This script runs the SAME statistic at both resolutions, so the resize is the
only difference, and adds a synthetic LOCALISED control whose ground-truth
amplitude is known -- which quantifies how much a localised signal loses in the
resize instead of leaving it to inference.

METHOD (matched to the recorded gate where it can be)
-----------------------------------------------------
* ROI = union box of the 12 nose+mouth landmarks at their MEDIAN position over
  the session, padded by ``roi_padding`` (the dataset's own convention; the
  recorded gate used the same median-landmark union).
* per-pixel scalar = HSV HUE (cv2 H, 0-179 -> 0-255). CAVEAT: the original
  sweep's palette LUT is NOT in the repo (its /tmp scripts were lost), so this is
  a RECONSTRUCTED palette coordinate. That is acceptable here because the whole
  point is a two-resolution comparison measured with one statistic; the 64x64
  row is printed alongside so it can be checked against the recorded
  0.46-0.77 / null 0.60-0.87.
* statistic = max over pixels AND over +-``--lags`` s of |r|, with r taken as the
  CIRCULAR normalised cross-correlation (both sides standardised), so every lag
  uses the full length and no edge weighting is needed.
* null = ``--nperm`` phase-scrambled belts: the FFT phases are randomised and the
  spectrum kept, so the belt's own autocorrelation -- the thing that makes a
  naive p-value meaningless here -- is preserved.
* reference belt = ``Resp_Volts.txt`` (1000 Hz) sampled at the frame times.

POSITIVE CONTROLS
-----------------
* ``motion``: the 12-landmark centroid (x, y) as the reference -- the recorded
  control. It is a GLOBAL signal, so it proves sensitivity to a whole-ROI
  modulation, not to a nostril-localised one.
* ``inject``: add ``amplitude * belt(t)`` to a small block of NATIVE pixels
  (default 4x4 at the ROI centre) BEFORE the resize, then measure at both
  resolutions. This is the missing control: it is localised, its ground truth is
  known exactly, and it gives the resize attenuation directly.

Usage (from ``code/``)::

    python analysis/tir_resp/native_resolution_gate.py --device cuda
    python analysis/tir_resp/native_resolution_gate.py --sessions F004_T2 --nperm 50
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from data import video_io as vio                                    # noqa: E402
from data.tir_resp_dataset import (TARGET_LANDMARKS, discover_sessions,
                                   parse_ir_features)               # noqa: E402

FS_BELT = 1000.0          # Resp_Volts.txt native rate
DEFAULT_SESSIONS = ('F001_T1', 'F001_T3', 'F002_T1', 'F003_T2',
                    'F004_T2', 'F004_T3', 'F004_T9', 'F004_T8')
DEFAULT_SIZES = (None, 64)      # None = native crop, no resize


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_belt(resp_file, n_frames, fps):
    """``Resp_Volts.txt`` (1000 Hz) sampled at the frame times -> [F]."""
    x = np.loadtxt(resp_file, ndmin=1).astype(np.float64).ravel()
    idx = np.round(np.arange(n_frames, dtype=np.float64) * FS_BELT / fps)
    idx = np.clip(idx.astype(np.int64), 0, x.size - 1)
    return x[idx]


def median_landmark_box(ir_file, pad):
    """Padded union box of the target landmarks at their median position."""
    ir = parse_ir_features(ir_file)
    lm = ir[:, [i - 1 for i in TARGET_LANDMARKS], :]
    med = np.nanmedian(lm, axis=0)                   # [12, 2]
    x0, x1 = float(med[:, 0].min()), float(med[:, 0].max())
    y0, y1 = float(med[:, 1].min()), float(med[:, 1].max())
    w, h = max(x1 - x0, 2.0), max(y1 - y0, 2.0)
    return dict(x=int(round(x0 - pad * w)), y=int(round(y0 - pad * h)),
                w=max(2, int(round(w * (1 + 2 * pad)))),
                h=max(2, int(round(h * (1 + 2 * pad)))),
                x0=x0, y0=y0, w0=w, h0=h)


def landmark_motion(ir_file, n_frames):
    """12-landmark centroid path -> [F, 2] (x, y) on the frame grid."""
    ir = parse_ir_features(ir_file)
    lm = ir[:n_frames][:, [i - 1 for i in TARGET_LANDMARKS], :]
    return np.nanmean(lm, axis=1)                    # [F, 2]


def roi_hue_frames(video, box, chunk=200, verbose=True):
    """Decode the box from every frame -> float32 hue [F, H, W]."""
    import cv2
    frames = []
    with vio.open_video(video) as r:
        n = int(r.num_frames)
        for s in range(0, n, chunk):
            part = r.read_range(s, min(chunk, n - s))       # [t,H,W,3] RGB
            for f in part:
                y0, x0 = max(0, box['y']), max(0, box['x'])
                y1 = min(f.shape[0], box['y'] + box['h'])
                x1 = min(f.shape[1], box['x'] + box['w'])
                if y1 - y0 < 2 or x1 - x0 < 2:
                    raise ValueError(
                        f'ROI box {box} falls outside the {f.shape[:2]} frame')
                sub = f[y0:y1, x0:x1]
                frames.append(cv2.cvtColor(sub, cv2.COLOR_RGB2HSV)[:, :, 0]
                              .astype(np.float32))
    H = min(a.shape[0] for a in frames)
    W = min(a.shape[1] for a in frames)
    out = np.stack([a[:H, :W] for a in frames])
    # CENTRE the hue. cv2 hue wraps at 0/179, and a warm face in a rainbow
    # palette sits near the 0 boundary -- so an injected +-amplitude oscillation
    # would wrap and corrupt the synthetic control. A constant shift leaves every
    # Pearson correlation unchanged, so this is free.
    med = float(np.median(out))
    out = out - med
    if verbose:
        p1, p99 = np.percentile(out, [1, 99])
        print(f'    decoded {out.shape[0]} frames, native ROI {H} x {W} '
              f'= {H * W} px | hue centred at {med:.0f}, '
              f'1-99 pct [{p1:.0f}, {p99:.0f}]')
    return out


def to_size(frames, size):
    """[F, H, W] -> [F, P]; ``size is None`` keeps native (no resize)."""
    import cv2
    if size is None:
        return frames.reshape(frames.shape[0], -1)
    return np.stack([cv2.resize(f, (size, size),
                                interpolation=cv2.INTER_AREA).reshape(-1)
                     for f in frames])


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def bandpass(x, fs, band, axis=0):
    if not band:
        return np.asarray(x, dtype=np.float64)
    from scipy import signal as sg
    b, a = sg.butter(4, [band[0] / (fs / 2.0), band[1] / (fs / 2.0)],
                     btype='band')
    padlen = min(3 * (max(len(a), len(b)) - 1), x.shape[axis] - 1)
    return sg.filtfilt(b, a, x, axis=axis, padlen=padlen)


def standardise(rows):
    """[P, F] -> unit-mean-zero / unit-std rows (plus the stds)."""
    m = rows.mean(axis=1, keepdims=True)
    s = rows.std(axis=1, keepdims=True)
    s = np.where(s < 1e-9, 1e-9, s)
    return (rows - m) / s, s


def phase_scramble(y, rng):
    """Randomise the FFT phases, keep the spectrum (=> keep autocorrelation)."""
    n = y.size
    sp = np.fft.rfft(y - y.mean())
    ph = rng.uniform(0.0, 2.0 * np.pi, sp.size)
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0
    out = np.fft.irfft(sp * np.exp(1j * ph), n=n)
    return ((out - out.mean()) / (out.std() + 1e-12)).astype(np.float32)


def max_abs_corr(X, y, lags, device):
    """max over pixels and lags of |r|; X [P,F] and y [F] standardised."""
    import torch
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(np.asarray(y, dtype=np.float32), device=device)
    Y = torch.stack([torch.roll(yt, int(l)) for l in lags], dim=1)   # [F, L]
    C = (Xt @ Y) / float(yt.numel())
    return float(C.abs().max())


def run_gate(series, y, fs, band, lags, nperm, device, rng, label):
    """Full gate for one (resolution, reference) pair."""
    if band:
        series = bandpass(series, fs, band, axis=0)
        y = bandpass(y, fs, band, axis=0)
    X, _ = standardise(series.T)                     # [P, F]
    y = np.asarray(y, dtype=np.float64)
    y = (y - y.mean()) / (y.std() + 1e-12)
    obs = max_abs_corr(X, y, lags, device)
    nul = np.empty(nperm, dtype=np.float64)
    for i in range(nperm):
        nul[i] = max_abs_corr(X, phase_scramble(y, rng), lags, device)
    p = float((nul >= obs).mean())
    q95 = float(np.percentile(nul, 95))
    print(f'    {label:34s} best|r| {obs:6.3f}   null95 {q95:6.3f}   '
          f'null median {np.median(nul):6.3f}   p {p:5.3f}   '
          f'{"DETECTED" if p < 0.05 else "not detected"}')
    return dict(best=obs, null95=q95, null_median=float(np.median(nul)), p=p,
                n_pixels=int(X.shape[0]))


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--raw_root', default='../data/raw/BP4D')
    ap.add_argument('--sessions', default=','.join(DEFAULT_SESSIONS))
    ap.add_argument('--sizes', default='native,64',
                    help="comma list: 'native' and/or side lengths (e.g. 64,96)")
    ap.add_argument('--band', default='0.1,0.6',
                    help="respiration band f_lo,f_hi; '' = no band-pass")
    ap.add_argument('--band2', default='',
                    help='optional second band, e.g. 0.1,0.5 (recorded gate)')
    ap.add_argument('--lags', default=3.0, type=float,
                    help='max |lag| in seconds (the recorded gate used +-3 s)')
    ap.add_argument('--fps', default=25.0, type=float)
    ap.add_argument('--nperm', default=200, type=int)
    ap.add_argument('--roi_padding', default=0.2, type=float)
    ap.add_argument('--inject_amps', default='1,2,4,8,16',
                    help='LOCALISED control amplitudes, in hue levels, added '
                         'to a small native block. The LOWEST amplitude that is '
                         'detected is the sensitivity threshold; comparing that '
                         'threshold across resolutions measures what the resize '
                         'costs a localised signal.')
    ap.add_argument('--inject_nperm', default=50, type=int,
                    help='permutations for the inject sweep (fewer than the '
                         'gate: only the threshold matters there)')
    ap.add_argument('--inject_blocks', default='1,2,4',
                    help='LOCALISED-control block sizes in native px. 1 is the '
                         'WORST case for a resize: a single native pixel is '
                         'split across neighbouring output pixels, whereas a '
                         'larger block stays contiguous. Empty -> skip the '
                         'control (for the full 8-session gate run).')
    ap.add_argument('--chunk', default=200, type=int)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='')
    args = ap.parse_args(argv)

    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('[gate] CUDA unavailable -> CPU')
        args.device = 'cpu'
    print(f'[gate] device {args.device} | nperm {args.nperm} | lags +-'
          f'{args.lags}s | band {args.band!r}')

    sizes = []
    for s in str(args.sizes).split(','):
        s = s.strip().lower()
        sizes.append(None if s in ('native', '0', '') else int(s))
    # ALWAYS include the un-band-passed ('raw') variant: the recorded head-motion
    # positive control got |r| 0.89-0.98 precisely because it was NOT band-limited
    # (motion is broadband). Band-limiting it here first measured only 0.71, which
    # would understate the sensitivity of the test.
    bands = [None] + [tuple(float(v) for v in b.split(','))
                      for b in (args.band, args.band2) if b.strip()]
    bname = lambda b: 'raw' if b is None else f'{b[0]}-{b[1]}'   # noqa: E731
    lags = np.arange(-int(round(args.lags * args.fps)),
                     int(round(args.lags * args.fps)) + 1)

    out = {}
    for name in [s.strip() for s in args.sessions.split(',') if s.strip()]:
        subj, task = name.split('_')
        spec = discover_sessions(args.raw_root, subjects=[subj],
                                 tasks=[task])[0]
        print(f'\n=== {name} ===')
        box = median_landmark_box(spec['ir_file'], args.roi_padding)
        print(f'  ROI median-landmark union: {box["w0"]:.0f} x {box["h0"]:.0f} '
              f'native -> padded box {box["w"]} x {box["h"]} '
              f'= {box["w"] * box["h"]} px')
        frames = roi_hue_frames(spec['video'], box, chunk=args.chunk)
        n_frames = frames.shape[0]
        belt = load_belt(spec['resp_file'], n_frames, args.fps)
        motion = landmark_motion(spec['ir_file'], n_frames)
        rec = {'box': box, 'n_frames': n_frames,
               'native_px': int(frames.shape[1] * frames.shape[2]),
               'distinct_hue_levels_per_pixel_median':
                   float(np.median([np.unique(frames[:, i, j]).size
                                    for i in range(0, frames.shape[1], 4)
                                    for j in range(0, frames.shape[2], 4)])),
               'gates': {}, 'controls': {}}

        # --- the gate, at every resolution, for every band ------------------ #
        for size in sizes:
            tag = 'native' if size is None else f'{size}x{size}'
            series = to_size(frames, size)
            for b in bands:
                for ref, yref in (('belt', belt),
                                  ('motion_x', motion[:, 0])):
                    key = f'{tag}|{bname(b)}|{ref}'
                    rec['gates'][key] = run_gate(
                        series, yref, args.fps, b, lags, args.nperm,
                        args.device, np.random.default_rng(0),
                        f'{tag} {bname(b):9s} vs {ref}')

        # --- synthetic LOCALISED control: detection THRESHOLD per resolution - #
        # inject INTO THE NATIVE FRAMES, then derive each size, so the resize is
        # the only thing that can attenuate it.
        blocks = [int(x) for x in str(args.inject_blocks).split(',') if x.strip()]
        amps = [float(a) for a in str(args.inject_amps).split(',') if a.strip()]
        inj_band = next((b for b in bands if b), None)
        if not blocks or not amps:
            out[name] = rec
            continue
        yb = bandpass(belt, args.fps, inj_band) if inj_band else belt
        yb = (yb - yb.mean()) / (yb.std() + 1e-12)
        H, W = frames.shape[1], frames.shape[2]
        print(f'  LOCALISED control: band {bname(inj_band)}, amplitudes {amps} '
              f'hue levels, blocks {blocks} px')
        for b0 in blocks:
            r0, c0 = (H - b0) // 2, (W - b0) // 2
            for size in sizes:
                tag = 'native' if size is None else f'{size}x{size}'
                for amp in amps:
                    inj = frames.copy()
                    inj[:, r0:r0 + b0, c0:c0 + b0] += (
                        amp * yb[:, None, None]).astype(np.float32)
                    a = f'{amp:g}'.replace('.', 'p')
                    rec['controls'][f'inject_{tag}_b{b0}_amp{a}'] = run_gate(
                        to_size(inj, size), belt, args.fps, inj_band, lags,
                        args.inject_nperm, args.device,
                        np.random.default_rng(1),
                        f'inject {tag:7s} {b0}x{b0} amp {amp:5.1f}')
        det = {}
        for b0 in blocks:
            for size in sizes:
                tag = 'native' if size is None else f'{size}x{size}'
                hit = [x for x in amps
                       if rec['controls'][
                           f'inject_{tag}_b{b0}_amp'
                           f'{f"{x:g}".replace(chr(46), chr(112))}']['p'] < 0.05]
                det[f'{tag}_block{b0}'] = min(hit) if hit else None
        rec['inject_threshold'] = det
        for b0 in blocks:
            print(f'  -> block {b0}x{b0} detection threshold: ' + ', '.join(
                f'{k.split("_")[0]} {v if v is not None else ">" + str(max(amps))}'
                for k, v in det.items() if k.endswith(f'block{b0}'))
                + ' hue levels')
        out[name] = rec

    # --- summary ------------------------------------------------------------ #
    print('\n' + '=' * 78)
    print('SUMMARY (p < 0.05 = detected above the phase-scrambled null)')
    print('=' * 78)
    for name, rec in out.items():
        print(f'{name}:')
        for k, v in rec['gates'].items():
            t, b, ref = k.split('|')
            print(f'   {t:7s} {b:9s} vs {ref:9s} best|r| {v["best"]:.3f}  '
                  f'null95 {v["null95"]:.3f}  p {v["p"]:.3f}  '
                  f'{"DETECTED" if v["p"] < 0.05 else "-"}')
        if rec.get('inject_threshold'):
            print(f'   localised-control threshold: {rec["inject_threshold"]}')
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(out, fh, indent=2)
        print(f'\n[gate] wrote {args.out}')


if __name__ == '__main__':
    main()
