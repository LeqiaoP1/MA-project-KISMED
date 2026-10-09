#!/usr/bin/env python
"""WHERE is the whole-frame gate's best pixel? (TIR-ROI -> RESP, open Q1).

WHY THIS EXISTS
---------------
`roi_sweep.py` (v2) found that the `frame` (WHOLE-FRAME) box detects the
respiration belt MORE strongly than ANY face box on F001_T3 -- detrended band
`frame` 0.909 vs `face` 0.813 vs `nose_mouth` 0.804, while the two off-face
background controls stay clean (`bg_left` 0.539, `bg_right` 0.489). Because
`frame` is a SUPERSET of `face`, `bg_left` and `bg_right`, the winning pixel
must lie OUTSIDE all of them -- somewhere else in the 726x480 frame. Since the
statistic is a MAX OVER PIXELS, a box's score is just the single best pixel it
happens to contain, so "frame > face" does not say the signal is spatially
diffuse; it says the best pixel is off-face.

That matters because a positive in this line cannot be believed until we know
WHERE the winning pixel is:
  * if it sits on the belt buckle / the thermal camera's own housing / a text
    overlay / a hot object in the background, the whole-frame detection is an
    ARTIFACT and every "DETECTED" belt row it drives is spurious;
  * if it sits on the exhaled-air plume (nostrils/upper-lip, i.e. in the AIR,
    which a skin-only landmark box cannot contain) then the landmark box was the
    wrong region and the line genuinely reopens;
  * if it sits on the FACE, the frame box only won because the sweep resizes
    every box to a common size, and the location question dissolves.

METHOD
------
Decode the WHOLE thermal frame at native resolution (optionally spatially
downscaled for memory), then compute, for EVERY pixel and RGB channel, the SAME
statistic the gate uses -- max over +-`--lags` s of |r| against the belt, where r
is the circular normalised cross-correlation of two standardised series (imported
`bandpass` / `standardise` / `phase_scramble` from `native_resolution_gate.py`,
so this is the gate's own statistic and no re-derivation can drift). Report the
ARGMAX pixel (x, y, channel, lag, |r|) and classify it against every ROI box that
`roi_sweep.build_candidates` builds (face / nose_mouth / bg_left / bg_right /
4x4 disjoint tiles), then dump a heatmap PNG with the argmax and the boxes drawn
on top.

The `--detrend` option subtracts the per-frame whole-frame channel mean first, so
a winner that survives detrending is LOCAL (not the camera's global auto-gain
channel). A `vs motion_x` map is produced too, because `motion_x` is the
positive control AND is itself not spatially specific (open Q2).

`--nperm` (default 0) additionally gives the phase-scrambled NULL of the whole
frame's max|r| -- the exact statistic whose argmax is reported -- so the LOCATION
is accompanied by a p-value immune to the multiple-comparison count.

Usage (from ``code/``, MUST run on a compute node -- it decodes the raw .wmv)::

    python analysis/tir_resp/frame_argmax.py --sessions F001_T3
    python analysis/tir_resp/frame_argmax.py --sessions F001_T3 --nperm 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from data import video_io as vio                                     # noqa: E402
from data.tir_resp_dataset import (discover_sessions,                # noqa: E402
                                   parse_ir_features)
from native_resolution_gate import (bandpass, landmark_motion,       # noqa: E402
                                    load_belt, phase_scramble,
                                    standardise)
from roi_sweep import build_candidates                               # noqa: E402


# --------------------------------------------------------------------------- #
# decode
# --------------------------------------------------------------------------- #
def decode_full_frame(video: str, downscale: int = 1, chunk: int = 200
                      ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Decode EVERY frame of ``video`` -> ``([F, H, W, 3] uint8, (H, W))``.

    One sequential pass; ``downscale`` > 1 spatially averages by that factor
    (INTER_AREA) so a long session cannot blow the memory budget. The returned
    frames are the RAW whole frame -- NOT the ROI crop -- because the whole point
    is to see the pixels outside the ROI box.
    """
    import cv2
    parts: List[np.ndarray] = []
    hw = (0, 0)
    with vio.open_video(video) as r:
        n = int(r.num_frames)
        for s in range(0, n, chunk):
            part = r.read_range(s, min(chunk, n - s))       # [t, H, W, 3] RGB
            if part is None or len(part) == 0:
                break
            out = part
            if downscale > 1:
                hw = (max(1, part.shape[1] // downscale),
                      max(1, part.shape[2] // downscale))
                out = np.stack([cv2.resize(f, (hw[1], hw[0]),
                                           interpolation=cv2.INTER_AREA)
                                for f in part])
            else:
                hw = (part.shape[1], part.shape[2])
            parts.append(np.ascontiguousarray(out))
    if not parts:
        raise RuntimeError(f'no frames decoded from {video}')
    return np.concatenate(parts, axis=0), hw


# --------------------------------------------------------------------------- #
# the gate statistic, but LOCATING its argmax
# --------------------------------------------------------------------------- #
def _max_abs_corr_locate(X: np.ndarray, y: np.ndarray, lags: np.ndarray,
                         device: str, block: int
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-ROW max over lags of |r| -- the gate statistic, one pixel at a time.

    ``X`` is ``[P, F]`` ALREADY standardised per row; ``y`` is ``[F]`` already
    standardised. Returns the per-row max|r| and the lag that achieved it, so a
    running ``.argmax()`` gives both the value and the LOCATION of the winner.
    Blocked over P so a full-frame map never materialises the [P, L] cross-
    correlation at once.
    """
    import torch
    P = X.shape[0]
    best = np.empty(P, dtype=np.float32)
    best_lag = np.empty(P, dtype=np.int32)
    yt = torch.as_tensor(np.asarray(y, dtype=np.float32), device=device)
    Y = torch.stack([torch.roll(yt, int(l)) for l in lags], dim=1)   # [F, L]
    lags_np = np.asarray(lags, dtype=np.int32)
    for p0 in range(0, P, block):
        Xb = torch.as_tensor(X[p0:p0 + block], dtype=torch.float32,
                             device=device)
        C = (Xb @ Y) / float(yt.numel())                 # [pb, L]
        v, idx = C.abs().max(dim=1)
        best[p0:p0 + block] = v.detach().cpu().numpy()
        best_lag[p0:p0 + block] = lags_np[idx.detach().cpu().numpy()]
    return best, best_lag


def _global_max(X: np.ndarray, y: np.ndarray, lags: np.ndarray, device: str,
                block: int) -> float:
    """Scalar max over pixels and lags of |r| -- the null statistic."""
    best, _ = _max_abs_corr_locate(X, y, lags, device, block)
    return float(best.max())


def build_series(frames: np.ndarray, gmean: Optional[np.ndarray],
                 band, fps: float) -> np.ndarray:
    """[F, H, W, 3] uint8 -> [P, F] float32, band-passed, standardised per row.

    ``gmean`` non-None subtracts the per-frame whole-frame channel mean (the
    DETRENDED variant). The column index maps C-order: ``p = h*(W*3) + w*3 + c``.
    """
    F, H, W, C = frames.shape
    P = H * W * C
    x = frames.reshape(F, P).T.astype(np.float32)          # [P, F]
    if gmean is not None:
        ch = np.arange(P) % C
        x = x - gmean[:F][:, ch].T.astype(np.float32)
    if band:
        x = bandpass(x, fps, band, axis=1).astype(np.float32)
    xs, _ = standardise(x)                                  # [P, F]
    return np.ascontiguousarray(xs, dtype=np.float32)


def pixel_xyc(p: int, W: int, C: int = 3) -> Tuple[int, int, int]:
    """Flat C-order index -> (y, x, channel) on the [H, W, C] frame grid."""
    c = p % C
    w = (p // C) % W
    h = p // (C * W)
    return h, w, c


def regions_of(x: int, y: int, boxes: Dict[str, Dict[str, int]]) -> List[str]:
    """Names of every candidate box that contains the pixel (x, y)."""
    return [name for name, b in boxes.items()
            if b['x'] <= x < b['x'] + b['w'] and b['y'] <= y < b['y'] + b['h']]


# --------------------------------------------------------------------------- #
# self-test: no data, no decode -- proves the argmax LOCATION is correct
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    print('[selftest] synthetic frames with a belt signal in a KNOWN off-face '
          'block; the argmax must land inside it')
    rng = np.random.default_rng(0)
    F, H, W = 400, 60, 90
    frames = rng.integers(0, 256, size=(F, H, W, 3), dtype=np.uint8)
    belt = np.sin(2 * np.pi * 0.25 * np.arange(F) / 25.0)
    # Inject a NEAR-PERFECT belt copy in a KNOWN block OUTSIDE every box below
    # (bottom-centre), on a flat 128 base so the correlation is unambiguous.
    iy, ix = 48, 40
    frames[:, iy:iy + 6, ix:ix + 6, 1] = np.clip(
        np.rint(128.0 + 90.0 * belt[:, None, None]), 0, 255).astype(np.uint8)
    # fake boxes, all DISJOINT from the injection: face top-left, the two
    # background controls bottom-left / bottom-right -- so the winner must fall
    # in 'frame' ONLY, i.e. off-face AND off both background controls.
    boxes = {'frame': {'x': 0, 'y': 0, 'w': W, 'h': H},
             'face': {'x': 5, 'y': 5, 'w': 30, 'h': 30},
             'bg_left': {'x': 0, 'y': 40, 'w': 20, 'h': 15},
             'bg_right': {'x': 70, 'y': 40, 'w': 20, 'h': 15}}
    lags = np.arange(-3, 4)
    F_ = frames.shape[0]
    gmean = frames.reshape(F_, -1, 3).mean(axis=1)          # [F, 3]
    X = build_series(frames, gmean, band=None, fps=25.0)
    y = np.asarray(belt, dtype=np.float64)
    y = (y - y.mean()) / (y.std() + 1e-12)
    best, blag = _max_abs_corr_locate(X, y, lags, device='cpu', block=20000)
    p = int(best.argmax())
    h, w, c = pixel_xyc(p, W)
    regs = regions_of(w, h, boxes)
    print(f'[selftest] argmax pixel (x={w}, y={h}, ch={c}) |r|={best[p]:.3f} '
          f'lag={blag[p]} regions={regs}')
    in_block = (iy <= h < iy + 6) and (ix <= w < ix + 6)
    ok = in_block and best[p] > 0.95 and regs == ['frame']
    print(f'[selftest] {"PASS" if ok else "FAIL"}: argmax inside the injected '
          f'off-face block {(ix, iy)}, |r|>0.95, contained by frame ONLY')
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selftest', action='store_true',
                    help='run the synthetic location check and exit (no data)')
    ap.add_argument('--raw_root', default=os.environ.get(
        'RAW_DATA_PATH', '/work/projects/l0003511/test'))
    ap.add_argument('--sessions', default='F001_T3',
                    help='the roi_sweep v2 winner by default')
    ap.add_argument('--downscale', default=1, type=int,
                    help='spatial INTER_AREA downscale factor; 1 = native')
    ap.add_argument('--grid', default=4, type=int,
                    help='face box is cut into grid x grid DISJOINT tiles')
    ap.add_argument('--band', default='0.1,0.6')
    ap.add_argument('--lags', default=3.0, type=float)
    ap.add_argument('--fps', default=25.0, type=float)
    ap.add_argument('--nperm', default=0, type=int,
                    help='phase-scrambled nulls of the whole-frame max|r|; '
                         '0 = locate only (significance is already in the gate)')
    ap.add_argument('--block', default=40000, type=int,
                    help='pixels per correlation block (memory guard)')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default='analysis/tir_resp/frame_argmax.json')
    ap.add_argument('--figure', default='analysis/tir_resp/frame_argmax.png')
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    lags = np.arange(-int(round(args.lags * args.fps)),
                     int(round(args.lags * args.fps)) + 1)
    band = None
    if args.band.strip():
        band = tuple(float(v) for v in args.band.split(','))
    print(f'[argmax] raw_root : {args.raw_root}')
    print(f'[argmax] sessions : {args.sessions}')
    print(f'[argmax] lags     : +-{args.lags}s = {lags.size} samples | '
          f'band {band} | downscale {args.downscale} | nperm {args.nperm}')

    out: Dict = {'lags_s': args.lags, 'band': band, 'downscale': args.downscale,
                 'nperm': args.nperm, 'sessions': {}}

    for name in [s.strip() for s in args.sessions.split(',') if s.strip()]:
        subj, task = name.split('_')
        spec = discover_sessions(args.raw_root, subjects=[subj],
                                 tasks=[task])[0]
        print(f'\n=== {name} ===')
        frames, (H, W) = decode_full_frame(spec['video'], args.downscale)
        n = int(frames.shape[0])
        scale = 1.0 / max(1, args.downscale)
        print(f'  decoded whole frame {H}x{W} (downscale {args.downscale}), '
              f'{n} frames')

        lm = parse_ir_features(spec['ir_file'])           # [F, 28, 2]
        boxes, ann = build_candidates(lm, (H, W), grid=args.grid)
        if args.downscale > 1:                            # scale boxes to grid
            boxes = {k: {'x': int(round(b['x'] * scale)),
                         'y': int(round(b['y'] * scale)),
                         'w': max(2, int(round(b['w'] * scale))),
                         'h': max(2, int(round(b['h'] * scale)))}
                     for k, b in boxes.items()}
        belt = load_belt(spec['resp_file'], n, args.fps)
        motion = landmark_motion(spec['ir_file'], n)
        n = int(min(n, belt.size, motion.shape[0]))
        frames = frames[:n]
        gmean = frames.reshape(n, -1, 3).mean(axis=1)     # [n, 3], per frame

        rec: Dict = {'frame_hw': [H, W], 'n_frames': n,
                     'boxes': boxes, 'anatomy': ann, 'variants': {}}
        refs = {'belt': belt[:n], 'motion_x': motion[:n, 0]}
        # primary map (for the figure): detrended band belt
        primary: Optional[np.ndarray] = None
        # detrended FIRST so the PRIMARY combo (the one whose location drives the
        # verdict) is printed before the diagnostic's cheaper side-combos.
        for variant in ('detrended', 'raw'):
            gm = None if variant == 'raw' else gmean
            X = build_series(frames, gm, band, args.fps)   # [P, n] float32
            for ref, y0 in refs.items():
                y = np.asarray(y0, dtype=np.float64)
                # Match run_gate EXACTLY: it band-passes BOTH the pixel series
                # and the reference. Skipping this on y leaves a BAND-LIMITED
                # series correlated against a BROADBAND motion_x, which crushes
                # the positive control (0.9 -> 0.34) for no physical reason.
                if band:
                    y = bandpass(y, args.fps, band)
                y = (y - y.mean()) / (y.std() + 1e-12)
                best, blag = _max_abs_corr_locate(X, y, lags, args.device,
                                                  args.block)
                p = int(best.argmax())
                h, w, c = pixel_xyc(p, W)
                regs = regions_of(w, h, boxes)
                # per-REGION argmax: the best pixel each named box contains,
                # so the face-vs-off-face comparison is at ONE resolution
                region_best: Dict[str, Dict] = {}
                for rname, b in boxes.items():
                    sub = best.reshape(H, W, 3)[b['y']:b['y'] + b['h'],
                                                b['x']:b['x'] + b['w'], :]
                    if sub.size == 0:
                        continue
                    flat = int(sub.reshape(-1).argmax())
                    rh, rw, rc = np.unravel_index(flat, sub.shape)
                    region_best[rname] = {
                        'best': float(sub.reshape(-1)[flat]),
                        'x': int(b['x'] + rw), 'y': int(b['y'] + rh),
                        'ch': int(rc)}
                res = {'best': float(best[p]), 'x': int(w), 'y': int(h),
                       'ch': int(c), 'lag': int(blag[p]), 'regions': regs,
                       'region_best': region_best,
                       'n_pixels': int(best.size)}
                # The phase-scrambled null is ~1 pass x nperm, so compute it ONLY
                # for the PRIMARY combo (detrended band vs belt) whose location
                # drives the verdict -- the other combos need the map, not a
                # p-value (the gate already established significance, and
                # null-ing motion_x/raw here would cost ~13x for no new claim).
                if args.nperm > 0 and variant == 'detrended' and ref == 'belt':
                    rng = np.random.default_rng(0)
                    nul = np.empty(args.nperm, dtype=np.float64)
                    for i in range(args.nperm):
                        ys = phase_scramble(y, rng)
                        nul[i] = _global_max(X, ys, lags, args.device,
                                             args.block)
                    res['null95'] = float(np.percentile(nul, 95))
                    res['null_median'] = float(np.median(nul))
                    res['p'] = float((nul >= best[p]).mean())
                    print(f'  [{variant:9s} vs {ref:8s}] null95 '
                          f'{res["null95"]:.3f} p {res["p"]:.3f}')
                rec['variants'][f'{variant}|{ref}'] = res
                print(f'  [{variant:9s} vs {ref:8s}] ARGMAX at (x={w}, y={h}, '
                      f'ch={c}) |r|={best[p]:.3f} lag={blag[p]}  '
                      f'regions={regs or ["<none>"]}')
                fb = region_best.get('face', {})
                print(f'      region max|r|: frame {region_best.get("frame",{}).get("best",float("nan")):.3f}'
                      f' | face {fb.get("best",float("nan")):.3f}'
                      f' | nose_mouth {region_best.get("nose_mouth",{}).get("best",float("nan")):.3f}'
                      f' | bg_left {region_best.get("bg_left",{}).get("best",float("nan")):.3f}'
                      f' | bg_right {region_best.get("bg_right",{}).get("best",float("nan")):.3f}')
                if variant == 'detrended' and ref == 'belt':
                    primary = best.reshape(H, W, 3)
            del X
        out['sessions'][name] = rec

        if primary is not None:
            base, ext = os.path.splitext(args.figure)
            _save_figure(f'{base}_{name}{ext}', name, primary.max(axis=2),
                         boxes, rec)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as fh:
            json.dump(out, fh, indent=2, default=float)
        print(f'\njson -> {args.out}')
    return 0


def _save_figure(path: str, name: str, m: np.ndarray,
                 boxes: Dict[str, Dict[str, int]], rec: Dict) -> None:
    """Heatmap of per-pixel max|r| (detrended belt), boxes + argmax overlaid."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as exc:                              # pragma: no cover
        print(f'[argmax] figure skipped ({exc})')
        return
    res = rec['variants'].get('detrended|belt', {})
    fig, ax = plt.subplots(figsize=(11, 8), dpi=120)
    im = ax.imshow(m, cmap='inferno', aspect='auto')
    fig.colorbar(im, ax=ax, label='per-pixel max |r| vs belt (detrended)')
    for rname, b in boxes.items():
        if rname.startswith('tile_'):
            continue
        ax.add_patch(Rectangle((b['x'], b['y']), b['w'], b['h'], fill=False,
                               edgecolor='cyan', lw=1.2))
        ax.text(b['x'], max(0, b['y'] - 4), rname, color='cyan', fontsize=8)
    if 'x' in res:
        ax.plot(res['x'], res['y'], marker='*', color='lime', ms=18, mec='k')
        ax.set_title(f'{name}: whole-frame argmax (x={res["x"]}, y={res["y"]}, '
                     f'ch={res["ch"]}) |r|={res["best"]:.3f} '
                     f'regions={res.get("regions")}')
    else:
        ax.set_title(f'{name}: per-pixel max |r| (detrended belt)')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f'figure -> {path}')


if __name__ == '__main__':
    sys.exit(main())
