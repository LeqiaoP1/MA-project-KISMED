#!/usr/bin/env python
"""ROI sweep for the TIR-ROI -> RESP line: WHICH region of the thermal video
carries respiration?

WHY THIS EXISTS
---------------
`TirROI_Resp_plan.md` section 12 closed the line on the 12-point NOSE+MOUTH
landmark union. But that box is a box on SKIN, and the strongest thermal
respiration signal is the EXHALED-AIR PLUME (37 C air against ~22 C ambient),
which lives in the AIR -- often against the BACKGROUND, and in projection
between the nostrils and the upper lip. A landmark-union box cannot contain
background AT ALL, and the union is dominated by the mouth, so the two nostril
points are diluted across a much larger area.

This SWEEPS boxes through the same gate (`run_gate` from
`native_resolution_gate.py`) and ranks them. It trains nothing, so a
hyperparameter can never flatter a candidate.

WHY v2 EXISTS -- three defects in v1, all fixed here
---------------------------------------------------
1. NESTED CANDIDATES CANNOT BE COMPARED. v1 used `nostril_40 subset nostril_80
   subset face_28`, and the statistic is a MAX OVER PIXELS. Every box containing
   the same best pixel returns the same number, so v1's near-identical rows said
   nothing about regions. FIX: a DISJOINT partition -- the face box is cut into
   a grid of non-overlapping tiles, so each number describes pixels no other box
   has. That is the only way a "which region" statement is meaningful.
2. A GLOBAL COMPONENT WAS NOT CONTROLLED. v1 found the background tracking
   `motion_x` at 0.899 (p=0.000): head position co-varies with a FRAME-WIDE
   signal (most likely thermal-camera auto-gain following the subject). Any box
   could then "detect" through that global channel. FIX: (a) a WHOLE-FRAME box
   quantifies the global component; (b) every box is ALSO measured with the
   per-frame whole-frame mean SUBTRACTED, so a result that survives detrending
   is LOCAL, and one that does not was the global channel.
3. `nperm 20` makes p=0.000 the weakest possible statement. FIX: default 100.

The `vs motion_x` rows are the positive control, with the caveat v1 exposed:
`motion_x` is NOT spatially specific (it fires off-face), so it proves the
pipeline has sensitivity, not that a region is informative.

The BELT reference is `Physiology/<S>/<T>/Resp_Volts.txt` (per task). Standing
confound: the belt and head motion are correlated via subject activity, so a
detected belt row is not yet respiration-specific.

Usage (from ``code/``)::

    python analysis/tir_resp/roi_sweep.py --sessions F001_T3,F002_T1 --grid 4
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
from data.tir_resp_dataset import (NUM_LANDMARKS, discover_sessions,  # noqa: E402
                                   parse_ir_features)
from native_resolution_gate import (landmark_motion, load_belt,      # noqa: E402
                                    run_gate)

#: 1-indexed landmark labels (user guide Fig. 3) -> short names, for annotating
#: which anatomy each DISJOINT tile covers.
LABELS = {9: 'nosebridge.R', 10: 'NOSTRIL.R', 11: 'mouthcorner.R',
          12: 'upperlip.R', 13: 'lowerlip.R', 20: 'nosebridge.L',
          21: 'NOSTRIL.L', 22: 'mouthcorner.L', 23: 'upperlip.L',
          24: 'lowerlip.L', 25: 'upperlip.C', 26: 'lowerlip.C'}


def _clip(b: Dict[str, int], fh: int, fw: int) -> Dict[str, int]:
    b = dict(b)
    b['x'] = max(0, min(int(b['x']), fw - 2))
    b['y'] = max(0, min(int(b['y']), fh - 2))
    b['w'] = max(2, min(int(b['w']), fw - b['x']))
    b['h'] = max(2, min(int(b['h']), fh - b['y']))
    return b


def _union(pts: np.ndarray, pad: float, fh: int, fw: int) -> Dict[str, int]:
    x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
    y0, y1 = float(pts[:, 1].min()), float(pts[:, 1].max())
    w, h = max(x1 - x0, 2.0), max(y1 - y0, 2.0)
    return _clip({'x': int(round(x0 - pad * w)), 'y': int(round(y0 - pad * h)),
                  'w': int(round(w * (1 + 2 * pad))),
                  'h': int(round(h * (1 + 2 * pad)))}, fh, fw)


def build_candidates(lm: np.ndarray, frame_hw: Tuple[int, int], grid: int = 4,
                     pad: float = 0.2
                     ) -> Tuple[Dict[str, Dict[str, int]], Dict[str, List[str]]]:
    """Boxes + an anatomy annotation per box.

    The face box (all 28 landmarks, pad 0.1) is partitioned into `grid x grid`
    DISJOINT tiles. The non-grid boxes are separate controls and are never
    ranked against the tiles -- only tiles are mutually exclusive.
    """
    fh, fw = frame_hw
    med = np.nanmedian(lm, axis=0)                        # [28, 2] -> (x, y)
    face = _union(med, 0.1, fh, fw)
    cands: Dict[str, Dict[str, int]] = {
        'frame': {'x': 0, 'y': 0, 'w': fw, 'h': fh},      # global control
        'face': face,                                     # whole-face reference
        'nose_mouth': _union(
            med[[8, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24, 25]], pad, fh, fw),
    }
    # off-face background controls: 2 face-widths to each side
    for side, dx in (('bg_left', -2.0), ('bg_right', 2.0)):
        cands[side] = _clip({'x': face['x'] + int(dx * face['w']),
                             'y': face['y'], 'w': face['w'], 'h': face['h']},
                            fh, fw)
    # DISJOINT tiles -- the only mutually comparable boxes
    tw, th = face['w'] / float(grid), face['h'] / float(grid)
    for r in range(grid):
        for c in range(grid):
            cands[f'tile_{r}{c}'] = _clip(
                {'x': int(round(face['x'] + c * tw)),
                 'y': int(round(face['y'] + r * th)),
                 'w': int(round(tw)), 'h': int(round(th))}, fh, fw)

    ann: Dict[str, List[str]] = {}
    for name, b in cands.items():
        names = []
        for lab, (x, y) in zip(range(1, NUM_LANDMARKS + 1), med):
            if (b['x'] <= x < b['x'] + b['w']) and \
                    (b['y'] <= y < b['y'] + b['h']):
                names.append(LABELS.get(lab, f'lm{lab}'))
        ann[name] = names
    return cands, ann


def decode_boxes(video: str, boxes: Dict[str, Dict[str, int]], size: int,
                 chunk: int = 200
                 ) -> Tuple[Dict[str, np.ndarray], np.ndarray, int,
                            Tuple[int, int]]:
    """Decode ONCE; crop every box per chunk; also track the global frame mean.

    One decode for N boxes: decoding per box would multiply the dominant cost by
    N. Returns ``(series, frame_mean, n_frames, (fh, fw))`` with
    ``series[name]`` shaped ``[F, size*size*3]`` uint8 and ``frame_mean`` shaped
    ``[F, 3]`` -- the whole-frame RGB mean per frame, i.e. exactly the global
    channel (camera auto-gain / illumination) that the `detrended` variant
    removes.
    """
    import cv2
    acc: Dict[str, List[np.ndarray]] = {k: [] for k in boxes}
    gmean: List[np.ndarray] = []
    n_seen, fh, fw = 0, 0, 0
    with vio.open_video(video) as r:
        n = int(r.num_frames)
        for s in range(0, n, chunk):
            part = r.read_range(s, min(chunk, n - s))    # [t, H, W, 3] RGB
            if part is None or len(part) == 0:
                break
            fh, fw = part.shape[1], part.shape[2]
            gmean.append(part.reshape(len(part), -1, 3).mean(axis=1))
            for k, b in boxes.items():
                y0, y1 = b['y'], b['y'] + b['h']
                x0, x1 = b['x'], b['x'] + b['w']
                for f in part:
                    sub = f[y0:y1, x0:x1]
                    if sub.shape[0] < 2 or sub.shape[1] < 2:
                        continue
                    # the DATASET's rule: INTER_AREA down, INTER_LINEAR up
                    interp = (cv2.INTER_AREA
                              if size <= min(sub.shape[0], sub.shape[1])
                              else cv2.INTER_LINEAR)
                    acc[k].append(cv2.resize(sub, (size, size),
                                             interpolation=interp).reshape(-1))
            n_seen += len(part)
    return ({k: np.stack(v) for k, v in acc.items() if v},
            np.concatenate(gmean) if gmean else np.zeros((0, 3)),
            n_seen, (fh, fw))


def probe_frame_hw(video: str) -> Tuple[int, int]:
    """Frame HxW without decoding the whole file (the boxes must be clipped)."""
    with vio.open_video(video) as r:
        one = r.read_range(0, 1)
    return int(one.shape[1]), int(one.shape[2])


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw_root', default=os.environ.get(
        'RAW_DATA_PATH', '/work/projects/l0003511/test'))
    ap.add_argument('--sessions', default='F001_T3,F002_T1')
    ap.add_argument('--grid', default=4, type=int,
                    help='the face box is cut into grid x grid DISJOINT tiles')
    ap.add_argument('--size', default=48, type=int,
                    help='every box is resized to this square, so the pixel '
                         'COUNT is constant and the comparison is of REGION, '
                         'not of pixel budget.')
    ap.add_argument('--detrend', choices=('both', 'raw', 'detrended'),
                    default='both',
                    help="'detrended' subtracts the per-frame whole-frame mean "
                         '(camera auto-gain / global illumination); a result '
                         'that survives it is LOCAL')
    ap.add_argument('--band', default='0.1,0.6')
    ap.add_argument('--lags', default=3.0, type=float)
    ap.add_argument('--fps', default=25.0, type=float)
    ap.add_argument('--nperm', default=100, type=int)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default='analysis/tir_resp/roi_sweep.json')
    args = ap.parse_args(argv)

    lags = np.arange(-int(round(args.lags * args.fps)),
                     int(round(args.lags * args.fps)) + 1)
    bands: List[Optional[Tuple[float, float]]] = [None]
    if args.band.strip():
        bands.append(tuple(float(v) for v in args.band.split(',')))
    bname = lambda b: 'raw' if b is None else f'{b[0]}-{b[1]}'    # noqa: E731
    size = args.size if args.size > 0 else None
    variants = (['raw', 'detrended'] if args.detrend == 'both'
                else [args.detrend])

    print(f'[roi] raw_root : {args.raw_root}')
    print(f'[roi] grid     : {args.grid}x{args.grid} DISJOINT face tiles')
    print(f'[roi] variants : {variants} | resize {size}px | '
          f'nperm {args.nperm}')
    print(f'[roi] lags     : +-{args.lags}s = {lags.size} samples\n')

    out: Dict = {'nperm': args.nperm, 'resize': size, 'sessions': {}}
    # row = (session, candidate, band, kind, best, null95, p)
    rows: List[Tuple] = []

    for name in [s.strip() for s in args.sessions.split(',') if s.strip()]:
        subj, task = name.split('_')
        spec = discover_sessions(args.raw_root, subjects=[subj],
                                 tasks=[task])[0]
        lm = parse_ir_features(spec['ir_file'])          # [F, 28, 2]
        # Frame size first: the boxes are clipped to the frame, so they cannot
        # be built before it is known.
        fh, fw = probe_frame_hw(spec['video'])
        cands, ann = build_candidates(lm, (fh, fw), grid=args.grid)
        series, gmean, n_frames, (fh, fw) = decode_boxes(spec['video'], cands,
                                                         size or 48)
        belt = load_belt(spec['resp_file'], n_frames, args.fps)
        motion = landmark_motion(spec['ir_file'], n_frames)
        n = int(min(n_frames, belt.size, motion.shape[0], gmean.shape[0]))
        if n < 100:
            print(f'=== {name} === too few frames ({n}) -> skip')
            continue
        print(f'=== {name} === {fh}x{fw} frame, {n} frames, {len(cands)} boxes, '
              f'variants {variants}')

        rec: Dict = {'frame_hw': [fh, fw], 'n_frames': n,
                     'boxes': {k: v for k, v in cands.items()},
                     'anatomy': ann, 'gates': {}}
        # per-channel index of every flattened pixel, for the detrending
        P = (size or 48) * (size or 48) * 3
        ch_idx = np.arange(P) % 3
        for cname, b in cands.items():
            if cname not in series:
                continue
            base = series[cname][:n].astype(np.float32)
            for variant in variants:
                x = base
                if variant == 'detrended':
                    # subtract the per-frame whole-frame mean, channel-wise:
                    # removes gain/illumination drift common to the whole frame
                    x = base - gmean[:n][:, ch_idx]
                for band in bands:
                    for ref, y in (('belt', belt[:n]),
                                   ('motion_x', motion[:n, 0])):
                        res = run_gate(
                            x, y, args.fps, band, lags, args.nperm,
                            args.device, np.random.default_rng(0),
                            f'{cname:14s} {variant:9s} {bname(band):9s} vs {ref}')
                        rec['gates'][
                            f'{cname}|{variant}|{bname(band)}|{ref}'] = res
                        rows.append((name, cname, variant, bname(band), ref,
                                     res['best'], res['null95'], res['p'],
                                     ','.join(ann.get(cname, [])) or '-'))
        out['sessions'][name] = rec

    def table(title: str, keep) -> None:
        print('\n' + '=' * 104)
        print(title)
        print('=' * 104)
        print(f'{"session":10s} {"box":14s} {"variant":9s} {"band":9s} '
              f'{"ref":9s} {"best|r|":>8s} {"null95":>8s} {"p":>7s}  '
              f'{"verdict":9s} anatomy')
        for s_, c_, v_, b_, k_, best, q95, p, a_ in rows:
            if not keep(c_, v_, b_, k_):
                continue
            print(f'{s_:10s} {c_:14s} {v_:9s} {b_:9s} {k_:9s} {best:8.3f} '
                  f'{q95:8.3f} {p:7.3f}  '
                  f'{"DETECTED" if p < 0.05 else "-":9s} {a_[:34]}')

    table('BELT, DETRENDED (local only) -- p<0.05 = this region carries it',
          lambda c, v, b, k: k == 'belt' and v == 'detrended')
    table('BELT, RAW (includes the global channel) -- compare row by row',
          lambda c, v, b, k: k == 'belt' and v == 'raw')
    table('CONTROLS (raw|raw): motion_x = sensitivity; frame/bg MUST NOT fire '
          'vs belt',
          lambda c, v, b, k: v == 'raw' and b == 'raw' and
          (k == 'motion_x' or c in ('frame', 'bg_left', 'bg_right')))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as fh_:
            json.dump(out, fh_, indent=2, default=float)
        print(f'\njson -> {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
