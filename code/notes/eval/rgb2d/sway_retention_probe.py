#!/usr/bin/env python
"""Phase 0 probe -- how much landmark sway can a hull window contain?

The 2D-ROI cache stores ONE square box per ``(session, clip)``: the union of the
per-frame boxes over that window. A window whose face translates a lot therefore
yields a box padded with background, and the retina is asked to attend to pixel
area that carries no texture. ``rgb2d_max_sway_px`` is the downstream gate that
drops such windows (the cache itself encodes no such decision).

This script measures the *distribution* of per-window sway on the real corpus so
that gate can be set from data instead of guessed. It is deliberately
independent of ``data/rgb2d_cache.py``: it re-unions the span itself and only
shares the ``.mat`` parser with ``data/rgb_features.py``. If the probe imported
the cache, a bug in the cache's hull would be invisible here.

Definitions
-----------
sway(px)  = max over target landmarks and over x/y of (max - min) across the
            frames of the window, i.e. the extent of the moving hull.
box_side  = max(sway_x, sway_y) * (1 + 2 * padding)   (the square the cache
            would store, before any frame clamping)

Usage
-----
    cd code
    ../.env/bin/python notes/eval/rgb2d/sway_retention_probe.py \
        --raw_root "$RAW_DATA_PATH" --limit 200

Writes ``code/notes/eval/rgb2d/sway_retention.json`` next to this file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from data import rgb_features as rf
from data import rgb_roi_dataset as rrd
from data import tir_resp_dataset as trd

THRESHOLDS = (5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw_root', default=os.environ.get('RAW_DATA_PATH', ''))
    p.add_argument('--subjects', default='', help='comma list; default all')
    p.add_argument('--limit', type=int, default=0, help='0 = all sessions')
    p.add_argument('--clip_seconds', type=float, default=8.0)
    p.add_argument('--clip_stride', type=float, default=None,
                   help='default = clip_seconds')
    p.add_argument('--fps', type=float, default=25.0)
    p.add_argument('--padding', type=float, default=0.10)
    p.add_argument('--landmarks', default='face')
    p.add_argument('--thresholds', default=','.join(str(t) for t in THRESHOLDS),
                   help='comma list of max_sway_px values for the curve')
    p.add_argument('--out', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'sway_retention.json'))
    return p.parse_args(argv)


def target_indices(spec: str) -> np.ndarray:
    """'face' = 1..49; otherwise a comma list of 1-based landmark labels."""
    if str(spec).strip() in ('', 'face'):
        return np.arange(rf.NUM_LANDMARKS_2D, dtype=np.int64)
    labels = [int(v) for v in str(spec).split(',') if v.strip()]
    idx = np.asarray([v - 1 for v in labels], dtype=np.int64)
    if idx.min() < 0 or idx.max() >= rf.NUM_LANDMARKS_2D:
        raise ValueError(f'landmarks out of 1..{rf.NUM_LANDMARKS_2D}: {spec}')
    return idx


def session_track(feat_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """``[n,49,2]`` points on the jpg grid + a ``[n]`` finite-row mask."""
    pts, _pose, frame, valid = rf.parse_2d_features(feat_file)
    order = np.argsort(frame)
    pts, frame, valid = pts[order], frame[order], valid[order]
    jpg = np.asarray([rf.jpg_index_of(int(f)) for f in frame], dtype=np.int64)
    keep = np.asarray(valid, bool) & (jpg >= 0)
    if not keep.any():
        return np.zeros((0, rf.NUM_LANDMARKS_2D, 2), np.float64), jpg[:0]
    uniq, first = np.unique(jpg[keep], return_index=True)
    sel = np.flatnonzero(keep)[first]
    return pts[sel], uniq


def window_sway(track: np.ndarray, idx: np.ndarray, w: int, hop: int,
                n_frames: int) -> List[Dict[str, float]]:
    """Per-window metrics of every stride-``hop`` window of ``w`` frames.

    ``sway`` is the GATE statistic and is computed by the very function the
    dataset calls (``tir_resp_dataset.landmark_sway``): the centroid excursion
    ``max_t ||c_t - median(c)||_2``. Using anything else here would calibrate the
    threshold against a different quantity -- notably a hull SPAN, which adds the
    face size to the motion and is ~2-3x larger.

    ``span`` is reported alongside for context: it is the union-box side, i.e.
    what the cache stores and therefore what gets squashed into ``input_size``.
    The two answer different questions ("may the model see this?" vs "how much
    of the crop is face?") and must not be substituted for one another.
    """
    rows: List[Dict[str, float]] = []
    if track.shape[0] == 0:
        return rows
    for f0 in range(0, max(1, n_frames - w + 1), hop):
        f1 = f0 + w
        lo = np.searchsorted(idx, f0)
        hi = np.searchsorted(idx, f1)
        if hi - lo < 2:
            continue
        win = track[lo:hi]
        sway = float(trd.landmark_sway(win, target_idx))     # inf = unusable
        sub = win[:, target_idx, :]
        good = sub[np.isfinite(sub).all(axis=(1, 2))]
        span = (float(np.max(good.max(axis=0) - good.min(axis=0)))
                if good.shape[0] >= 2 else float('nan'))
        rows.append({'start': int(f0), 'sway': sway, 'span': span})
    return rows


def n_jpgs(session: Dict[str, object]) -> int:
    """Frame count: the spec already carries it, else list the source."""
    declared = int(session.get('n_jpg') or 0)
    if declared > 0:
        return declared
    src_dir = session.get('rgb_dir')
    if src_dir and os.path.isdir(str(src_dir)):
        return len([f for f in os.listdir(str(src_dir))
                    if str(f).lower().endswith('.jpg')])
    zip_path, member = session.get('zip'), str(session.get('member') or '')
    if zip_path and os.path.isfile(str(zip_path)):
        import zipfile
        with zipfile.ZipFile(str(zip_path)) as zf:
            return len([n for n in zf.namelist()
                        if n.startswith(member) and n.lower().endswith('.jpg')])
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not args.raw_root:
        print('--raw_root is required (or export RAW_DATA_PATH)', file=sys.stderr)
        return 2

    sessions = rrd.discover_rgb_sessions(args.raw_root)
    if args.subjects:
        want = tuple(s.strip() for s in args.subjects.split(',') if s.strip())
        sessions = [s for s in sessions
                    if str(s.get('subject', '')).startswith(want)]
    if args.limit:
        sessions = sessions[:int(args.limit)]
    if not sessions:
        print('no 2D sessions found under ' + args.raw_root, file=sys.stderr)
        return 2

    global target_idx
    target_idx = target_indices(args.landmarks)

    w = max(1, int(round(args.clip_seconds * args.fps)))
    hop = max(1, int(round((args.clip_stride or args.clip_seconds) * args.fps)))

    per_window: List[Dict[str, float]] = []
    per_session: Dict[str, Dict[str, float]] = {}
    t0 = time.time()
    n_missing = 0
    for k, s in enumerate(sessions):
        name = f"{s['subject']}_{s['task']}"
        feat = str(s.get('feat_file') or '')
        if not os.path.isfile(feat):
            n_missing += 1
            continue
        n = n_jpgs(s)
        if n <= 0:
            n_missing += 1
            continue
        track, idx = session_track(feat)
        rows = window_sway(track, idx, w, hop, n)
        for r in rows:
            r['session'] = name                           # type: ignore[assignment]
        per_window.extend(rows)
        if rows:
            sways = [r['sway'] for r in rows]
            per_session[name] = {
                'n_windows': len(rows),
                'max_sway': max(sways),
                'min_sway': min(sways),
                'median_sway': float(np.median(sways)),
            }
        if (k + 1) % 50 == 0:
            print(f'  {k + 1}/{len(sessions)} sessions, '
                  f'{len(per_window)} windows, {time.time() - t0:.1f}s',
                  flush=True)

    if not per_window:
        print('no windows measured', file=sys.stderr)
        return 1

    sway = np.asarray([r['sway'] for r in per_window])
    span = np.asarray([r['span'] for r in per_window])
    span = span[np.isfinite(span)]
    sess_max = np.asarray([v['max_sway'] for v in per_session.values()])
    sess_min = np.asarray([v['min_sway'] for v in per_session.values()])
    curve = []
    for th in [float(v) for v in str(args.thresholds).split(',') if v.strip()]:
        curve.append({'max_sway_px': th,
                      'windows_kept': round(float((sway <= th).mean()), 4),
                      # strict: every window of the session survives
                      'sessions_kept': round(float((sess_max <= th).mean()), 4),
                      # the dataset gate is per CLIP: a session still trains as
                      # long as ONE of its windows survives
                      'sessions_any_clip_kept': round(
                          float((sess_min <= th).mean()), 4)})

    def pct(a: np.ndarray, q: float) -> float:
        return float(np.percentile(a, q))

    report = {
        'raw_root': os.path.abspath(args.raw_root),
        'clip_seconds': args.clip_seconds,
        'clip_stride_s': args.clip_stride or args.clip_seconds,
        'fps': args.fps,
        'padding': args.padding,
        'landmarks': args.landmarks,
        'n_target_landmarks': int(target_idx.size),
        'sessions': len(sessions),
        'sessions_measured': len(per_session),
        'sessions_skipped': n_missing,
        'windows': len(per_window),
        'sway_px': {f'p{q}': round(pct(sway, q), 2)
                    for q in (50, 75, 90, 95, 99, 100)},
        'span_px': {f'p{q}': round(pct(span, q), 2)
                    for q in (50, 90, 100)} if span.size else {},
        'session_max_sway_px': {f'p{q}': round(pct(sess_max, q), 2)
                                for q in (50, 75, 90, 95, 99, 100)},
        'retention': curve,
        'runtime_s': round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps(report, indent=2))
    print('\nretention curve (windows / sessions-all / sessions-any-clip kept):')
    for row in curve:
        print(f"  {row['max_sway_px']:>4.0f} px  "
              f"{row['windows_kept'] * 100:6.2f}%  "
              f"{row['sessions_kept'] * 100:6.2f}%  "
              f"{row['sessions_any_clip_kept'] * 100:6.2f}%")
    print(f'\nwrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
