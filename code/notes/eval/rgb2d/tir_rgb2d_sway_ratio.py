#!/usr/bin/env python
"""What ``rgb2d_max_sway_px`` reproduces the TIR gate's ``max_sway_px = 40``?

The two gates use the SAME statistic (``tir_resp_dataset.landmark_sway``,
centroid excursion inside one window), but in each modality's own source pixels:
the thermal camera delivers 726x480, the visible one 1392x1040, so 40 px on TIR
is roughly half the head travel that 40 px means on RGB. Copying the number
would make the RGB2D gate ~2x stricter than the TIR gate; copying nothing leaves
the two unrelated.

This script measures the conversion factor **empirically on the same sessions**:
for every ``(subject, task)`` that has BOTH an ``IRFeatures`` thermal track and a
``2DFeatures`` visible track, it computes the per-window sway of each and takes
the ratio. That ratio is the only defensible answer to "the same amount of head
motion", because it compares the two measurements of one physical movement
instead of assuming a FOV ratio.

Outputs two candidate thresholds:

* ``scale``  -- ``40 * median(sway_rgb) / median(sway_tir)``: same *physical*
  head motion. This is "match the 40 px".
* ``quantile`` -- the RGB value whose *retention* equals the TIR gate's, i.e. the
  RGB threshold that rejects the same fraction of clips TIR rejects. Use this if
  the intent is "the same filter strength" rather than "the same motion".

Usage
-----
    cd code
    ../.env/bin/python notes/eval/rgb2d/tir_rgb2d_sway_ratio.py \
        --raw_root "$RAW_DATA_PATH" --limit 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from data import rgb_features as rf
from data import rgb_roi_dataset as rrd
from data import tir_resp_dataset as trd

TIR_THRESHOLD = 40.0
FACE_2D = list(range(1, 50))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw_root', default=os.environ.get('RAW_DATA_PATH', ''))
    p.add_argument('--limit', type=int, default=0, help='0 = all sessions')
    p.add_argument('--subjects', default='')
    p.add_argument('--clip_seconds', type=float, default=8.0)
    p.add_argument('--fps', type=float, default=25.0)
    p.add_argument('--tir_threshold', type=float, default=TIR_THRESHOLD)
    p.add_argument('--out', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'tir_rgb2d_sway_ratio.json'))
    return p.parse_args(argv)


def track_2d(feat_file: str) -> np.ndarray:
    pts, _pose, frame, valid = rf.parse_2d_features(feat_file)
    order = np.argsort(frame)
    return pts[order][np.asarray(valid, bool)[order]]


def windows(track: np.ndarray, target_idx: np.ndarray, w: int, hop: int,
            n_frames: int) -> List[float]:
    """Sway of every stride-``hop`` window; ``n_frames`` is the frame count."""
    out: List[float] = []
    n = min(int(n_frames), int(track.shape[0]))
    for f0 in range(0, max(1, n - w + 1), hop):
        win = track[f0:f0 + w]
        if win.shape[0] < 2:
            continue
        out.append(float(trd.landmark_sway(win, target_idx)))
    return out


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

    w = max(1, int(round(args.clip_seconds * args.fps)))
    idx_2d = np.asarray([i - 1 for i in FACE_2D], dtype=np.int64)
    idx_tir = np.asarray(list(trd.TARGET_LANDMARKS), dtype=np.int64) - 1

    pair: List[Dict[str, object]] = []
    n_no_tir = n_no_2d = 0
    for s in sessions:
        subj, task = str(s['subject']), str(s['task'])
        ir_file = trd.find_ir_features(args.raw_root, subj, task)
        feat = str(s.get('feat_file') or '')
        if ir_file is None:
            n_no_tir += 1
            continue
        if not os.path.isfile(feat):
            n_no_2d += 1
            continue
        try:
            tir_track = trd.parse_ir_features(ir_file)
            two_track = track_2d(feat)
        except (ValueError, OSError) as exc:
            print(f'  skip {subj}_{task}: {exc}')
            continue
        n = int(s.get('n_jpg') or 0) or min(tir_track.shape[0], two_track.shape[0])
        sw_tir = windows(tir_track, idx_tir, w, w, n)
        sw_2d = windows(two_track, idx_2d, w, w, n)
        for i in range(min(len(sw_tir), len(sw_2d))):
            a, b = sw_tir[i], sw_2d[i]
            if not (np.isfinite(a) and np.isfinite(b)) or a <= 0.0:
                continue
            pair.append({'session': f'{subj}_{task}', 'start': i * w,
                         'tir': a, 'rgb2d': b, 'ratio': b / a})
        if len(pair) and len(pair) % 500 < 2:
            print(f'  {subj}_{task}: {len(pair)} windows paired', flush=True)

    if not pair:
        print('no session has BOTH tracks -- nothing to compare', file=sys.stderr)
        return 1

    tir = np.asarray([p['tir'] for p in pair], dtype=np.float64)
    rgb = np.asarray([p['rgb2d'] for p in pair], dtype=np.float64)
    ratio = rgb / tir
    kept_tir = float((tir <= args.tir_threshold).mean())
    matched = float(np.quantile(rgb, kept_tir)) if 0.0 < kept_tir < 1.0 else float('nan')
    per_sess: Dict[str, Dict[str, float]] = {}
    for p in pair:
        e = per_sess.setdefault(str(p['session']), {'tir': 0.0, 'rgb2d': 0.0})
        e['tir'] = max(e['tir'], float(p['tir']))
        e['rgb2d'] = max(e['rgb2d'], float(p['rgb2d']))

    report = {
        'raw_root': os.path.abspath(args.raw_root),
        'clip_seconds': args.clip_seconds,
        'tir_threshold_px': args.tir_threshold,
        'sessions_with_both_tracks': len(per_sess),
        'sessions_missing_tir': n_no_tir,
        'sessions_missing_2d': n_no_2d,
        'windows_paired': len(pair),
        'tir_sway_px': {f'p{q}': round(float(np.percentile(tir, q)), 2)
                        for q in (50, 75, 90, 95, 99)},
        'rgb2d_sway_px': {f'p{q}': round(float(np.percentile(rgb, q)), 2)
                          for q in (50, 75, 90, 95, 99)},
        'ratio_rgb2d_over_tir': {f'p{q}': round(float(np.percentile(ratio, q)), 3)
                                 for q in (10, 25, 50, 75, 90)},
        'tir_retention_at_threshold': round(kept_tir, 4),
        'scale_matched_rgb2d_px': round(
            float(args.tir_threshold * np.median(rgb) / np.median(tir)), 1),
        'retention_matched_rgb2d_px': round(matched, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    print(f'\nwrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
