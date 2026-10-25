#!/usr/bin/env python3
"""Scan the ROI cache for the thermal-appearance features of every session.

Companion to ``t3_spread_explained.py`` (which can compute the same features for
one task). This one scans the WHOLE corpus so the per-SUBJECT trait can be
measured on tasks other than the one being explained: the T3 spread turned out to
be dominated by a same-subject-everywhere effect, so the natural question is
whether that trait is the subject's thermal appearance (how much palette detail
the face actually offers) rather than their breathing.

Features per subject-task (all from the model's OWN input crop, no decode of the
video -- the shard already holds it):

* ``sat_frac``          fraction of crop pixels with the red channel >= 250
                        (the palette's hot end is saturated on this corpus)
* ``crop_g_std``        median per-frame spatial std of the green channel
* ``roi_g_std``         same, restricted to the 12-point nose+mouth box
* ``roi_sat_frac``      saturation fraction inside that box
* ``roi_thermal_tot``   temporal std of the box mean, AFTER subtracting the
                        per-frame whole-crop mean (removes AGC / global gain)
* ``roi_thermal_band``  the 0.1-0.6 Hz part of that
* ``roi_thermal_frac``  its share of the total

Frames are subsampled (``--stride``) -- the band of interest is far below the
resulting rate, so nothing is lost and the scan stays I/O bound.

Usage (from ``code/``)::

    ../.env/bin/python notes/eval/tir_resp/roi_contrast_scan.py --workers 16 \\
        --out ../output/eval/resp_kfold/roi_contrast_all.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_CODE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_CODE_DIR))

DEFAULT_CACHE = ('/work/scratch/ne95ocyg/tir_roi_cache/'
                 'tirroi_nose_mouth_pad0.20_minmax_src726x480_native_v2_9d0214aefa')


def contrast_one(args: Tuple[str, float, Tuple[float, float], int]
                 ) -> Optional[dict]:
    """One shard -> feature dict (module-level so it can be pickled)."""
    path, fps, band, stride = args
    try:
        from data import roi_cache as rc
        from data import tir_resp_dataset as trd
        from evaluation.metrics import bandpass_fft
        sh = rc.read_shard(path)
    except Exception:                                       # pragma: no cover
        return None
    frames = sh['frames'][::stride]
    if frames.shape[0] < 8:
        return None
    lm = sh['landmarks'][::stride] if sh['landmarks'] is not None else None
    x0, y0 = sh['box'][0], sh['box'][2]
    H, W = frames.shape[1], frames.shape[2]
    f = frames.astype(np.float32)
    g = f[..., 1]
    glob = g.reshape(g.shape[0], -1).mean(axis=1)
    gq = g - glob[:, None, None]

    out = {
        'session': str(sh['session']),
        'frames': int(sh['n_frames']),
        'crop_w': int(W), 'crop_h': int(H),
        'sat_frac': float((f[..., 0] >= 250).mean()),
        'crop_g_std': float(np.median(g.reshape(g.shape[0], -1).std(axis=1))),
    }
    if lm is not None:
        tgt = lm[:, trd.TARGET_LANDMARK_IDX, :].copy()
        tgt[..., 0] -= x0
        tgt[..., 1] -= y0
        n = g.shape[0]
        q = np.full(n, np.nan)
        spread = np.full(n, np.nan)
        sat = np.full(n, np.nan)
        for i in range(n):
            p = tgt[i]
            if not np.isfinite(p).all():
                continue
            a = int(np.clip(np.floor(p[:, 0].min()), 0, W - 2))
            b = int(np.clip(np.ceil(p[:, 0].max()), a + 1, W - 1))
            c = int(np.clip(np.floor(p[:, 1].min()), 0, H - 2))
            d = int(np.clip(np.ceil(p[:, 1].max()), c + 1, H - 1))
            patch = gq[i, c:d, a:b]
            if patch.size < 4:
                continue
            q[i] = float(patch.mean())
            spread[i] = float(patch.std())
            sat[i] = float((f[i, c:d, a:b, 0] >= 250).mean())
        if np.isfinite(q).sum() > 8:
            qq = q[np.isfinite(q)]
            qq = qq - qq.mean()
            rate = fps / float(stride)
            bp = bandpass_fft(qq[None, :], rate, band)[0]
            out.update(
                roi_thermal_tot=float(np.std(qq)),
                roi_thermal_band=float(np.sqrt(np.mean(bp ** 2))),
                roi_thermal_frac=float(np.mean(bp ** 2) /
                                       max(float(np.mean(qq ** 2)), 1e-12)),
                roi_g_std=float(np.nanmedian(spread)),
                roi_sat_frac=float(np.nanmedian(sat)),
            )
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--roi_cache', default=os.environ.get('TIR_ROI_CACHE',
                                                         DEFAULT_CACHE))
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--stride', type=int, default=10)
    p.add_argument('--fps', type=float, default=25.0)
    p.add_argument('--band', default='0.1,0.6')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--out', required=True)
    a = p.parse_args()

    band = tuple(float(v) for v in a.band.split(','))
    tasks_dir = os.path.join(a.roi_cache, 'tasks')
    paths = sorted(str(q) for q in Path(tasks_dir).glob('*.npz'))
    if a.limit:
        paths = paths[:a.limit]
    print(f'[contrast] {len(paths)} shards, {a.workers} workers, '
          f'stride {a.stride}')

    jobs = [(q, a.fps, band, a.stride) for q in paths]
    rows: List[dict] = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, res in enumerate(ex.map(contrast_one, jobs, chunksize=8), 1):
            if res is not None:
                rows.append(res)
            if i % 200 == 0:
                print(f'[contrast]   {i}/{len(jobs)}')

    op = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, 'w') as fh:
        json.dump(rows, fh, indent=1)
    print(f'[contrast] wrote {op} ({len(rows)} shards)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
