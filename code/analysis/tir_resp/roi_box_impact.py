"""How much did the TIR-ROI box move? -- impact analysis for `roi_quantile`.

`roi_quantile` (an outlier-robust percentile box) was REMOVED on 2026-10-08: it
was 0.0 in every config, i.e. numerically inert, and the removal was bit-exact.
This script is the measurement that was taken first, and it is the reference for
the magnitudes if the knob is ever re-added (see `analysis/hpc_precomputation.md`
section 8, which also carries the re-add recipe and the test that was missing).

Question it answers: how much work WAS ``roi_quantile`` doing, and what would the
two competing box rules cost?

It needs **LANDMARKS ONLY -- no video decode**, because the box is a pure function
of the ``IRFeatures`` track (``roi_box_from_landmarks``). The whole corpus can
therefore be scanned in minutes on a CPU-only node, which is why this is the cheap
gate before touching the contract.

Background. ``data/tir_resp_dataset.DEFAULT_ROI_QUANTILE`` carries a measured note
(2026-09-26): "the min/max union box grows with head motion; sessions with a SMALL
box + low landmark jitter reconstruct at Pearson ~0.7 while large/jerky ones sit
at ~0.0 (possible because the inflated crop is mostly STATIC BACKGROUND, so the
mean-pooled tokens encode pose rather than nostril temperature)". ``roi_quantile``
is the recorded mitigation for that, and every shipped config sets it to ``0.0``.

Reported per subject-task, then aggregated:

* ``clip`` box at the SHIPPED Stage-2 geometry (8 s window, 2 s hop, pad 0.2, q=0)
* ``task`` box = min/max over the WHOLE task + the same padding -- i.e. the
  "one bounding box per task, applied to all its clips" proposal
* **linear inflation** ``sqrt(task_area / clip_area)``: how much facial detail a
  task box gives up at a fixed ``input_size`` (a bigger box is downsampled harder)
* **hop sensitivity**: clip boxes at the Stage-3 hop (1 s) vs Stage-2 (2 s)
* **quantile impact**: linear reduction of the clip box at q in {0.02,0.05,0.1,0.2}
* **landmark quality** (the user's premise): rate of NaN rows and of ``(0,0)``
  tracker-sentinel rows -- "high quality landmarks" is a claim about THESE, while
  the quantile question is about MOTION; they are independent
* **jitter**: mean per-landmark std over the task, i.e. the "jerky" axis of the
  2026-09-26 note, to see whether the two box regimes it describes exist

Usage (from ``code/``; CPU-only, no Slurm needed for a sample):
    python analysis/tir_resp/roi_box_impact.py --limit 40
    python analysis/tir_resp/roi_box_impact.py --workers 16
"""
import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from data import tir_resp_dataset as trd                    # noqa: E402

#: Thermal frame size (measured; used only for clamping the box to the frame).
SRC_W, SRC_H = 726, 480
#: The clipped-landmark set the ROI is built from (12 points, nose + mouth).
TGT = trd.TARGET_LANDMARK_IDX


def _box(pts: np.ndarray, padding: float
         ) -> Optional[Tuple[int, int, int, int]]:
    """Min/max box of one window, or ``None`` when it has no usable landmarks.

    This is the PRODUCTION path: the box is a pure function of the landmark
    track (``roi_box_from_landmarks``), so no video has to be decoded.
    """
    try:
        return trd.roi_box_from_landmarks(pts.reshape(-1, 2), SRC_W, SRC_H,
                                          padding)
    except ValueError:
        return None


def _percentile_box(pts: np.ndarray, padding: float, q: float
                    ) -> Optional[Tuple[int, int, int, int]]:
    """PERCENTILE box -- LOCAL re-implementation of the REMOVED ``roi_quantile``.

    Production no longer has this branch (``roi_quantile`` was removed on
    2026-10-08; see ``analysis/hpc_precomputation.md`` section 8), so it lives
    here ONLY to keep the "what the knob bought" table reproducible. Padding,
    rounding and clamping go through the same helpers as the min/max path
    (``trd._clamp_box``), so the two boxes differ only in their bounds.
    Validated by reproducing the recorded magnitudes: q=0.05 -> x0.835,
    q=0.10 -> x0.713, q=0.20 -> x0.536.
    """
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    p = p[np.isfinite(p).all(axis=1)]
    if p.size == 0:
        raise ValueError('no finite landmark coordinates in this window')
    lo = np.percentile(p, 100.0 * q, axis=0)
    hi = np.percentile(p, 100.0 * (1.0 - q), axis=0)
    x0, x1 = float(lo[0]), float(hi[0])
    y0, y1 = float(lo[1]), float(hi[1])
    w, h = x1 - x0, y1 - y0
    ix0, ix1 = trd._clamp_box(math.floor(x0 - padding * w),
                              math.ceil(x1 + padding * w) + 1, SRC_W)
    iy0, iy1 = trd._clamp_box(math.floor(y0 - padding * h),
                              math.ceil(y1 + padding * h) + 1, SRC_H)
    return ix0, ix1, iy0, iy1


def _linear(box: Tuple[int, int, int, int]) -> float:
    """Geometric-mean side length of a box, in source pixels."""
    x0, x1, y0, y1 = box
    return float(np.sqrt(max(x1 - x0, 1) * max(y1 - y0, 1)))


def _stats(task: dict, window: int, hops: List[int], padding: float,
           quantiles: List[float], drop_sentinel: bool = False
           ) -> Optional[Dict[str, float]]:
    """All box statistics for one subject-task.

    ``drop_sentinel`` masks ``(0,0)`` tracker-failure frames to NaN so
    ``roi_box_from_landmarks`` ignores them (it drops non-finite rows). Without
    it a SINGLE failed frame -- which sits far from the face -- blows up a
    min/max box, which would both distort these statistics and be a real hazard
    for a one-box-per-task rule.
    """
    ir_file = task['ir_file']
    if not ir_file or not os.path.isfile(ir_file):
        return None
    try:
        ir = trd.parse_ir_features(ir_file)
    except (ValueError, OSError):
        return None
    pts_all = ir[:, TGT, :]                                  # [T, 12, 2]
    n = pts_all.shape[0]
    if n < window:
        return None

    finite = np.isfinite(pts_all).all(axis=2)                # [T, 12]
    sentinel = (np.abs(pts_all) < 1e-9).all(axis=2)          # [T, 12] (0,0) rows
    if drop_sentinel:
        pts_all = np.where(sentinel[..., None], np.nan, pts_all)
    out: Dict[str, float] = {
        'frames': float(n),
        'nan_frac': float(1.0 - finite.all(axis=1).mean()),
        'sentinel_frac': float(sentinel.all(axis=1).mean()),
    }

    # task box: min/max over the WHOLE task (the one-box-per-task proposal)
    tb = _box(pts_all.reshape(-1, 2), padding)
    if tb is None:
        return None
    out['task_lin'] = _linear(tb)

    # clip boxes at every hop
    for hop in hops:
        boxes = []
        for start in range(0, n - window + 1, hop):
            b = _box(pts_all[start:start + window], padding)
            if b is not None:
                boxes.append(b)
        if not boxes:
            continue
        lin = np.array([_linear(b) for b in boxes])
        out[f'clip_lin_{hop}'] = float(np.median(lin))
        out[f'clip_lin_p90_{hop}'] = float(np.percentile(lin, 90))
        # how much the box varies WITHIN a task (per-clip adaptivity)
        out[f'clip_spread_{hop}'] = float(lin.max() / max(lin.min(), 1e-6))
        out[f'clip_n_{hop}'] = float(len(lin))
        # the resolution a task box would cost, in linear terms
        out[f'infl_{hop}'] = out['task_lin'] / max(float(np.median(lin)), 1.0)

    # what roi_quantile WOULD buy on the SHIPPED hop (removed knob, local impl.)
    med_main = out.get(f'clip_lin_{hops[0]}')
    for q in quantiles:
        red = []
        for start in range(0, n - window + 1, hops[0]):
            w_pts = pts_all[start:start + window]
            b0 = _box(w_pts, padding)
            try:
                bq = _percentile_box(w_pts, padding, q)
            except ValueError:
                bq = None
            if b0 is not None and bq is not None:
                red.append(_linear(bq) / max(_linear(b0), 1.0))
        if red and med_main:
            out[f'q{int(q * 100)}_ratio'] = float(np.median(red))

    # jitter: the "jerky" axis of the 2026-09-26 note
    ok = pts_all[finite.all(axis=1)]
    out['jitter'] = float(np.nanmean(np.nanstd(ok, axis=0))) if ok.size else np.nan
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw_root', default=os.environ.get(
        'RAW_DATA_PATH', '/work/projects/l0003511/test'))
    p.add_argument('--padding', type=float, default=0.2,
                   help='the shipped roi_padding (configs use 0.2)')
    p.add_argument('--fps', type=float, default=25.0)
    p.add_argument('--clip_seconds', type=float, default=8.0)
    p.add_argument('--hops', type=float, nargs='+', default=[2.0, 1.0],
                   help='window hops in seconds: Stage-2 ships 2.0, Stage-3 1.0')
    p.add_argument('--quantiles', type=float, nargs='+',
                   default=[0.02, 0.05, 0.10, 0.20])
    p.add_argument('--limit', type=int, default=0, help='0 = every subject-task')
    p.add_argument('--drop_sentinel', action='store_true',
                   help='exclude (0,0) tracker-failure frames from the boxes '
                        '(one failed frame otherwise blows up a min/max box)')
    p.add_argument('--workers', type=int, default=0,
                   help='0 = single process (multiprocessing needs care on Slurm)')
    p.add_argument('--csv', default='', help='optional per-task CSV output path')
    args = p.parse_args()

    sessions = trd.discover_sessions(args.raw_root)
    if args.limit:
        sessions = sessions[:args.limit]
    window = int(round(args.clip_seconds * args.fps))
    hops = [int(round(h * args.fps)) for h in args.hops]

    print(f'[box-impact] raw_root   : {args.raw_root}')
    print(f'[box-impact] subject-tasks: {len(sessions)}')
    print(f'[box-impact] geometry   : {window} frame window, '
          f'hops {hops} frames = {args.hops} s, padding {args.padding}')
    print(f'[box-impact] quantiles  : {args.quantiles}')
    print(f'[box-impact] drop_sentinel: {args.drop_sentinel}')
    print()

    work = [(s, window, hops, args.padding, args.quantiles, args.drop_sentinel)
            for s in sessions]
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            rows = pool.starmap(_stats, work)
    else:
        rows = [_stats(*w) for w in work]

    names = [s['session'] for s in sessions]
    pairs = [(nm, r) for nm, r in zip(names, rows) if r]
    if not pairs:
        print('[box-impact] no usable IRFeatures tracks found')
        return 1

    keys = sorted({k for _, r in pairs for k in r})
    agg = {k: float(np.nanmedian([r[k] for _, r in pairs if k in r]))
           for k in keys}

    def col(k: str) -> np.ndarray:
        return np.array([r[k] for _, r in pairs if k in r], dtype=float)

    print(f'[box-impact] scanned    : {len(pairs)}/{len(sessions)} subject-tasks '
          f'({len(sessions) - len(pairs)} skipped: no/!parseable IRFeatures)')
    print(f'[box-impact] total frames: {int(agg["frames"] * len(pairs)):,}')
    print()

    print('--- landmark quality (the "landmarks are high quality" premise) ---')
    print(f'  rows that are NaN/unparseable : {100 * agg["nan_frac"]:6.3f} %'
          f'   (p90 {100 * np.nanpercentile(col("nan_frac"), 90):.3f} %, '
          f'max {100 * np.nanmax(col("nan_frac")):.2f} %)')
    print(f'  rows that are a (0,0) sentinel: {100 * agg["sentinel_frac"]:6.3f} %'
          f'   (p90 {100 * np.nanpercentile(col("sentinel_frac"), 90):.3f} %, '
          f'max {100 * np.nanmax(col("sentinel_frac")):.2f} %)')
    print()

    print('--- box size (source px, geometric-mean side) ---')
    print(f'  task box (min/max over the whole task): median {agg["task_lin"]:6.1f}'
          f'  p10 {np.nanpercentile(col("task_lin"), 10):6.1f}'
          f'  p90 {np.nanpercentile(col("task_lin"), 90):6.1f}')
    for hop, hs in zip(hops, args.hops):
        k = f'clip_lin_{hop}'
        if k not in agg:
            continue
        print(f'  clip box (hop {hs:g} s)               : median {agg[k]:6.1f}'
              f'  p10 {np.nanpercentile(col(k), 10):6.1f}'
              f'  p90 {np.nanpercentile(col(k), 90):6.1f}')
    print()

    print('--- what a ONE-BOX-PER-TASK rule costs (linear inflation) ---')
    for hop, hs in zip(hops, args.hops):
        k = f'infl_{hop}'
        if k not in agg:
            continue
        v = col(k)
        print(f'  hop {hs:g} s: median x{agg[k]:.3f}   p90 x{np.nanpercentile(v, 90):.3f}'
              f'   p99 x{np.nanpercentile(v, 99):.3f}   max x{np.nanmax(v):.2f}')
        for thr in (1.2, 1.5, 2.0):
            print(f'      tasks losing >= x{thr:.1f} detail: '
                  f'{100.0 * (v >= thr).mean():5.1f} %')
    print()

    print('--- per-clip adaptivity (how much the clip box varies INSIDE a task) ---')
    for hop, hs in zip(hops, args.hops):
        k = f'clip_spread_{hop}'
        if k not in agg:
            continue
        v = col(k)
        print(f'  hop {hs:g} s: max/min clip-box side  median x{agg[k]:.2f}'
              f'   p90 x{np.nanpercentile(v, 90):.2f}   max x{np.nanmax(v):.2f}')
    print()

    print('--- what the REMOVED roi_quantile would buy (clip box, linear) ---')
    print('    (local re-implementation; expected q=0.05 x0.835, q=0.10 x0.713,'
          ' q=0.20 x0.536)')
    for q in args.quantiles:
        k = f'q{int(q * 100)}_ratio'
        if k in agg:
            v = col(k)
            print(f'  q={q:<5g}: median x{agg[k]:.3f}   p10 x'
                  f'{np.nanpercentile(v, 10):.3f}   (1.000 = no effect)')
    print()

    print('--- the 2026-09-26 mechanism: does a largER box track MORE jitter? ---')
    j, tl, cl = col('jitter'), col('task_lin'), col(f'clip_lin_{hops[0]}')
    m = np.isfinite(j) & np.isfinite(tl) & np.isfinite(cl)
    if m.sum() > 10:
        print(f'  corr(task box side, jitter)  = {np.corrcoef(tl[m], j[m])[0, 1]:+.3f}')
        print(f'  corr(clip box side, jitter)  = {np.corrcoef(cl[m], j[m])[0, 1]:+.3f}')
        q = np.quantile(cl[m], [0.1, 0.9])
        lo, hi = cl[m] <= q[0], cl[m] >= q[1]
        print(f'  small-box decile: median clip side {np.median(cl[m][lo]):5.1f} px,'
              f' jitter {np.median(j[m][lo]):5.2f}')
        print(f'  large-box decile: median clip side {np.median(cl[m][hi]):5.1f} px,'
              f' jitter {np.median(j[m][hi]):5.2f}')
        print(f'  -> the two regimes the note describes '
              f'{"EXIST" if np.median(cl[m][hi]) / max(np.median(cl[m][lo]), 1) > 1.3 else "are NOT clearly separated"}')

    if args.csv:
        with open(args.csv, 'w') as fh:
            fh.write('session,' + ','.join(keys) + '\n')
            for (nm, r) in pairs:
                fh.write(nm + ',' + ','.join(
                    f'{r.get(k, float("nan")):.6g}' for k in keys) + '\n')
        print(f'\n[box-impact] per-task CSV -> {args.csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
