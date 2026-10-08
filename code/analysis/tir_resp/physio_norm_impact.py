"""clip- vs session-level physio normalisation: the measured consequences.

Context. ``--physio_norm {none, clip|zscore, session}`` is DATASET-owned (the
model is pinned to identity for the ROI lineages by
``run_pretrain.pin_roi_target_norm``). The HPC pair ships ``session``; every
local tir_roi/rgb_roi config ships ``clip``. This script quantifies what that
choice actually costs, so it is not decided by intuition.

NO VIDEO IS DECODED. The respiration target comes from
``Physiology/<S>/<T>/Resp_Volts.txt`` and the window lattice from the dataset
(which itself only reads video *metadata*), so the whole corpus can be scanned
with the CPU alone.

It answers three questions, per subject-task:

1. **How much does identity explain?** A variance decomposition of the raw kept
   samples: between-session vs between-clip-within-session vs within-clip. This
   is what a ``session`` map removes (and a ``clip`` map removes more).
2. **Does the session map compress the good clips?** ``session`` statistics are
   taken over the WHOLE raw file (``_resp_full``), including railed / dead
   samples and any span outside the clip lattice. ``compression =
   std(kept windows) / std(whole file)``: well below 1 means the clips that
   survive the rail/spread filters are squashed toward zero by the statistics of
   the samples that did NOT survive.
3. **How discontinuous is the target at clip boundaries?** The assembled
   waveform is only a coherent series under ``session``. ``jump_ratio`` compares
   the normalised step across a clip boundary with the typical step *inside* a
   clip, for each convention: ``session`` keeps the real signal continuity,
   ``clip`` adds the change of affine map.

Usage (from ``code/``; CPU only, no Slurm needed for a sample):
    python analysis/tir_resp/physio_norm_impact.py --limit 40
    python analysis/tir_resp/physio_norm_impact.py --workers 8 --csv out.csv
"""
import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from data import tir_resp_dataset as trd                    # noqa: E402

#: the shipped cleaning rule: a window touching |x| >= this is dropped.
RAIL_TOUCH_V = 9.9
_EPS = 1e-12


def _stats_for_session(y: np.ndarray, windows: List[Tuple[int, int]]) -> Dict:
    """All normalisation statistics for one subject-task."""
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[0]
    win = sorted((max(0, a), min(n, b)) for a, b in windows if b > a)
    if not win:
        return {}

    mu_file, sd_file = float(np.nanmean(y)), float(np.nanstd(y))
    kept = np.concatenate([y[a:b] for a, b in win])
    mu_kept, sd_kept = float(np.nanmean(kept)), float(np.nanstd(kept))

    out: Dict[str, float] = {
        'file_samples': float(n),
        'clips': float(len(win)),
        'file_std': sd_file,
        'kept_std': sd_kept,
        # < 1 => the surviving clips are compressed by the whole-file statistics
        'compression': sd_kept / (sd_file + _EPS),
        'rail_frac_file': float(np.mean(np.abs(y) >= RAIL_TOUCH_V)),
        'rail_frac_kept': float(np.mean(np.abs(kept) >= RAIL_TOUCH_V)),
        # union coverage (overlapping windows must NOT be double counted) and
        # the overlap factor that explains why a raw sum would exceed 100 %
        'span_frac': float((win[-1][1] - win[0][0]) / max(n, 1)),
        'overlap': float(sum(b - a for a, b in win)
                       / max(win[-1][1] - win[0][0], 1)),
    }

    # --- per-clip statistics -------------------------------------------------
    means, stds = [], []
    for a, b in win:
        w = y[a:b]
        means.append(float(np.nanmean(w)))
        stds.append(float(np.nanstd(w)))
    means_a, stds_a = np.asarray(means), np.asarray(stds)

    # target std a clip gets under `session` (divide by the session's std):
    # 1.0 == the clip convention, which forces it to exactly 1 by construction.
    out['clip_std_under_session_median'] = float(np.nanmedian(stds_a / (sd_file + _EPS)))
    out['clip_std_under_session_p10'] = float(np.nanpercentile(stds_a / (sd_file + _EPS), 10))
    out['clip_std_under_session_p90'] = float(np.nanpercentile(stds_a / (sd_file + _EPS), 90))

    # --- variance decomposition WITHIN this session --------------------------
    # between clips (their means differ) vs within clips (shape inside a window)
    var_total = float(np.nanvar(kept))
    var_between = float(np.nanvar(means_a)) if len(win) > 1 else 0.0
    var_within = float(np.nanmean(stds_a ** 2))
    out['between_clip_frac'] = var_between / (var_total + _EPS)
    out['within_clip_frac'] = var_within / (var_total + _EPS)

    # --- boundary discontinuity, both conventions ---------------------------
    # windows are ordered by start and overlap; the step is from the last sample
    # of one window to the first sample of the next.
    steps_sess, steps_clip, inside_sess = [], [], []
    for idx, (a0, b0) in enumerate(win[:-1]):
        a1, b1 = win[idx + 1]
        # The SAME two samples under both conventions: the last sample of one
        # window and the first of the next. They are `hop` seconds apart, so the
        # absolute jump is large for BOTH -- what matters is the EXTRA step the
        # per-clip affine map adds on top of the signal itself.
        y_prev, y_next = float(y[b0 - 1]), float(y[a1])
        steps_sess.append(abs(y_next - y_prev) / (sd_file + _EPS))
        z_prev = (y_prev - means_a[idx]) / (stds_a[idx] + _EPS)
        z_next = (y_next - means_a[idx + 1]) / (stds_a[idx + 1] + _EPS)
        steps_clip.append(abs(z_next - z_prev))
        inside_sess.append(
            float(np.nanmedian(np.abs(np.diff(y[a0:b0])))) / (sd_file + _EPS))
    if steps_sess:
        med_s = float(np.nanmedian(steps_sess))
        med_c = float(np.nanmedian(steps_clip))
        out['jump_session'] = med_s
        out['jump_clip'] = med_c
        # the discontinuity the `clip` convention adds, relative to the signal's
        # own change over the same hop: 1.0 = no extra step at all
        out['jump_excess'] = med_c / (med_s + _EPS)
        # reference scale: a 1 ms step inside a clip, same normalisation
        out['inside_step'] = float(np.nanmedian(inside_sess))
    out['n_boundaries'] = float(len(steps_sess))
    return out


def _session_task(job) -> Optional[Tuple[str, Dict]]:
    """One worker's unit of work: (session, resp_file, windows).

    Takes only cheap, picklable data -- passing the dataset itself into a Pool
    would serialise it once per session.
    """
    sess, resp_file, windows = job
    if not windows:
        return None
    try:
        y = trd._load_1d(resp_file)
    except (OSError, ValueError):
        return None
    st = _stats_for_session(y, windows)
    return (sess, st) if st else None


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw_root', default=os.environ.get(
        'RAW_DATA_PATH', '/work/projects/l0003511/test'))
    p.add_argument('--clip_seconds', type=float, default=8.0)
    p.add_argument('--clip_stride', type=float, default=2.0,
                   help='the HPC Stage-2 hop')
    p.add_argument('--fps', type=float, default=25.0)
    p.add_argument('--resp_fs', type=float, default=1000.0)
    p.add_argument('--min_signal_spread', type=float, default=0.1)
    p.add_argument('--rail_touch_v', type=float, default=RAIL_TOUCH_V)
    p.add_argument('--subjects', default='', help='comma list; empty = all')
    p.add_argument('--limit', type=int, default=0,
                   help='0 = all; else cap the number of CLIPS built (fast smoke)')
    p.add_argument('--workers', type=int, default=0,
                   help='0 = single process')
    p.add_argument('--csv', default='', help='per-session CSV output path')
    args = p.parse_args()

    subs = [s.strip() for s in args.subjects.split(',') if s.strip()] or None
    print(f'[physio-norm] building the dataset (metadata only, no decode) ...')
    ds = trd.BP4DPlusTIRRespDataset(
        raw_root=args.raw_root, subjects=subs,
        clip_seconds=args.clip_seconds, clip_stride=args.clip_stride,
        fps=args.fps, resp_fs=args.resp_fs, input_size=64,
        min_signal_spread=args.min_signal_spread,
        rail_touch_v=args.rail_touch_v, norm='none',
        max_entries=args.limit or None, verbose=False)

    entries_for: Dict[str, List[dict]] = {}
    for e in ds.entries:
        entries_for.setdefault(e['session'], []).append(e)
    sessions = [s for s in ds.sessions if s in entries_for]
    print(f'[physio-norm] {len(sessions)} subject-task(s) with kept clips, '
          f'{len(ds.entries)} clip(s) total')

    jobs = []
    for s in sessions:
        spec = ds.sessions[s]
        if not spec.get('resp_file'):
            continue
        windows = [(int(round(e['t_start'] * ds.resp_fs)),
                    int(round(e['t_start'] * ds.resp_fs)) + ds.resp_len)
                   for e in entries_for[s]]
        jobs.append((s, spec['resp_file'], windows))

    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            out = pool.map(_session_task, jobs)
    else:
        out = [_session_task(j) for j in jobs]
    rows = [r for r in out if r]
    if not rows:
        print('[physio-norm] nothing to report (no usable respiration windows)')
        return 1

    names = [n for n, _ in rows]
    keys = sorted({k for _, r in rows for k in r})
    cols = {k: np.array([r.get(k, np.nan) for _, r in rows], dtype=float)
            for k in keys}

    def agg(k: str, fn=np.nanmedian) -> float:
        return float(fn(cols[k])) if k in cols else float('nan')

    print(f'\n[physio-norm] analysed {len(rows)} subject-task(s), '
          f'{int(np.nansum(cols["clips"]))} clip(s)')
    print()

    print('--- 1. variance decomposition of the RAW kept samples ---')
    print(f'  within-session, between-clip : median {100 * agg("between_clip_frac"):5.1f} %'
          f'   (p90 {100 * np.nanpercentile(cols["between_clip_frac"], 90):5.1f} %)')
    print(f'  within-session, within-clip  : median {100 * agg("within_clip_frac"):5.1f} %'
          f'   (p90 {100 * np.nanpercentile(cols["within_clip_frac"], 90):5.1f} %)')
    print('  -> the between-clip share is what a `clip` map removes on top of')
    print('     what a `session` map already removed (the session mean).')
    print()

    print('--- 2. does the SESSION map squash the surviving clips? ---')
    print('  compression = std(kept windows) / std(whole raw file)')
    c = cols['compression']
    print(f'  median x{agg("compression"):.3f}   p10 x{np.nanpercentile(c, 10):.3f}'
          f'   min x{np.nanmin(c):.3f}')
    for thr in (0.5, 0.7, 0.9):
        print(f'      sessions below x{thr:<4}: {100.0 * np.nanmean(c < thr):5.1f} %')
    print(f'  rail fraction: whole file median {100 * agg("rail_frac_file"):5.2f} %'
          f'   kept windows median {100 * agg("rail_frac_kept"):5.2f} %')
    print(f'  clip union coverage of the file : median {100 * agg("span_frac"):5.1f} %'
          '   (the samples OUTSIDE it still set the session mean/std)')
    print(f'  window overlap factor          : median x{agg("overlap"):.1f}'
          '   (8 s window on a 2 s hop)')
    print()

    print('--- 3. per-clip target std under `session` (1.0 == the clip convention) ---')
    print(f'  median {agg("clip_std_under_session_median"):.3f}'
          f'   p10 {agg("clip_std_under_session_p10"):.3f}'
          f'   p90 {agg("clip_std_under_session_p90"):.3f}')
    print('  -> deviate from 1 => alpha*StdLoss is a real amplitude term in Stage 3')
    print()

    print('--- 4. boundary continuity of the ASSEMBLED waveform ---')
    print('  normalised step between the last sample of one window and the first')
    print('  of the next (they are one hop apart, so both are non-zero):')
    print(f'  `session` convention: median {agg("jump_session"):.3f}   (signal change only)')
    print(f'  `clip`    convention: median {agg("jump_clip"):.3f}   (signal + map change)')
    print(f'  excess added by `clip`: median x{agg("jump_excess"):.2f}'
          '   (1.0 = the map adds nothing)')
    print(f'  reference: a 1 ms step inside a clip = {agg("inside_step"):.5f}'
          f'  ({int(np.nansum(cols["n_boundaries"]))} boundaries measured)')

    if args.csv:
        with open(args.csv, 'w') as fh:
            fh.write('session,' + ','.join(keys) + '\n')
            for nm, r in rows:
                fh.write(nm + ',' + ','.join(
                    f'{r.get(k, float("nan")):.6g}' for k in keys) + '\n')
        print(f'\n[physio-norm] per-session CSV -> {args.csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
