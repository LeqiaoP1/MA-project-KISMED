"""Geometry ledger for the ROI crop contract -- the numbers the design quotes.

Reproduces, from the RAW ``IRFeatures`` tracks and the EMITTED clip list, every
geometry figure in ``notes/design/ROI_crop_and_rescaling.md``:

* the SQUARE crop side before and after the motion gate (median / p95 / max);
* the isotropicity ledger: ``sigma = s / side``, and the share of clips that are
  genuinely downscaled vs upsampled, for ``s in {64, 80, 96, 112, 128}`` and
  ``p in {0.0, 0.1, 0.15, 0.2}`` -- the SHIPPED pair is ``p = 0.10, s = 96``,
  since ``sigma`` is a density fixed by ``s / side`` and the two knobs must move
  together (see the design doc, 4.1);
* the motion gate's attrition at ``max_sway_px in {30, 40, 60, 100}``: clips,
  sessions and subjects kept, plus the worst task -- the gate CHANGES THE
  CORPUS, so any per-task result must say whether it was computed before or
  after it.

WHERE the clip list comes from. The population the design measured is "the union
of the five subject-disjoint Stage-3 fold dumps" = 1260 sessions / 138 subjects /
21 231 clips. Those dumps carry the reservoir of clips that survived the sentinel
/ rail / flat gates, which is exactly the pre-motion-gate corpus, and they are
cheap and exact -- re-deriving them would need every ``Resp_Volts.txt`` and every
container probe for no extra information. So: read the dumps, read the tracks.

Everything here is a pure function of (landmark track, frame size, padding,
crop_rule) -- no video decode, no cache, no GPU, and no dependence on the v3 ROI
cache, so it can run BEFORE the cache rebuild and gate it.

Usage (from ``code/``)::

    python notes/eval/tir_resp/roi_geometry_ledger.py \
        --raw_root "$RAW_DATA_PATH" \
        --json ../output/eval/tir_resp/roi_geometry_ledger.json

Exits 1 when a published figure is not reproduced (within tolerance), so it can
be used as a gate rather than as a printout.
"""
import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
_REPO_DIR = os.path.dirname(_CODE_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from data import tir_resp_dataset as trd                     # noqa: E402

#: The published figures (design doc, 2026-10-30: 21 231 clips / 1260 sessions,
#: shipped contract `roi_padding 0.10` + `input_size 96`).
#: Tolerances: +-1 px on percentiles (the integer-bound convention below can
#: differ from the design's `ceil(side)` by one), +-1 pp on shares.
EXPECT = {
    'n_clips': 21231,
    'n_sessions': 1260,
    'n_subjects': 138,
    'side_median': 95,
    'side_p95': 131,
    'side_max': 353,
    'gated40_median': 95,
    'gated40_p95': 125,
    'gated40_max': 174,
    'kept30': 94.0,
    'kept40': 96.8,
    'kept60': 98.9,
    'kept100': 99.9,
    'sessions_kept40': 1238,
    'worst_task40': 'T4',
    'worst_task40_pct': 80,
}


# --------------------------------------------------------------------------- #
# the emitted clip list
# --------------------------------------------------------------------------- #
def load_clips(pattern: str) -> List[dict]:
    """Union of the fold ``entries.json`` dumps, asserting they are disjoint."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f'[ledger] no entries dump matches {pattern!r}; pass '
                         f'--entries with the fold glob')
    clips: List[dict] = []
    seen_sessions: Dict[str, str] = {}
    for path in paths:
        with open(path) as fh:
            rows = json.load(fh)
        for row in rows:
            sess = str(row['session'])
            prev = seen_sessions.get(sess)
            if prev is not None and prev != path:
                raise SystemExit(f'[ledger] session {sess} appears in BOTH '
                                 f'{prev} and {path} -- the folds must be '
                                 f'subject-disjoint, so the union would '
                                 f'double-count')
            seen_sessions[sess] = path
            clips.append({'session': sess, 'subject': sess.rsplit('_', 1)[0],
                          'task': str(row['task']),
                          't_start': float(row['t_start'])})
    clips.sort(key=lambda c: (c['session'], c['t_start']))
    print(f'[ledger] {len(clips)} clip(s) from {len(paths)} fold dump(s)')
    return clips


# --------------------------------------------------------------------------- #
# per-clip geometry
# --------------------------------------------------------------------------- #
def measure(clips: Sequence[dict], raw_root: str, fps: float,
            clip_frames: int, paddings: Sequence[float],
            w: int, h: int) -> Tuple[List[dict], Dict[str, int]]:
    """Side (per padding) and sway for every clip; cached per session track."""
    idx = np.asarray(trd.TARGET_LANDMARK_IDX, dtype=np.int64)
    tracks: Dict[str, Optional[np.ndarray]] = {}
    missing = {'ir_features': 0, 'short_track': 0, 'box_out_of_frame': 0}
    out: List[dict] = []
    for i, c in enumerate(clips):
        sess = c['session']
        if sess not in tracks:
            path = trd.find_ir_features(raw_root, c['subject'], c['task'])
            tracks[sess] = trd.parse_ir_features(path) if path else None
            if tracks[sess] is None:
                missing['ir_features'] += 1
        ir = tracks[sess]
        if ir is None:
            continue
        fs = int(round(c['t_start'] * fps))
        fe = fs + clip_frames
        if fe > ir.shape[0]:
            missing['short_track'] += 1
            continue
        pts = ir[fs:fe][:, idx, :]
        row = {'session': sess, 'subject': c['subject'], 'task': c['task'],
               'frame_start': fs,
               'sway_px': trd.landmark_sway(pts, np.arange(pts.shape[1])),
               # the face at ONE instant, as opposed to the clip-wide envelope:
               # the gap between the two is the head travel inside the window
               'face_w': float(np.median(np.ptp(pts[..., 0], axis=1))),
               'face_h': float(np.median(np.ptp(pts[..., 1], axis=1)))}
        ok = True
        for p in paddings:
            try:
                x0, x1, y0, y1 = trd.roi_box_from_landmarks(
                    pts, w, h, p, crop_rule=trd.CROP_RULE_SQUARE)
            except trd.RoiBoxOutOfFrame:
                missing['box_out_of_frame'] += 1
                ok = False
                break
            row[f'side_p{p:.2f}'] = int(x1 - x0)
        if ok:
            out.append(row)
        if (i + 1) % 5000 == 0:
            print(f'[ledger]   {i + 1}/{len(clips)} clip(s) measured',
                  flush=True)
    return out, missing


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def _pct(values: Sequence[int], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def side_stats(rows: Sequence[dict], key: str) -> Dict[str, float]:
    sides = [r[key] for r in rows]
    if not sides:
        return {'n': 0}
    return {'n': len(sides),
            'median': _pct(sides, 50), 'p95': _pct(sides, 95),
            'max': int(max(sides)), 'min': int(min(sides)),
            'mean': float(np.mean(sides))}


def gate_stats(rows: Sequence[dict], key: str, thr: float) -> Dict[str, object]:
    kept = [r for r in rows if r['sway_px'] <= thr]
    per_task: Dict[str, Dict[str, int]] = {}
    for r in rows:
        t = per_task.setdefault(r['task'], {'kept': 0, 'total': 0})
        t['total'] += 1
    for r in kept:
        per_task[r['task']]['kept'] += 1
    task_pct = {k: 100.0 * v['kept'] / v['total']
                for k, v in sorted(per_task.items()) if v['total']}
    worst = min(task_pct, key=task_pct.get) if task_pct else None
    return {
        'threshold': float(thr),
        'clips_kept': len(kept),
        'clips_kept_pct': 100.0 * len(kept) / max(len(rows), 1),
        'sessions_kept': len({r['session'] for r in kept}),
        'sessions_total': len({r['session'] for r in rows}),
        'subjects_kept': len({r['subject'] for r in kept}),
        'subjects_total': len({r['subject'] for r in rows}),
        'worst_task': worst,
        'worst_task_pct': (round(task_pct[worst], 1) if worst else None),
        'per_task_pct': {k: round(v, 1) for k, v in task_pct.items()},
        'side': side_stats(kept, key),
    }


def sigma_table(rows: Sequence[dict], key: str, sizes: Sequence[int]
                ) -> Dict[str, Dict[str, float]]:
    """``sigma = s / side`` per size: median and the up/down-sample shares."""
    out: Dict[str, Dict[str, float]] = {}
    for s in sizes:
        sides = np.asarray([r[key] for r in rows], dtype=np.float64)
        if sides.size == 0:
            continue
        sigma = s / sides
        out[str(s)] = {
            'median_sigma': float(np.median(sigma)),
            'downscaled_pct': float(100.0 * np.mean(sigma < 1.0)),
            'upsampled_pct': float(100.0 * np.mean(sigma > 1.0)),
            'native_pct': float(100.0 * np.mean(sigma == 1.0)),
        }
    return out


def face_sweep(rows: Sequence[dict], paddings: Sequence[float],
               gates: Sequence[Tuple[float, float]], size: int = 96
               ) -> List[Dict[str, object]]:
    """PADDING x GATE: how much of the patch is face, and at what resample cost.

    ``occupancy`` is the area of ONE FRAME's target envelope over the square box,
    so it separates the three things a loose crop is made of: the face itself, the
    head travel inside the window, and the padding. ``sigma = size / side`` says
    whether the extra occupancy is MEASURED or INTERPOLATED -- below 1 the encoder
    samples the same photons more often, which does not add information.

    :param gates: ``(max_sway_px, padding)`` pairs; ``max_sway_px <= 0`` disables
        the gate for that row.
    """
    out: List[Dict[str, object]] = []
    for gate, pad in gates:
        key = f'side_p{pad:.2f}'
        keep = [r for r in rows if gate <= 0 or r['sway_px'] <= gate]
        if not keep or key not in keep[0]:
            continue
        sides = np.asarray([r[key] for r in keep], dtype=np.float64)
        occ = np.asarray([100.0 * r['face_w'] * r['face_h'] / max(r[key], 1) ** 2
                          for r in keep], dtype=np.float64)
        sigma = size / sides
        out.append({
            'max_sway_px': float(gate), 'padding': float(pad),
            'clips': len(keep),
            'clips_pct': 100.0 * len(keep) / max(len(rows), 1),
            'side_median': float(np.median(sides)),
            'side_p95': float(np.percentile(sides, 95)),
            'side_max': int(sides.max()),
            'occupancy_median': float(np.median(occ)),
            'occupancy_p10': float(np.percentile(occ, 10)),
            'face_px_at_size': float(np.median(occ)) * size * size / 100.0,
            'median_sigma': float(np.median(sigma)),
            'downscaled_pct': float(100.0 * np.mean(sigma < 1.0)),
            'heavy_upsample_pct': float(100.0 * np.mean(sigma > 1.25)),
        })
    return out


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _check(name: str, got: float, want: float, tol: float,
           lines: List[str], fails: List[str]) -> None:
    ok = abs(float(got) - float(want)) <= tol
    lines.append(f'  {name:<26} measured {got:>8.2f}   design {want:>8.2f}   '
                 f'{"OK" if ok else "MISMATCH"}')
    if not ok:
        fails.append(f'{name}: measured {got}, design {want} (tol {tol})')


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw_root',
                   default=os.environ.get('RAW_DATA_PATH', ''))
    p.add_argument('--entries',
                   default=os.path.join(_REPO_DIR, 'output', 'finetune',
                                        'resp_tir_roi_hpc_fold*',
                                        'entries.json'))
    p.add_argument('--clip_seconds', type=float, default=8.0)
    p.add_argument('--fps', type=float, default=25.0)
    p.add_argument('--paddings', type=float, nargs='+',
                   default=[0.10, 0.15, 0.20, 0.00])
    p.add_argument('--sizes', type=int, nargs='+', default=[64, 80, 96, 112, 128])
    p.add_argument('--thresholds', type=float, nargs='+',
                   default=[30.0, 40.0, 60.0, 100.0])
    p.add_argument('--width', type=int, default=726)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--json', default='')
    p.add_argument('--face_sweep', action='store_true',
                   help='also sweep PADDING x GATE and report how much of the '
                        'patch is one frame\'s face, next to sigma = s/side '
                        '(are the extra pixels MEASURED or INTERPOLATED?)')
    args = p.parse_args(argv)

    if not args.raw_root or not os.path.isdir(args.raw_root):
        print(f'[ledger] raw_root not usable: {args.raw_root!r} (pass '
              f'--raw_root or set $RAW_DATA_PATH)', file=sys.stderr)
        return 2
    ir_dir = os.path.join(args.raw_root, trd.IR_TREE)
    if not os.path.isdir(ir_dir):
        print(f'[ledger] no {trd.IR_TREE}/ under {args.raw_root}', file=sys.stderr)
        return 2
    print(f'[ledger] raw_root : {args.raw_root}')
    print(f'[ledger] source   : {args.width}x{args.height}, clip '
          f'{args.clip_seconds:g} s @ {args.fps:g} fps '
          f'({int(round(args.clip_seconds * args.fps))} frames)')

    clips = load_clips(args.entries)
    cf = int(round(args.clip_seconds * args.fps))
    rows, missing = measure(clips, args.raw_root, args.fps, cf,
                            args.paddings, args.width, args.height)
    print(f'[ledger] measured {len(rows)} clip(s); unusable: {missing}')

    # the PRODUCTION padding, not the largest one: since 2026-10-30 the shipped
    # contract is p = 0.10, so picking `max(paddings)` would gate the ledger on a
    # retired geometry.
    key = f'side_p{trd.DEFAULT_ROI_PADDING:.2f}'
    report: Dict[str, object] = {
        'raw_root': args.raw_root, 'entries': args.entries,
        'n_clips': len(rows),
        'n_sessions': len({r['session'] for r in rows}),
        'n_subjects': len({r['subject'] for r in rows}),
        'missing': missing,
        'paddings': list(args.paddings),
        'sizes': list(args.sizes),
        'sides': {f'p{p:.2f}': side_stats(rows, f'side_p{p:.2f}')
                  for p in args.paddings},
        'sway': {'median': _pct([r['sway_px'] for r in rows], 50),
                 'p95': _pct([r['sway_px'] for r in rows], 95),
                 'max': float(max(r['sway_px'] for r in rows))},
        'gates': {f'{t:g}': gate_stats(rows, key, t)
                  for t in args.thresholds},
        'sigma': {f'p{p:.2f}': sigma_table(rows, f'side_p{p:.2f}', args.sizes)
                  for p in args.paddings},
    }

    lines: List[str] = []
    fails: List[str] = []
    lines.append('--- corpus ---')
    _check('n_clips', len(rows), EXPECT['n_clips'], 0, lines, fails)
    _check('n_sessions', report['n_sessions'], EXPECT['n_sessions'], 0,
           lines, fails)
    _check('n_subjects', report['n_subjects'], EXPECT['n_subjects'], 0,
           lines, fails)

    main_pad = trd.DEFAULT_ROI_PADDING
    s0 = report['sides'][f'p{main_pad:.2f}']
    lines.append(f'--- square side, p={main_pad:g}, pre-gate '
                 f'(n={s0.get("n")}) ---')
    _check('median', s0['median'], EXPECT['side_median'], 2, lines, fails)
    _check('p95', s0['p95'], EXPECT['side_p95'], 3, lines, fails)
    _check('max', s0['max'], EXPECT['side_max'], 3, lines, fails)

    g40 = report['gates']['40']
    _check('gated40 median', g40['side']['median'], EXPECT['gated40_median'],
           2, lines, fails)
    _check('gated40 p95', g40['side']['p95'], EXPECT['gated40_p95'], 3,
           lines, fails)
    _check('gated40 max', g40['side']['max'], EXPECT['gated40_max'], 3,
           lines, fails)

    lines.append('--- motion gate attrition ---')
    for t in ('30', '40', '60', '100'):
        g = report['gates'][t]
        lines.append(f'  max_sway_px={t:>3}  clips {g["clips_kept"]:6d} '
                     f'({g["clips_kept_pct"]:5.1f} %)  '
                     f'sessions {g["sessions_kept"]:4d}/{g["sessions_total"]}  '
                     f'subjects {g["subjects_kept"]:3d}/{g["subjects_total"]}  '
                     f'worst task {g["worst_task"]} {g["worst_task_pct"]} %')
    _check('kept@30 %', report['gates']['30']['clips_kept_pct'],
           EXPECT['kept30'], 1.0, lines, fails)
    _check('kept@40 %', g40['clips_kept_pct'], EXPECT['kept40'], 1.0,
           lines, fails)
    _check('kept@60 %', report['gates']['60']['clips_kept_pct'],
           EXPECT['kept60'], 1.0, lines, fails)
    _check('kept@100 %', report['gates']['100']['clips_kept_pct'],
           EXPECT['kept100'], 1.0, lines, fails)
    _check('sessions@40', g40['sessions_kept'], EXPECT['sessions_kept40'], 2,
           lines, fails)
    if g40['worst_task'] != EXPECT['worst_task40']:
        fails.append(f'worst task @40 is {g40["worst_task"]}, design says '
                     f'{EXPECT["worst_task40"]}')
    lines.append(f'  worst task @40            {g40["worst_task"]} '
                 f'{g40["worst_task_pct"]} %   design {EXPECT["worst_task40"]} '
                 f'{EXPECT["worst_task40_pct"]} %   '
                 f'{"OK" if g40["worst_task"] == EXPECT["worst_task40"] else "MISMATCH"}')
    lines.append('  per task @40: ' + ', '.join(
        f'{k} {v}' for k, v in g40['per_task_pct'].items()))

    lines.append('--- isotropicity: sigma = s / side ---')
    for pad, table in report['sigma'].items():
        for size, st in table.items():
            lines.append(f'  p={pad[1:]} s={size:>3}  median sigma '
                         f'{st["median_sigma"]:4.2f}  downscaled '
                         f'{st["downscaled_pct"]:5.1f} %  upsampled '
                         f'{st["upsampled_pct"]:5.1f} %  native '
                         f'{st["native_pct"]:4.1f} %')

    if args.face_sweep:
        lo = min(args.paddings)
        gates = [(40.0, main_pad), (30.0, main_pad), (20.0, main_pad),
                 (40.0, 0.15), (40.0, lo)]
        sweep = face_sweep(rows, args.paddings, gates, size=96)
        report['face_sweep'] = sweep
        lines.append('--- padding x gate: how much of the 96^2 patch is ONE '
                     'frame\'s face ---')
        lines.append(f'  {"gate":>5}{"pad":>6}{"clips":>8}{"side":>6}'
                     f'{"p95":>5}{"max":>5}{"face%":>7}{"p10":>6}'
                     f'{"facePx":>8}{"sigma":>7}{"down%":>7}{"up>1.25%":>9}')
        for s in sweep:
            lines.append(
                f'  {s["max_sway_px"]:5.0f}{s["padding"]:6.2f}{s["clips"]:8d}'
                f'{s["side_median"]:6.0f}{s["side_p95"]:5.0f}{s["side_max"]:5d}'
                f'{s["occupancy_median"]:7.1f}{s["occupancy_p10"]:6.1f}'
                f'{s["face_px_at_size"]:8.0f}{s["median_sigma"]:7.2f}'
                f'{s["downscaled_pct"]:7.1f}{s["heavy_upsample_pct"]:9.1f}')

    print('\n'.join(lines))
    print(f'\n[ledger] VERDICT: {"PASS" if not fails else "FAIL"} '
          f'({len(fails)} mismatch(es))')
    for f in fails:
        print(f'    ! {f}')
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, 'w') as fh:
            json.dump({**report, 'checks_failed': fails}, fh, indent=2)
        print(f'[ledger] JSON -> {args.json}')
    return 0 if not fails else 1


if __name__ == '__main__':
    raise SystemExit(main())
