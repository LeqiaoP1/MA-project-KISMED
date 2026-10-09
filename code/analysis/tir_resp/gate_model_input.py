#!/usr/bin/env python
"""Respiration gate measured in THE MODEL'S OWN INPUT SPACE (offline ROI cache).

WHY THIS EXISTS
---------------
The recorded gate (`TirROI_Resp_plan.md` section 12, 2026-09-26) found respiration
undetectable in the thermal ROI, which closed the TIR-ROI -> RESP line. Two things
weaken that conclusion for the CURRENT setup:

  1. it used a HAND-DERIVED scalar per pixel (HSV hue) whose palette LUT is LOST
     -- `native_resolution_gate.py` reconstructs it and says so -- so its absolute
     numbers are not comparable to anything measured now;
  2. it ran on 4 subjects with a 40-epoch encoder. The HPC run has 139 subjects
     and a 150-epoch encoder.

This script removes the representation ambiguity entirely: it reads the OFFLINE
ROI CACHE, i.e. the exact pixels the encoder consumes, and resizes them with the
DATASET'S OWN interpolation rule (`INTER_AREA` when downscaling, `INTER_LINEAR`
when upscaling -- `to_size()` in the sibling script hardcodes INTER_AREA, which
is wrong for the 112 upscale and is NOT used here).

It then applies the SAME statistic and the SAME phase-scrambled null as the
recorded gate, via `run_gate` imported from `native_resolution_gate.py`, so the
two differ ONLY in the pixel representation and the subject pool.

The statistic: max over (pixel x channel) and over lags of |r| against the belt,
null = phase-scrambled belts with the spectrum preserved. The `motion_x` rows are
the POSITIVE CONTROL and must read DETECTED, else nothing else is interpretable.

Answering "does the information exist in the tensor the model sees?" is the only
version of the question that bears on whether a loss term can exploit it: the
data-processing inequality says no function of the input can recover information
the input does not carry.

Usage (from ``code/``)::

    python analysis/tir_resp/gate_model_input.py --sessions F001_T1,F004_T9
    python analysis/tir_resp/gate_model_input.py --nperm 200 --out gate.json

Needs no GPU and no video decode -- the pixels come from the cache, so a session
costs seconds, not the minutes that decoding .wmv costs in the sibling script.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from data import roi_cache as rc                                  # noqa: E402
from data.tir_resp_dataset import discover_sessions               # noqa: E402
# Reuse the statistic, the null and the reference loaders verbatim, so this run
# differs from the recorded gate ONLY in the pixel source.
from native_resolution_gate import (load_belt, landmark_motion,  # noqa: E402
                                    run_gate)

#: the recorded gate's 8 sessions, so the rows line up with the recorded 0/8.
DEFAULT_SESSIONS = ('F001_T1', 'F001_T3', 'F002_T1', 'F003_T2',
                    'F004_T2', 'F004_T3', 'F004_T9', 'F004_T8')


def find_cache_dir(root: str) -> str:
    """Locate the ONE cache key directory under ``root``.

    Refuses an ambiguous match rather than picking one: the cache is keyed on the
    ROI geometry, and silently reading a different key's pixels is exactly the
    silent-drift failure the cache's own gate exists to prevent.
    """
    hits = sorted(glob.glob(os.path.join(root, '*', 'params.json')))
    if len(hits) != 1:
        raise SystemExit(
            f'expected exactly ONE cache key dir under {root}, found '
            f'{len(hits)}: {[os.path.dirname(h) for h in hits]}. Pass '
            f'--cache_dir explicitly.')
    return os.path.dirname(hits[0])


def resize_like_dataset(frames: np.ndarray, size: Optional[int],
                        pad: float = 0.2) -> np.ndarray:
    """[F, H, W, C] -> [F, P] using the DATASET's interpolation rule.

    ``data/tir_resp_dataset.py`` picks ``INTER_AREA`` when the target is at most
    the source extent and ``INTER_LINEAR`` otherwise. For the HPC geometry
    (input_size 112 vs a ~72x93 ROI box) that is an UPSCALE, so INTER_LINEAR --
    using INTER_AREA there, as the sibling script does, would measure a different
    image than the model is fed.
    """
    import cv2
    f, h, w, c = frames.shape
    if size is None:                      # native box, no resize at all
        return frames.reshape(f, -1)
    if size == h and size == w:
        return frames.reshape(f, -1)
    interp = (cv2.INTER_AREA if size <= min(h, w) else cv2.INTER_LINEAR)
    out = np.empty((f, size, size, c), dtype=frames.dtype)
    for i in range(f):
        # plain assignment, not cv2's `dst=`: the Python binding does not accept
        # `dst` as a keyword, and it requires a preallocated array otherwise.
        out[i] = cv2.resize(frames[i], (size, size), interpolation=interp)
    return out.reshape(f, -1)


def sample_sessions(cache_dir: str, per_task: int) -> List[str]:
    """``per_task`` session-tasks per TASK LABEL, spread across the cache.

    Deterministic and SUBJECT-SPREAD: the shards of each task are sorted and then
    every k-th is taken, so the sample is not clustered on the first few subjects
    (which would confound the task label with the subject).
    """
    names = sorted(
        os.path.basename(p)[:-len(rc.SHARD_SUFFIX)]
        for p in glob.glob(os.path.join(cache_dir, rc.SUBDIR_TASKS,
                                        '*' + rc.SHARD_SUFFIX)))
    by_task: Dict[str, List[str]] = {}
    for n in names:
        by_task.setdefault(n.split('_')[-1], []).append(n)
    out: List[str] = []
    for task in sorted(by_task):
        lst = by_task[task]
        k = max(1, len(lst) // per_task)
        out += lst[::k][:per_task]
    return out


def task_of(name: str) -> str:
    return name.split('_')[-1]


def group_of(task: str, spec: str) -> str:
    for part in str(spec).split(';'):
        if '=' not in part:
            continue
        g, tasks = part.split('=', 1)
        if task in [t.strip() for t in tasks.split(',')]:
            return g.strip()
    return '?'


def estimate_translation(gray: np.ndarray) -> np.ndarray:
    """Per-frame LOCAL (dx, dy) ROI translation via phase correlation.

    ``gray`` is ``[n, h, w]`` float32; returns ``[n, 2]`` where row t is the
    frame-to-frame shift of frame t w.r.t. t-1 (row 0 = 0). A TRANSLATING
    texture makes each pixel's intensity change ~ grad(I).(dx, dy), so
    partialling these two series out removes the dominant motion channel -- the
    exact confound for a pulse-band hit (ballistocardiographic head motion).
    """
    import cv2
    n = int(gray.shape[0])
    out = np.zeros((n, 2), dtype=np.float64)
    for t in range(1, n):
        (dx, dy), _ = cv2.phaseCorrelate(np.ascontiguousarray(gray[t - 1]),
                                         np.ascontiguousarray(gray[t]))
        out[t] = (float(dx), float(dy))
    return out


def partial_out(series: np.ndarray, reg: np.ndarray) -> np.ndarray:
    """Remove the LSQ projection of every column of ``series`` [n, P] onto
    ``reg`` [n, K] + an intercept; returns the residual [n, P] float32."""
    A = np.column_stack([np.ones(series.shape[0], dtype=np.float64),
                         np.asarray(reg, dtype=np.float64)])
    beta, *_ = np.linalg.lstsq(A, np.asarray(series, dtype=np.float64),
                               rcond=None)
    return (np.asarray(series, dtype=np.float64) - A @ beta).astype(np.float32)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw_root', default=os.environ.get(
        'RAW_DATA_PATH', '/work/projects/l0003511/test'))
    ap.add_argument('--cache_root', default=os.environ.get(
        'ROI_CACHE', '/work/scratch/ne95ocyg/tir_roi_cache'))
    ap.add_argument('--cache_dir', default='',
                    help='exact cache key dir; unset -> the only one under '
                         '--cache_root')
    ap.add_argument('--sessions', default=','.join(DEFAULT_SESSIONS))
    ap.add_argument('--sample-per-task', default=0, type=int,
                    help='instead of --sessions, draw N session-tasks PER TASK '
                         'LABEL from the cache (all 10 tasks). Subject-spread, '
                         'so the sample is not clustered on a few subjects.')
    ap.add_argument('--groups',
                    default='low=T2,T3;moderate=T4,T7,T8,T10;high=T1,T5,T6,T9',
                    help='task -> group mapping used for the per-group tally '
                         '(must mirror the Stage-2/3 `task_groups:` block).')
    ap.add_argument('--sizes', default='112,box',
                    help="comma list: an int (resize to NxN, the model's input) "
                         "or 'box' (native cached box, NO resize). Default "
                         'measures BOTH so resize vs representation separate.')
    ap.add_argument('--band', default='0.1,0.6')
    ap.add_argument('--refs', default='belt,motion_x,bp',
                    help='references to gate against: "belt" = respiration, '
                         '"motion_x" = positive control, "bp" = BP_mmHg (a '
                         'SECOND vital signal, read from the dir of Resp_Volts). '
                         'Comma-separated.')
    ap.add_argument('--bp_band', default='0.7,2.5',
                    help='band for the BP reference (arterial PULSE band, Hz); '
                         'its raw (tonic/mean-drift) row is gated too. "" = raw '
                         'only.')
    ap.add_argument('--lags', default=3.0, type=float,
                    help='lag search half-width in SECONDS')
    ap.add_argument('--fps', default=25.0, type=float)
    ap.add_argument('--stride', default=1, type=int,
                    help='temporal decimation; the model uses temporal_stride 2, '
                         'but 1 (default) measures the NATIVE frame rate, i.e. an '
                         'upper bound on the available information')
    ap.add_argument('--nperm', default=200, type=int)
    ap.add_argument('--partial-motion', action='store_true',
                    help='partial out the LOCAL ROI motion (per-frame phase-'
                         'correlation translation) from every pixel series '
                         'before the gate -- this separates a THERMAL signal '
                         'from an apparent-MOTION one (e.g. ballisto-'
                         'cardiographic head motion, which also sits at the '
                         'pulse rate). Re-run WITHOUT it to compare.')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default='analysis/tir_resp/gate_model_input.json')
    args = ap.parse_args(argv)

    cache_dir = args.cache_dir or find_cache_dir(args.cache_root)
    params = json.load(open(os.path.join(cache_dir, 'params.json')))
    print(f'[gate] cache    : {cache_dir}')
    print(f'[gate] cache key: pad {params["roi_padding"]} | '
          f'{params["roi_landmarks_name"]} | {params["pixel_format"]} | '
          f'src {params["src_w"]}x{params["src_h"]} | '
          f'{params["store_resolution"]}')
    if abs(float(params['roi_padding']) - 0.2) > 1e-9:
        print(f'[gate] WARNING: cache roi_padding {params["roi_padding"]} != the '
              f'Stage-2/3 value 0.2 -- the cached box is NOT the trained ROI.')

    fps_eff = args.fps / max(1, int(args.stride))
    lags = np.arange(-int(round(args.lags * fps_eff)),
                     int(round(args.lags * fps_eff)) + 1)
    sizes = [(None if s.strip().lower() in ('box', 'native') else int(s))
             for s in str(args.sizes).split(',') if s.strip()]
    bands: List[Optional[Tuple[float, float]]] = [None]
    if args.band.strip():
        bands.append(tuple(float(v) for v in args.band.split(',')))
    # BP is a DIFFERENT vital signal with a DIFFERENT band: a pulsatile arterial
    # waveform at heart rate (~0.7-2.5 Hz) vs the 0.1-0.6 Hz respiration band.
    # Gate BOTH its pulse band AND raw -- raw catches the tonic/mean drift that
    # the pulse band excludes, and a raw-only BP hit is NOT a pulse.
    bp_bands: List[Optional[Tuple[float, float]]] = [None]
    if args.bp_band.strip():
        bp_bands.append(tuple(float(v) for v in args.bp_band.split(',')))
    ref_names = [r.strip() for r in str(args.refs).split(',') if r.strip()]
    bname = lambda b: 'raw' if b is None else f'{b[0]}-{b[1]}'   # noqa: E731

    print(f'[gate] raw_root : {args.raw_root}')
    print(f'[gate] fps      : {fps_eff:g} (native {args.fps:g} / stride '
          f'{args.stride}) | lags +-{args.lags}s = {lags.size} samples')
    print(f'[gate] sizes    : {[("box" if s is None else s) for s in sizes]}')
    print(f'[gate] refs     : {ref_names} | bp_band {args.bp_band!r}')
    print(f'[gate] nperm    : {args.nperm}\n')

    out: Dict = {'cache_dir': cache_dir, 'cache_params': params,
                 'sizes': [('box' if s is None else s) for s in sizes],
                 'nperm': args.nperm, 'fps': fps_eff, 'stride': args.stride,
                 'sessions': {}}
    verdicts: List[Tuple[str, str, str, str, float, float]] = []
    bp_verdicts: List[Tuple[str, str, str, str, float, float]] = []

    if args.sample_per_task > 0:
        name_list = sample_sessions(cache_dir, args.sample_per_task)
        print(f'[gate] sampled {len(name_list)} session-tasks: '
              f'{args.sample_per_task} per task label, subject-spread\n')
    else:
        name_list = [s.strip() for s in args.sessions.split(',') if s.strip()]

    for name in name_list:
        subj, task = name.split('_')
        path = rc.shard_path(cache_dir, name)
        if not os.path.isfile(path):
            print(f'=== {name} === NOT CACHED (skipped shard) -> skip')
            continue
        sh = rc.read_shard(path)
        frames = sh['frames']                      # [n, h, w, 3] uint8, native
        if args.stride > 1:
            frames = frames[::args.stride]
        n_frames = int(frames.shape[0])
        box = sh['box']
        print(f'=== {name} === cached box {box} -> {frames.shape[1]}x'
              f'{frames.shape[2]} px, {n_frames} frames')

        spec = discover_sessions(args.raw_root, subjects=[subj],
                                 tasks=[task])[0]
        belt_full = load_belt(spec['resp_file'], int(sh['n_frames']), args.fps)
        motion_full = landmark_motion(spec['ir_file'], int(sh['n_frames']))
        bp_full = None
        if 'bp' in ref_names:
            bp_path = os.path.join(os.path.dirname(spec['resp_file']),
                                   'BP_mmHg.txt')
            if os.path.isfile(bp_path):
                # load_belt is just "1-D text file sampled at frame times" --
                # reusable verbatim for BP_mmHg.txt (also 1000 Hz nominal).
                bp_full = load_belt(bp_path, int(sh['n_frames']), args.fps)
            else:
                print(f'    [bp] BP_mmHg.txt not found next to '
                      f'{spec["resp_file"]} -- skipping the BP reference')
        belt = belt_full[::args.stride] if args.stride > 1 else belt_full
        motion = (motion_full[::args.stride] if args.stride > 1 else motion_full)
        bp = (bp_full[::args.stride]
              if (bp_full is not None and args.stride > 1) else bp_full)
        if belt.size < n_frames:                   # belt shorter than the video
            n_frames = int(belt.size)
            frames = frames[:n_frames]

        motion_xy = None
        if args.partial_motion:
            gray = frames[:n_frames].astype(np.float32).mean(axis=3)  # [n,h,w]
            motion_xy = estimate_translation(gray)
            mag = np.abs(motion_xy).mean(axis=0)
            print(f'    [motion] local ROI translation mean |dx|,|dy| = '
                  f'{mag[0]:.3f}, {mag[1]:.3f} px/frame')

        rec: Dict = {'box': box, 'n_frames': n_frames,
                     'cache_key': os.path.basename(cache_dir), 'gates': {}}
        series_by_ref = {'belt': belt[:n_frames],
                         'motion_x': motion[:n_frames, 0]}
        if bp is not None:
            series_by_ref['bp'] = bp[:n_frames]
        bands_by_ref = {'belt': bands, 'motion_x': bands, 'bp': bp_bands}
        for size in sizes:
            tag = 'box' if size is None else f'{size}x{size}'
            series = resize_like_dataset(frames, size).astype(np.float32)
            if motion_xy is not None:
                series = partial_out(series, motion_xy)
            for ref in ref_names:
                if ref not in series_by_ref:
                    continue
                for b in bands_by_ref.get(ref, bands):
                    key = f'{tag}|{bname(b)}|{ref}'
                    res = run_gate(series, series_by_ref[ref], fps_eff, b, lags,
                                   args.nperm, args.device,
                                   np.random.default_rng(0),
                                   f'{tag:8s} {bname(b):9s} vs {ref}')
                    rec['gates'][key] = res
                    row = (name, tag, bname(b),
                           'DETECTED' if res['p'] < 0.05 else '-',
                           res['best'], res['null95'])
                    if ref == 'belt':
                        verdicts.append(row)
                    elif ref == 'bp':
                        bp_verdicts.append(row)
        out['sessions'][name] = rec

    print('\n' + '=' * 78)
    print('VERDICTS -- respiration ("vs belt") only; p < 0.05 = DETECTED')
    print('=' * 78)
    print(f'{"session":10s} {"size":8s} {"band":9s} {"best|r|":>8s} '
          f'{"null95":>8s}  verdict')
    for name, tag, band, v, best, q95 in verdicts:
        print(f'{name:10s} {tag:8s} {band:9s} {best:8.3f} {q95:8.3f}  {v}')
    n_det = sum(1 for *_, v, _, _ in verdicts if v == 'DETECTED')
    print(f'\n{n_det} of {len(verdicts)} belt rows DETECTED '
          f'({100.0 * n_det / max(1, len(verdicts)):.0f} %)')

    if bp_verdicts:
        print('\n' + '=' * 78)
        print('BP VERDICTS -- vs BP_mmHg (a SECOND vital signal); '
              'p < 0.05 = DETECTED')
        print('=' * 78)
        print(f'{"session":10s} {"size":8s} {"band":9s} {"best|r|":>8s} '
              f'{"null95":>8s}  verdict')
        for name, tag, band, v, best, q95 in bp_verdicts:
            print(f'{name:10s} {tag:8s} {band:9s} {best:8.3f} {q95:8.3f}  {v}')
        nb = sum(1 for *_, v, _, _ in bp_verdicts if v == 'DETECTED')
        print(f'\n{nb} of {len(bp_verdicts)} BP rows DETECTED '
              f'({100.0 * nb / max(1, len(bp_verdicts)):.0f} %). Read the PULSE '
              f'band vs raw SEPARATELY: a raw-only BP hit can be the tonic '
              f'drift, which is not a pulse.')

    # ---- per-TASK and per-GROUP tally: the whole point of the stratified run --
    for level, label in ((task_of, 'TASK'),
                         (lambda n: group_of(task_of(n), args.groups), 'GROUP')):
        tally: Dict[str, List[int]] = {}
        for name, _tag, _band, v, _b, _q in verdicts:
            key = level(name)
            t = tally.setdefault(key, [0, 0])
            t[0] += 1
            t[1] += int(v == 'DETECTED')
        if not tally:
            continue
        print(f'\nper {label} detection rate:')
        for key in sorted(tally):
            n_, d_ = tally[key]
            bar = '#' * int(round(20.0 * d_ / max(1, n_)))
            print(f'  {key:10s} {d_:3d}/{n_:3d}  {100.0 * d_ / max(1, n_):5.0f} %  {bar}')

    print('\nReminder: a run is only interpretable if the motion_x rows were '
          'DETECTED (the positive control).')

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as fh:
            json.dump(out, fh, indent=2, default=float)
        print(f'\njson -> {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
