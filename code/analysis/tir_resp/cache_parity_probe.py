"""Parity probe for the offline TIR-ROI cache.

Spec: ``analysis/tir_resp/hpc_precomputation.md`` section 9.7. Three checks, one
of which can still change the design:

  T1 CONTAINMENT   every kept clip box must lie inside the whole-task UNION box
                   the cache writer stores -- at BOTH hops (2.0 s for Stage 2,
                   1.0 s for Stage 3), plus the synthetic cases the real corpus
                   cannot reach (``_clamp_box``'s ``min_size`` re-centring
                   branch: every real box is ~100 px wide, so it never fires).
  T2 PIXEL PARITY  sub-cropping the clip box out of a NATIVE-resolution union
                   crop and resizing must equal the live decode ->
                   ``_roi_patches`` path, byte for byte.
  T3 SEEK PARITY   the SEQUENTIAL decode a cache WRITER would use must agree with
                   the SEEKING ``read_range`` the LIVE dataset uses.

T3 is the one check that is not expected to pass by construction. ``read_all()``
decodes linearly from frame 0; ``read_range(start, n)`` issues
``CAP_PROP_POS_FRAMES`` whenever ``start > 0``. For an inter-frame codec those
two can differ at non-keyframe positions, in which case the live pipeline has no
single canonical decode of a source frame -- it reads the same frame ~4x under
different ``frame_start`` values -- and the cache would be DEFINING the pixels
rather than reproducing them. That is a design-relevant finding, not a bug to
hide behind a tolerance, so it is reported first in the summary.

T2, by contrast, IS expected to pass: cropping a sub-rectangle out of a
translated rectangle is the same pixels, and the interpolation decision in
``_roi_patches`` depends only on the box EXTENT, which translation preserves. Its
value is catching coordinate/clamping arithmetic errors -- and, importantly,
failing loudly if the union box is NOT a superset, because numpy would then
silently WRAP negative indices instead of raising. T2 asserts that explicitly.

No cache files are needed: the union crop is simulated in memory, so this runs
BEFORE the writer exists.

Usage (from ``code/``; CPU only, no GPU, nothing written):

    python analysis/tir_resp/cache_parity_probe.py --tasks 40
    python analysis/tir_resp/cache_parity_probe.py --tasks 0     # full sweep

Exit code 0 == every check passed.
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from data import tir_resp_dataset as trd                        # noqa: E402
from data import video_io as vio                                # noqa: E402

#: Thermal frame size. Only a DEFAULT: the probe derives the true size from a
#: decoded frame (T2) and cross-checks the production ``clip_roi_box`` against
#: the landmarks-only computation on a sample, so a wrong constant cannot slip
#: through the T1 sweep silently.
SRC_W, SRC_H = 726, 480

TGT = trd.TARGET_LANDMARK_IDX
Box = Tuple[int, int, int, int]


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def _union_box(ir: np.ndarray, padding: float, w: int, h: int
               ) -> Optional[Box]:
    """The box the cache WRITER would store for one subject-task.

    Min/max over the task's **valid** rows only: non-finite rows dropped and
    ``(0, 0)`` tracker-sentinel rows excluded (spec section 2.3). Safe because a
    clip overlapping any sentinel/missing target landmark is dropped entirely by
    ``missing_frame_mask``, so no KEPT clip's frames are removed here.
    """
    pts = np.asarray(ir)[:, TGT, :]
    finite = np.isfinite(pts).all(axis=2)
    sentinel = (np.abs(pts) < 1e-9).all(axis=2)
    keep = finite.all(axis=1) & ~sentinel.all(axis=1)
    if not keep.any():
        return None
    return trd.roi_box_from_landmarks(pts[keep], w, h, padding)


def _clip_box(ir: np.ndarray, start: int, end: int, padding: float,
              w: int, h: int) -> Box:
    """The clip box -- exactly what ``BP4DPlusTIRRespDataset.clip_roi_box``
    computes, minus its decode (which exists only to learn ``w, h``)."""
    pts = np.asarray(ir)[start:end][:, TGT, :]
    return trd.roi_box_from_landmarks(pts, w, h, padding)


def _contains(outer: Box, inner: Box) -> bool:
    ox0, ox1, oy0, oy1 = outer
    ix0, ix1, iy0, iy1 = inner
    return bool(ox0 <= ix0 and ix1 <= ox1 and oy0 <= iy0 and iy1 <= oy1)


def _area(b: Box) -> int:
    return max(0, b[1] - b[0]) * max(0, b[3] - b[2])


# --------------------------------------------------------------------------- #
# T1 -- containment
# --------------------------------------------------------------------------- #
def test_clamp_corner(padding: float, w: int, h: int, trials: int = 4000
                      ) -> Dict:
    """Synthetic containment, aimed at ``_clamp_box``'s degenerate branch.

    Real clips never produce a sub-2-px box, so the ``min_size`` re-centring
    branch is unreachable from the corpus; it is also the only place the box
    construction is not obviously monotone in its inputs. Drive it directly:
    draw an inner landmark cloud inside an outer one, near the frame EDGES as
    well as in the middle, and require containment after both are clamped.
    """
    rng = np.random.default_rng(12345)
    degenerate = 0
    fails: List[str] = []
    for _ in range(trials):
        x = float(rng.uniform(0.0, w))
        y = float(rng.uniform(0.0, h))
        half = float(rng.choice([0.0, 0.25, 1.0, 5.0]))
        inner = np.array([[x - half, y - half], [x + half, y + half]])
        gx = float(rng.uniform(0.0, 3.0))
        gy = float(rng.uniform(0.0, 3.0))
        outer = np.array([[x - half - gx, y - half - gy],
                          [x + half + gx, y + half + gy]])
        try:
            ib = trd.roi_box_from_landmarks(inner, w, h, padding)
            ob = trd.roi_box_from_landmarks(outer, w, h, padding)
        except ValueError:
            continue
        # Would _clamp_box have had to grow a side to min_size (2)? Measure the
        # PRE-clamp padded extent -- the CLAMPED box is always >= 2 px, so
        # testing the result would count 0 hits and claim coverage it lacks.
        raw_ext = 2.0 * half * (1.0 + 2.0 * padding)
        if raw_ext < 2.0:
            degenerate += 1
        if not _contains(ob, ib):
            fails.append(f'synthetic: inner {ib} not inside outer {ob}')
            if len(fails) >= 5:
                break
    return {'trials': trials, 'degenerate_hits': degenerate, 'fails': fails}


def test_containment(base, hops: Sequence[float], w: int, h: int,
                     padding: float) -> Dict:
    """Sweep the real kept clips of every hop and require containment."""
    out: Dict = {'clips': 0, 'tasks': 0, 'fails': [], 'worst_margin': None,
                 'per_hop': {}}
    seen_tasks = set()
    for hop, ds in hops:
        n_clips, n_bad, worst = 0, 0, None
        for i, e in enumerate(ds.entries):
            sess = e['session']
            ir = ds._landmarks(sess)
            ub = _union_box(ir, padding, w, h)
            if ub is None:
                continue
            cb = _clip_box(ir, e['frame_start'], e['frame_end'], padding, w, h)
            n_clips += 1
            out['clips'] += 1
            if sess not in seen_tasks:
                seen_tasks.add(sess)
                out['tasks'] += 1
            if not _contains(ub, cb):
                n_bad += 1
                if len(out['fails']) < 10:
                    out['fails'].append(
                        f'{sess} hop {hop}s frames {e["frame_start"]}..'
                        f'{e["frame_end"]}: clip {cb} not inside union {ub}')
            # how much slack the union leaves on the tightest axis, in px
            m = min(cb[0] - ub[0], ub[1] - cb[1], cb[2] - ub[2], ub[3] - cb[3])
            worst = m if worst is None else min(worst, m)
        out['per_hop'][f'{hop:g}s'] = {'clips': n_clips, 'failed': n_bad,
                                       'tightest_margin_px': worst}
        if out['worst_margin'] is None or (worst is not None
                                           and worst < out['worst_margin']):
            out['worst_margin'] = worst
    return out


def cross_check_clip_box(base, n: int, padding: float, w: int, h: int) -> Dict:
    """Validate the SRC_W/SRC_H constant against production ``clip_roi_box``.

    ``clip_roi_box(i)`` decodes the clip purely to learn the frame size, then
    builds the box from landmarks. Comparing it with the landmarks-only
    computation proves the constant used by the whole T1 sweep is the real
    decoded size, and that the frame size is uniform across tasks.
    """
    out: Dict = {'checked': 0, 'fails': [], 'frame_shapes': [], 'indices': []}
    for i, e in enumerate(base.entries):
        if out['checked'] >= n:
            break
        proc = tuple(base.clip_roi_box(i))            # decodes -> authoritative
        ir = base._landmarks(e['session'])
        mine = _clip_box(ir, e['frame_start'], e['frame_end'], padding, w, h)
        frames = base._frames(e)
        shape = tuple(frames.shape[1:])
        out['frame_shapes'].append(shape)
        out['indices'].append(int(i))
        if tuple(mine) != proc:
            out['fails'].append(
                f'clip {i} ({e["session"]}): landmarks-only {mine} != '
                f'production {proc}')
        out['checked'] += 1
    return out


# --------------------------------------------------------------------------- #
# T2 -- pixel parity
# --------------------------------------------------------------------------- #
def test_pixel_parity(base, n_clips: int, padding: float, seed: int) -> Dict:
    """Byte-equality of the union-crop path against the live decode path.

    Simulates the cache: ``cached = frames[:, uy0:uy1, ux0:ux1]`` at NATIVE
    resolution, then sub-crops the clip box out of it and resizes -- reusing the
    PRODUCTION ``_roi_patches`` for both sides so no resize logic is duplicated.
    The only difference handed to it is the coordinate frame, shifted by the
    union origin; ``_roi_patches`` derives its interpolation filter from the box
    EXTENT, which the shift preserves.
    """
    rng = np.random.default_rng(seed)
    by_sess: Dict[str, List[int]] = {}
    for i, e in enumerate(base.entries):
        by_sess.setdefault(e['session'], []).append(i)
    sess = sorted(by_sess)
    picks: List[int] = []
    for s in rng.permutation(len(sess)):
        if len(picks) >= n_clips:
            break
        picks.append(int(rng.choice(by_sess[sess[s]])))

    out: Dict = {'clips': 0, 'sessions': len({base.entries[i]['session']
                                              for i in picks}),
                 'fails': [], 'max_abs_diff': 0, 'frame_shape': None,
                 'raw_ok': 0}
    for i in picks:
        e = base.entries[i]
        frames = base._frames(e)                       # the LIVE decode
        h, w = int(frames.shape[1]), int(frames.shape[2])
        out['frame_shape'] = [h, w, int(frames.shape[3])]
        ir = base._landmarks(e['session'])
        ub = _union_box(ir, padding, w, h)
        if ub is None:
            continue
        cb = _clip_box(ir, e['frame_start'], e['frame_end'], padding, w, h)
        ux0, ux1, uy0, uy1 = ub
        shifted = (cb[0] - ux0, cb[1] - ux0, cb[2] - uy0, cb[3] - uy0)
        label = f'{e["session"]} frames {e["frame_start"]}..{e["frame_end"]}'
        # A negative shifted coord means the union box is NOT a superset; numpy
        # would wrap the index and silently return wrong pixels. Fail loudly.
        if min(shifted) < 0:
            out['fails'].append(
                f'{label}: clip {cb} maps to {shifted} in union {ub} -- NEGATIVE, '
                f'the union box does not contain the clip box')
            continue
        if uy1 > h or ux1 > w or uy0 < 0 or ux0 < 0:
            out['fails'].append(f'{label}: union {ub} outside the frame {w}x{h}')
            continue
        cached = frames[:, uy0:uy1, ux0:ux1]            # native, no resize
        live = base._roi_patches(e, frames, cb)
        cach = base._roi_patches(e, cached, shifted)
        if live.shape != cach.shape:
            out['fails'].append(f'{label}: shape {live.shape} != {cach.shape}')
            continue
        if not np.array_equal(live, cach):
            d = int(np.abs(live.astype(np.int16) - cach.astype(np.int16)).max())
            out['max_abs_diff'] = max(out['max_abs_diff'], d)
            out['fails'].append(f'{label}: cached != live (max |d| {d})')
        else:
            out['raw_ok'] += 1
        out['clips'] += 1
    return out


# --------------------------------------------------------------------------- #
# T3 -- seek vs sequential decode
# --------------------------------------------------------------------------- #
def test_seek_parity(video: str, num_frames: int, clip_frames: int,
                     n_probe: int, n_ir: Optional[int], seed: int) -> Dict:
    """Sequential (writer) vs seeking (live) decode of the SAME frame indices."""
    rng = np.random.default_rng(seed)
    out: Dict = {'n_probe': 0, 'n_mismatch': 0, 'max_abs_diff': 0,
                 'mismatch_frames': 0, 'total_frames': 0, 'fails': [],
                 'read_all_frames': 0, 'num_frames': int(num_frames),
                 'n_ir': None if n_ir is None else int(n_ir)}

    r1 = vio.open_video(video)
    try:
        seq = r1.read_all()
    finally:
        r1.close()
    out['read_all_frames'] = int(seq.shape[0])
    N = int(seq.shape[0])
    if N != int(num_frames):
        out['fails'].append(
            f'read_all decoded {N} frames but the container reports '
            f'{int(num_frames)}')

    span = max(1, N - clip_frames)
    fracs = np.linspace(0.05, 0.95, num=max(1, n_probe - 2)) if n_probe > 2 \
        else np.array([0.5])
    starts = sorted({int(round(f * span)) for f in fracs}
                    | {int(rng.integers(0, span + 1)) for _ in range(2)})
    starts = [s for s in starts if 0 <= s < N]

    for st in starts:
        k = min(clip_frames, N - st)
        if k <= 0:
            continue
        r2 = vio.open_video(video)
        try:
            got = r2.read_range(st, k)
        except Exception as exc:                  # short read / seek refused
            out['n_mismatch'] += 1
            out['mismatch_frames'] += int(k)
            out['fails'].append(f'start {st}: read_range raised {exc!r}')
            continue
        finally:
            r2.close()
        want = seq[st:st + k]
        out['n_probe'] += 1
        out['total_frames'] += int(k)
        if got.shape != want.shape:
            out['n_mismatch'] += 1
            out['mismatch_frames'] += int(k)
            out['fails'].append(
                f'start {st}: shape {got.shape} != {want.shape}')
            continue
        bad = np.any(got != want, axis=(1, 2, 3))
        n_bad = int(bad.sum())
        if n_bad:
            d = int(np.abs(got.astype(np.int16) - want.astype(np.int16)).max())
            out['n_mismatch'] += 1
            out['mismatch_frames'] += n_bad
            out['max_abs_diff'] = max(out['max_abs_diff'], d)
            first = int(np.argmax(bad))
            out['fails'].append(
                f'start {st}: {n_bad}/{k} frames differ from the sequential '
                f'decode (first at index {first}, max |d| {d})')
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _build(raw_root: str, hop: float, args,
           subjects: Optional[Sequence[str]] = None
           ) -> trd.TirRoiRespPretrainDataset:
    """One Stage-2 geometry view at a given hop, cleaning OFF by default.

    Cleaning is deliberately off unless ``--cleaning``: the rail/spread rules
    only REMOVE clips, so testing without them covers a superset of the shipped
    kept set (conservative for containment) and avoids loading a 1000 Hz
    respiration series per task, which is what makes the full sweep cheap.
    """
    return trd.TirRoiRespPretrainDataset(
        raw_root=raw_root, streams=('tir', 'resp'),
        clip_duration=args.clip_seconds, clip_stride=hop,
        temporal_stride=args.temporal_stride, tubelet_t=2,
        input_size=args.input_size, roi_padding=args.padding,
        min_signal_spread=(0.1 if args.cleaning else 0.0),
        rail_touch_v=(9.9 if args.cleaning else 0.0),
        physio_norm='session', subjects=list(subjects) if subjects else None,
        max_entries=args.max_entries or None, verbose=False)


def _select_subjects(raw_root: str, n_tasks: int, explicit) -> Optional[List[str]]:
    """``--tasks N`` -> the distinct subjects of the first N discovered tasks."""
    if explicit:
        return [s.strip() for s in str(explicit).split(',') if s.strip()]
    if n_tasks <= 0:
        return None
    found = trd.discover_sessions(raw_root)
    return sorted({r['subject'] for r in found[:n_tasks]})


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw_root', default=os.environ.get('RAW_DATA_PATH', ''))
    p.add_argument('--tasks', type=int, default=40,
                   help='number of subject-tasks to cover; 0 = the whole corpus')
    p.add_argument('--subjects', default='', help='explicit subject list (a,b)')
    p.add_argument('--max_entries', type=int, default=0,
                   help='cap the entries the dataset builds (0 = no cap)')
    p.add_argument('--hops', type=float, nargs='+', default=[2.0, 1.0],
                   help='clip hops in seconds: Stage 2 ships 2.0, Stage 3 1.0')
    p.add_argument('--clip_seconds', type=float, default=8.0)
    p.add_argument('--temporal_stride', type=int, default=2)
    p.add_argument('--input_size', type=int, default=112)
    p.add_argument('--padding', type=float, default=0.2)
    p.add_argument('--cleaning', action='store_true',
                   help='enable the shipped rail/spread rules (0.1 V / 9.9 V); '
                        'slower, and only REMOVES clips')
    p.add_argument('--pixel_clips', type=int, default=8)
    p.add_argument('--t3_tasks', type=int, default=2)
    p.add_argument('--t3_probes', type=int, default=4)
    p.add_argument('--max_video_frames', type=int, default=1600,
                   help='skip T3 on longer videos: read_all holds the whole '
                        'task in RAM (~0.9 GB at 851 frames of 726x480x3)')
    p.add_argument('--clipbox_checks', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--json', default='', help='optional JSON report path')
    args = p.parse_args(argv)

    if not args.raw_root or not os.path.isdir(args.raw_root):
        print(f'[probe] raw_root not usable: {args.raw_root!r}; pass --raw_root '
              f'or set $RAW_DATA_PATH', file=sys.stderr)
        return 2

    print(f'[probe] raw_root : {args.raw_root}')
    print(f'[probe] hops     : {args.hops} s; clip {args.clip_seconds}s; '
          f'input_size {args.input_size}; padding {args.padding}; '
          f'cleaning {"ON" if args.cleaning else "off"}')

    subs = _select_subjects(args.raw_root, args.tasks, args.subjects)
    print(f'[probe] subjects : {len(subs) if subs else "ALL"}')
    report: Dict = {'raw_root': args.raw_root, 'args': vars(args)}

    hops: List[Tuple[float, object]] = []
    for hop in args.hops:
        ds = _build(args.raw_root, hop, args, subs)
        base = ds.base
        print(f'[probe] hop {hop:g}s: {len(base.entries)} clips over '
              f'{len(base.sessions)} subject-task sessions')
        hops.append((hop, base))
    if not hops or not hops[0][1].entries:
        print('[probe] no clips built -- check --subjects / --raw_root',
              file=sys.stderr)
        return 2

    w, h = SRC_W, SRC_H
    base0 = hops[0][1]

    # ---- cross-check the frame-size constant against production -------------
    print('\n--- T0 frame size / production clip_roi_box cross-check ---')
    t0 = cross_check_clip_box(base0, args.clipbox_checks, args.padding, w, h)
    if t0['frame_shapes']:
        hh, ww, cc = t0['frame_shapes'][0]
        print(f'  decoded frame shape : {hh}x{ww}x{cc}')
        if any(s != t0['frame_shapes'][0] for s in t0['frame_shapes']):
            t0['fails'].append(f'non-uniform decoded shapes '
                               f'{sorted(set(t0["frame_shapes"]))}')
        w, h = ww, hh
    print(f'  constant {SRC_W}x{SRC_H} vs decoded {w}x{h}: '
          f'{"MATCH" if (w, h) == (SRC_W, SRC_H) else "DIFFERENT -- sweep uses"}')
    print(f'  production clip_roi_box compared on {t0["checked"]} clip(s): '
          f'{"PASS" if not t0["fails"] else "FAIL"}')
    for f in t0['fails'][:5]:
        print(f'    ! {f}')
    report['T0'] = t0

    # ---- T1 ----------------------------------------------------------------
    print(f'\n--- T1 CONTAINMENT (union box must contain every clip box) ---')
    t1 = test_containment(base0, hops, w, h, args.padding)
    t1['clamp_corner'] = test_clamp_corner(args.padding, w, h)
    for key, v in t1['per_hop'].items():
        print(f'  hop {key:>5}: {v["clips"]:7d} clips, {v["failed"]} outside, '
              f'tightest margin {v["tightest_margin_px"]} px')
    cc_hits = t1['clamp_corner']['degenerate_hits']
    print(f'  synthetic min_size corner: {cc_hits} degenerate case(s) of '
          f'{t1["clamp_corner"]["trials"]} trials, '
          f'{len(t1["clamp_corner"]["fails"])} failure(s)')
    print(f'  => {"PASS" if not t1["fails"] and not t1["clamp_corner"]["fails"] else "FAIL"}'
          f'  ({t1["clips"]} clips, {t1["tasks"]} subject-tasks)')
    for f in (t1['fails'] + t1['clamp_corner']['fails'])[:5]:
        print(f'    ! {f}')
    report['T1'] = t1

    # ---- T2 ----------------------------------------------------------------
    print(f'\n--- T2 PIXEL PARITY (native union crop vs live decode) ---')
    t2 = test_pixel_parity(base0, args.pixel_clips, args.padding, args.seed)
    print(f'  frame shape {t2["frame_shape"]}, {t2["clips"]} clip(s) over '
          f'{t2["sessions"]} session(s)')
    print(f'  => {"PASS" if not t2["fails"] else "FAIL"}'
          f'  ({t2["raw_ok"]}/{t2["clips"]} byte-identical, '
          f'max |d| {t2["max_abs_diff"]})')
    for f in t2['fails'][:5]:
        print(f'    ! {f}')
    report['T2'] = t2

    # ---- T3 ----------------------------------------------------------------
    print(f'\n--- T3 SEEK PARITY (sequential writer decode vs seeking live '
          f'decode) ---')
    sess = sorted(base0.sessions)
    cand = [s for s in sess
            if 0 < int(base0.sessions[s].get('n_vid') or 0)
            <= args.max_video_frames]
    rng = np.random.default_rng(args.seed + 1)
    order = rng.permutation(len(cand)) if cand else []
    t3_all: List[Dict] = []
    for j in order[:args.t3_tasks]:
        s = cand[int(j)]
        meta = base0.sessions[s]
        r = test_seek_parity(meta['video'], int(meta['n_vid']),
                             base0.clip_frames, args.t3_probes,
                             int(meta.get('n_ir') or 0), args.seed + int(j))
        r['session'] = s
        t3_all.append(r)
        print(f'  {s:<12} n_vid {r["num_frames"]:5d} '
              f'n_ir {r["n_ir"] if r["n_ir"] is not None else "?":>5} '
              f'read_all {r["read_all_frames"]:5d} | '
              f'{r["n_probe"]} probe(s), {r["n_mismatch"]} mismatching, '
              f'{r["mismatch_frames"]}/{r["total_frames"]} frames differ, '
              f'max |d| {r["max_abs_diff"]}')
        for f in r['fails'][:3]:
            print(f'    ! {f}')
    if t3_all:
        worst = max(t3_all, key=lambda r: r['max_abs_diff'])
        tot = sum(r['total_frames'] for r in t3_all)
        bad = sum(r['mismatch_frames'] for r in t3_all)
        print(f'  => {"PASS" if bad == 0 else "FAIL"}  '
              f'({tot - bad}/{tot} frames identical, max |d| '
              f'{worst["max_abs_diff"]})')
    else:
        print(f'  SKIPPED: no session with <= {args.max_video_frames} frames')
    report['T3'] = t3_all

    # ---- summary -----------------------------------------------------------
    ok = (not t0['fails'] and not t1['fails']
          and not t1['clamp_corner']['fails'] and not t2['fails']
          and not [r for r in t3_all if r['mismatch_frames']])
    print('\n' + '=' * 68)
    print(f'VERDICT: {"PASS" if ok else "FAIL"}   '
          f'(T0 {"ok" if not t0["fails"] else "FAIL"}, '
          f'T1 {"ok" if not t1["fails"] and not t1["clamp_corner"]["fails"] else "FAIL"}, '
          f'T2 {"ok" if not t2["fails"] else "FAIL"}, '
          f'T3 {"ok" if t3_all and not [r for r in t3_all if r["mismatch_frames"]] else "FAIL"})')
    if t3_all and any(r['mismatch_frames'] for r in t3_all):
        print('\nT3 FAILED: the live pipeline seeks and a cache writer would not,\n'
              'and the two disagree. The cache therefore cannot "reproduce" the\n'
              'live pixels -- it would DEFINE them. Report this before writing\n'
              'the cache (analysis/hpc_precomputation.md section 3.3).')
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f'[probe] JSON report -> {args.json}')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
