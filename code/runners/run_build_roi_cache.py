"""Build / inspect the offline TIR-ROI pixel cache.

Spec: ``analysis/tir_resp/hpc_precomputation.md`` section 9. Design validated by
``analysis/tir_resp/cache_parity_probe.py`` (section 10).

The cache is a pure derived artifact: it stores NATIVE-resolution union crops of
each subject-task so the training loader never has to decode a ``.wmv``. It
encodes no downstream decision -- see ``data/roi_cache.py`` -- so nothing here
takes a ``clip_stride``, an ``input_size`` or a cleaning knob.

Modes (exactly one):

``--init``    Probe the source frame size and write ``params.json``. Idempotent,
              so an array job can call it from every element safely.
``--build``   Cache the subject-tasks assigned to this array element. DEFAULT.
``--status``  Filesystem truth: cached / skipped / missing, bytes, skip reasons.
``--index``   Rebuild ``manifest.json`` from the shards (the derived index).
``--check``   Re-verify written shards against the LIVE dataset, byte for byte.

Usage (from ``code/``)::

    # once
    python runners/run_build_roi_cache.py --init --out_root $WORK_SCRATCH/tir_roi_cache

    # one array element (32 of them)
    python runners/run_build_roi_cache.py --build --array_index 0 --array_count 32 \
        --out_root $WORK_SCRATCH/tir_roi_cache

    # after the array job
    python runners/run_build_roi_cache.py --index  --out_root $WORK_SCRATCH/tir_roi_cache
    python runners/run_build_roi_cache.py --check  --out_root $WORK_SCRATCH/tir_roi_cache

CPU only. No GPU, no YAML config, and no torch import unless ``--check`` runs --
a build element must stay cheap to start.
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import roi_cache as rc                                # noqa: E402
from data import tir_resp_dataset as trd                        # noqa: E402
from data import video_io as vio                                # noqa: E402


def env_or(name: str, default: str = '') -> str:
    return os.environ.get(name, default)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def resolve_landmarks(spec) -> Sequence[int]:
    return trd.resolve_roi_landmarks(spec)


def array_position(args) -> tuple:
    """``(index, count)`` from the CLI, else from the Slurm array environment."""
    idx = args.array_index
    cnt = args.array_count
    if idx < 0:
        idx = int(env_or('SLURM_ARRAY_TASK_ID', '0') or 0)
    if cnt <= 0:
        cnt = int(env_or('SLURM_ARRAY_TASK_COUNT', '1') or 1)
    if cnt < 1:
        cnt = 1
    if not (0 <= idx < cnt):
        raise SystemExit(f'--array_index {idx} outside [0, {cnt})')
    return idx, cnt


def my_sessions(sessions: List[dict], idx: int, cnt: int) -> List[dict]:
    """Round-robin partition: balanced even when task cost varies."""
    return [s for i, s in enumerate(sessions) if i % cnt == idx]


def probe_frame_size(sessions: Sequence[dict], n_probe: int = 3) -> tuple:
    """Decode ONE frame from up to ``n_probe`` tasks to learn the source size.

    The size is a KEY field (it is what the boxes are clamped to), so it has to
    come from a real decode rather than a constant. Uniformity is asserted too:
    a corpus mixing frame sizes would break containment for every task but one,
    and the failure would be silent.
    """
    seen: Dict[tuple, List[str]] = {}
    for s in sessions:
        if len(seen) >= n_probe and all(len(v) for v in seen.values()):
            break
        if not s.get('video') or not os.path.isfile(s['video']):
            continue
        try:
            reader = vio.open_video(s['video'])
            try:
                frame = reader.read_range(0, 1)[0]
            finally:
                reader.close()
        except Exception:
            continue
        seen.setdefault(tuple(frame.shape), []).append(str(s['session']))
        if sum(len(v) for v in seen.values()) >= n_probe:
            break
    if not seen:
        raise SystemExit('[cache] could not decode a single probe frame; check '
                         '--raw_root and the OpenCV WMV support')
    if len(seen) > 1:
        detail = '; '.join(f'{k}: {v[:3]}' for k, v in seen.items())
        raise SystemExit(f'[cache] NON-UNIFORM source frame shapes across the '
                         f'corpus: {detail}. The union-box design assumes one '
                         f'source geometry; refusing to build.')
    (h, w, c), names = next(iter(seen.items()))
    return w, h, c, names


def make_params(args, corpus_sessions) -> Dict:
    """The FROZEN parameters. ``corpus_sessions`` is the UNFILTERED discovery.

    The key is a property of the CORPUS, not of the subset this invocation
    happens to build: ``--subjects``/``--limit`` restrict the WORK, they must
    not create a second cache identity. A partial cache is then simply the same
    cache, partially built -- and ``--status`` reports the rest as missing.
    """
    w, h, c, names = probe_frame_size(corpus_sessions)
    if c != 3:
        raise SystemExit(f'[cache] probe decoded {c} channels, expected 3 (RGB). '
                         f'Set VIDEO_BACKEND / check the reader.')
    params = rc.build_params(args.raw_root, corpus_sessions,
                             resolve_landmarks(args.roi_landmarks),
                             args.roi_padding, w, h)
    print(f'[cache] source frame  : {w}x{h}x{c} (probed on {", ".join(names)})')
    return params


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def mode_init(args, corpus, sessions) -> int:
    params = make_params(args, corpus)
    key = rc.cache_key(params)
    root = rc.cache_root(args.out_root, key)
    if (root / rc.PARAMS_NAME).is_file():
        rc.check_params_compatible(root, params)      # raises on any mismatch
        print(f'[cache] params already present and identical: {root}')
    else:
        rc.write_params(root, params)
        print(f'[cache] wrote {root / rc.PARAMS_NAME}')
    print(f'[cache] key           : {key}')
    print(f'[cache] corpus        : {params["corpus"]} '
          f'({params["n_sessions_discovered"]} subject-tasks discovered)')
    print(f'[cache] landmarks     : {params["roi_landmarks_name"]} '
          f'({len(params["roi_landmarks"])} points), padding '
          f'{params["roi_padding"]}')
    print(f'[cache] decoder       : {params["decoder"][:80]}')
    print(f'\n[cache] next: --build --array_index N --array_count M --out_root '
          f'{args.out_root}')
    return 0


def mode_build(args, corpus, sessions) -> int:
    idx, cnt = array_position(args)
    params = make_params(args, corpus)
    key = rc.cache_key(params)
    root = rc.cache_root(args.out_root, key)
    if not (root / rc.PARAMS_NAME).is_file():
        raise SystemExit(f'[cache] {root}/params.json not found -- run --init '
                         f'first (same --roi_padding / --roi_landmarks).')
    rc.check_params_compatible(root, params)          # FAIL LOUDLY

    todo = my_sessions(sessions, idx, cnt)
    print(f'[cache] key           : {key}')
    print(f'[cache] element       : {idx}/{cnt} -> {len(todo)} subject-task(s)')
    t0 = time.time()
    n_ok = n_skip = 0
    reasons: Dict[str, int] = {}
    for j, spec in enumerate(todo, 1):
        session = str(spec['session'])
        if rc.shard_path(root, session).is_file() and not args.overwrite:
            n_ok += 1
            continue                                       # resumable
        t1 = time.time()
        payload, reason = rc.build_task(spec, resolve_landmarks(args.roi_landmarks),
                                        args.roi_padding,
                                        with_landmarks=not args.no_landmarks,
                                        max_task_bytes=int(args.max_task_gb * 1e9))
        if payload is None:
            rc.write_skip(root, session, reason or 'unknown')
            n_skip += 1
            reasons[reason or 'unknown'] = reasons.get(reason or 'unknown', 0) + 1
            print(f'  [{j}/{len(todo)}] {session:<12} SKIP  {reason}', flush=True)
            continue
        path = rc.write_shard(root, session, payload, compress=args.compress)
        n_ok += 1
        mb = path.stat().st_size / 1e6
        print(f'  [{j}/{len(todo)}] {session:<12} ok    '
              f'{payload["n_frames"]:5d} fr  box {tuple(int(v) for v in payload["box"])} '
              f'{mb:6.1f} MB  {time.time() - t1:5.1f}s', flush=True)
    dt = time.time() - t0
    print(f'[cache] element {idx} done: {n_ok} cached, {n_skip} skipped in '
          f'{dt / 60:.1f} min')
    for r, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f'    skip {r:28s} {n}')
    return 0


def mode_status(args, corpus, sessions) -> int:
    params = make_params(args, corpus)
    root = rc.cache_root(args.out_root, rc.cache_key(params))
    if not (root / rc.PARAMS_NAME).is_file():
        raise SystemExit(f'[cache] nothing built at {root}')
    info = rc.scan(root, sessions)
    print(f'[cache] root          : {root}')
    print(f'[cache] cached        : {info["cached"]}')
    print(f'[cache] skipped       : {info["skipped"]}')
    print(f'[cache] missing       : {info["missing"]}')
    print(f'[cache] size          : {info["bytes"] / 1e9:.1f} GB '
          f'({info["bytes"] / 2 ** 30:.1f} GiB)')
    if info['missing']:
        print(f'[cache] e.g. missing  : {", ".join(info["missing_names"][:8])}')
    if info['skipped_names']:
        hist: Dict[str, int] = {}
        for name in info['skipped_names']:
            with open(root / rc.SUBDIR_SKIPPED / f'{name}.json') as fh:
                r = json.load(fh)['reason'].split(':')[0]
            hist[r] = hist.get(r, 0) + 1
        for r, n in sorted(hist.items(), key=lambda kv: -kv[1]):
            print(f'    skip {r:28s} {n}')
    return 0


def mode_index(args, corpus, sessions) -> int:
    params = make_params(args, corpus)
    root = rc.cache_root(args.out_root, rc.cache_key(params))
    m = rc.build_manifest(root, sessions, params)
    print(f'[cache] manifest -> {root / rc.MANIFEST_NAME}')
    print(f'[cache] ok {m["n_ok"]}  skipped {m["n_skipped"]}  '
          f'missing {m["n_missing"]}  '
          f'{m["total_bytes"] / 1e9:.1f} GB')
    if m['n_missing']:
        print('[cache] NOTE: missing shards are either not yet built or '
              'geometry-fatal without a skip file')
    return 0


def mode_check(args, corpus, sessions) -> int:
    """Re-verify written shards against the LIVE pipeline, byte for byte.

    This is gate item 2 of the spec, applied to the ARTIFACT rather than to an
    in-memory simulation: the shard's native crop is sub-cropped and resized and
    must equal the live decode -> ``_roi_patches`` result exactly.
    """
    params = make_params(args, corpus)
    root = rc.cache_root(args.out_root, rc.cache_key(params))
    rc.check_params_compatible(root, params)

    import numpy as np
    rng = np.random.default_rng(args.seed)
    have = [s for s in sessions
            if rc.shard_path(root, str(s['session'])).is_file()]
    if not have:
        raise SystemExit(f'[cache] no shards under {root}')
    pick = [have[int(i)] for i in
            rng.permutation(len(have))[:min(args.check_sessions, len(have))]]

    idx, cnt = array_position(args)
    fails: List[str] = []
    n_px = 0
    for hop in args.hops:
        ds = trd.TirRoiRespPretrainDataset(
            raw_root=args.raw_root, streams=('tir', 'resp'),
            clip_duration=args.clip_seconds, clip_stride=hop,
            temporal_stride=2, tubelet_t=2, input_size=args.input_size,
            roi_padding=args.roi_padding,
            min_signal_spread=0.0, rail_touch_v=0.0, physio_norm='session',
            subjects=None, verbose=False)
        base = ds.base
        by_sess: Dict[str, List[int]] = {}
        for i, e in enumerate(base.entries):
            by_sess.setdefault(e['session'], []).append(i)
        print(f'\n--- hop {hop:g}s ---')
        for spec in pick:
            session = str(spec['session'])
            rows = by_sess.get(session)
            if not rows:
                continue
            shard = rc.read_shard(rc.shard_path(root, session))
            box_u = shard['box']
            print(f'  {session:<12} box_u {box_u} shape '
                  f'{tuple(shard["frames"].shape)} n_frames '
                  f'{shard["n_frames"]}')
            if shard['landmarks'] is not None:
                live_lm = base._landmarks(session)[:shard['n_frames']]
                if not np.array_equal(np.asarray(shard['landmarks'], np.float32),
                                      np.asarray(live_lm, np.float32)):
                    fails.append(f'{session}: stored landmarks != live '
                                 f'IRFeatures[:{shard["n_frames"]}]')
            for i in rows[:args.check_clips]:
                e = base.entries[i]
                frames = base._frames(e)                # LIVE seeking decode
                box = base.clip_roi_box(i)              # authoritative
                f0, f1 = e['frame_start'], e['frame_end']
                if f1 > shard['n_frames']:
                    fails.append(f'{session} clip {i}: frames {f0}..{f1} beyond '
                                 f'the stored {shard["n_frames"]}')
                    continue
                shifted = rc.shift_box(box, (box_u[0], box_u[2]))
                if min(shifted) < 0 or not rc.box_contains(box_u, box):
                    fails.append(f'{session} clip {i}: clip {box} not inside '
                                 f'union {box_u}')
                    continue
                live = base._roi_patches(e, frames, box)
                cach = base._roi_patches(e, shard['frames'][f0:f1], shifted)
                n_px += 1
                if live.shape != cach.shape:
                    fails.append(f'{session} clip {i}: {live.shape} != '
                                 f'{cach.shape}')
                elif not np.array_equal(live, cach):
                    d = int(np.abs(live.astype(np.int16)
                                   - cach.astype(np.int16)).max())
                    fails.append(f'{session} clip {i}: cached != live '
                                 f'(max abs d {d})')
    print(f'\n[cache] checked {n_px} clip(s) over {len(pick)} session(s) x '
          f'{len(args.hops)} hop(s)')
    print(f'[cache] VERDICT: {"PASS" if not fails else "FAIL"}')
    for f in fails[:10]:
        print(f'    ! {f}')
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'checked_clips': n_px, 'sessions': len(pick),
                       'fails': fails}, fh, indent=2)
        print(f'[cache] JSON -> {args.json}')
    return 0 if not fails else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--init', action='store_true',
                   help='probe the source frame size and write params.json')
    g.add_argument('--build', action='store_true', help='cache this shard (default)')
    g.add_argument('--status', action='store_true', help='report progress')
    g.add_argument('--index', action='store_true', help='rebuild manifest.json')
    g.add_argument('--check', action='store_true',
                   help='re-verify shards against the live dataset')

    p.add_argument('--raw_root', default=env_or('RAW_DATA_PATH', ''))
    p.add_argument('--out_root',
                   default=os.path.join(env_or('WORK_SCRATCH',
                                               env_or('OUTPUT_DIR', '.')),
                                        'tir_roi_cache'))
    p.add_argument('--subjects', default='', help='comma list ("" = all)')
    p.add_argument('--limit', type=int, default=0,
                   help='first N discovered subject-tasks (0 = all; smoke tests)')

    # ROI CONTRACT -- these two are part of the cache key
    p.add_argument('--roi_padding', type=float, default=0.2)
    p.add_argument('--roi_landmarks', default='',
                   help="'' or a preset name (nose_mouth, nostrils, "
                        "nostril_mouth, nose_tip) or an explicit CSV of indices")

    # array partition
    p.add_argument('--array_index', type=int, default=-1,
                   help='default: $SLURM_ARRAY_TASK_ID')
    p.add_argument('--array_count', type=int, default=0,
                   help='default: $SLURM_ARRAY_TASK_COUNT')
    p.add_argument('--overwrite', action='store_true',
                   help='re-cache shards that already exist')

    # build options
    p.add_argument('--compress', action='store_true',
                   help='np.savez_compressed (smaller, slower; not the default '
                        'because the 92 GB figure assumes raw uint8)')
    p.add_argument('--no_landmarks', action='store_true',
                   help='omit the landmark track from the shard')
    p.add_argument('--max_task_gb', type=float, default=12.0,
                   help='REFUSE (skip, with a reason) a task whose full-frame '
                        'decode would exceed this many GB. The frames array is '
                        'the whole task in RAM: 726x480x3 = 1.05 MB/frame, so '
                        'the corpus longest task (5149 frames) needs ~5.4 GB. '
                        'Set this BELOW the request so an oversized video is '
                        'skipped loudly instead of OOM-killing the element.')

    # check options
    p.add_argument('--hops', type=float, nargs='+', default=[2.0, 1.0])
    p.add_argument('--clip_seconds', type=float, default=8.0)
    p.add_argument('--input_size', type=int, default=112)
    p.add_argument('--check_sessions', type=int, default=4)
    p.add_argument('--check_clips', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--json', default='')
    args = p.parse_args(argv)

    if not args.raw_root or not os.path.isdir(args.raw_root):
        print(f'[cache] raw_root not usable: {args.raw_root!r}; pass --raw_root '
              f'or set $RAW_DATA_PATH', file=sys.stderr)
        return 2

    subjects = [s.strip() for s in args.subjects.split(',') if s.strip()] or None
    # UNFILTERED: the cache key is a property of the corpus (see make_params).
    corpus = trd.discover_sessions(args.raw_root)
    sessions = trd.discover_sessions(args.raw_root, subjects=subjects)
    if args.limit:
        sessions = sessions[:args.limit]
    print(f'[cache] out_root      : {args.out_root}')
    print(f'[cache] raw_root      : {args.raw_root}')
    print(f'[cache] corpus (key)  : {len(corpus)} subject-task(s)')
    print(f'[cache] scope         : {len(sessions)} subject-task(s) this run')

    if args.init:
        return mode_init(args, corpus, sessions)
    if args.status:
        return mode_status(args, corpus, sessions)
    if args.index:
        return mode_index(args, corpus, sessions)
    if args.check:
        return mode_check(args, corpus, sessions)
    return mode_build(args, corpus, sessions)


if __name__ == '__main__':
    raise SystemExit(main())
