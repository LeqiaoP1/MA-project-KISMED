"""Build the offline 2D-texture face-ROI pixel cache (``rgb2d``).

Spec: ``notes/design/2DTexture_ROI_cache_preprocessing.md`` (v2).
Build order: ``notes/plan/2DTexture_ROI_cache_implementation_plan.md`` (Phase 2).

Modes (mutually exclusive; ``--build`` is the fall-through):

    --init      probe the source frame size, write/revalidate ``params.json``
    --build     cache every session of THIS array element (resumable)
    --status    counts, skip-reason histogram, total bytes
    --index     rebuild ``manifest.json`` by scanning the shards
    --check N   re-build N random sessions from raw and compare byte-for-byte

Corpus identity discipline (identical to the TIR builder): ``params.json`` is
keyed on the UNFILTERED discovery, so ``--subjects`` / ``--limit`` restrict the
WORK of this element without creating a second cache identity -- a partial cache
is the same cache, partially built, and ``--status`` reports the rest as
``missing``.

Sharding across the corpus uses the Slurm array: ``SLURM_ARRAY_TASK_ID`` /
``SLURM_ARRAY_TASK_COUNT`` (overridable with ``--array_index`` /
``--array_count``), round-robin ``i % cnt == idx`` so the elements stay balanced
even though task cost varies.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import rgb2d_cache as c                                   # noqa: E402
from data import rgb_features as rf                                 # noqa: E402
from data import rgb_roi_dataset as rrd                             # noqa: E402

OUT_SUBDIR = 'rgb2d_roi_cache'


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def default_raw_root() -> str:
    root = os.environ.get('RAW_DATA_PATH')
    if root:
        return root
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    return os.path.join(repo, 'data', 'raw', 'BP4D')


def default_out_root() -> str:
    base = os.environ.get('WORK_SCRATCH') or os.environ.get('OUTPUT_DIR')
    if not base:
        repo = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        base = os.path.join(repo, 'output')
    return os.path.join(base, OUT_SUBDIR)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Build the rgb2d (2D-texture face-ROI) pixel cache.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--init', action='store_true',
                      help='probe the frame size and write params.json')
    mode.add_argument('--build', action='store_true', help='cache this element')
    mode.add_argument('--status', action='store_true', help='report progress')
    mode.add_argument('--index', action='store_true', help='rebuild manifest.json')
    mode.add_argument('--check', nargs='?', type=int, const=20, default=None,
                      metavar='N',
                      help='re-build N sessions from raw and compare bytes')

    p.add_argument('--raw_root', default=default_raw_root())
    p.add_argument('--out_root', default=default_out_root())
    p.add_argument('--subjects', default='',
                   help='CSV of subject ids; restricts the WORK, not the key')
    p.add_argument('--limit', type=int, default=0,
                   help='first N discovered tasks (smoke runs)')
    p.add_argument('--roi_padding', type=float, default=0.10)
    p.add_argument('--roi_landmarks', default='face',
                   help="'face' (1..49), a 2D preset name, or a CSV of labels")
    p.add_argument('--crop_rule', default='square',
                   choices=['square', 'box'])
    p.add_argument('--clip_seconds', type=float, default=8.0,
                   help='window length the hull unions over (a KEY field)')
    p.add_argument('--fps', type=float, default=25.0)
    p.add_argument('--source_policy', default=c.SOURCE_POLICY,
                   choices=[c.SOURCE_POLICY])
    p.add_argument('--array_index', type=int, default=-1)
    p.add_argument('--array_count', type=int, default=0)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--max_task_gb', type=float, default=12.0,
                   help='refuse-with-reason above this artefact size (GB)')
    p.add_argument('--seed', type=int, default=0, help='--check only')
    p.add_argument('--json', default='', help='--check/--status dump path')
    return p


def array_position(args) -> Tuple[int, int]:
    idx = args.array_index
    cnt = args.array_count
    if idx < 0:
        idx = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0') or 0)
    if cnt <= 0:
        cnt = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', '1') or 1)
    return int(idx), max(1, int(cnt))


def my_sessions(sessions: Sequence[dict], idx: int, cnt: int) -> List[dict]:
    return [s for i, s in enumerate(sessions) if i % cnt == idx]


def resolve_landmarks(spec: str) -> Tuple[int, ...]:
    text = str(spec or '').strip()
    if text in ('', 'face') or text.lower() == 'face':
        return tuple(range(1, rf.NUM_LANDMARKS_2D + 1))
    return rf.resolve_roi_landmarks_2d(text)


# --------------------------------------------------------------------------- #
# params
# --------------------------------------------------------------------------- #
def make_params(args, corpus: Sequence[dict]) -> Dict:
    """Probe the source geometry, then build the frozen parameter set."""
    probe: List[Tuple[int, int, int]] = []
    for spec in corpus[:3]:
        try:
            probe.append(c.probe_frame_size(spec))
        except Exception as exc:
            print(f'[rgb2d] probe skipped {spec["session"]}: {exc}')
    if not probe:
        raise SystemExit(
            f'rgb2d: cannot probe a frame under {args.raw_root}; nothing to build.')
    shapes = {p for p in probe}
    if len(shapes) > 1:
        raise SystemExit(
            f'rgb2d: the corpus is NOT uniform in frame size: {sorted(shapes)}. '
            f'The hull design assumes one source geometry; refusing to build.')
    fh, fw, _ch = probe[0]
    print(f'[rgb2d] source frame {fw}x{fh} (h x w = {fh} x {fw}), '
          f'probed {len(probe)} session(s)')
    return c.build_params(args.raw_root, corpus, resolve_landmarks(
        args.roi_landmarks), args.roi_padding, fw, fh,
        crop_rule=args.crop_rule, clip_seconds=args.clip_seconds,
        fps=args.fps, source_policy=args.source_policy)


def _open_root(args) -> Tuple[Path, Dict, List[dict]]:
    """Resolve the cache dir from ``params.json`` and the WORK session list."""
    corpus = c.discover_sessions(args.raw_root)
    want = c.loader_params(args.raw_root, corpus,
                           resolve_landmarks(args.roi_landmarks),
                           args.roi_padding, crop_rule=args.crop_rule,
                           clip_seconds=args.clip_seconds, fps=args.fps,
                           source_policy=args.source_policy)
    root = c.find_cache(args.out_root, want)
    params = c.read_params(root)
    subjects = [s.strip() for s in str(args.subjects or '').split(',')
                if s.strip()]
    work = (c.discover_sessions(args.raw_root, subjects=subjects)
            if subjects else corpus)
    if args.limit:
        work = work[:int(args.limit)]
    return root, params, work


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def mode_init(args) -> int:
    corpus = c.discover_sessions(args.raw_root)
    if not corpus:
        raise SystemExit(f'rgb2d: no 2D+3D sessions under {args.raw_root}')
    # Report -- never record -- subjects whose archive is not readable yet. A
    # 0-byte or half-copied ``<S>.zip`` is a delivery state, not a corpus fact:
    # measured 2026-11-10, F069.zip went 0 B -> 41.78 GB in ~40 min. A skip
    # marker written here would outlive the copy and keep excluding a subject
    # whose pixels are on disk, so the exclusion stays run-local.
    for subject, why in rrd.unreadable_archives(args.raw_root):
        print(f'[rgb2d] {subject}.zip {why} -- its sessions are missing from '
              f'this build; rerun once the copy is complete')
    params = make_params(args, corpus)
    key = c.cache_key(params)
    root = c.cache_root(args.out_root, key)
    if (root / c.PARAMS_NAME).is_file():
        # Idempotent across array elements: revalidate instead of rewriting.
        c.check_params_compatible(root, params)
        print(f'[rgb2d] params OK (existing): {root}')
    else:
        c.write_params(root, params)
        print(f'[rgb2d] wrote {root / c.PARAMS_NAME}')
    print(f'[rgb2d] key = {key}')
    return 0


def mode_build(args) -> int:
    root, params, work = _open_root(args)
    idx, cnt = array_position(args)
    mine = my_sessions(work, idx, cnt)
    pad = float(params['roi_padding'])
    labels = tuple(int(v) for v in params['roi_landmarks'])
    max_bytes = int(float(args.max_task_gb) * 1e9) if args.max_task_gb else 0
    stage = root / c.SUBDIR_TMP / f'stage.{os.getpid()}'
    stage.mkdir(parents=True, exist_ok=True)

    print(f'[rgb2d] element {idx}/{cnt}: {len(mine)} of {len(work)} session(s)'
          f' -> {root}')
    n_ok = n_skip = n_have = n_err = 0
    t0 = time.time()
    for k, spec in enumerate(mine, 1):
        session = str(spec['session'])
        if c.shard_path(root, session).is_file() and not args.overwrite:
            n_have += 1
            continue
        try:
            meta, reason = c.build_task(
                spec, labels, pad, stage, crop_rule=params['crop_rule'],
                clip_seconds=float(params['clip_seconds']),
                fps=float(params['fps']), max_task_bytes=max_bytes)
            if meta is None:
                c.write_skip(root, session, reason or 'unknown')
                n_skip += 1
                print(f'  [{k}/{len(mine)}] SKIP {session}: {reason}')
            else:
                c.commit_shard(root, session, meta, stage)
                n_ok += 1
                print(f'  [{k}/{len(mine)}] ok   {session} '
                      f'{meta["hull_hw"][0]}x{meta["hull_hw"][1]} '
                      f'x{meta["n_frames"]}f = '
                      f'{meta["frames"]["nbytes"] / 1e6:.1f} MB')
        except (IOError, OSError, ValueError) as exc:
            # One unreadable session must not kill a 44-session element.
            c.write_skip(root, session, f'build_error: {exc}')
            n_err += 1
            print(f'  [{k}/{len(mine)}] ERR  {session}: {exc}')
        if k % 25 == 0:
            dt = time.time() - t0
            print(f'  ... {k}/{len(mine)}  {dt / k:.2f} s/session')
    # A failed build still leaves the stage dir behind for inspection.
    try:
        for p in stage.iterdir():
            p.unlink()
        stage.rmdir()
    except OSError:
        pass
    print(f'[rgb2d] element {idx}/{cnt} done: {n_ok} built, {n_have} already '
          f'there, {n_skip} skipped, {n_err} errors, '
          f'{time.time() - t0:.1f} s')
    return 0


def mode_status(args) -> int:
    root, params, work = _open_root(args)
    rep = c.scan(root, work)
    print(f'[rgb2d] {root}')
    print(f'  params: {json.dumps({k: params[k] for k in ("roi_landmarks_name", "roi_padding", "crop_rule", "clip_seconds", "fps", "store_resolution", "src_w", "src_h")}, sort_keys=True)}')
    print(f'  cached {rep["cached"]}  skipped {rep["skipped"]}  '
          f'missing {rep["missing"]}  incomplete {rep["incomplete"]}  '
          f'truncated {rep["truncated"]}')
    print(f'  bytes  {rep["bytes"] / 1e9:.3f} GB')
    hist: Dict[str, int] = {}
    sk_dir = Path(root) / c.SUBDIR_SKIPPED
    if sk_dir.is_dir():
        for p in sorted(sk_dir.glob('*.json')):
            try:
                with open(p) as fh:
                    reason = str(json.load(fh).get('reason') or 'unknown')
            except Exception:
                reason = 'unreadable'
            hist[reason.split(':')[0]] = hist.get(reason.split(':')[0], 0) + 1
    if hist:
        print('  skip reasons:')
        for k in sorted(hist, key=lambda x: -hist[x]):
            print(f'    {hist[k]:5d}  {k}')
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'root': str(root), 'scan': {k: v for k, v in rep.items()
                                                  if not k.endswith('_names')},
                       'skip_reasons': hist}, fh, indent=2)
        print(f'  wrote {args.json}')
    return 0


def mode_index(args) -> int:
    root, params, work = _open_root(args)
    man = c.build_manifest(root, work, params)
    print(f'[rgb2d] manifest: {man["n_ok"]} ok, {man["n_skipped"]} skipped, '
          f'{man["n_missing"]} missing, {man["total_bytes"] / 1e9:.3f} GB')
    return 0


def _arrays_equal(path_a: Path, path_b: Path, shape, dtype, chunk: int) -> bool:
    """Chunked comparison so the check never holds two full arrays."""
    a = np.memmap(str(path_a), dtype=dtype, mode='r', shape=shape)
    b = np.memmap(str(path_b), dtype=dtype, mode='r', shape=shape)
    try:
        for start in range(0, shape[0], chunk):
            stop = min(start + chunk, shape[0])
            if not np.array_equal(a[start:stop], b[start:stop]):
                return False
        return True
    finally:
        del a
        del b


def mode_check(args) -> int:
    root, params, work = _open_root(args)
    have = [s for s in work if c.shard_path(root, str(s['session'])).is_file()]
    if not have:
        print('[rgb2d] --check: nothing cached yet')
        return 1
    rng = random.Random(int(args.seed))
    picks = rng.sample(have, min(int(args.check or 20), len(have)))
    pad = float(params['roi_padding'])
    labels = tuple(int(v) for v in params['roi_landmarks'])
    max_bytes = int(float(args.max_task_gb) * 1e9) if args.max_task_gb else 0
    stage = root / c.SUBDIR_TMP / f'check.{os.getpid()}'
    stage.mkdir(parents=True, exist_ok=True)
    fails: List[str] = []
    for spec in picks:
        session = str(spec['session'])
        meta, reason = c.build_task(
            spec, labels, pad, stage, crop_rule=params['crop_rule'],
            clip_seconds=float(params['clip_seconds']),
            fps=float(params['fps']), max_task_bytes=max_bytes)
        if meta is None:
            fails.append(f'{session}: rebuild failed ({reason})')
            continue
        try:
            with open(c.shard_path(root, session)) as fh:
                on_disk = json.load(fh)
            for key in ('box', 'pad', 'n_frames', 'src_hw', 'hull_hw'):
                got, want = on_disk[key], meta[key]
                if isinstance(want, list):
                    same = [int(v) for v in got] == [int(v) for v in want]
                else:
                    same = int(got) == int(want)
                if not same:
                    fails.append(f'{session}: {key} {got} != {want}')
            for key, suffix in (('frames', c.FRAMES_SUFFIX),
                                ('landmarks', c.LANDMARKS_SUFFIX)):
                shape = tuple(int(v) for v in on_disk[key]['shape'])
                dtype = np.dtype(on_disk[key]['dtype'])
                if not _arrays_equal(
                        Path(root) / c.SUBDIR_TASKS / f'{session}{suffix}',
                        stage / f'{session}{suffix}', shape, dtype,
                        c.CHUNK_FRAMES):
                    fails.append(f'{session}: {key} bytes differ')
        finally:
            for suffix in (c.FRAMES_SUFFIX, c.LANDMARKS_SUFFIX):
                try:
                    (stage / f'{session}{suffix}').unlink()
                except OSError:
                    pass
    try:
        stage.rmdir()
    except OSError:
        pass
    verdict = 'PASS' if not fails else 'FAIL'
    print(f'VERDICT: {verdict}  ({len(picks) - len({f.split(":")[0] for f in fails})}'
          f'/{len(picks)} sessions identical, {len(fails)} failure(s))')
    for f in fails:
        print(f'  FAIL {f}')
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'verdict': verdict, 'checked': len(picks),
                       'failures': fails}, fh, indent=2)
    return 0 if not fails else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check is not None:
        return mode_check(args)
    if args.status:
        return mode_status(args)
    if args.index:
        return mode_index(args)
    if args.init:
        return mode_init(args)
    return mode_build(args)


if __name__ == '__main__':
    raise SystemExit(main())
