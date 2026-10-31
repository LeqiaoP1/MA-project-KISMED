"""Build / inspect the offline respiration-grid cache.

Contract: ``code/notes/design/RESP_session_preprocess.md``. Build order:
``code/notes/plan/RESP_session_preprocess_implementation_plan.md``.

The cache is a pure derived artefact: it stores one whole-session signal grid
plus a per-sample validity mask, both at the output rate, so the training loader
never re-reads ``Resp_Volts.txt`` and never re-interpolates a window. It encodes
no downstream decision (no clip length, hop, split or normalisation), which is
what lets ONE cache serve Stage 2 and Stage 3.

Modes (exactly one):

``--init``    Write ``params.json``. Idempotent, so an array job can call it from
              every element safely.
``--build``   Cache the subject-tasks assigned to this array element. DEFAULT.
``--status``  Filesystem truth: cached / skipped / missing, bytes, skip reasons.
``--index``   Rebuild ``manifest.json`` from the shards (the derived index).
``--check``   Re-preprocess N sessions from raw and assert the shard is
              BIT-IDENTICAL (the parity gate that caught the ROI cache's bugs).

Usage (from ``code/``)::

    python runners/run_build_resp_cache.py --init --build \\
        --out_root $WORK_SCRATCH/resp_cache --raw_root $RAW_DATA_PATH
    python runners/run_build_resp_cache.py --status --out_root $WORK_SCRATCH/resp_cache
    python runners/run_build_resp_cache.py --check 20 --out_root $WORK_SCRATCH/resp_cache

CPU only, single process, no GPU / no YAML / no torch import.
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import resp_cache as respc                           # noqa: E402
from data import resp_preprocess as rpp                        # noqa: E402
from data import tir_resp_dataset as trd                        # noqa: E402


def env_or(name: str, default: str = '') -> str:
    return os.environ.get(name, default)


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


def make_params(args, corpus_sessions) -> Dict:
    """The FROZEN parameters. ``corpus_sessions`` is the UNFILTERED discovery.

    The key is a property of the CORPUS, not of the subset this invocation
    happens to build: ``--subjects``/``--limit`` restrict the WORK, they must
    not create a second cache identity.
    """
    return respc.build_params(
        args.raw_root, corpus_sessions,
        fs_in=args.fs_in, fs_out=args.fs_out,
        sat_v=args.sat_v, sat_min_s=args.sat_min_s,
        flat_window_s=args.flat_window_s, flat_ptp_v=args.flat_ptp_v,
        min_run_s=args.min_run_s, half_taps=args.half_taps,
        dtype=respc.STORE_DTYPE)


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def mode_init(args, corpus, sessions) -> int:
    params = make_params(args, corpus)
    key = respc.cache_key(params)
    root = respc.cache_root(args.out_root, key)
    if (root / respc.PARAMS_NAME).is_file():
        respc.check_params_compatible(root, params)   # raises on any mismatch
        print(f'[resp_cache] params already present and identical: {root}')
    else:
        respc.write_params(root, params)
        print(f'[resp_cache] wrote {root / respc.PARAMS_NAME}')
    print(f'[resp_cache] key       : {key}')
    print(f'[resp_cache] corpus    : {params["corpus"]} '
          f'({params["n_sessions_discovered"]} subject-tasks discovered)')
    print(f'[resp_cache] grid      : src {params["fs_in"]} Hz -> out '
          f'{params["fs_out"]} Hz, {params["resample"]} '
          f'(half_taps {params["half_taps"]})')
    print(f'[resp_cache] validity  : sat |V|>{params["sat_v"]:g} for >= '
          f'{params["sat_min_s"]:g} s; flat ptp<{params["flat_ptp_v"]:g} V '
          f'over {params["flat_window_s"]:g} s; runs < '
          f'{params["min_run_s"]:g} s are holes')
    print(f'\n[resp_cache] next: --build --out_root {args.out_root}')
    return 0


def mode_build(args, corpus, sessions) -> int:
    idx, cnt = array_position(args)
    params = make_params(args, corpus)
    root = respc.cache_root(args.out_root, respc.cache_key(params))
    if not (root / respc.PARAMS_NAME).is_file():
        raise SystemExit(f'[resp_cache] {root}/params.json not found -- run '
                         f'--init first (same validity knobs).')
    respc.check_params_compatible(root, params)       # FAIL LOUDLY

    todo = my_sessions(sessions, idx, cnt)
    print(f'[resp_cache] key       : {respc.cache_key(params)}')
    print(f'[resp_cache] element   : {idx}/{cnt} -> {len(todo)} subject-task(s)')
    t0 = time.time()
    n_ok = n_skip = 0
    reasons: Dict[str, int] = {}
    valid_fracs: List[float] = []
    for j, spec in enumerate(todo, 1):
        session = str(spec['session'])
        if respc.shard_path(root, session).is_file() and not args.overwrite:
            n_ok += 1
            continue                                  # resumable
        t1 = time.time()
        payload, reason = respc.build_task(spec, params)
        if payload is None:
            respc.write_skip(root, session, reason or 'unknown')
            n_skip += 1
            reasons[reason or 'unknown'] = reasons.get(reason or 'unknown', 0) + 1
            print(f'  [{j}/{len(todo)}] {session:<12} SKIP  {reason}', flush=True)
            continue
        path = respc.write_shard(root, session, payload, compress=args.compress)
        n_ok += 1
        valid_fracs.append(float(payload['valid_frac']))
        mb = path.stat().st_size / 1e6
        print(f'  [{j}/{len(todo)}] {session:<12} ok  '
              f'{payload["n_src"]:6d} src -> {payload["signal"].size:5d} out  '
              f'valid {100.0 * payload["valid_frac"]:5.1f}%  '
              f'runs {payload["n_runs"]:2d}  {mb:5.2f} MB  '
              f'{time.time() - t1:5.2f}s', flush=True)
    dt = time.time() - t0
    print(f'[resp_cache] element {idx} done: {n_ok} cached, {n_skip} skipped '
          f'in {dt / 60:.1f} min')
    if valid_fracs:
        print(f'[resp_cache] valid_frac this element: median {sorted(valid_fracs)[len(valid_fracs) // 2]:.3f}')
    for r, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f'    skip {r:28s} {n}')
    return 0


def mode_status(args, corpus, sessions) -> int:
    params = make_params(args, corpus)
    root = respc.cache_root(args.out_root, respc.cache_key(params))
    if not (root / respc.PARAMS_NAME).is_file():
        raise SystemExit(f'[resp_cache] nothing built at {root}')
    info = respc.scan(root, sessions)
    print(f'[resp_cache] root     : {root}')
    print(f'[resp_cache] cached   : {info["cached"]}')
    print(f'[resp_cache] skipped  : {info["skipped"]}')
    print(f'[resp_cache] missing  : {info["missing"]}')
    print(f'[resp_cache] size     : {info["bytes"] / 1e6:.1f} MB')
    if info['missing']:
        print(f'[resp_cache] e.g. missing  : '
              f'{", ".join(info["missing_names"][:8])}')
    if info['skipped_names']:
        hist: Dict[str, int] = {}
        for name in info['skipped_names']:
            with open(root / respc.SUBDIR_SKIPPED / f'{name}.json') as fh:
                r = json.load(fh)['reason'].split(':')[0]
            hist[r] = hist.get(r, 0) + 1
        for r, n in sorted(hist.items(), key=lambda kv: -kv[1]):
            print(f'    skip {r:28s} {n}')
    return 0


def mode_index(args, corpus, sessions) -> int:
    params = make_params(args, corpus)
    root = respc.cache_root(args.out_root, respc.cache_key(params))
    m = respc.build_manifest(root, sessions, params)
    print(f'[resp_cache] manifest -> {root / respc.MANIFEST_NAME}')
    print(f'[resp_cache] ok {m["n_ok"]}  skipped {m["n_skipped"]}  '
          f'missing {m["n_missing"]}  {m["total_bytes"] / 1e6:.1f} MB')
    vf = m['valid_frac']
    print(f'[resp_cache] valid_frac mean {vf["mean"]:.3f}  median '
          f'{vf["median"]:.3f}  min {vf["min"]:.3f}  max {vf["max"]:.3f}')
    if m.get('cause_totals'):
        total = sum(m['cause_totals'].values()) or 1
        print('[resp_cache] invalid-sample share: ' + ', '.join(
            f'{k} {100.0 * v / total:.1f}%' for k, v in m['cause_totals'].items()))
    if m['n_missing']:
        print('[resp_cache] NOTE: missing shards are either not yet built or '
              'have no skip file -- see --status')
    return 0


def mode_check(args, corpus, sessions) -> int:
    """Re-preprocess sessions from raw and compare with the stored shard.

    The comparison is BIT-IDENTICAL on the stored dtype (``np.array_equal`` plus
    an explicit NaN-position check), not approximate: a resampler or mask change
    must show up as a parity failure even when it is numerically small.
    """
    params = make_params(args, corpus)
    root = respc.cache_root(args.out_root, respc.cache_key(params))
    respc.check_params_compatible(root, params)

    import numpy as np
    rng = np.random.default_rng(args.seed)
    have = [s for s in sessions
            if respc.shard_path(root, str(s['session'])).is_file()]
    if not have:
        raise SystemExit(f'[resp_cache] no shards under {root}')
    n = min(int(args.check), len(have))
    pick = [have[int(i)] for i in rng.permutation(len(have))[:n]]

    fails: List[str] = []
    for k, spec in enumerate(pick, 1):
        session = str(spec['session'])
        shard = respc.read_shard(respc.shard_path(root, session))
        fresh, reason = respc.build_task(spec, params)
        if fresh is None:
            fails.append(f'{session}: rebuild failed ({reason})')
            continue
        same_sig = np.array_equal(np.asarray(shard['signal']),
                                  np.asarray(fresh['signal']), equal_nan=True)
        same_mask = np.array_equal(np.asarray(shard['mask']),
                                   np.asarray(fresh['mask']))
        same_cause = np.array_equal(np.asarray(shard['cause']),
                                    np.asarray(fresh['cause']))
        if not (same_sig and same_mask and same_cause):
            fails.append(f'{session}: shard != fresh preprocess '
                         f'(signal {"ok" if same_sig else "DIFF"}, '
                         f'mask {"ok" if same_mask else "DIFF"}, '
                         f'cause {"ok" if same_cause else "DIFF"})')
        else:
            print(f'  [{k}/{n}] {session:<12} identical '
                  f'({shard["signal"].size} out, valid '
                  f'{100.0 * shard["valid_frac"]:5.1f}%)', flush=True)
    print(f'\n[resp_cache] checked {n} session(s)')
    print(f'[resp_cache] VERDICT: {"PASS" if not fails else "FAIL"}')
    for f in fails[:10]:
        print(f'    ! {f}')
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'checked': n, 'fails': fails}, fh, indent=2)
        print(f'[resp_cache] JSON -> {args.json}')
    return 0 if not fails else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group('modes (default: --build)')
    g.add_argument('--init', action='store_true',
                   help='write params.json (idempotent; combine with --build)')
    g.add_argument('--build', action='store_true',
                   help='cache this shard (default)')
    g.add_argument('--status', action='store_true', help='report progress')
    g.add_argument('--index', action='store_true', help='rebuild manifest.json')
    g.add_argument('--check', nargs='?', type=int, const=20, default=0,
                   metavar='N',
                   help='re-preprocess N sessions (default 20) and compare '
                        'bit for bit')

    p.add_argument('--raw_root', default=env_or('RAW_DATA_PATH', ''))
    p.add_argument('--out_root',
                   default=os.path.join(env_or('WORK_SCRATCH',
                                               env_or('OUTPUT_DIR', '.')),
                                        'resp_cache'))
    p.add_argument('--subjects', default='', help='comma list ("" = all)')
    p.add_argument('--limit', type=int, default=0,
                   help='first N discovered subject-tasks (0 = all; smoke tests)')

    # VALIDITY / GRID CONTRACT -- every field below is part of the cache key
    p.add_argument('--fs_in', type=float, default=rpp.FS_IN,
                   help='nominal source rate of Resp_Volts.txt (1000 Hz)')
    p.add_argument('--fs_out', type=float, default=rpp.FS_OUT,
                   help='output grid rate; MUST equal the run --fs (100 Hz)')
    p.add_argument('--sat_v', type=float, default=rpp.SAT_V,
                   help='rail threshold (volts, magnitude)')
    p.add_argument('--sat_min_s', type=float, default=rpp.SAT_MIN_S,
                   help='a saturation event shorter than this is KEPT')
    p.add_argument('--flat_window_s', type=float, default=rpp.FLAT_WINDOW_S,
                   help='flatline probe length (seconds)')
    p.add_argument('--flat_ptp_v', type=float, default=rpp.FLAT_PTP_V,
                   help='flatline peak-to-peak threshold (volts)')
    p.add_argument('--min_run_s', type=float, default=rpp.MIN_RUN_S,
                   help='valid runs shorter than this become holes')
    p.add_argument('--half_taps', type=int, default=rpp.RESAMPLE_HALF_TAPS,
                   help='source samples replicated on each run edge')

    # array partition
    p.add_argument('--array_index', type=int, default=-1,
                   help='default: $SLURM_ARRAY_TASK_ID')
    p.add_argument('--array_count', type=int, default=0,
                   help='default: $SLURM_ARRAY_TASK_COUNT')
    p.add_argument('--overwrite', action='store_true',
                   help='re-cache shards that already exist')

    # build options
    p.add_argument('--compress', action='store_true',
                   help='np.savez_compressed (not the default: the raw grid is '
                        'small and uncompressed reads faster)')

    # check options
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--json', default='')
    args = p.parse_args(argv)

    if not args.raw_root or not os.path.isdir(args.raw_root):
        print(f'[resp_cache] raw_root not usable: {args.raw_root!r}; pass '
              f'--raw_root or set $RAW_DATA_PATH', file=sys.stderr)
        return 2

    subjects = [s.strip() for s in args.subjects.split(',') if s.strip()] or None
    # UNFILTERED: the cache key is a property of the corpus (see make_params).
    corpus = trd.discover_sessions(args.raw_root)
    sessions = trd.discover_sessions(args.raw_root, subjects=subjects)
    if args.limit:
        sessions = sessions[:args.limit]
    print(f'[resp_cache] out_root : {args.out_root}')
    print(f'[resp_cache] raw_root : {args.raw_root}')
    print(f'[resp_cache] corpus   : {len(corpus)} subject-task(s) (key)')
    print(f'[resp_cache] scope    : {len(sessions)} subject-task(s) this run')

    if args.init:
        rc = mode_init(args, corpus, sessions)
        if rc:
            return rc
        if not args.build:                 # --init alone: stop after params
            return 0
    if args.status:
        return mode_status(args, corpus, sessions)
    if args.index:
        return mode_index(args, corpus, sessions)
    if args.check:
        return mode_check(args, corpus, sessions)
    return mode_build(args, corpus, sessions)


if __name__ == '__main__':
    raise SystemExit(main())
