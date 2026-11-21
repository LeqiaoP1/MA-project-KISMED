"""Build / inspect the offline blood-pressure session artefact (ADR-004).

Contract: ``code/notes/design/ADR-004-bp-session-cache-artefact.md``. Build order:
``code/notes/plan/features/ADR-004-bp-session-cache-plan.md``.

The artefact is a pure derived artefact: one whole-session signal grid, a
per-sample validity mask, a per-sample cause code and a segment table, all on the
unchanged 1000 Hz source grid, so the training loader never re-reads
``BP_mmHg.txt`` and never interpolates a window. It encodes no downstream
decision (no clip length, hop, split or normalisation), which is what lets ONE
artefact serve every consumer.

Modes (exactly one):

``--init``            Write ``params.json``. Idempotent, so an array job can call
                      it from every element safely.
``--build``           Cache the sessions assigned to this array element. DEFAULT.
``--status``          Filesystem truth: cached / skipped / missing, bytes, reasons.
``--index``           Rebuild ``manifest.json`` from the shards (derived index).
``--check N``         Re-preprocess N sessions from raw and require the shard to
                      be BIT-IDENTICAL on signal, mask, cause and segment table.
``--check-imports``   Report whether the build's import graph loaded ``torch``
                      (it must not) and exit.

Usage (from ``code/``)::

    python runners/run_build_bp_artefact.py --init --build \\
        --out_root $WORK_SCRATCH/bp_artefact --raw_root $RAW_DATA_PATH
    python runners/run_build_bp_artefact.py --status --out_root $WORK_SCRATCH/bp_artefact
    python runners/run_build_bp_artefact.py --check 20 --out_root $WORK_SCRATCH/bp_artefact

**Torch-free by construction.** This runner puts ``code/data`` on ``sys.path``
and imports the modules as top-level modules instead of importing the ``data``
package, whose ``__init__`` pulls in torch through the dataset classes. That is
how Criterion 12's "CPU only, no accelerator library" stays structural; the
modules' own imports are guarded for both contexts.

CPU only, single process, no GPU / no YAML / no torch import.
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(CODE_DIR, 'data')
if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)

import bp_preprocess as bpp                                      # noqa: E402
import bp_artefact as bpa                                        # noqa: E402


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
    """Round-robin partition: balanced even when session cost varies."""
    return [s for i, s in enumerate(sessions) if i % cnt == idx]


def make_params(args, corpus) -> Dict:
    """The FROZEN parameters. ``corpus`` is the UNFILTERED discovery.

    The key is a property of the CORPUS, not of the subset this invocation
    happens to build: ``--subjects`` / ``--limit`` restrict the WORK, they must
    not create a second artefact identity.
    """
    return bpa.build_params(
        corpus,
        fs_in=args.fs_in, fs_out=args.fs_out,
        gate1_min_mmhg=args.gate1_min_mmhg,
        gate1_max_mmhg=args.gate1_max_mmhg,
        flatline_max_ms=args.flatline_max_ms,
        rail_dwell_min_ms=args.rail_dwell_min_ms,
        min_segment_s=args.min_segment_s)


def discovered(args) -> List[dict]:
    subjects = [s for s in str(args.subjects).replace(',', ' ').split() if s]
    return bpa.discover_sessions(args.raw_root, subjects or None)


def scoped(args, corpus) -> List[dict]:
    """The sessions THIS invocation works on (``--subjects`` / ``--limit``)."""
    sessions = corpus
    if args.limit:
        sessions = sessions[: int(args.limit)]
    return sessions


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def mode_init(args, corpus) -> int:
    params = make_params(args, corpus)
    key = bpa.cache_key(params)
    root = bpa.cache_root(args.out_root, key)
    if (root / bpa.PARAMS_NAME).is_file():
        bpa.check_params_compatible(root, params)      # raises on any mismatch
        print(f'[bp_artefact] params already present and identical: {root}')
    else:
        bpa.write_params(root, params)
        print(f'[bp_artefact] wrote {root / bpa.PARAMS_NAME}')
    print(f'[bp_artefact] key      : {key}')
    print(f'[bp_artefact] corpus   : {params["corpus"]} '
          f'({len(corpus)} session directories discovered)')
    print(f'[bp_artefact] grid     : src {params["fs_in"]} Hz -> out '
          f'{params["fs_out"]} Hz (no resampling; grid index == source index)')
    print(f'[bp_artefact] gates    : range {params["gate1_min_mmhg"]:g} .. '
          f'{params["gate1_max_mmhg"]:g} mmHg; flatline run > '
          f'{params["flatline_max_ms"]} ms; rail dwell >= '
          f'{params["rail_dwell_min_ms"]} ms; segments >= '
          f'{params["min_segment_s"]:g} s')
    print(f'\n[bp_artefact] next: --build --out_root {args.out_root}')
    return 0


def mode_build(args, corpus) -> int:
    idx, cnt = array_position(args)
    params = make_params(args, corpus)
    root = bpa.cache_root(args.out_root, bpa.cache_key(params))
    if not (root / bpa.PARAMS_NAME).is_file():
        raise SystemExit(f'[bp_artefact] {root}/params.json not found -- run '
                         f'--init first (same knobs).')
    bpa.check_params_compatible(root, params)          # FAIL LOUDLY

    todo = my_sessions(scoped(args, corpus), idx, cnt)
    print(f'[bp_artefact] key     : {bpa.cache_key(params)}')
    print(f'[bp_artefact] element : {idx}/{cnt} -> {len(todo)} session(s)')
    t0 = time.time()
    n_ok = n_skip = 0
    reasons: Dict[str, int] = {}
    fracs: List[float] = []
    for j, spec in enumerate(todo, 1):
        session = str(spec['session'])
        if bpa.shard_path(root, session).is_file() and not args.overwrite:
            n_ok += 1
            continue                                   # resumable
        t1 = time.time()
        payload, reason = bpa.build_task(spec, params)
        if payload is None:
            bpa.write_skip(root, session, reason or 'unknown')
            n_skip += 1
            key_reason = (reason or 'unknown').split(':')[0]
            reasons[key_reason] = reasons.get(key_reason, 0) + 1
            print(f'  [{j}/{len(todo)}] {session:<14} SKIP  {reason}', flush=True)
            continue
        path = bpa.write_shard(root, session, payload, compress=args.compress)
        n_ok += 1
        fracs.append(float(payload['valid_frac']))
        mb = path.stat().st_size / 1e6
        print(f'  [{j}/{len(todo)}] {session:<14} ok  '
              f'{int(payload["n_src"]):7d} src  valid '
              f'{100.0 * float(payload["valid_frac"]):5.1f}%  segs '
              f'{int(payload["n_segments"]):3d}  {mb:5.2f} MB  '
              f'{time.time() - t1:5.2f}s', flush=True)
    dt = time.time() - t0
    print(f'[bp_artefact] element {idx} done: {n_ok} cached, {n_skip} skipped '
          f'in {dt / 60:.1f} min')
    if fracs:
        print(f'[bp_artefact] valid_frac this element: '
              f'median {sorted(fracs)[len(fracs) // 2]:.3f}')
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f'    skip {reason:24s} {n}')
    return 0


def mode_status(args, corpus) -> int:
    params = make_params(args, corpus)
    root = bpa.cache_root(args.out_root, bpa.cache_key(params))
    if not (root / bpa.PARAMS_NAME).is_file():
        raise SystemExit(f'[bp_artefact] nothing built at {root}')
    info = bpa.scan(root, scoped(args, corpus))
    print(f'[bp_artefact] root     : {root}')
    print(f'[bp_artefact] cached   : {info["cached"]}')
    print(f'[bp_artefact] skipped  : {info["skipped"]}')
    print(f'[bp_artefact] missing  : {info["missing"]}')
    print(f'[bp_artefact] size     : {info["bytes"] / 1e6:.1f} MB')
    if info['missing']:
        print(f'[bp_artefact] e.g. missing : '
              f'{", ".join(info["missing_names"][:8])}')
    if info['skipped_names']:
        hist: Dict[str, int] = {}
        for name in info['skipped_names']:
            with open(root / bpa.SUBDIR_SKIPS / f'{name}.json') as fh:
                reason = json.load(fh)['reason'].split(':')[0]
            hist[reason] = hist.get(reason, 0) + 1
        for reason, n in sorted(hist.items(), key=lambda kv: -kv[1]):
            print(f'    skip {reason:24s} {n}')
    return 0


def mode_index(args, corpus) -> int:
    params = make_params(args, corpus)
    root = bpa.cache_root(args.out_root, bpa.cache_key(params))
    m = bpa.build_manifest(root, scoped(args, corpus), params)
    print(f'[bp_artefact] manifest -> {root / bpa.MANIFEST_NAME}')
    print(f'[bp_artefact] ok {m["n_ok"]}  skipped {m["n_skipped"]}  '
          f'missing {m["n_missing"]}  {m["total_bytes"] / 1e6:.1f} MB')
    vf = m['valid_frac']
    print(f'[bp_artefact] valid_frac mean {vf["mean"]:.3f}  median '
          f'{vf["median"]:.3f}  min {vf["min"]:.3f}  max {vf["max"]:.3f}')
    print(f'[bp_artefact] zero-segment shards: {m["n_zero_segment"]}')
    seg = m.get('segments') or {}
    if seg.get('n_total'):
        print(f'[bp_artefact] segments: {seg["n_total"]} total, median '
              f'{seg["median"]:.1f} s, p10 {seg["p10"]:.1f} s, p90 '
              f'{seg["p90"]:.1f} s, max {seg["max"]:.1f} s; '
              f'<8 s {100.0 * seg["frac_below_8s"]:.1f}%')
    if m.get('cause_totals'):
        total = sum(m['cause_totals'].values()) or 1
        print('[bp_artefact] sample share by cause: ' + ', '.join(
            f'{k} {100.0 * v / total:.1f}%' for k, v in m['cause_totals'].items()))
    if m['n_missing']:
        print('[bp_artefact] NOTE: missing shards are either not yet built or '
              'have no skip file -- see --status')
    return 0


def mode_check(args, corpus) -> int:
    """Re-preprocess sessions from raw and compare with the stored shard.

    The comparison is BIT-IDENTICAL on the stored dtype, not approximate: a gate,
    threshold or segment-rule change must show up as a parity failure even when
    it is numerically small.
    """
    import numpy as np

    params = make_params(args, corpus)
    root = bpa.cache_root(args.out_root, bpa.cache_key(params))
    bpa.check_params_compatible(root, params)

    rng = np.random.default_rng(args.seed)
    have = [s for s in corpus
            if bpa.shard_path(root, str(s['session'])).is_file()]
    if not have:
        raise SystemExit(f'[bp_artefact] no shards under {root}')
    n = min(int(args.check), len(have))
    pick = [have[int(i)] for i in rng.permutation(len(have))[:n]]

    fails: List[str] = []
    for k, spec in enumerate(pick, 1):
        session = str(spec['session'])
        shard = bpa.read_shard(bpa.shard_path(root, session))
        fresh, reason = bpa.build_task(spec, params)
        if fresh is None:
            fails.append(f'{session}: rebuild failed ({reason})')
            continue
        same, diffs = bpa.shards_equal(shard, fresh)
        if not same:
            fails.append(f'{session}: shard != fresh preprocess ({", ".join(diffs)})')
        else:
            print(f'  [{k}/{n}] {session:<14} identical '
                  f'({shard["signal"].size} samples, '
                  f'{shard["n_segments"]} segments, valid '
                  f'{100.0 * shard["valid_frac"]:5.1f}%)', flush=True)
    print(f'\n[bp_artefact] checked {n} session(s)')
    print(f'[bp_artefact] VERDICT: {"PASS" if not fails else "FAIL"}')
    for f in fails[:10]:
        print(f'    ! {f}')
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'checked': n, 'fails': fails}, fh, indent=2)
        print(f'[bp_artefact] JSON -> {args.json}')
    return 0 if not fails else 1


def mode_check_imports(args, corpus) -> int:
    """Report whether the build's import graph pulled an accelerator library."""
    loaded = 'torch' in sys.modules
    print(f'[bp_artefact] torch loaded: {loaded}')
    print('[bp_artefact] VERDICT: '
          f'{"FAIL -- the build must not import an accelerator library" if loaded else "PASS"}')
    return 1 if loaded else 0


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
                   help='cache the sessions of this array element (default)')
    g.add_argument('--status', action='store_true', help='report progress')
    g.add_argument('--index', action='store_true', help='rebuild manifest.json')
    g.add_argument('--check', nargs='?', type=int, const=20, default=0,
                   metavar='N',
                   help='re-preprocess N sessions (default 20) and compare '
                        'bit for bit')
    g.add_argument('--check-imports', action='store_true',
                   help='report whether the import graph loaded torch, then exit')

    p.add_argument('--raw_root', default=env_or('RAW_DATA_PATH', ''))
    p.add_argument('--out_root',
                   default=os.path.join(env_or('WORK_SCRATCH',
                                               env_or('OUTPUT_DIR', '.')),
                                        'bp_artefact'))
    p.add_argument('--subjects', default='', help='comma list ("" = all)')
    p.add_argument('--limit', type=int, default=0,
                   help='first N discovered sessions (0 = all; smoke tests)')
    p.add_argument('--overwrite', action='store_true',
                   help='rebuild a shard that already exists')
    p.add_argument('--compress', action='store_true',
                   help='compress the shard container (NOT the frozen default)')
    p.add_argument('--seed', type=int, default=0, help='--check sampling seed')
    p.add_argument('--json', default='', help='--check report path')
    p.add_argument('--array_index', type=int, default=-1)
    p.add_argument('--array_count', type=int, default=0)

    # GATE / GRID CONTRACT -- every field below is part of the artefact key
    p.add_argument('--fs_in', type=float, default=bpp.FS_IN,
                   help='nominal source rate of BP_mmHg.txt (1000 Hz)')
    p.add_argument('--fs_out', type=float, default=bpp.FS_OUT,
                   help='output grid rate; MUST equal --fs_in (no resampling)')
    p.add_argument('--gate1_min_mmhg', type=float, default=bpp.GATE1_MIN_MMHG)
    p.add_argument('--gate1_max_mmhg', type=float, default=bpp.GATE1_MAX_MMHG)
    p.add_argument('--flatline_max_ms', type=float, default=bpp.FLATLINE_MAX_MS)
    p.add_argument('--rail_dwell_min_ms', type=float,
                   default=bpp.RAIL_DWELL_MIN_MS)
    p.add_argument('--min_segment_s', type=float, default=bpp.MIN_SEGMENT_S)

    args = p.parse_args(argv)

    if args.check_imports:
        return mode_check_imports(args, [])

    if not args.raw_root:
        raise SystemExit('--raw_root (or $RAW_DATA_PATH) is required')
    corpus = discovered(args)

    if args.status:
        return mode_status(args, corpus)
    if args.index:
        return mode_index(args, corpus)
    if args.check:
        return mode_check(args, corpus)
    if args.init:
        rc = mode_init(args, corpus)
        if not args.build:
            return rc
    return mode_build(args, corpus)


if __name__ == '__main__':
    sys.exit(main())
