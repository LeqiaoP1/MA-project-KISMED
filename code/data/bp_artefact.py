"""Identity, storage and lookup for the blood-pressure session artefact (ADR-004).

The artefact is a cached, per-session **whole-session grid**: the segment-mean-
removed BP trace on the unchanged 1000 Hz source grid, a per-sample validity
mask, a per-sample cause code and a segment table, in absolute session time.
Consumers read it and cut windows from it; they never clean the raw trace again.

This module owns three things and no others:

* the **identity** — the frozen parameter set, its readable key, and the
  fail-loudly comparison a consumer must run before reading a shard;
* the **storage** — one ``sessions/<SESSION>.npz`` per session, written
  atomically, and one ``skips/<SESSION>.json`` for a session with no readable
  trace;
* the **filesystem truth** — ``scan`` / ``build_manifest``, which never trust a
  derived index while a build is running.

It does NOT decide validity (that is :mod:`data.bp_preprocess`) and does not
interpret the mask (that is :mod:`data.bp_windows`).

Layout (ADR §2.6 — one shard per session, one manifest derived from the shards)::

    bpsession_src1000_range30-250_flat200ms_rail30ms_seg4s_v1_<sha1[:10]>/
        params.json                # the frozen contract (--init)
        sessions/<SESSION>.npz     # THE SOURCE OF TRUTH
        skips/<SESSION>.json       # {"session","reason"} — missing/unreadable/empty ONLY
        manifest.json              # derived index (--index)
"""
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:                                     # inside the ``data`` package (tests)
    from . import bp_preprocess as bpp
except ImportError:                      # loaded standalone by the torch-free builder
    import bp_preprocess as bpp          # noqa: F401  (``code/data`` is on sys.path)

__all__ = [
    'BP_ARTEFACT_FORMAT_VERSION', 'COMPARE_KEYS', 'SUBDIR_SESSIONS',
    'SUBDIR_SKIPS', 'SUBDIR_TMP', 'SHARD_SUFFIX', 'PARAMS_NAME',
    'MANIFEST_NAME', 'SKIP_REASONS', 'BP_SOURCE_NAME',
    'discover_sessions', 'corpus_digest', 'build_params', 'cache_key',
    'cache_root', 'write_params', 'read_params', 'check_params_compatible',
    'find_cache', 'build_task', 'write_shard', 'write_skip', 'read_shard',
    'shard_path', 'shards_equal', 'scan', 'build_manifest',
]

#: Bump when a change alters the STORED BYTES or the shard schema. The cache key
#: is a hash of PARAMETERS, not of the code, so this field is what makes a stale
#: artefact impossible to reuse silently.
BP_ARTEFACT_FORMAT_VERSION = 1

SUBDIR_SESSIONS = 'sessions'
SUBDIR_SKIPS = 'skips'
SUBDIR_TMP = '.tmp'
SHARD_SUFFIX = '.npz'
PARAMS_NAME = 'params.json'
MANIFEST_NAME = 'manifest.json'

#: The raw file this artefact is built from, inside ``Physiology/<S>/<T>/``.
BP_SOURCE_NAME = 'BP_mmHg.txt'

#: The ONLY reasons a session may be skipped. A session whose every sample is
#: rejected is still a shard — the artefact never rules a session unusable, only
#: a consumer may (ADR Criterion 8).
SKIP_REASONS = ('missing_bp_mmhg', 'unreadable_bp_mmhg', 'empty_bp_mmhg')

_TASK_RE = re.compile(r'^[Tt](\d+)$')

#: Every field a consumer must agree on: the WHOLE identity, in canonical order.
COMPARE_KEYS = ('format_version', 'corpus', 'fs_in', 'fs_out',
                'gate1_min_mmhg', 'gate1_max_mmhg', 'flatline_max_ms',
                'rail_dwell_min_ms', 'min_segment_s', 'store')


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def _task_key(task: str):
    m = _TASK_RE.match(str(task))
    return (0, int(m.group(1))) if m else (1, str(task))


# --------------------------------------------------------------------------- #
# discovery — the corpus, enumerated by DIRECTORY, never by file glob
# --------------------------------------------------------------------------- #
def discover_sessions(raw_root: str,
                      subjects: Optional[Sequence[str]] = None) -> List[dict]:
    """Every ``<raw_root>/Physiology/<SUBJECT>/<TASK>/`` directory, unfiltered.

    Discovery enumerates session **directories** and resolves ``BP_mmHg.txt``
    inside each. It never globs for the file: a glob would make the
    ``missing_bp_mmhg`` skip unreachable by construction and would silently
    shrink the denominator the corpus gate reports.

    A missing trace yields ``source_file = None`` (the path is still recorded);
    the builder turns that into a skip record. Nothing is filtered here — not by
    subject, not by whether a trace exists, not by label content.
    """
    root = Path(raw_root)
    phys = root / 'Physiology'
    if not phys.is_dir():
        raise FileNotFoundError(
            f'bp_artefact: no Physiology/ tree under {root} — check --raw_root.')
    want = {str(s) for s in subjects} if subjects else None

    out: List[dict] = []
    for subj_dir in sorted(p for p in phys.iterdir() if p.is_dir()):
        subject = subj_dir.name
        if want is not None and subject not in want:
            continue
        task_dirs = sorted((p for p in subj_dir.iterdir() if p.is_dir()),
                           key=lambda p: _task_key(p.name))
        for task_dir in task_dirs:
            path = task_dir / BP_SOURCE_NAME
            out.append({
                'session': f'{subject}_{task_dir.name}',
                'subject': subject,
                'task': task_dir.name,
                'source_path': str(path),
                'source_file': str(path) if path.is_file() else None,
            })
    if not out:
        raise FileNotFoundError(
            f'bp_artefact: no session directories under {phys}'
            + (f' for subjects {sorted(want)}' if want else ''))
    return out


def corpus_digest(sessions: Sequence[dict]) -> str:
    """Stable digest of the discovered session list.

    Uses the session id and **whether a trace was found**, never the absolute
    path: the key must describe the corpus, not the machine the build ran on.
    """
    rows = sorted(f"{s['session']}\t{'1' if s.get('source_file') else '0'}"
                  for s in sessions)
    return _sha1('\n'.join(rows))[:16]


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #
def build_params(sessions: Sequence[dict],
                 fs_in: float = bpp.FS_IN, fs_out: float = bpp.FS_OUT,
                 gate1_min_mmhg: float = bpp.GATE1_MIN_MMHG,
                 gate1_max_mmhg: float = bpp.GATE1_MAX_MMHG,
                 flatline_max_ms: float = bpp.FLATLINE_MAX_MS,
                 rail_dwell_min_ms: float = bpp.RAIL_DWELL_MIN_MS,
                 min_segment_s: float = bpp.MIN_SEGMENT_S,
                 store: str = bpp.STORE) -> Dict:
    """The FROZEN parameter set — exactly the identity fields of ADR §2.6.

    Every field here changes a stored sample, so every field is hashed into the
    directory name and compared by every consumer. Nothing that merely selects
    or rescales a window (clip length, hop, split, task subset, normalisation) is
    present: that absence is what lets ONE artefact serve every consumer.

    ``sessions`` MUST be the UNFILTERED discovery: the key is a property of the
    CORPUS, so ``--subjects`` / ``--limit`` may narrow a BUILD without creating a
    second identity.

    The rates and the two millisecond thresholds serialise as ``int``, the
    mmHg/seconds thresholds as ``float``, so ``1000`` and ``1000.0`` cannot
    produce two keys for one artefact.
    """
    if str(store) != bpp.STORE:
        raise ValueError(
            f'bp_artefact: store {store!r} is not the frozen {bpp.STORE!r}; a '
            f'different representation is a different artefact (bump '
            f'BP_ARTEFACT_FORMAT_VERSION and the constant together).')
    if float(fs_out) != float(fs_in):
        raise ValueError(
            f'bp_artefact: fs_out={float(fs_out):g} != fs_in={float(fs_in):g}. '
            f'This artefact stores the source grid unchanged (ADR §2.2 step 5); '
            f'no resampling geometry is supported.')
    if float(gate1_min_mmhg) >= float(gate1_max_mmhg):
        raise ValueError(
            f'bp_artefact: gate1_min_mmhg={gate1_min_mmhg!r} must be below '
            f'gate1_max_mmhg={gate1_max_mmhg!r}.')
    return {
        'format_version': BP_ARTEFACT_FORMAT_VERSION,
        'corpus': corpus_digest(sessions),
        'fs_in': int(round(float(fs_in))),
        'fs_out': int(round(float(fs_out))),
        'gate1_min_mmhg': float(gate1_min_mmhg),
        'gate1_max_mmhg': float(gate1_max_mmhg),
        'flatline_max_ms': int(round(float(flatline_max_ms))),
        'rail_dwell_min_ms': int(round(float(rail_dwell_min_ms))),
        'min_segment_s': float(min_segment_s),
        'store': str(store),
    }


def cache_key(params: Dict) -> str:
    """Readable slug + content hash. The hash covers EVERY frozen field."""
    body = json.dumps(params, sort_keys=True, default=str)
    slug = (f"bpsession_src{int(params['fs_in'])}"
            f"_range{float(params['gate1_min_mmhg']):g}"
            f"-{float(params['gate1_max_mmhg']):g}"
            f"_flat{int(params['flatline_max_ms'])}ms"
            f"_rail{int(params['rail_dwell_min_ms'])}ms"
            f"_seg{float(params['min_segment_s']):g}s"
            f"_v{int(params['format_version'])}")
    return f'{slug}_{_sha1(body)[:10]}'


def cache_root(out_root: str, key: str) -> Path:
    return Path(out_root) / key


def write_params(root: Path, params: Dict) -> None:
    """Write ``params.json`` atomically (``mkstemp`` + ``os.replace``)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(root), prefix=PARAMS_NAME + '.',
                                    suffix='.tmp')
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, 'w') as fh:
            json.dump(params, fh, indent=2, sort_keys=True)
        os.replace(tmp, root / PARAMS_NAME)
    finally:
        if tmp.exists():                 # only if the replace failed
            try:
                tmp.unlink()
            except OSError:
                pass


def read_params(root: Path) -> Dict:
    with open(Path(root) / PARAMS_NAME) as fh:
        return json.load(fh)


def _param_diff(have: Dict, want: Dict,
                keys: Sequence[str] = COMPARE_KEYS) -> List[str]:
    return [f'{k}: artefact={have.get(k)!r} expected={want.get(k)!r}'
            for k in keys if have.get(k) != want.get(k)]


def check_params_compatible(root: Path, params: Dict,
                            keys: Sequence[str] = COMPARE_KEYS) -> None:
    """Raise with a field-by-field diff if ``root`` was built for something else.

    A consumer that read a slightly different grid would invalidate every
    comparison while looking healthy at every step, so a mismatch REFUSES rather
    than adapts.
    """
    root = Path(root)
    if not (root / PARAMS_NAME).is_file():
        raise FileNotFoundError(
            f'bp_artefact: {root}/params.json not found. Build the artefact with '
            f'`runners/run_build_bp_artefact.py --init --build`.')
    diffs = _param_diff(read_params(root), params)
    if diffs:
        raise ValueError(
            f'bp_artefact: {root} was built with DIFFERENT parameters:\n  '
            + '\n  '.join(diffs)
            + '\nRefusing to read a different grid. Rebuild with the knobs you '
              'asked for, or point --out_root at the matching artefact.')


def find_cache(out_root, want: Dict, keys: Sequence[str] = COMPARE_KEYS,
               require_session: Optional[str] = None) -> Path:
    """Locate the artefact directory whose parameters match ``want``.

    Distinguishes a **mismatch** (a candidate exists with other fields →
    ``ValueError`` with the diff) from an **absence** (nothing there →
    ``FileNotFoundError``): only the first means "you asked for another grid".
    """
    out_root = Path(out_root)
    if not out_root.is_dir():
        raise FileNotFoundError(
            f'bp_artefact: out_root does not exist: {out_root}. Build it with '
            f'`runners/run_build_bp_artefact.py --init --build`.')
    matches: List[Path] = []
    near: List[Tuple[Path, List[str]]] = []
    lacking: List[Tuple[Path, str]] = []
    for child in sorted(out_root.iterdir()):
        if not child.is_dir() or not (child / PARAMS_NAME).is_file():
            continue
        diffs = _param_diff(read_params(child), want, keys)
        if diffs:
            near.append((child, diffs))
            continue
        if require_session is not None:
            name = str(require_session)
            if not (shard_path(child, name).is_file() or
                    (child / SUBDIR_SKIPS / f'{name}.json').is_file()):
                lacking.append((child, name))
                continue
        matches.append(child)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            'bp_artefact: ambiguous match — several artefacts satisfy the '
            'request:\n  ' + '\n  '.join(str(m) for m in matches)
            + '\nDelete the stale one or pass the directory explicitly.')
    if lacking:
        raise FileNotFoundError(
            f'bp_artefact: {lacking[0][0]} matches the parameters but holds no '
            f'shard for session {lacking[0][1]!r} — the build is incomplete or '
            f'covers a different scope.')
    if near:
        best = min(near, key=lambda kv: len(kv[1]))
        raise ValueError(
            f'bp_artefact: {out_root} holds no artefact matching the requested '
            f'parameters. Nearest: {best[0]}\n  ' + '\n  '.join(best[1])
            + '\nRefusing to read a different grid.')
    raise FileNotFoundError(
        f'bp_artefact: no artefact at all under {out_root}. Build it with '
        f'`runners/run_build_bp_artefact.py --init --build`.')


# --------------------------------------------------------------------------- #
# build one session
# --------------------------------------------------------------------------- #
def _load_bp_mmhg(path: str) -> np.ndarray:
    """One value per line → ``float64``; a malformed token becomes ``NaN``.

    Deliberately local: the builder must not depend on a torch-importing dataset
    module just to read a text column.
    """
    vals: List[float] = []
    with open(path, 'r') as fh:
        for line in fh:
            tok = line.strip()
            if not tok:
                continue
            try:
                vals.append(float(tok.split()[0]))
            except ValueError:
                vals.append(float('nan'))
    return np.asarray(vals, dtype=np.float64)


def build_task(spec: dict, params: Dict
               ) -> Tuple[Optional[dict], Optional[str]]:
    """Preprocess ONE session into a shard payload.

    Returns ``(payload, None)``, or ``(None, reason)`` for the three reasons in
    :data:`SKIP_REASONS`. It NEVER skips on label content: a session whose every
    sample is rejected is still a shard with ``valid_frac == 0``.
    """
    session = str(spec['session'])
    source = spec.get('source_file') or spec.get('source_path')
    if not source or not os.path.isfile(str(source)):
        return None, 'missing_bp_mmhg'
    try:
        x = _load_bp_mmhg(str(source))
    except Exception as exc:                  # unreadable / malformed file
        return None, f'unreadable_bp_mmhg: {exc}'
    if x.size == 0:
        return None, 'empty_bp_mmhg'

    out = bpp.preprocess_session(
        x, fs_in=params['fs_in'], fs_out=params['fs_out'],
        gate1_min_mmhg=params['gate1_min_mmhg'],
        gate1_max_mmhg=params['gate1_max_mmhg'],
        flatline_max_ms=params['flatline_max_ms'],
        rail_dwell_min_ms=params['rail_dwell_min_ms'],
        min_segment_s=params['min_segment_s'],
        dtype=bpp.STORE_DTYPE)
    payload = dict(out)
    payload.update({
        'n_src': np.int64(out['n_src']),
        'fs_in': np.int64(out['fs_in']),
        'fs_out': np.int64(out['fs_out']),
        'n_segments': np.int64(out['n_segments']),
        'valid_frac': np.float64(out['valid_frac']),
        'n_nonfinite': np.int64(out['n_nonfinite']),
        'session': session,
        'source_file': str(source),
    })
    return payload, None


# --------------------------------------------------------------------------- #
# shard IO
# --------------------------------------------------------------------------- #
def write_shard(root: Path, session: str, payload: Dict,
                compress: bool = False) -> Path:
    """Write one shard atomically (staging dir + ``os.replace``).

    ``compress`` defaults to False: the container is uncompressed, because
    compression costs build time and would make the ADR's uncompressed byte
    budget unmeasurable.
    """
    sessions_dir = Path(root) / SUBDIR_SESSIONS
    tmp_dir = Path(root) / SUBDIR_TMP
    sessions_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    final = sessions_dir / f'{session}{SHARD_SUFFIX}'
    tmp = tmp_dir / f'{session}.{os.getpid()}{SHARD_SUFFIX}'
    saver = np.savez_compressed if compress else np.savez
    with open(tmp, 'wb') as fh:
        saver(fh, **payload)
    os.replace(tmp, final)
    return final


def write_skip(root: Path, session: str, reason: str) -> Path:
    """Record WHY a session has no shard. Creates ``.tmp`` itself."""
    d = Path(root) / SUBDIR_SKIPS
    tmp_dir = Path(root) / SUBDIR_TMP
    d.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = d / f'{session}.json'
    tmp = tmp_dir / f'{session}.{os.getpid()}.json'
    with open(tmp, 'w') as fh:
        json.dump({'session': session, 'reason': reason}, fh, indent=2)
    os.replace(tmp, path)
    return path


def shard_path(root: Path, session: str) -> Path:
    return Path(root) / SUBDIR_SESSIONS / f'{session}{SHARD_SUFFIX}'


def read_shard(path) -> Dict:
    """Load one shard and re-assert its invariants.

    The loader re-checks ``isnan(signal) == ~mask``, ``mask ⇔ cause == 0`` and
    the segment table's shape and bounds, because a shard is the ONLY statement
    of validity downstream: a corrupted or foreign file must fail here, not
    become a training target.
    """
    with np.load(str(path), allow_pickle=False) as z:
        out = {
            'signal': z['signal'],
            'mask': np.asarray(z['mask'], dtype=bool),
            'cause': np.asarray(z['cause'], dtype=np.uint8),
            'seg_start': np.asarray(z['seg_start'], dtype=np.int64),
            'seg_stop': np.asarray(z['seg_stop'], dtype=np.int64),
            'seg_mean_mmhg': np.asarray(z['seg_mean_mmhg'], dtype=np.float64),
            'seg_seconds': np.asarray(z['seg_seconds'], dtype=np.float64),
            'n_src': int(z['n_src']),
            'fs_in': int(z['fs_in']),
            'fs_out': int(z['fs_out']),
            'n_segments': int(z['n_segments']),
            'valid_frac': float(z['valid_frac']),
            'n_nonfinite': int(z['n_nonfinite']),
            'session': str(z['session']),
            'source_file': str(z['source_file']),
        }

    n = out['signal'].size
    if out['mask'].shape != (n,) or out['cause'].shape != (n,):
        raise ValueError(
            f'bp_artefact: {path} has inconsistent lengths '
            f'(signal {out["signal"].shape}, mask {out["mask"].shape}, '
            f'cause {out["cause"].shape}).')
    if not np.array_equal(np.isnan(out['signal']), ~out['mask']):
        raise ValueError(
            f'bp_artefact: {path} violates `isnan(signal) == ~mask`; refusing to '
            f'serve a shard whose mask is not the sole authority.')
    if not np.array_equal(out['mask'], out['cause'] == bpp.CAUSE_VALID):
        raise ValueError(f'bp_artefact: {path} violates `mask <=> cause == 0`.')

    lengths = [out[k].size for k in ('seg_start', 'seg_stop', 'seg_mean_mmhg',
                                     'seg_seconds')]
    if len(set(lengths)) != 1 or lengths[0] != out['n_segments']:
        raise ValueError(
            f'bp_artefact: {path} has a malformed segment table '
            f'(lengths {lengths}, n_segments {out["n_segments"]}).')
    start, stop = out['seg_start'], out['seg_stop']
    if start.size and (np.any(stop <= start) or np.any(start < 0) or
                       np.any(stop > n)):
        raise ValueError(
            f'bp_artefact: {path} has out-of-range segment bounds.')
    if start.size > 1 and np.any(start[1:] < stop[:-1]):
        raise ValueError(
            f'bp_artefact: {path} has overlapping or unsorted segments.')
    return out


def shards_equal(a: Dict, b: Dict) -> Tuple[bool, List[str]]:
    """Bit-parity between two shard payloads — the ``--check`` comparison.

    Exact on ``signal`` (NaN-aware), ``mask``, ``cause`` and the segment table,
    not approximate: a gate or threshold change must show up as a parity failure
    even when it is numerically small.
    """
    diffs: List[str] = []
    if not np.array_equal(np.asarray(a['signal']), np.asarray(b['signal']),
                          equal_nan=True):
        diffs.append('signal')
    for key in ('mask', 'cause', 'seg_start', 'seg_stop', 'seg_mean_mmhg',
                'seg_seconds'):
        if not np.array_equal(np.asarray(a[key]), np.asarray(b[key])):
            diffs.append(key)
    return (not diffs), diffs


# --------------------------------------------------------------------------- #
# filesystem truth
# --------------------------------------------------------------------------- #
def scan(root: Path, sessions: Sequence[dict]) -> Dict:
    """Cached / skipped / missing. Never trusts the manifest."""
    root = Path(root)
    cached, skipped, missing = [], [], []
    n_bytes = 0
    for s in sessions:
        name = str(s['session'])
        p = shard_path(root, name)
        if p.is_file():
            cached.append(name)
            n_bytes += p.stat().st_size
            continue
        if (root / SUBDIR_SKIPS / f'{name}.json').is_file():
            skipped.append(name)
            continue
        missing.append(name)
    return {'cached': len(cached), 'skipped': len(skipped),
            'missing': len(missing), 'bytes': n_bytes,
            'cached_names': cached, 'skipped_names': skipped,
            'missing_names': missing}


def build_manifest(root: Path, sessions: Sequence[dict], params: Dict) -> Dict:
    """Derived index, cheap enough to rebuild: it reads headers, not payloads."""
    root = Path(root)
    entries: List[Dict] = []
    all_seg_seconds: List[float] = []
    for s in sessions:
        name = str(s['session'])
        p = shard_path(root, name)
        if p.is_file():
            with np.load(str(p), allow_pickle=False) as z:
                entries.append({
                    'session': name, 'status': 'ok',
                    'bytes': p.stat().st_size,
                    'n_src': int(z['n_src']),
                    'n_segments': int(z['n_segments']),
                    'valid_frac': float(z['valid_frac']),
                    'n_nonfinite': int(z['n_nonfinite']),
                    'seg_seconds_max': (float(np.max(z['seg_seconds']))
                                        if int(z['n_segments']) else 0.0),
                    'cause': {str(int(k)): int(v) for k, v in
                              zip(*np.unique(z['cause'], return_counts=True))},
                })
                all_seg_seconds.extend(np.asarray(z['seg_seconds']).ravel().tolist())
            continue
        sp = root / SUBDIR_SKIPS / f'{name}.json'
        if sp.is_file():
            with open(sp) as fh:
                entries.append(dict(json.load(fh), status='skipped'))
        else:
            entries.append({'session': name, 'status': 'missing'})

    ok = [e for e in entries if e['status'] == 'ok']
    fracs = np.asarray([e['valid_frac'] for e in ok], dtype=np.float64)
    # Only sessions that HAVE a segment contribute to the length statistics:
    # a zero-segment shard must not be reported as a 0-second segment, or the
    # "min segment length" would read as a real 0 s segment.
    seg_len = np.asarray([e['seg_seconds_max'] for e in ok if e['n_segments'] > 0],
                         dtype=np.float64)
    cause_totals: Dict[str, int] = {}
    for e in ok:
        for code, count in e['cause'].items():
            cause_totals[code] = cause_totals.get(code, 0) + int(count)

    def _stats(values: np.ndarray) -> Dict:
        if not values.size:
            return {'mean': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0}
        return {'mean': float(values.mean()), 'median': float(np.median(values)),
                'min': float(values.min()), 'max': float(values.max())}

    seg_all = np.asarray(all_seg_seconds, dtype=np.float64)
    segment_stats = _stats(seg_all)
    segment_stats.update({
        'n_total': int(seg_all.size),
        'p10': float(np.percentile(seg_all, 10)) if seg_all.size else 0.0,
        'p90': float(np.percentile(seg_all, 90)) if seg_all.size else 0.0,
        # how much of the admitted time cannot carry a clip of a given length;
        # an artefact STATISTIC, not a policy -- no clip length is stored
        'frac_below_8s': float((seg_all < 8.0).mean()) if seg_all.size else 0.0,
        'frac_below_16s': float((seg_all < 16.0).mean()) if seg_all.size else 0.0,
    })

    manifest = {
        'key': cache_key(params),
        'params': params,
        'n_sessions': len(sessions),
        'n_ok': len(ok),
        'n_skipped': sum(1 for e in entries if e['status'] == 'skipped'),
        'n_missing': sum(1 for e in entries if e['status'] == 'missing'),
        'n_zero_segment': sum(1 for e in ok if e['n_segments'] == 0),
        'total_bytes': sum(int(e.get('bytes') or 0) for e in entries),
        'valid_frac': _stats(fracs),
        'seg_seconds_max': _stats(seg_len),
        'segments': segment_stats,
        'cause_totals': {bpp.CAUSE_NAMES.get(int(k), str(k)): v
                         for k, v in sorted(cause_totals.items())},
        'entries': entries,
    }
    tmp = root / (MANIFEST_NAME + '.tmp')
    with open(tmp, 'w') as fh:
        json.dump(manifest, fh, indent=2)
    os.replace(tmp, root / MANIFEST_NAME)
    return manifest
