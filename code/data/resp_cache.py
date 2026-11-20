"""Identity, storage and lookup for the respiration session artefact (ADR-003).

The artefact is a cached, per-session **whole-session grid**: the offset-removed
signal on a fixed output grid plus a per-sample validity mask in absolute session
time. Consumers read it and cut windows from it; they never clean the raw trace
again (ADR §2).

This module owns three things and no others:

* the **identity** — the frozen parameter set, its readable key and the
  fail-loudly comparison a consumer must run before reading a shard (I6, I9);
* the **storage** — one ``tasks/<SESSION>.npz`` per session, written atomically,
  and one ``skipped/<SESSION>.json`` for a session that has no readable trace;
* the **filesystem truth** — ``scan`` / ``build_manifest``, which never trust a
  derived index while an array job is running.

It does NOT decide validity (that is ``data.resp_preprocess``) and does not
interpret the mask (that is ``data.resp_window``).

Layout (ADR §2.1's "one shard per session", plan step 11)::

    respgrid_src1000_fs100_sat9.95-1s_flat0.3-5s_run8s_v1_<sha1[:10]>/
        params.json            # the frozen contract (--init)
        tasks/<SESSION>.npz    # THE SOURCE OF TRUTH
        skipped/<SESSION>.json # {"session","reason"} — missing/unreadable ONLY
        manifest.json          # derived index (--index)
"""
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import resp_preprocess as rpp
from .roi_cache import corpus_digest

__all__ = [
    'RESP_CACHE_FORMAT_VERSION', 'COMPARE_KEYS', 'SUBDIR_TASKS',
    'SUBDIR_SKIPPED', 'SUBDIR_TMP', 'SHARD_SUFFIX', 'PARAMS_NAME',
    'MANIFEST_NAME', 'SKIP_REASONS',
    'resample_spec', 'build_params', 'cache_key', 'cache_root',
    'write_params', 'read_params', 'check_params_compatible', 'loader_params',
    'EVAL_COMPARE_KEYS', 'eval_params',
    'find_cache', 'build_task', 'write_shard', 'write_skip', 'read_shard',
    'shard_path', 'scan', 'build_manifest',
]

#: Bump when a change alters the STORED BYTES or the shard schema. The key is a
#: hash of PARAMETERS, not of the code, so this field is what makes a stale
#: artefact impossible to reuse silently (ADR I9).
RESP_CACHE_FORMAT_VERSION = 1

SUBDIR_TASKS = 'tasks'
SUBDIR_SKIPPED = 'skipped'
SUBDIR_TMP = '.tmp'
SHARD_SUFFIX = '.npz'
PARAMS_NAME = 'params.json'
MANIFEST_NAME = 'manifest.json'

#: The ONLY reasons a session may be skipped. A session whose every sample is
#: invalid is still a shard (ADR Criterion 7): the artefact never rules a session
#: unusable, only a consumer may.
SKIP_REASONS = ('missing_resp_volts', 'unreadable_resp_volts', 'empty_resp_volts')


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #
def resample_spec(fs_in=rpp.FS_IN, fs_out=rpp.FS_OUT_DEFAULT,
                  taps_per_phase=rpp.RESAMPLE_TAPS_PER_PHASE) -> Dict:
    """The frozen description of the resampler, derived from the validated pair.

    ``kernel_radius_src`` is the polyphase filter's support in SOURCE samples. It
    is recorded because it decides how much of a run's edge is replicated and
    therefore what the stored values are — not because it is a knob.
    """
    up, down, radius = rpp.resolve_geometry(fs_in, fs_out, taps_per_phase)
    return {
        'kind': rpp.RESAMPLE_KIND,
        'taps_per_phase': int(taps_per_phase),
        'up': int(up),
        'down': int(down),
        'kernel_radius_src': int(radius),
    }


def build_params(raw_root: str, sessions: Sequence[dict],
                 fs_in: float = rpp.FS_IN,
                 fs_out: float = rpp.FS_OUT_DEFAULT,
                 sat_v: float = rpp.SAT_V,
                 sat_min_s: float = rpp.SAT_MIN_S,
                 flat_window_s: float = rpp.FLAT_WINDOW_S,
                 flat_ptp_v: float = rpp.FLAT_PTP_V,
                 min_run_s: float = rpp.MIN_RUN_S,
                 taps_per_phase: int = rpp.RESAMPLE_TAPS_PER_PHASE,
                 dtype: str = rpp.STORE_DTYPE) -> Dict:
    """The FROZEN parameter set — exactly the identity fields of ADR §4.1.

    Every field here changes a stored sample, so every field is hashed into the
    directory name and compared by every consumer. Nothing that merely selects or
    rescales a window (clip length, hop, split, task subset, normalisation) is
    present: that absence is what lets ONE artefact serve the 2.0 s-hop and the
    1.0 s-hop stages (ADR Criterion 5).

    ``sessions`` MUST be the UNFILTERED discovery: the key is a property of the
    CORPUS, so ``--subjects`` / ``--limit`` may narrow a BUILD without creating a
    second identity.

    Raises ``ValueError`` from :func:`resp_preprocess.resolve_geometry` when the
    rate pair or the tap count is refused (plan A7/A8).
    """
    if str(dtype) != rpp.STORE_DTYPE:
        raise ValueError(
            f'resp_cache: store dtype {dtype!r} is not the frozen '
            f'{rpp.STORE_DTYPE!r}; a different representation is a different '
            f'artefact, bump RESP_CACHE_FORMAT_VERSION and STORE_DTYPE together.')
    return {
        'format_version': RESP_CACHE_FORMAT_VERSION,
        'corpus': corpus_digest(sessions),
        'fs_in': int(round(float(fs_in))),
        'fs_out': int(round(float(fs_out))),
        'sat_v': float(sat_v),
        'sat_min_s': float(sat_min_s),
        'flat_window_s': float(flat_window_s),
        'flat_ptp_v': float(flat_ptp_v),
        'min_run_s': float(min_run_s),
        'resample': resample_spec(fs_in, fs_out, taps_per_phase),
        'store': {'kind': rpp.STORE_KIND, 'dtype': rpp.STORE_DTYPE},
    }


#: Every field a consumer must agree on. Deliberately the WHOLE identity: this
#: artefact has no decode stack and no source-path field, so there is nothing a
#: loader cannot recompute (contrast ``roi_cache.COMPARE_KEYS``).
COMPARE_KEYS = ('format_version', 'corpus', 'fs_in', 'fs_out', 'sat_v',
                'sat_min_s', 'flat_window_s', 'flat_ptp_v', 'min_run_s',
                'resample', 'store')


def cache_key(params: Dict) -> str:
    """Readable slug + content hash. The hash is over EVERY frozen field."""
    body = json.dumps(params, sort_keys=True, default=str)
    rs = params['resample']
    slug = (f'respgrid_src{int(params["fs_in"])}'
            f'_fs{int(params["fs_out"])}'
            f'_sat{float(params["sat_v"]):g}-{float(params["sat_min_s"]):g}s'
            f'_flat{float(params["flat_ptp_v"]):g}-'
            f'{float(params["flat_window_s"]):g}s'
            f'_run{float(params["min_run_s"]):g}s'
            f'_v{int(params["format_version"])}')
    assert rs['kind'] and rs['taps_per_phase'], 'resample spec incomplete'
    return f'{slug}_{_sha1(body)[:10]}'


def cache_root(out_root: str, key: str) -> Path:
    return Path(out_root) / key


def write_params(root: Path, params: Dict) -> None:
    """Write ``params.json`` atomically (``mkstemp`` + ``os.replace``).

    The staging name is unique per writer so two array elements racing on
    ``--init`` cannot truncate each other's partial file; ``os.replace`` makes
    the publication atomic either way.
    """
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
        if tmp.exists():                    # only if the replace failed
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

    This is the fail-loudly gate (ADR I6): a consumer that reads a slightly
    different grid would invalidate every comparison while looking healthy at
    every step, so a mismatch REFUSES rather than adapts.
    """
    root = Path(root)
    if not (root / PARAMS_NAME).is_file():
        raise FileNotFoundError(
            f'resp_cache: {root}/params.json not found. Build the artefact with '
            f'`runners/run_build_resp_cache.py --init --build`.')
    have = read_params(root)
    diffs = _param_diff(have, params)
    if diffs:
        raise ValueError(
            f'resp_cache: {root} was built with DIFFERENT parameters:\n  '
            + '\n  '.join(diffs)
            + '\nRefusing to read a different grid. Rebuild with the knobs you '
              'asked for, or point --resp_cache at the matching directory.')


#: The fields an EVALUATOR can verify: everything except ``corpus``. The report
#: process does not know which training corpus produced the artefact, and
#: verifying a corpus it guessed would be worse than skipping that one field.
EVAL_COMPARE_KEYS = tuple(k for k in COMPARE_KEYS if k != 'corpus')


def eval_params(fs_in: float = rpp.FS_IN,
                fs_out: float = rpp.FS_OUT_DEFAULT,
                sat_v: float = rpp.SAT_V,
                sat_min_s: float = rpp.SAT_MIN_S,
                flat_window_s: float = rpp.FLAT_WINDOW_S,
                flat_ptp_v: float = rpp.FLAT_PTP_V,
                min_run_s: float = rpp.MIN_RUN_S,
                taps_per_phase: int = rpp.RESAMPLE_TAPS_PER_PHASE) -> Dict:
    """The identity subset a report process can reconstruct on its own.

    ``corpus`` is excluded (see :data:`EVAL_COMPARE_KEYS`): the evaluator has no
    training discovery to hash. Every field that changes a stored sample IS
    checked, so a reference read from a different grid still aborts.
    """
    full = build_params('', [], fs_in=fs_in, fs_out=fs_out, sat_v=sat_v,
                        sat_min_s=sat_min_s, flat_window_s=flat_window_s,
                        flat_ptp_v=flat_ptp_v, min_run_s=min_run_s,
                        taps_per_phase=taps_per_phase)
    return {k: full[k] for k in EVAL_COMPARE_KEYS}


def loader_params(raw_root: str, corpus_sessions: Sequence[dict],                  fs_out: float = rpp.FS_OUT_DEFAULT,
                  fs_in: float = rpp.FS_IN,
                  sat_v: float = rpp.SAT_V,
                  sat_min_s: float = rpp.SAT_MIN_S,
                  flat_window_s: float = rpp.FLAT_WINDOW_S,
                  flat_ptp_v: float = rpp.FLAT_PTP_V,
                  min_run_s: float = rpp.MIN_RUN_S,
                  taps_per_phase: int = rpp.RESAMPLE_TAPS_PER_PHASE) -> Dict:
    """The frozen parameters a CONSUMER can recompute on its own.

    ``corpus_sessions`` must be the unfiltered discovery, for the same reason as
    :func:`build_params`.
    """
    full = build_params(raw_root, corpus_sessions, fs_in=fs_in, fs_out=fs_out,
                        sat_v=sat_v, sat_min_s=sat_min_s,
                        flat_window_s=flat_window_s, flat_ptp_v=flat_ptp_v,
                        min_run_s=min_run_s, taps_per_phase=taps_per_phase)
    return {k: full[k] for k in COMPARE_KEYS}


def find_cache(out_root, want: Dict, keys: Sequence[str] = COMPARE_KEYS,
               require_session: Optional[str] = None) -> Path:
    """Locate the artefact directory whose parameters match ``want``.

    Raises ``FileNotFoundError`` when nothing matches (with a field-by-field diff
    against the nearest candidate), and refuses an ambiguous match rather than
    picking one.

    :param require_session: when given, a candidate must also be able to serve
        that session (a shard, or a ``skipped/`` record). The evaluator passes it
        so a cache that simply lacks the session is refused as a MATCH rather
        than failing later with a missing file.
    """
    out_root = Path(out_root)
    if not out_root.is_dir():
        raise FileNotFoundError(
            f'resp_cache: out_root does not exist: {out_root}. Build it with '
            f'`runners/run_build_resp_cache.py --init --build`.')
    matches: List[Path] = []
    near: List[Tuple[Path, Dict]] = []
    lacking: List[Tuple[Path, str]] = []
    for child in sorted(out_root.iterdir()):
        if not child.is_dir() or not (child / PARAMS_NAME).is_file():
            continue
        have = read_params(child)
        diffs = _param_diff(have, want, keys)
        if diffs:
            near.append((child, diffs))
            continue
        if require_session is not None:
            name = str(require_session)
            if not (shard_path(child, name).is_file() or
                    (child / SUBDIR_SKIPPED / f'{name}.json').is_file()):
                lacking.append((child, name))
                continue
        matches.append(child)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            'resp_cache: ambiguous match — several artefacts satisfy the request:\n  '
            + '\n  '.join(str(m) for m in matches)
            + '\nDelete the stale one or pass the directory explicitly.')
    if lacking:
        raise FileNotFoundError(
            f'resp_cache: {lacking[0][0]} matches the parameters but holds no '
            f'shard for session {lacking[0][1]!r} -- the build is incomplete or '
            f'covers a different scope.')
    if near:
        # an artefact exists but is NOT the one requested: that is a mismatch,
        # not an absence, and it must read like one
        best = min(near, key=lambda kv: len(kv[1]))
        raise ValueError(
            f'resp_cache: {out_root} holds no artefact matching the requested '
            f'parameters. Nearest: {best[0]}\n  '
            + '\n  '.join(best[1])
            + '\nRefusing to read a different grid. Rebuild with '
              '`runners/run_build_resp_cache.py --init --build`, or pass the '
              'knobs that produced the existing artefact.')
    raise FileNotFoundError(
        f'resp_cache: no artefact at all under {out_root}. Build it with '
        f'`runners/run_build_resp_cache.py --init --build`.')


# --------------------------------------------------------------------------- #
# build one session
# --------------------------------------------------------------------------- #
def _load_resp_volts(path: str) -> np.ndarray:
    """One value per line -> ``float32``; a malformed token becomes ``NaN``.

    Deliberately local: the builder and the cache must not depend on the
    torch-importing dataset module just to read a text column.
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
    return np.asarray(vals, dtype=np.float32)


def build_task(spec: dict, params: Dict
               ) -> Tuple[Optional[Dict], Optional[str]]:
    """Preprocess ONE subject-task into a shard payload.

    Returns ``(payload, None)``, or ``(None, reason)`` for the three narrow
    reasons in :data:`SKIP_REASONS`. It NEVER skips on label content: a session
    whose every sample is invalid is still a shard with ``valid_frac == 0``
    (ADR Criterion 7).
    """
    session = str(spec['session'])
    resp_file = spec.get('resp_file')
    if not resp_file or not os.path.isfile(str(resp_file)):
        return None, 'missing_resp_volts'
    try:
        x = _load_resp_volts(str(resp_file))
    except Exception as exc:                   # unreadable / malformed file
        return None, f'unreadable_resp_volts: {exc}'
    if x.size == 0:
        return None, 'empty_resp_volts'

    out = rpp.preprocess_session(
        x, fs_in=params['fs_in'], fs_out=params['fs_out'],
        sat_v=params['sat_v'], sat_min_s=params['sat_min_s'],
        flat_window_s=params['flat_window_s'], flat_ptp_v=params['flat_ptp_v'],
        min_run_s=params['min_run_s'],
        taps_per_phase=params['resample']['taps_per_phase'],
        dtype=params['store']['dtype'])
    payload = {
        'signal': out['signal'],
        'mask': np.asarray(out['mask'], dtype=bool),
        'cause': np.asarray(out['cause'], dtype=np.uint8),
        'n_src': np.int64(out['n_src']),
        'fs_in': np.int64(out['fs_in']),
        'fs_out': np.int64(out['fs_out']),
        'n_runs': np.int64(out['n_runs']),
        'valid_frac': np.float64(out['valid_frac']),
        'n_nonfinite': np.int64(out['n_nonfinite']),
        'session': session,
        'resp_file': str(resp_file),
    }
    return payload, None


# --------------------------------------------------------------------------- #
# shard IO
# --------------------------------------------------------------------------- #
def write_shard(root: Path, session: str, payload: Dict,
                compress: bool = False) -> Path:
    """Write one shard atomically (staging dir + ``os.replace``).

    The staging name carries the PID so two processes can never share it.
    """
    tasks_dir = Path(root) / SUBDIR_TASKS
    tmp_dir = Path(root) / SUBDIR_TMP
    tasks_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    final = tasks_dir / f'{session}{SHARD_SUFFIX}'
    tmp = tmp_dir / f'{session}.{os.getpid()}{SHARD_SUFFIX}'
    saver = np.savez_compressed if compress else np.savez
    with open(tmp, 'wb') as fh:
        saver(fh, **payload)
    os.replace(tmp, final)
    return final


def write_skip(root: Path, session: str, reason: str) -> Path:
    """Record WHY a session has no shard. Creates ``.tmp`` itself."""
    d = Path(root) / SUBDIR_SKIPPED
    tmp_dir = Path(root) / SUBDIR_TMP
    d.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = d / f'{session}.json'
    tmp = tmp_dir / f'{session}.{os.getpid()}.json'
    with open(tmp, 'w') as fh:
        json.dump({'session': session, 'reason': reason}, fh, indent=2)
    os.replace(tmp, path)
    return path


def read_shard(path) -> Dict:
    """Load one shard and verify its two internal invariants.

    The loader re-asserts ``isnan(signal) == ~mask`` and ``mask <=> cause``
    because a shard is the ONLY statement of validity downstream: a corrupted or
    foreign file must fail here, not become a training target (ADR I4).
    """
    with np.load(str(path), allow_pickle=False) as z:
        out = {
            'signal': z['signal'],
            'mask': np.asarray(z['mask'], dtype=bool),
            'cause': np.asarray(z['cause'], dtype=np.uint8),
            'n_src': int(z['n_src']),
            'fs_in': int(z['fs_in']),
            'fs_out': int(z['fs_out']),
            'n_runs': int(z['n_runs']),
            'valid_frac': float(z['valid_frac']),
            'n_nonfinite': int(z['n_nonfinite']),
            'session': str(z['session']),
            'resp_file': str(z['resp_file']),
        }
    if not np.array_equal(np.isnan(out['signal']), ~out['mask']):
        raise ValueError(
            f'resp_cache: {path} violates `isnan(signal) == ~mask`; refusing to '
            f'serve a shard whose mask is not the sole authority.')
    if not np.array_equal(out['mask'], out['cause'] == rpp.CAUSE_VALID):
        raise ValueError(
            f'resp_cache: {path} violates `mask <=> cause == 0`.')
    return out


def shard_path(root: Path, session: str) -> Path:
    return Path(root) / SUBDIR_TASKS / f'{session}{SHARD_SUFFIX}'


def scan(root: Path, sessions: Sequence[dict]) -> Dict:
    """Filesystem truth: cached / skipped / missing. Never trusts the manifest."""
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
        if (root / SUBDIR_SKIPPED / f'{name}.json').is_file():
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
    for s in sessions:
        name = str(s['session'])
        p = shard_path(root, name)
        if p.is_file():
            with np.load(str(p), allow_pickle=False) as z:
                entries.append({
                    'session': name, 'status': 'ok',
                    'bytes': p.stat().st_size,
                    'n_src': int(z['n_src']),
                    'n_out': int(z['signal'].size),
                    'n_runs': int(z['n_runs']),
                    'valid_frac': float(z['valid_frac']),
                    'n_nonfinite': int(z['n_nonfinite']),
                    'cause': {str(int(k)): int(v) for k, v in
                              zip(*np.unique(z['cause'], return_counts=True))},
                })
            continue
        sp = root / SUBDIR_SKIPPED / f'{name}.json'
        if sp.is_file():
            with open(sp) as fh:
                entries.append(dict(json.load(fh), status='skipped'))
        else:
            entries.append({'session': name, 'status': 'missing'})

    ok = [e for e in entries if e['status'] == 'ok']
    fracs = np.asarray([e['valid_frac'] for e in ok], dtype=np.float64)
    cause_totals: Dict[str, int] = {}
    for e in ok:
        for code, count in e['cause'].items():
            cause_totals[code] = cause_totals.get(code, 0) + int(count)
    manifest = {
        'key': cache_key(params),
        'params': params,
        'n_sessions': len(sessions),
        'n_ok': len(ok),
        'n_skipped': sum(1 for e in entries if e['status'] == 'skipped'),
        'n_missing': sum(1 for e in entries if e['status'] == 'missing'),
        'n_zero_valid': sum(1 for e in ok if e['valid_frac'] == 0.0),
        'total_bytes': sum(int(e.get('bytes') or 0) for e in entries),
        'valid_frac': ({
            'mean': float(fracs.mean()), 'median': float(np.median(fracs)),
            'min': float(fracs.min()), 'max': float(fracs.max()),
        } if fracs.size else {'mean': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0}),
        'cause_totals': {rpp.CAUSE_NAMES.get(int(k), str(k)): v
                         for k, v in sorted(cause_totals.items())},
        'entries': entries,
    }
    tmp = root / (MANIFEST_NAME + '.tmp')
    with open(tmp, 'w') as fh:
        json.dump(manifest, fh, indent=2)
    os.replace(tmp, root / MANIFEST_NAME)
    return manifest
