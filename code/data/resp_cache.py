"""Offline respiration-grid cache: key, params, shard IO, verification.

Contract of record: ``code/notes/design/RESP_session_preprocess.md``.
Build order: ``code/notes/plan/RESP_session_preprocess_implementation_plan.md``.

**The cache is ONE STEP of the pipeline and encodes no downstream decision.**
It answers "which volts are trustworthy at which instant" faster than re-reading
``Resp_Volts.txt`` every epoch, and nothing else (design §4 D2). That principle
produces every rule here:

* :func:`cache_key` covers ONLY what changes the STORED SAMPLES -- the source
  and output rates, the four validity thresholds, the resampler shape, the mask
  rule, the dtype and the corpus identity.
* Everything that only SELECTS WINDOWS (``clip_seconds``, ``clip_stride``,
  ``task_set``/split) or RESCALES (``physio_norm``, ``input_size``) is
  deliberately absent from the key AND from the writer. That is exactly what
  lets ONE cache serve Stage 2 (2.0 s hop) and Stage 3 (1.0 s hop).
* The build never skips a session because of its LABEL: every session with a
  readable ``Resp_Volts.txt`` gets a shard, even an all-invalid one (design §3,
  plan C7). Only a consumer may rule a session unusable.
* The two caches (ROI pixels, respiration grid) have DISJOINT key fields and
  INDEPENDENT format versions (D9), so rebuilding one never forces the other.

Layout (mirrors :mod:`data.roi_cache`)::

    <out_root>/<CACHE_KEY>/
        params.json            # the frozen parameters (--init)
        tasks/F001_T1.npz      # THE SOURCE OF TRUTH (~0.3 MB mean)
        skipped/F082_T1.json   # {"session", "reason"} -- missing/unreadable only
        manifest.json          # DERIVED index (--index)

Shards, not the manifest, are authoritative: an array job writes only its own
disjoint set of shards, and ``--index`` rebuilds the manifest by scanning. Every
shard appears via ``os.replace``, so a killed job cannot leave a half-written one
behind.

Shard contents (``np.savez``, uncompressed -- 65 k float32 is 260 kB):

* ``signal``  float32 ``[N_out]`` run-mean-removed volts; ``NaN`` exactly where
  ``mask`` is False
* ``mask``    bool ``[N_out]``    ``True`` = usable
* ``cause``   uint8 ``[N_out]``   0 valid, 1 saturation, 2 flatline, 3 short run,
  4 non-finite -- restores the per-rule accounting the dataset prints
* ``n_src`` / ``fs_in`` / ``fs_out`` -- grid identity, self-verifying
* ``n_runs`` / ``valid_frac`` / ``n_nonfinite`` -- build statistics
* ``session`` / ``resp_file`` -- provenance
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import resp_preprocess as rpp                        # noqa: E402
from data import roi_cache as rc                               # noqa: E402
from data import tir_resp_dataset as trd                       # noqa: E402

#: Bump when a change alters the STORED SAMPLES or the shard schema.
#:
#: Independent of ``roi_cache.CACHE_FORMAT_VERSION`` ON PURPOSE (design D9): the
#: two caches are orthogonal and a rebuild of one must not invalidate the other.
#:
#: v1 -> v2: the per-run resample length is now derived from the SESSION GRID
#: (``floor((n_src-1)*fs_out/fs_in)+1``) instead of ``round(len*fs_out/fs_in)``
#: per run, so a run reaching EOF fills the session's last grid sample instead
#: of leaving it ``NaN``. Same length, one sample changed -> v1 shards must not
#: be mixed with v2 ones.
RESP_CACHE_FORMAT_VERSION = 2

#: Bump when the MASK rule changes meaning; part of the key.
MASK_RULE = rpp.MASK_RULE
RESAMPLE_RULE = rpp.RESAMPLE_RULE

STORE_RULE = 'grid+mask'
STORE_DTYPE = 'float32'

SUBDIR_TASKS = 'tasks'
SUBDIR_SKIPPED = 'skipped'
SUBDIR_TMP = '.tmp'
SHARD_SUFFIX = '.npz'
PARAMS_NAME = 'params.json'
MANIFEST_NAME = 'manifest.json'


# --------------------------------------------------------------------------- #
# cache identity
# --------------------------------------------------------------------------- #
def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def corpus_digest(sessions: Sequence[dict]) -> str:
    """Corpus identity -- the SAME digest the ROI cache uses (one digest, D9).

    Imported rather than copied so the two caches can never disagree about which
    sessions they were built from.
    """
    return rc.corpus_digest(sessions)


def build_params(raw_root: str, sessions: Sequence[dict],
                 fs_in: float = rpp.FS_IN, fs_out: float = rpp.FS_OUT,
                 sat_v: float = rpp.SAT_V, sat_min_s: float = rpp.SAT_MIN_S,
                 flat_window_s: float = rpp.FLAT_WINDOW_S,
                 flat_ptp_v: float = rpp.FLAT_PTP_V,
                 min_run_s: float = rpp.MIN_RUN_S,
                 half_taps: int = rpp.RESAMPLE_HALF_TAPS,
                 dtype: str = 'float32') -> Dict:
    """The FROZEN parameter set. Everything here changes the stored samples."""
    fi, fo = rpp._rates_integral(fs_in, fs_out)
    return {
        'format_version': RESP_CACHE_FORMAT_VERSION,
        'raw_root': str(raw_root),
        'corpus': corpus_digest(sessions),
        'n_sessions_discovered': len(sessions),
        'fs_in': int(fi),
        'fs_out': int(fo),
        'sat_v': float(sat_v),
        'sat_min_s': float(sat_min_s),
        'flat_window_s': float(flat_window_s),
        'flat_ptp_v': float(flat_ptp_v),
        'min_run_s': float(min_run_s),
        'resample': RESAMPLE_RULE,
        'half_taps': int(half_taps),
        'mask_rule': MASK_RULE,
        'dtype': str(dtype),
        'store': STORE_RULE,
    }


def cache_key(params: Dict) -> str:
    """Readable slug + content hash. The hash is over EVERY frozen field."""
    body = json.dumps(params, sort_keys=True, default=str)
    slug = (f'respgrid_src{int(params["fs_in"])}'
            f'_fs{int(params["fs_out"])}'
            f'_sat{float(params["sat_v"]):g}-{float(params["sat_min_s"]):g}s'
            f'_flat{float(params["flat_ptp_v"]):g}'
            f'-{float(params["flat_window_s"]):g}s'
            f'_run{float(params["min_run_s"]):g}s'
            f'_v{params["format_version"]}')
    return f'{slug}_{_sha1(body)[:10]}'


def cache_root(out_root: str, key: str) -> Path:
    return Path(out_root) / key


def write_params(root: Path, params: Dict) -> None:
    """Write ``params.json`` atomically. Concurrent identical writes are safe."""
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / (PARAMS_NAME + '.tmp')
    with open(tmp, 'w') as fh:
        json.dump(params, fh, indent=2, sort_keys=True)
    os.replace(tmp, root / PARAMS_NAME)


def read_params(root: Path) -> Dict:
    with open(Path(root) / PARAMS_NAME) as fh:
        return json.load(fh)


#: The fields a LOADER can verify without reading the raw signal. Every one of
#: them changes a stored sample, so a mismatch means the cache is wrong for the
#: run (design G6) -- never "close enough".
COMPARE_KEYS = ('format_version', 'corpus', 'fs_in', 'fs_out', 'sat_v',
                'sat_min_s', 'flat_window_s', 'flat_ptp_v', 'min_run_s',
                'resample', 'half_taps', 'mask_rule', 'dtype', 'store')

#: Fields an EVALUATOR can verify without knowing the training corpus: the whole
#: GRID CONTRACT minus ``corpus``. An evaluation dump may legitimately be scored
#: against a cache built for a different session set (the intersection with the
#: dump is what matters), while the grid identity MUST match or the reference and
#: the label would come from two different preprocessings (design C8).
EVAL_COMPARE_KEYS = tuple(k for k in COMPARE_KEYS if k != 'corpus')


def loader_params(raw_root: str, corpus_sessions: Sequence[dict],
                  fs_in: float = rpp.FS_IN, fs_out: float = rpp.FS_OUT,
                  sat_v: float = rpp.SAT_V, sat_min_s: float = rpp.SAT_MIN_S,
                  flat_window_s: float = rpp.FLAT_WINDOW_S,
                  flat_ptp_v: float = rpp.FLAT_PTP_V,
                  min_run_s: float = rpp.MIN_RUN_S,
                  half_taps: int = rpp.RESAMPLE_HALF_TAPS,
                  dtype: str = 'float32') -> Dict:
    """The subset of the frozen parameters a consumer can compute on its own.

    ``corpus_sessions`` MUST be the UNFILTERED discovery: the cache key is a
    property of the CORPUS, not of the subset a given run trains on.
    """
    full = build_params(raw_root, corpus_sessions, fs_in=fs_in, fs_out=fs_out,
                        sat_v=sat_v, sat_min_s=sat_min_s,
                        flat_window_s=flat_window_s, flat_ptp_v=flat_ptp_v,
                        min_run_s=min_run_s, half_taps=half_taps, dtype=dtype)
    return {k: full[k] for k in COMPARE_KEYS}


def eval_params(fs_in: float = rpp.FS_IN, fs_out: float = rpp.FS_OUT,
                sat_v: float = rpp.SAT_V, sat_min_s: float = rpp.SAT_MIN_S,
                flat_window_s: float = rpp.FLAT_WINDOW_S,
                flat_ptp_v: float = rpp.FLAT_PTP_V,
                min_run_s: float = rpp.MIN_RUN_S,
                half_taps: int = rpp.RESAMPLE_HALF_TAPS,
                dtype: str = STORE_DTYPE) -> Dict:
    """The GRID CONTRACT without the corpus digest, for evaluators.

    An evaluation dump names its own sessions, so the corpus is not part of what
    it can verify; every OTHER frozen field still must agree, because a different
    ``fs_out`` / ``sat_min_s`` means a different label space (design C8/G6).
    """
    full = build_params('', [], fs_in=fs_in, fs_out=fs_out, sat_v=sat_v,
                        sat_min_s=sat_min_s, flat_window_s=flat_window_s,
                        flat_ptp_v=flat_ptp_v, min_run_s=min_run_s,
                        half_taps=half_taps, dtype=dtype)
    return {k: full[k] for k in EVAL_COMPARE_KEYS}


def find_cache(out_root, want: Dict, keys: Sequence[str] = COMPARE_KEYS,
               require_session: Optional[str] = None) -> Path:
    """Locate the cache directory whose parameters match ``want``.

    THIS IS THE FAIL-LOUDLY GATE (plan G6). A run whose ``fs_out`` / ``sat_min_s``
    / corpus differ must REFUSE the cache rather than silently read a different
    label space: silent data drift is the one failure mode that invalidates every
    downstream comparison while looking healthy.

    :param keys: which frozen fields to compare. The DATASET uses the full
        :data:`COMPARE_KEYS` (corpus included); an EVALUATOR uses
        :data:`EVAL_COMPARE_KEYS` because it does not know the training corpus.
    :param require_session: when given, keep only directories that actually hold
        ``tasks/<session>.npz``. That is how an evaluation picks the right cache
        out of several built for different session sets instead of guessing.

    Raises with a field-by-field diff when nothing matches, and refuses an
    ambiguous match rather than picking one.
    """
    keys = tuple(keys)
    out_root = Path(out_root)
    if not out_root.is_dir():
        raise FileNotFoundError(
            f'resp_cache: out_root does not exist: {out_root}. Build it with '
            f'`runners/run_build_resp_cache.py --init --build`.')
    matches: List[Path] = []
    near: List[Tuple[Path, Dict]] = []
    for d in sorted(out_root.iterdir()):
        pf = d / PARAMS_NAME
        if not (d.is_dir() and pf.is_file()):
            continue
        try:
            with open(pf) as fh:
                have = json.load(fh)
        except Exception:
            continue
        diffs = {k: (have.get(k), want.get(k)) for k in keys
                 if have.get(k) != want.get(k)}
        if diffs:
            near.append((d, diffs))
        else:
            matches.append(d)
    if require_session:
        with_shard = [d for d in matches
                      if shard_path(d, require_session).is_file()]
        if not with_shard:
            raise FileNotFoundError(
                f'resp_cache: {len(matches)} cache(s) under {out_root} match '
                f'the grid contract but NONE holds a shard for '
                f'{require_session} (the build does not cover it, or it was '
                f'skipped as unreadable).')
        matches = with_shard
    if len(matches) == 1:
        return matches[0]
    if not matches:
        lines = [f'resp_cache: no cache under {out_root} matches this run. '
                 f'Wanted (key fields):']
        lines += [f'    {k} = {want[k]!r}' for k in keys]
        if near:
            lines.append('  Closest existing cache(es):')
            for d, diffs in near[:3]:
                lines.append(f'    {d.name}')
                for k, (h, w) in diffs.items():
                    lines.append(f'        {k}: cache={h!r} run={w!r}')
        else:
            lines.append('  (the directory holds no cache at all)')
        raise ValueError('\n'.join(lines))
    names = ', '.join(m.name for m in matches)
    raise ValueError(
        f'resp_cache: {len(matches)} caches under {out_root} match this run '
        f'({names}); refusing to guess. Pass --resp_cache_key to pin one.')


def check_params_compatible(root: Path, params: Dict) -> None:
    """FAIL LOUDLY if the on-disk cache was built with different parameters."""
    have = read_params(root)
    if have == params:
        return
    diffs = [f'{k}: cache={have.get(k)!r} run={params.get(k)!r}'
             for k in sorted(set(have) | set(params))
             if have.get(k) != params.get(k)]
    raise ValueError(
        f'resp_cache: {root} was built with DIFFERENT parameters:\n  '
        + '\n  '.join(diffs)
        + '\nRefusing to mix shards. Use a different --out_root or delete it.')


# --------------------------------------------------------------------------- #
# build one session
# --------------------------------------------------------------------------- #
def build_task(spec: dict, params: Dict
               ) -> Tuple[Optional[Dict], Optional[str]]:
    """Preprocess ONE subject-task and return its shard payload.

    Returns ``(payload, None)`` on success or ``(None, reason)`` when the task
    has no usable ``Resp_Volts.txt``. The three reasons are deliberately narrow
    -- ``missing_resp_volts``, ``unreadable_resp_volts``, ``empty_resp_volts``
    -- because :func:`data.resp_preprocess.preprocess_session` NEVER rules a
    session unusable (design §3): an all-invalid session is still a shard with
    ``valid_frac == 0``.
    """
    session = str(spec['session'])
    resp_file = spec.get('resp_file')
    if not resp_file or not os.path.isfile(resp_file):
        return None, 'missing_resp_volts'
    try:
        x = trd._load_1d(str(resp_file))
    except Exception as exc:                        # unreadable / malformed
        return None, f'unreadable_resp_volts: {exc}'
    if x.size == 0:
        return None, 'empty_resp_volts'

    out = rpp.preprocess_session(
        x, fs_in=params['fs_in'], fs_out=params['fs_out'],
        sat_v=params['sat_v'], sat_min_s=params['sat_min_s'],
        flat_window_s=params['flat_window_s'], flat_ptp_v=params['flat_ptp_v'],
        min_run_s=params['min_run_s'], half_taps=params['half_taps'],
        dtype=params['dtype'])
    payload = {
        'signal': out['signal'],
        'mask': np.asarray(out['mask'], dtype=bool),
        'cause': np.asarray(out['cause'], dtype=np.uint8),
        'n_src': int(out['n_src']),
        'fs_in': int(out['fs_in']),
        'fs_out': int(out['fs_out']),
        'n_runs': int(out['n_runs']),
        'valid_frac': float(out['valid_frac']),
        'n_nonfinite': int(out['n_nonfinite']),
        'session': session,
        'resp_file': str(resp_file),
    }
    return payload, None


# --------------------------------------------------------------------------- #
# shard IO
# --------------------------------------------------------------------------- #
def _assert_nan_matches_mask(signal: np.ndarray, mask: np.ndarray,
                             where: str) -> None:
    """Design D4: invalid samples are ``NaN``, never 0, on BOTH sides."""
    sig = np.asarray(signal)
    m = np.asarray(mask, dtype=bool)
    if sig.shape != m.shape:
        raise ValueError(f'resp_cache: {where}: signal {sig.shape} != mask '
                         f'{m.shape}')
    if not np.array_equal(np.isnan(sig), ~m):
        raise ValueError(
            f'resp_cache: {where}: signal/mask mismatch -- invalid samples must '
            f'be NaN and every NaN must be invalid')


def write_shard(root: Path, session: str, payload: Dict,
                compress: bool = False) -> Path:
    """Write one shard atomically (staging dir + ``os.replace``).

    The staging name carries the PID, so two processes can never share it.
    """
    _assert_nan_matches_mask(payload['signal'], payload['mask'],
                             f'{session} (writer)')
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
    """Record WHY a session was not cached (missing / unreadable only).

    Creates ``.tmp`` itself, exactly like :func:`data.roi_cache.write_skip`: the
    first task of the first array element can legitimately be a skip.
    """
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
    """Load one shard. Arrays are materialised before the archive closes."""
    with np.load(str(path), allow_pickle=False) as z:
        signal = z['signal']
        mask = np.asarray(z['mask'], dtype=bool)
        out = {
            'signal': np.asarray(signal),
            'mask': mask,
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
    _assert_nan_matches_mask(out['signal'], out['mask'],
                             f'{out["session"]} (loader)')
    return out


def shard_path(root: Path, session: str) -> Path:
    return Path(root) / SUBDIR_TASKS / f'{session}{SHARD_SUFFIX}'


def scan(root: Path, sessions: Sequence[dict]) -> Dict:
    """Filesystem truth: which sessions are cached, skipped, or still missing."""
    root = Path(root)
    cached, skipped, missing, n_bytes = [], [], [], 0
    for s in sessions:
        name = str(s['session'])
        p = shard_path(root, name)
        if p.is_file():
            cached.append(name)
            n_bytes += p.stat().st_size
            continue
        sp = root / SUBDIR_SKIPPED / f'{name}.json'
        if sp.is_file():
            skipped.append(name)
            continue
        missing.append(name)
    return {'cached': len(cached), 'skipped': len(skipped),
            'missing': len(missing), 'bytes': n_bytes,
            'cached_names': cached, 'skipped_names': skipped,
            'missing_names': missing}


def cause_histogram(cause: np.ndarray) -> Dict[str, int]:
    """Invalid-sample share per rule, for the manifest (plan Step 9)."""
    c = np.asarray(cause, dtype=np.uint8).ravel()
    return {name: int((c == code).sum())
            for code, name in sorted(rpp.CAUSE_NAMES.items())}


def build_manifest(root: Path, sessions: Sequence[dict], params: Dict) -> Dict:
    """Derived index. Reads only the SMALL members of each shard."""
    root = Path(root)
    entries: List[Dict] = []
    for s in sessions:
        name = str(s['session'])
        p = shard_path(root, name)
        if p.is_file():
            with np.load(str(p), allow_pickle=False) as z:
                sig = z['signal']
                mask = np.asarray(z['mask'], dtype=bool)
                cause = np.asarray(z['cause'], dtype=np.uint8)
                entries.append({
                    'session': name, 'status': 'ok',
                    'bytes': p.stat().st_size,
                    'n_out': int(sig.size),
                    'n_src': int(z['n_src']),
                    'n_runs': int(z['n_runs']),
                    'valid_frac': float(z['valid_frac']),
                    'n_nonfinite': int(z['n_nonfinite']),
                    'n_valid': int(mask.sum()),
                    'cause_counts': cause_histogram(cause),
                })
            continue
        sp = root / SUBDIR_SKIPPED / f'{name}.json'
        if sp.is_file():
            with open(sp) as fh:
                entries.append(dict(json.load(fh), status='skipped'))
        else:
            entries.append({'session': name, 'status': 'missing'})
    ok = [e for e in entries if e['status'] == 'ok']
    valid = [float(e['valid_frac']) for e in ok]
    manifest = {
        'key': cache_key(params),
        'params': params,
        'n_sessions': len(sessions),
        'n_ok': len(ok),
        'n_skipped': sum(1 for e in entries if e['status'] == 'skipped'),
        'n_missing': sum(1 for e in entries if e['status'] == 'missing'),
        'total_bytes': sum(int(e.get('bytes') or 0) for e in entries),
        'valid_frac': {'mean': float(np.mean(valid)) if valid else 0.0,
                       'median': float(np.median(valid)) if valid else 0.0,
                       'min': float(np.min(valid)) if valid else 0.0,
                       'max': float(np.max(valid)) if valid else 0.0},
        'cause_totals': {
            name: int(sum(e['cause_counts'][name] for e in ok))
            for name in rpp.CAUSE_NAMES.values()} if ok else {},
        'sessions': entries,
    }
    with open(root / MANIFEST_NAME, 'w') as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


__all__ = [
    'RESP_CACHE_FORMAT_VERSION', 'MASK_RULE', 'RESAMPLE_RULE', 'SUBDIR_TASKS',
    'SUBDIR_SKIPPED', 'SUBDIR_TMP', 'SHARD_SUFFIX', 'PARAMS_NAME',
    'MANIFEST_NAME', 'COMPARE_KEYS', 'EVAL_COMPARE_KEYS', 'corpus_digest',
    'build_params', 'eval_params', 'cache_key', 'cache_root', 'write_params',
    'read_params', 'loader_params', 'find_cache', 'check_params_compatible',
    'build_task', 'write_shard', 'write_skip', 'read_shard', 'shard_path',
    'scan', 'cause_histogram', 'build_manifest',
]
