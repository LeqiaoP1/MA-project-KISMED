"""Unit tests for ``data/resp_cache.py`` + ``runners/run_build_resp_cache.py``.

The cache holds one whole-session signal grid plus a validity mask per
subject-task. These tests pin the key contract (what changes the stored bytes
and what must NOT), the fail-loudly lookup, the atomic shard IO and the
``--check`` parity gate. No video is ever decoded: the builder only reads
``Resp_Volts.txt``.
"""
import os
import sys

import numpy as np
import pytest

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from data import resp_cache as respc
from data import resp_preprocess as rpp
from runners import run_build_resp_cache as rb

FS = 1000.0
OUT = 100.0


def _sine(seconds: float, f: float, amp: float = 1.0, dc: float = 0.0) -> np.ndarray:
    t = np.arange(int(round(seconds * FS))) / FS
    return dc + amp * np.sin(2 * np.pi * f * t)


def _write_raw(root, sessions: dict) -> str:
    """Minimal raw tree. Values are 1-D respiration traces."""
    for name, x in sessions.items():
        subj, task = name.split('_')
        (root / 'Thermal' / subj).mkdir(parents=True, exist_ok=True)
        (root / 'Thermal' / subj / f'{task}.wmv').write_bytes(b'\x00')
        phys = root / 'Physiology' / subj / task
        phys.mkdir(parents=True, exist_ok=True)
        if x is None:
            (phys / 'Resp_Volts.txt').write_text('')
        else:
            (phys / 'Resp_Volts.txt').write_text(
                '\n'.join(f'{v:.5f}' for v in x) + '\n')
    return str(root)


@pytest.fixture
def corpus(tmp_path):
    raw = tmp_path / 'raw'
    _write_raw(raw, {
        'S001_T1': _sine(20.0, 0.3, amp=2.0, dc=3.0),
        'S001_T2': _sine(20.0, 0.4, amp=1.0, dc=-2.0),
        'S002_T1': np.full(20000, 5.0),          # all-invalid (flatline)
    })
    return raw


def _params(raw, **kw):
    sessions = rb.trd.discover_sessions(str(raw))
    return respc.build_params(str(raw), sessions, **kw)


# --------------------------------------------------------------------------- #
# key contract
# --------------------------------------------------------------------------- #
def test_key_is_stable_and_changes_with_every_frozen_field(corpus):
    base = _params(corpus)
    assert respc.cache_key(_params(corpus)) == respc.cache_key(base)
    changes = {
        'fs_out': 50.0,
        'fs_in': 500.0,
        'sat_v': 9.5,
        'sat_min_s': 3.0,
        'flat_window_s': 4.0,
        'flat_ptp_v': 0.06,
        'min_run_s': 6.0,
        'half_taps': 10,
    }
    for field, value in changes.items():
        assert respc.cache_key(_params(corpus, **{field: value})) != \
            respc.cache_key(base), f'{field} did not change the key'


def test_key_does_not_depend_on_downstream_knobs(corpus):
    """``clip_seconds`` / ``clip_stride`` / ``physio_norm`` are NOT parameters.

    That is the property that lets ONE artefact serve Stage 2 and Stage 3.
    """
    p = _params(corpus)
    for forbidden in ('clip_seconds', 'clip_stride', 'physio_norm',
                      'input_size', 'task_set', 'tasks', 'max_entries'):
        assert forbidden not in p
    # and the loader half is a strict subset of the writer half
    sessions = rb.trd.discover_sessions(str(corpus))
    loader = respc.loader_params(str(corpus), sessions)
    assert set(loader) == set(respc.COMPARE_KEYS)
    assert all(p[k] == v for k, v in loader.items())


def test_key_slug_is_readable(corpus):
    p = _params(corpus)
    key = respc.cache_key(p)
    assert key.startswith('respgrid_src1000_fs100_')
    ver = respc.RESP_CACHE_FORMAT_VERSION
    assert f'_v{ver}_' in key
    assert key.endswith(f'_v{ver}_' + key.rsplit('_', 1)[1])
    assert f"sat{p['sat_v']:g}-{p['sat_min_s']:g}s" in key
    assert f"flat{p['flat_ptp_v']:g}-{p['flat_window_s']:g}s" in key
    assert 'run8s' in key


# --------------------------------------------------------------------------- #
# lookup gate
# --------------------------------------------------------------------------- #
def test_find_cache_matches_and_diffs_loudly(corpus, tmp_path):
    out = tmp_path / 'out'
    params = _params(corpus)
    root = respc.cache_root(out, respc.cache_key(params))
    respc.write_params(root, params)

    sessions = rb.trd.discover_sessions(str(corpus))
    want = respc.loader_params(str(corpus), sessions)
    assert respc.find_cache(out, want) == root

    wrong = respc.loader_params(str(corpus), sessions, fs_out=50.0)
    with pytest.raises(ValueError) as exc:
        respc.find_cache(out, wrong)
    msg = str(exc.value)
    assert 'fs_out' in msg and 'cache=100' in msg and 'run=50' in msg

    with pytest.raises(FileNotFoundError):
        respc.find_cache(tmp_path / 'nope', want)


def test_find_cache_refuses_an_ambiguous_match(corpus, tmp_path):
    out = tmp_path / 'out'
    params = _params(corpus)
    for name in ('a', 'b'):
        respc.write_params(out / name, params)
    sessions = rb.trd.discover_sessions(str(corpus))
    want = respc.loader_params(str(corpus), sessions)
    with pytest.raises(ValueError) as exc:
        respc.find_cache(out, want)
    assert 'refusing to guess' in str(exc.value)


def test_check_params_compatible_raises_on_drift(corpus, tmp_path):
    root = tmp_path / 'c'
    params = _params(corpus)
    respc.write_params(root, params)
    respc.check_params_compatible(root, params)        # no raise
    with pytest.raises(ValueError):
        respc.check_params_compatible(
            root, _params(corpus, sat_min_s=3.0))


# --------------------------------------------------------------------------- #
# shard IO
# --------------------------------------------------------------------------- #
def test_shard_round_trip_including_cause(corpus, tmp_path):
    params = _params(corpus)
    spec = [s for s in rb.trd.discover_sessions(str(corpus))
            if s['session'] == 'S001_T1'][0]
    payload, reason = respc.build_task(spec, params)
    assert reason is None
    root = tmp_path / 'cache'
    path = respc.write_shard(root, 'S001_T1', payload)
    assert path.is_file()
    sh = respc.read_shard(path)
    assert np.array_equal(sh['signal'], payload['signal'], equal_nan=True)
    assert np.array_equal(sh['mask'], payload['mask'])
    assert np.array_equal(sh['cause'], payload['cause'])
    assert sh['session'] == 'S001_T1'
    assert sh['fs_in'] == 1000 and sh['fs_out'] == 100
    assert sh['n_runs'] == 1
    assert np.array_equal(np.isnan(sh['signal']), ~sh['mask'])


def test_write_shard_leaves_no_staging_file(tmp_path):
    payload = {
        'signal': np.array([1.0, np.nan], np.float32),
        'mask': np.array([True, False]),
        'cause': np.array([0, 2], np.uint8),
        'n_src': 2, 'fs_in': 1000, 'fs_out': 100, 'n_runs': 1,
        'valid_frac': 0.5, 'n_nonfinite': 0,
        'session': 'S001_T1', 'resp_file': '/x/Resp_Volts.txt',
    }
    root = tmp_path / 'cache'
    respc.write_shard(root, 'S001_T1', payload)
    assert list((root / respc.SUBDIR_TMP).glob('*')) == []


def test_write_shard_refuses_zero_filled_invalid_samples(tmp_path):
    bad = {
        'signal': np.array([1.0, 0.0], np.float32),   # should be NaN
        'mask': np.array([True, False]),
        'cause': np.array([0, 2], np.uint8),
        'n_src': 2, 'fs_in': 1000, 'fs_out': 100, 'n_runs': 1,
        'valid_frac': 0.5, 'n_nonfinite': 0,
        'session': 'S001_T1', 'resp_file': '/x/Resp_Volts.txt',
    }
    with pytest.raises(ValueError):
        respc.write_shard(tmp_path / 'cache', 'S001_T1', bad)


def test_write_skip_creates_the_staging_dir(tmp_path):
    root = tmp_path / 'fresh'
    respc.write_skip(root, 'F082_T1', 'missing_resp_volts')
    assert (root / respc.SUBDIR_SKIPPED / 'F082_T1.json').is_file()
    assert list((root / respc.SUBDIR_TMP).glob('*')) == []


# --------------------------------------------------------------------------- #
# build_task reasons + the "never skip on label content" rule
# --------------------------------------------------------------------------- #
def test_build_task_reasons(corpus, tmp_path):
    params = _params(corpus)
    # a spec without a resp file is a data skip, not a programming error
    assert respc.build_task({'session': 'X_T1', 'resp_file': None},
                            params) == (None, 'missing_resp_volts')
    missing = str(tmp_path / 'nope.txt')
    assert respc.build_task({'session': 'X_T1', 'resp_file': missing},
                            params) == (None, 'missing_resp_volts')

    empty = tmp_path / 'empty.txt'
    empty.write_text('')
    assert respc.build_task({'session': 'X_T1', 'resp_file': str(empty)},
                            params) == (None, 'empty_resp_volts')


def test_all_invalid_session_still_gets_a_shard(corpus, tmp_path):
    """Design §3 / plan C7: the build never rules a session unusable."""
    out = tmp_path / 'out'
    assert rb.main(['--init', '--build', '--out_root', str(out),
                    '--raw_root', str(corpus)]) == 0
    params = _params(corpus)
    root = respc.cache_root(out, respc.cache_key(params))
    shard = respc.shard_path(root, 'S002_T1')
    assert shard.is_file()
    sh = respc.read_shard(shard)
    assert sh['valid_frac'] == 0.0
    assert not sh['mask'].any()
    assert np.isnan(sh['signal']).all()
    # ... and only the truly unreadable are recorded as skips
    assert list((root / respc.SUBDIR_SKIPPED).glob('*.json')) == []


def test_build_status_index_and_check_parity(corpus, tmp_path):
    out = tmp_path / 'out'
    common = ['--out_root', str(out), '--raw_root', str(corpus)]
    assert rb.main(['--init'] + common) == 0
    assert rb.main(['--build'] + common) == 0
    assert rb.main(['--status'] + common) == 0
    assert rb.main(['--index'] + common) == 0
    params = _params(corpus)
    root = respc.cache_root(out, respc.cache_key(params))
    import json
    with open(root / respc.MANIFEST_NAME) as fh:
        m = json.load(fh)
    assert m['n_ok'] == 3 and m['n_missing'] == 0
    assert 0.0 <= m['valid_frac']['mean'] <= 1.0
    assert rb.main(['--check', '3'] + common) == 0


def test_scan_and_manifest_accounting(corpus, tmp_path):
    out = tmp_path / 'out'
    common = ['--out_root', str(out), '--raw_root', str(corpus)]
    assert rb.main(['--init', '--build'] + common) == 0
    params = _params(corpus)
    root = respc.cache_root(out, respc.cache_key(params))
    sessions = rb.trd.discover_sessions(str(corpus))
    info = respc.scan(root, sessions)
    assert info['cached'] == 3 and info['missing'] == 0
    m = respc.build_manifest(root, sessions, params)
    assert m['n_ok'] == 3
    # the all-invalid session contributes its flatline samples to the histogram
    assert m['cause_totals']['flatline'] > 0
