"""Phase 3 gates — artefact identity, shard IO and the builder's contract.

Gate map: G0 (automated schema), G5 (no consumer choice in the artefact),
G6 (nominal source rate only), G7 (nothing ruled unusable), G10 (loud mismatch).
"""
import json
import os

import numpy as np
import pytest

from data import resp_cache as respc
from data import resp_preprocess as rpp

#: The canonical identity of ADR §4.1 — frozen here ON PURPOSE (plan G0). Adding,
#: renaming or removing a key field must fail this test until §4.1 is updated too.
CANONICAL_KEYS = {
    'format_version', 'corpus', 'fs_in', 'fs_out', 'sat_v', 'sat_min_s',
    'flat_window_s', 'flat_ptp_v', 'min_run_s', 'resample', 'store',
}

#: Fields that select or rescale a window. They may NEVER appear in the identity
#: (ADR §4.2) — that absence is what lets one artefact serve every hop.
CONSUMER_FIELDS = {
    'clip_seconds', 'clip_stride', 'stride', 'clip_frames', 'task_set', 'tasks',
    'task_groups', 'split', 'fold', 'n_folds', 'physio_norm', 'target_norm',
    'input_size', 'max_entries', 'temporal_stride', 'roi_padding', 'crop_rule',
}


def corpus(n: int = 3):
    return [{'session': f'F{i:03d}_T1', 'subject': f'F{i:03d}', 'task': 'T1'}
            for i in range(1, n + 1)]


def make_params(**kw):
    return respc.build_params('raw', corpus(), **kw)


def write_resp_file(path, values):
    with open(path, 'w') as fh:
        for v in values:
            fh.write(f'{v}\n')


# --------------------------------------------------------------------------- #
# G0 — the identity is the ADR's field set, no more and no less
# --------------------------------------------------------------------------- #
def test_identity_schema_is_exact():
    params = make_params()
    assert set(params) == CANONICAL_KEYS, (
        'build_params() must return exactly the ADR §4.1 identity; '
        f'unexpected {sorted(set(params) - CANONICAL_KEYS)}, '
        f'missing {sorted(CANONICAL_KEYS - set(params))}')
    assert set(params['resample']) == {'kind', 'taps_per_phase', 'up', 'down',
                                       'kernel_radius_src'}
    assert set(params['store']) == {'kind', 'dtype'}


def test_no_consumer_field_enters_the_identity():
    params = make_params()
    flat = set(params) | set(params['resample']) | set(params['store'])
    assert not (flat & CONSUMER_FIELDS), (
        f'{sorted(flat & CONSUMER_FIELDS)} must never enter the artefact '
        '(ADR §4.2 / Criterion 5)')
    body = json.dumps(params)
    for field in CONSUMER_FIELDS:
        assert field not in body, f'{field} leaked into params.json'


def test_the_key_changes_iff_a_key_field_changes():
    base = make_params()
    base_key = respc.cache_key(base)
    assert respc.cache_key(make_params()) == base_key, 'the key must be stable'

    mutations = {
        'fs_out': dict(fs_out=50),
        'sat_v': dict(sat_v=9.0),
        'sat_min_s': dict(sat_min_s=2.0),
        'flat_window_s': dict(flat_window_s=4.0),
        'flat_ptp_v': dict(flat_ptp_v=0.4),
        'min_run_s': dict(min_run_s=9.0),
        'taps_per_phase': dict(taps_per_phase=12),
    }
    for name, kw in mutations.items():
        assert respc.cache_key(make_params(**kw)) != base_key, (
            f'{name} changes stored bytes, so it must change the key (ADR I9)')

    assert respc.cache_key(
        respc.build_params('raw', corpus(4))) != base_key, \
        'the corpus digest is part of the identity'


def test_fs_in_is_nominal_and_no_measured_rate_is_recorded():
    params = make_params()
    assert params['fs_in'] == 1000
    assert 'measured' not in json.dumps(params).lower(), (
        'the measured source rate must not be recorded in the artefact '
        '(ADR §2.3); it belongs in the evidence note only')


def test_geometry_refusals_reach_the_builder():
    with pytest.raises(ValueError, match='must divide'):
        make_params(fs_out=128)
    with pytest.raises(ValueError, match='taps_per_phase'):
        make_params(taps_per_phase=4)
    with pytest.raises(ValueError, match='store dtype'):
        make_params(dtype='float64')


# --------------------------------------------------------------------------- #
# lookup — fail loudly, never adapt
# --------------------------------------------------------------------------- #
def test_find_cache_reports_a_field_by_field_diff(tmp_path):
    """A mismatched artefact is REFUSED with the differing field named.

    A present-but-wrong artefact is a ``ValueError`` (mismatch); a missing one is
    a ``FileNotFoundError`` (absence). The two must not be confused, because only
    the first one means "you asked for a different grid".
    """
    want = make_params()
    other = make_params(fs_out=50)
    root = respc.cache_root(tmp_path, respc.cache_key(other))
    respc.write_params(root, other)
    with pytest.raises(ValueError) as exc:
        respc.find_cache(tmp_path, want)
    msg = str(exc.value)
    assert 'fs_out' in msg and 'Nearest' in msg, msg


def test_find_cache_refuses_an_ambiguous_match(tmp_path):
    want = make_params()
    for name in ('a', 'b'):
        d = tmp_path / name
        d.mkdir()
        respc.write_params(d, want)
    with pytest.raises(ValueError, match='ambiguous'):
        respc.find_cache(tmp_path, want)


def test_find_cache_returns_the_single_match(tmp_path):
    want = make_params()
    root = respc.cache_root(tmp_path, respc.cache_key(want))
    respc.write_params(root, want)
    assert respc.find_cache(tmp_path, want) == root


def test_check_params_compatible_names_the_difference(tmp_path):
    root = respc.cache_root(tmp_path, 'x')
    respc.write_params(root, make_params())
    respc.check_params_compatible(root, make_params())          # no raise
    with pytest.raises(ValueError, match='min_run_s'):
        respc.check_params_compatible(root, make_params(min_run_s=12.0))


# --------------------------------------------------------------------------- #
# shard IO
# --------------------------------------------------------------------------- #
def _payload(session='F001_T1', invalid=False):
    x = np.sin(2 * np.pi * 0.3 * np.arange(20000) / 1000.0) + 4.5
    x[5000:6500] = -10.0
    if invalid:
        x = np.full(20000, -10.0)
    out = rpp.preprocess_session(x, fs_in=1000.0, fs_out=100.0)
    return {
        'signal': out['signal'], 'mask': out['mask'], 'cause': out['cause'],
        'n_src': np.int64(out['n_src']), 'fs_in': np.int64(out['fs_in']),
        'fs_out': np.int64(out['fs_out']), 'n_runs': np.int64(out['n_runs']),
        'valid_frac': np.float64(out['valid_frac']),
        'n_nonfinite': np.int64(out['n_nonfinite']),
        'session': session, 'resp_file': '/raw/Physiology/F001/T1/Resp_Volts.txt',
    }


def test_shard_round_trip_is_lossless_and_leaves_no_temp(tmp_path):
    payload = _payload()
    path = respc.write_shard(tmp_path, 'F001_T1', payload)
    assert path.is_file()
    assert list((tmp_path / respc.SUBDIR_TMP).iterdir()) == [], \
        'the staging file must be replaced, never left behind'
    shard = respc.read_shard(path)
    np.testing.assert_array_equal(shard['signal'], payload['signal'])
    np.testing.assert_array_equal(shard['mask'], payload['mask'])
    np.testing.assert_array_equal(shard['cause'], payload['cause'])
    assert shard['n_src'] == 20000 and shard['fs_out'] == 100
    assert shard['session'] == 'F001_T1'


def test_read_shard_refuses_a_zero_filled_hole(tmp_path):
    """A corrupted shard must fail loudly, not become a training target."""
    payload = _payload()
    bad = dict(payload)
    bad['signal'] = np.nan_to_num(payload['signal'], nan=0.0)
    path = tmp_path / 'bad.npz'
    with open(path, 'wb') as fh:
        np.savez(fh, **bad)
    with pytest.raises(ValueError, match='isnan'):
        respc.read_shard(path)


def test_read_shard_refuses_mask_cause_disagreement(tmp_path):
    payload = _payload()
    bad = dict(payload)
    bad['cause'] = np.zeros_like(payload['cause'])
    path = tmp_path / 'bad.npz'
    with open(path, 'wb') as fh:
        np.savez(fh, **bad)
    with pytest.raises(ValueError, match='mask'):
        respc.read_shard(path)


def test_write_skip_records_the_reason(tmp_path):
    path = respc.write_skip(tmp_path, 'F001_T1', 'missing_resp_volts')
    with open(path) as fh:
        assert json.load(fh) == {'session': 'F001_T1',
                                'reason': 'missing_resp_volts'}


# --------------------------------------------------------------------------- #
# G7 — the build never rules a session unusable
# --------------------------------------------------------------------------- #
def test_build_task_reports_the_three_narrow_reasons(tmp_path):
    params = make_params()
    assert respc.build_task({'session': 'X', 'resp_file': None}, params) == \
        (None, 'missing_resp_volts')
    missing = tmp_path / 'nope.txt'
    assert respc.build_task({'session': 'X', 'resp_file': str(missing)}, params) \
        == (None, 'missing_resp_volts')
    empty = tmp_path / 'empty.txt'
    empty.write_text('')
    assert respc.build_task({'session': 'X', 'resp_file': str(empty)}, params) == \
        (None, 'empty_resp_volts')
    assert set(respc.SKIP_REASONS) == {'missing_resp_volts',
                                       'unreadable_resp_volts',
                                       'empty_resp_volts'}


def test_an_all_invalid_session_still_produces_a_shard(tmp_path):
    """ADR Criterion 7: only a CONSUMER may call a session unusable."""
    params = make_params()
    path = tmp_path / 'Resp_Volts.txt'
    write_resp_file(path, [-10.0] * 20000)
    payload, reason = respc.build_task(
        {'session': 'F001_T1', 'resp_file': str(path)}, params)
    assert reason is None and payload is not None
    assert float(payload['valid_frac']) == 0.0
    assert np.isnan(payload['signal']).all()
    respc.write_shard(tmp_path / 'cache', 'F001_T1', payload)
    shard = respc.read_shard(
        respc.shard_path(tmp_path / 'cache', 'F001_T1'))
    assert shard['valid_frac'] == 0.0


def test_build_task_writes_float32_signal_and_uint8_cause(tmp_path):
    params = make_params()
    path = tmp_path / 'Resp_Volts.txt'
    values = list(np.sin(2 * np.pi * 0.3 * np.arange(20000) / 1000.0) + 4.5)
    values[5000:6500] = [-10.0] * 1500
    write_resp_file(path, values)
    payload, reason = respc.build_task(
        {'session': 'F002_T1', 'resp_file': str(path)}, params)
    assert reason is None
    assert payload['signal'].dtype == np.float32
    assert payload['cause'].dtype == np.uint8
    assert payload['mask'].dtype == np.bool_
    assert payload['fs_in'] == 1000 and payload['fs_out'] == 100
    assert payload['n_runs'] >= 1


# --------------------------------------------------------------------------- #
# filesystem truth
# --------------------------------------------------------------------------- #
def test_scan_and_manifest_split_cached_skipped_missing(tmp_path):
    sessions = corpus(3)
    params = respc.build_params('raw', sessions)
    root = respc.cache_root(tmp_path, respc.cache_key(params))
    respc.write_params(root, params)
    respc.write_shard(root, 'F001_T1', _payload('F001_T1'))
    respc.write_skip(root, 'F002_T1', 'missing_resp_volts')

    info = respc.scan(root, sessions)
    assert (info['cached'], info['skipped'], info['missing']) == (1, 1, 1)

    manifest = respc.build_manifest(root, sessions, params)
    assert manifest['n_ok'] == 1 and manifest['n_skipped'] == 1
    assert manifest['n_missing'] == 1
    assert manifest['n_zero_valid'] == 0
    assert manifest['entries'][0]['status'] == 'ok'
    assert (root / respc.MANIFEST_NAME).is_file()


def test_manifest_counts_wholly_invalid_sessions(tmp_path):
    sessions = corpus(2)
    params = respc.build_params('raw', sessions)
    root = respc.cache_root(tmp_path, respc.cache_key(params))
    respc.write_params(root, params)
    respc.write_shard(root, 'F001_T1', _payload('F001_T1', invalid=True))
    manifest = respc.build_manifest(root, sessions, params)
    assert manifest['n_zero_valid'] == 1
    assert manifest['valid_frac']['min'] == 0.0


# --------------------------------------------------------------------------- #
# CLI composition — the documented recipe is `--status --index`
# --------------------------------------------------------------------------- #
def _tiny_corpus(tmp_path):
    raw = tmp_path / 'raw'
    (raw / 'Thermal' / 'S001').mkdir(parents=True)
    (raw / 'Thermal' / 'S001' / 'S001_T1.wmv').write_bytes(b'\x00')
    ph = raw / 'Physiology' / 'S001' / 'T1'
    ph.mkdir(parents=True)
    vals = np.sin(2 * np.pi * 0.3 * np.arange(12000) / 1000.0) + 4.5
    (ph / 'Resp_Volts.txt').write_text(
        '\n'.join(f'{v:.5f}' for v in vals) + '\n')
    return raw


def test_status_and_index_both_run_from_one_invocation(tmp_path):
    """`--status --index` must do BOTH reports, not just the first."""
    from data import tir_resp_dataset as trd
    from runners import run_build_resp_cache as rb

    raw = _tiny_corpus(tmp_path)
    out = tmp_path / 'cache'
    assert rb.main(['--init', '--build', '--out_root', str(out),
                    '--raw_root', str(raw)]) == 0
    root = respc.cache_root(
        out, respc.cache_key(respc.build_params(
            str(raw), trd.discover_sessions(str(raw)))))
    assert not (root / respc.MANIFEST_NAME).is_file()
    assert rb.main(['--status', '--index', '--out_root', str(out),
                    '--raw_root', str(raw)]) == 0
    assert (root / respc.MANIFEST_NAME).is_file(), \
        '--index was skipped because --status returned first'


def test_init_alone_writes_params_and_no_shard(tmp_path):
    from data import tir_resp_dataset as trd
    from runners import run_build_resp_cache as rb

    raw = _tiny_corpus(tmp_path)
    out = tmp_path / 'cache'
    assert rb.main(['--init', '--out_root', str(out), '--raw_root', str(raw)]) == 0
    root = respc.cache_root(
        out, respc.cache_key(respc.build_params(
            str(raw), trd.discover_sessions(str(raw)))))
    assert (root / respc.PARAMS_NAME).is_file()
    assert not (root / respc.SUBDIR_TASKS).exists(), '--init must not build'
