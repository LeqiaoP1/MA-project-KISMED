"""Identity, storage and builder tests for the BP artefact (ADR-004 Criterion 8–12).

The identity tests are deliberately rigid: adding, renaming or removing a key
field must fail here until §3.1 of the plan and the shard schema move together.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import bp_synth as bs
from data import bp_artefact as bpa
from data import bp_preprocess as bpp

#: The frozen key set, transcribed from ADR §2.6 / plan §3.1.
CANONICAL_KEYS = {
    'format_version', 'corpus', 'fs_in', 'fs_out', 'gate1_min_mmhg',
    'gate1_max_mmhg', 'flatline_max_ms', 'rail_dwell_min_ms', 'min_segment_s',
    'store',
}

#: Fields that select or rescale a window and may NEVER appear (ADR §2.8).
FORBIDDEN_KEYS = {
    'clip_seconds', 'clip_stride', 'clip_hop', 'physio_norm', 'input_size',
    'max_entries', 'task_set', 'task_groups', 'fold', 'split', 'subject',
    'normalisation', 'norm',
}

CORPUS = [{'session': 'S1_T1', 'source_file': '/raw/Physiology/S1/T1/BP_mmHg.txt'},
          {'session': 'S1_T2', 'source_file': '/raw/Physiology/S1/T2/BP_mmHg.txt'}]


def _params(sessions=None, **over):
    """Params for the frozen corpus, optionally over one or more identity fields."""
    return bpa.build_params(CORPUS if sessions is None else sessions, **over)


def _payload(**over):
    base = dict(
        signal=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        mask=np.array([True, True, True, True]),
        cause=np.zeros(4, dtype=np.uint8),
        seg_start=np.array([0], dtype=np.int64),
        seg_stop=np.array([4], dtype=np.int64),
        seg_mean_mmhg=np.array([90.0], dtype=np.float64),
        seg_seconds=np.array([0.004], dtype=np.float64),
        n_src=np.int64(4), fs_in=np.int64(1000), fs_out=np.int64(1000),
        n_segments=np.int64(1), valid_frac=np.float64(1.0),
        n_nonfinite=np.int64(0), session='S1_T1',
        source_file='/raw/Physiology/S1/T1/BP_mmHg.txt',
    )
    base.update(over)
    return base


# --------------------------------------------------------------------------- #
# identity (G0)
# --------------------------------------------------------------------------- #
def test_identity_schema_is_exactly_the_frozen_key_set():
    params = _params()
    assert set(params) == CANONICAL_KEYS
    assert not (set(params) & FORBIDDEN_KEYS)


def test_identity_has_no_derived_nesting_and_numeric_types_are_frozen():
    params = _params()
    assert isinstance(params['fs_in'], int)
    assert isinstance(params['fs_out'], int)
    assert isinstance(params['flatline_max_ms'], int)
    assert isinstance(params['rail_dwell_min_ms'], int)
    assert isinstance(params['min_segment_s'], float)
    assert params['store'] == bpp.STORE
    assert all(not isinstance(v, dict) for v in params.values())


def test_float_and_int_spelling_of_a_rate_produce_the_same_key():
    a = _params()
    b = bpa.build_params(CORPUS, fs_in=1000.0, fs_out=1000.0,
                         flatline_max_ms=200.0, rail_dwell_min_ms=30.0)
    assert a == b
    assert bpa.cache_key(a) == bpa.cache_key(b)


def test_key_is_readable_and_changes_with_every_gate():
    baseline = bpa.cache_key(_params())
    assert baseline.startswith('bpsession_src1000_range30-250_flat200ms_rail30ms_seg4s_v1_')

    variants = [
        dict(gate1_min_mmhg=35.0),
        dict(gate1_max_mmhg=240.0),
        dict(flatline_max_ms=250),
        dict(rail_dwell_min_ms=40),
        dict(min_segment_s=5.0),
    ]
    for over in variants:
        assert bpa.cache_key(_params(**over)) != baseline, over


def test_resampling_geometry_is_refused():
    with pytest.raises(ValueError, match='no resampling'):
        bpa.build_params(CORPUS, fs_out=500)
    with pytest.raises(ValueError, match='no resampling'):
        bpa.build_params(CORPUS, fs_in=1000, fs_out=2000)


def test_foreign_store_is_refused():
    with pytest.raises(ValueError, match='frozen'):
        bpa.build_params(CORPUS, store='signal+mask')


def test_corpus_digest_is_machine_independent(tmp_path):
    here = tmp_path / 'here'
    there = tmp_path / 'elsewhere' / 'deeper'
    for root in (here, there):
        for subj, task in (('S1', 'T1'), ('S1', 'T2')):
            d = root / 'Physiology' / subj / task
            d.mkdir(parents=True)
            (d / bpa.BP_SOURCE_NAME).write_text('90.0\n')

    a = bpa.discover_sessions(str(here))
    b = bpa.discover_sessions(str(there))
    assert bpa.corpus_digest(a) == bpa.corpus_digest(b)


# --------------------------------------------------------------------------- #
# discovery enumerates DIRECTORIES, never a file glob (G8)
# --------------------------------------------------------------------------- #
def test_discovery_lists_task_directories_without_a_trace(tmp_path):
    for subj, task in (('S1', 'T1'), ('S1', 'T2'), ('S1', 'T10')):
        (tmp_path / 'Physiology' / subj / task).mkdir(parents=True)
    (tmp_path / 'Physiology' / 'S1' / 'T1' / bpa.BP_SOURCE_NAME).write_text('90\n')

    sessions = bpa.discover_sessions(str(tmp_path))
    assert [s['session'] for s in sessions] == ['S1_T1', 'S1_T2', 'S1_T10']
    assert sessions[0]['source_file'] is not None
    assert sessions[1]['source_file'] is None
    assert sessions[1]['source_path'].endswith('T2/BP_mmHg.txt')


def test_discovery_missing_tree_is_loud(tmp_path):
    with pytest.raises(FileNotFoundError, match='Physiology'):
        bpa.discover_sessions(str(tmp_path))


# --------------------------------------------------------------------------- #
# find_cache: mismatch vs absence (G9)
# --------------------------------------------------------------------------- #
def test_find_cache_returns_the_matching_directory(tmp_path):
    params = _params()
    root = bpa.cache_root(tmp_path, bpa.cache_key(params))
    bpa.write_params(root, params)
    assert bpa.find_cache(tmp_path, params) == root


def test_find_cache_reports_a_mismatch_with_a_field_diff(tmp_path):
    params = _params()
    root = bpa.cache_root(tmp_path, bpa.cache_key(params))
    bpa.write_params(root, params)

    other = _params(gate1_min_mmhg=35.0)
    with pytest.raises(ValueError, match='gate1_min_mmhg'):
        bpa.find_cache(tmp_path, other)


def test_find_cache_distinguishes_absence_from_mismatch(tmp_path):
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match='no artefact'):
        bpa.find_cache(empty, _params())


def test_find_cache_refuses_an_ambiguous_match(tmp_path):
    params = _params()
    for name in ('a', 'b'):
        bpa.write_params(tmp_path / name, params)
    with pytest.raises(ValueError, match='ambiguous'):
        bpa.find_cache(tmp_path, params)


def test_check_params_compatible_names_the_differing_field(tmp_path):
    params = _params()
    root = bpa.cache_root(tmp_path, bpa.cache_key(params))
    bpa.write_params(root, params)
    with pytest.raises(ValueError, match='flatline_max_ms'):
        bpa.check_params_compatible(root, _params(flatline_max_ms=250))


# --------------------------------------------------------------------------- #
# shard IO (G10, G11)
# --------------------------------------------------------------------------- #
def test_shard_roundtrip_preserves_every_member(tmp_path):
    payload = _payload()
    path = bpa.write_shard(tmp_path, 'S1_T1', payload)
    assert path == bpa.shard_path(tmp_path, 'S1_T1')

    back = bpa.read_shard(path)
    for key in ('signal', 'mask', 'cause', 'seg_start', 'seg_stop',
                'seg_mean_mmhg', 'seg_seconds'):
        assert np.array_equal(back[key], payload[key]), key
    assert back['n_src'] == 4
    assert back['fs_in'] == back['fs_out'] == 1000
    assert back['session'] == 'S1_T1'


def test_write_shard_is_atomic_and_leaves_no_staging_file(tmp_path):
    bpa.write_shard(tmp_path, 'S1_T1', _payload())
    assert list((tmp_path / bpa.SUBDIR_TMP).iterdir()) == []


def test_read_shard_rejects_signal_that_is_not_nan_where_masked(tmp_path):
    path = tmp_path / 'bad.npz'
    np.savez(path, **_payload(signal=np.array([1.0, 2.0, 3.0, 4.0], np.float32),
                             mask=np.array([True, True, False, True])))
    with pytest.raises(ValueError, match='isnan'):
        bpa.read_shard(path)


def test_read_shard_rejects_a_mask_that_disagrees_with_cause(tmp_path):
    path = tmp_path / 'bad.npz'
    np.savez(path, **_payload(cause=np.array([0, 3, 0, 0], np.uint8)))
    with pytest.raises(ValueError, match='cause'):
        bpa.read_shard(path)


def test_read_shard_rejects_a_malformed_segment_table(tmp_path):
    path = tmp_path / 'bad.npz'
    np.savez(path, **_payload(seg_stop=np.array([], dtype=np.int64),
                             seg_start=np.array([], dtype=np.int64),
                             seg_mean_mmhg=np.array([], dtype=np.float64),
                             seg_seconds=np.array([], dtype=np.float64),
                             n_segments=np.int64(1)))
    with pytest.raises(ValueError, match='segment table'):
        bpa.read_shard(path)


def test_read_shard_rejects_out_of_range_segment_bounds(tmp_path):
    path = tmp_path / 'bad.npz'
    np.savez(path, **_payload(seg_stop=np.array([99], dtype=np.int64)))
    with pytest.raises(ValueError, match='bounds'):
        bpa.read_shard(path)


def test_shards_equal_detects_a_segment_table_difference():
    a = _payload()
    b = _payload(seg_mean_mmhg=np.array([91.0], dtype=np.float64))
    same, diffs = bpa.shards_equal(a, b)
    assert not same and diffs == ['seg_mean_mmhg']

    same, diffs = bpa.shards_equal(_payload(), _payload())
    assert same and diffs == []


# --------------------------------------------------------------------------- #
# build_task: skips are narrow, content never skips (G8)
# --------------------------------------------------------------------------- #
def _write_trace(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(x, dtype=np.float64), fmt='%.6f')
    return path


def test_build_task_produces_a_payload(tmp_path):
    f = _write_trace(tmp_path / 'Physiology' / 'S1' / 'T1' / bpa.BP_SOURCE_NAME,
                     bs.sine(6000))
    payload, reason = bpa.build_task(
        {'session': 'S1_T1', 'source_file': str(f)}, _params())
    assert reason is None
    assert payload['n_src'] == 6000
    assert payload['n_segments'] == 1
    assert payload['valid_frac'] == pytest.approx(1.0)


def test_build_task_skips_a_missing_trace(tmp_path):
    payload, reason = bpa.build_task(
        {'session': 'S1_T1',
         'source_file': str(tmp_path / 'nope' / bpa.BP_SOURCE_NAME)}, _params())
    assert payload is None and reason == 'missing_bp_mmhg'


def test_build_task_skips_an_empty_trace(tmp_path):
    f = tmp_path / bpa.BP_SOURCE_NAME
    f.write_text('')
    payload, reason = bpa.build_task(
        {'session': 'S1_T1', 'source_file': str(f)}, _params())
    assert payload is None and reason == 'empty_bp_mmhg'


def test_build_task_skips_an_unreadable_trace(tmp_path, monkeypatch):
    f = tmp_path / bpa.BP_SOURCE_NAME
    f.write_text('90.0\n')

    def _boom(_path):
        raise OSError('boom')

    monkeypatch.setattr(bpa, '_load_bp_mmhg', _boom)
    payload, reason = bpa.build_task(
        {'session': 'S1_T1', 'source_file': str(f)}, _params())
    assert payload is None and reason.startswith('unreadable_bp_mmhg')


def test_a_wholly_invalid_session_is_still_a_shard(tmp_path):
    f = _write_trace(tmp_path / bpa.BP_SOURCE_NAME, bs.constant(5000, 20.0))
    payload, reason = bpa.build_task(
        {'session': 'S1_T1', 'source_file': str(f)}, _params())
    assert reason is None
    assert payload['valid_frac'] == 0.0
    assert payload['n_segments'] == 0
    assert payload['seg_start'].dtype == np.int64


def test_malformed_tokens_become_non_finite_and_are_counted(tmp_path):
    f = tmp_path / bpa.BP_SOURCE_NAME
    lines = ['%.6f' % v for v in bs.sine(5000)]
    lines[10] = 'not-a-number'
    f.write_text('\n'.join(lines) + '\n')
    payload, reason = bpa.build_task(
        {'session': 'S1_T1', 'source_file': str(f)}, _params())
    assert reason is None
    assert payload['n_nonfinite'] == 1
    assert payload['cause'][10] == bpp.CAUSE_NONFINITE


# --------------------------------------------------------------------------- #
# scan / manifest
# --------------------------------------------------------------------------- #
def test_scan_separates_cached_skipped_and_missing(tmp_path):
    params = _params()
    root = bpa.cache_root(tmp_path, bpa.cache_key(params))
    bpa.write_params(root, params)
    bpa.write_shard(root, 'S1_T1', _payload(session='S1_T1'))
    bpa.write_skip(root, 'S1_T2', 'missing_bp_mmhg')

    info = bpa.scan(root, CORPUS + [{'session': 'S9_T9'}])
    assert info['cached'] == 1
    assert info['skipped'] == 1
    assert info['missing'] == 1
    assert info['bytes'] > 0


def test_manifest_is_derived_from_the_shards(tmp_path):
    params = _params()
    root = bpa.cache_root(tmp_path, bpa.cache_key(params))
    bpa.write_params(root, params)
    bpa.write_shard(root, 'S1_T1', _payload(session='S1_T1'))
    bpa.write_skip(root, 'S1_T2', 'missing_bp_mmhg')

    m = bpa.build_manifest(root, CORPUS, params)
    assert m['n_ok'] == 1 and m['n_skipped'] == 1 and m['n_missing'] == 0
    assert m['key'] == bpa.cache_key(params)
    assert m['cause_totals'] == {'valid': 4}
    assert (root / bpa.MANIFEST_NAME).is_file()


def test_manifest_does_not_report_a_zero_segment_shard_as_a_zero_second_segment(tmp_path):
    params = _params()
    root = bpa.cache_root(tmp_path, bpa.cache_key(params))
    bpa.write_params(root, params)
    bpa.write_shard(root, 'S1_T1', _payload(session='S1_T1'))        # 1 segment
    bpa.write_shard(root, 'S1_T2', _payload(                           # 0 segments
        session='S1_T2',
        signal=np.full(4, np.nan, dtype=np.float32),
        mask=np.zeros(4, dtype=bool),
        cause=np.full(4, bpp.CAUSE_RANGE, dtype=np.uint8),
        seg_start=np.empty(0, dtype=np.int64),
        seg_stop=np.empty(0, dtype=np.int64),
        seg_mean_mmhg=np.empty(0, dtype=np.float64),
        seg_seconds=np.empty(0, dtype=np.float64),
        n_segments=np.int64(0), valid_frac=np.float64(0.0)))

    m = bpa.build_manifest(root, CORPUS, params)
    assert m['n_zero_segment'] == 1
    # the 0.004 s segment is the only one; the empty shard must not pull min to 0
    assert m['seg_seconds_max']['min'] == pytest.approx(0.004)
    assert m['seg_seconds_max']['max'] == pytest.approx(0.004)


# --------------------------------------------------------------------------- #
# the builder CLI (Criterion 12, G12)
# --------------------------------------------------------------------------- #
def _make_corpus(root, n=5):
    for i in range(n):
        _write_trace(root / 'Physiology' / f'S{i:02d}' / 'T1' / bpa.BP_SOURCE_NAME,
                     bs.sine(6000, level=90.0 + i))


def _run(code_dir, *args):
    runner = os.path.join(code_dir, 'runners', 'run_build_bp_artefact.py')
    return subprocess.run([sys.executable, runner, *args], cwd=code_dir,
                          capture_output=True, text=True)


def test_cli_builds_and_reverifies_a_five_session_subset(code_dir, tmp_path):
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    _make_corpus(raw)

    built = _run(code_dir, '--init', '--build',
                 '--raw_root', str(raw), '--out_root', str(out))
    assert built.returncode == 0, built.stdout + built.stderr

    artefacts = [p for p in out.iterdir() if p.is_dir()]
    assert len(artefacts) == 1
    root = artefacts[0]
    assert (root / bpa.PARAMS_NAME).is_file()
    shards = list((root / bpa.SUBDIR_SESSIONS).glob('*.npz'))
    assert len(shards) == 5

    # a rebuild of the same 5 must be bit-identical on the stored content
    checked = _run(code_dir, '--check', '5',
                   '--raw_root', str(raw), '--out_root', str(out))
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert 'VERDICT: PASS' in checked.stdout

    indexed = _run(code_dir, '--index',
                   '--raw_root', str(raw), '--out_root', str(out))
    assert indexed.returncode == 0, indexed.stdout + indexed.stderr
    manifest = json.loads((root / bpa.MANIFEST_NAME).read_text())
    assert manifest['n_ok'] == 5


def test_cli_status_reports_a_skip(code_dir, tmp_path):
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    _make_corpus(raw, n=2)
    (raw / 'Physiology' / 'S09' / 'T1').mkdir(parents=True)   # no trace

    _run(code_dir, '--init', '--build',
         '--raw_root', str(raw), '--out_root', str(out))
    status = _run(code_dir, '--status',
                  '--raw_root', str(raw), '--out_root', str(out))
    assert status.returncode == 0, status.stdout + status.stderr
    assert 'missing_bp_mmhg' in status.stdout


def test_build_imports_no_accelerator_library(code_dir):
    result = _run(code_dir, '--check-imports')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'torch loaded: False' in result.stdout
