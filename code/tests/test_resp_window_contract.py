"""Contract tests for the respiration cache as a DATASET consumer.

Pins the window rule (`mask[s:s+n].all()`), the reporting counters, the
fail-loudly attachment gates (fs mismatch, legacy knobs, incomplete cache) and
the two-cache coverage intersection. No video is decoded: ``video_io`` is
stubbed exactly like ``test_tir_resp_rail_filter.py`` does.
"""
import json
import os
import sys

import numpy as np
import pytest

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from data import roi_cache as rc
from data import tir_resp_dataset as trd
from data import video_io as vio
from runners import run_build_resp_cache as rb

FPS = 25.0
RESP_FS = 1000.0
SECONDS = 24.0
N_FRAMES = int(round(SECONDS * FPS))          # 600
N_RESP = int(round(SECONDS * RESP_FS))        # 24000
CLIP_S = 8.0
SRC_W, SRC_H = 64, 64


class _FakeReader:
    """Container-probe + decode stand-in over a tiny 64x64 RGB frame."""

    def __init__(self, path):
        self.num_frames = N_FRAMES
        self.fps = FPS

    def read_range(self, start, count):
        n = max(0, min(count, self.num_frames - start))
        y, x = np.mgrid[0:SRC_H, 0:SRC_W]
        frame = np.stack([x, y, (x + y) % 255], axis=-1).astype(np.uint8)
        return np.repeat(frame[None], n, axis=0)

    def read_all(self):
        return self.read_range(0, self.num_frames)

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _stub_video(monkeypatch):
    monkeypatch.setattr(vio, 'open_video', lambda path: _FakeReader(path))
    yield


def _breathing(seconds: float = SECONDS) -> np.ndarray:
    t = np.arange(int(round(seconds * RESP_FS))) / RESP_FS
    return 3.0 + 2.0 * np.sin(2 * np.pi * 0.3 * t)


def _write_raw(root, traces: dict) -> str:
    for name, x in traces.items():
        subj, task = name.split('_')
        (root / 'Thermal' / subj).mkdir(parents=True, exist_ok=True)
        (root / 'Thermal' / subj / f'{task}.wmv').write_bytes(b'\x00')
        (root / 'IRFeatures').mkdir(exist_ok=True)
        # 28 landmarks in a small ring inside the 64x64 frame
        pts = []
        for i in range(28):
            ang = 2 * np.pi * i / 28.0
            pts += [f'{32 + 6 * np.cos(ang):.2f}', f'{32 + 6 * np.sin(ang):.2f}']
        line = ' '.join(pts)
        (root / 'IRFeatures' / f'{name}.txt').write_text(
            '\n'.join([line] * N_FRAMES) + '\n')
        phys = root / 'Physiology' / subj / task
        phys.mkdir(parents=True, exist_ok=True)
        (phys / 'Resp_Volts.txt').write_text(
            '\n'.join(f'{v:.5f}' for v in x) + '\n')
    return str(root)


def _build_resp_cache(raw, out):
    assert rb.main(['--init', '--build', '--out_root', str(out),
                    '--raw_root', str(raw)]) == 0
    return out


def _key_root(raw, out):
    sessions = trd.discover_sessions(str(raw))
    params = rb.respc.build_params(str(raw), sessions)
    return rb.respc.cache_root(out, rb.respc.cache_key(params))


def _make_resp_cache(raw, out, traces):
    _write_raw(raw, traces)
    return _build_resp_cache(raw, out)


def _base(raw, cache, **kw):
    kw.setdefault('resp_fs', 100.0)
    return trd.BP4DPlusTIRRespDataset(
        raw_root=str(raw), clip_seconds=CLIP_S, clip_stride=CLIP_S,
        input_size=112, roi_padding=0.2, crop_rule='square',
        fps=FPS, resp_cache=str(cache), allow_empty=True, **kw)


# --------------------------------------------------------------------------- #
# window rule
# --------------------------------------------------------------------------- #
def test_window_inside_a_valid_run_is_kept(tmp_path):
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    cache = _make_resp_cache(raw, out, {'S001_T1': _breathing()})
    ds = _base(raw, cache)
    assert len(ds) == 3                        # 0/8/16 s on a 24 s session
    assert ds.stats['resp_cache'] == str(_key_root(raw, cache))
    assert ds.stats['resp_cache_key'] == _key_root(raw, cache).name
    assert ds.stats['resp_fs'] == 100.0
    item = ds[0]
    assert item['resp_signal'].shape == (800,)
    assert np.isfinite(item['resp_signal'].numpy()).all()


def test_window_over_a_saturation_hole_is_dropped_with_the_right_counter(tmp_path):
    x = _breathing()
    x[int(10 * RESP_FS):int(13 * RESP_FS)] = 12.0     # 3 s event
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    cache = _make_resp_cache(raw, out, {'S001_T1': x})
    ds = _base(raw, cache)
    assert len(ds) == 2                        # 0-8 s and 16-24 s survive
    assert ds.stats['clips_dropped_invalid_sat'] == 1
    assert ds.stats['clips_dropped_rail'] == 0
    assert ds.stats['clips_dropped_flat'] == 0
    sess = ds.sessions['S001_T1']
    assert sess['clips_dropped_invalid_sat'] == 1
    assert 0.0 < sess['resp_valid_frac'] < 1.0


def test_brief_rail_graze_keeps_the_window(tmp_path):
    """C1 headline: a 0.5 s graze (< the 1 s alarm) no longer drops the window."""
    x = _breathing()
    x[int(10 * RESP_FS):int(10.5 * RESP_FS)] = 12.0   # 0.5 s < sat_min_s
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    cache = _make_resp_cache(raw, out, {'S001_T1': x})
    ds = _base(raw, cache)
    assert len(ds) == 3
    assert ds.stats['clips_dropped_invalid_sat'] == 0


def test_nonfinite_samples_drop_the_window(tmp_path):
    x = _breathing()
    x[int(10 * RESP_FS)] = np.nan
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    cache = _make_resp_cache(raw, out, {'S001_T1': x})
    ds = _base(raw, cache)
    assert len(ds) == 2
    assert ds.stats['clips_dropped_invalid_nonfinite'] == 1


# --------------------------------------------------------------------------- #
# fail-loudly attachment gates
# --------------------------------------------------------------------------- #
def test_fs_out_mismatch_raises(tmp_path):
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    cache = _make_resp_cache(raw, out, {'S001_T1': _breathing()})
    with pytest.raises(ValueError) as exc:
        _base(raw, cache, resp_fs=50.0)
    assert 'fs_out' in str(exc.value)


def test_legacy_knobs_with_a_cache_raise(tmp_path):
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    cache = _make_resp_cache(raw, out, {'S001_T1': _breathing()})
    with pytest.raises(ValueError) as exc:
        _base(raw, cache, min_signal_spread=0.1)
    assert 'min_signal_spread' in str(exc.value)
    with pytest.raises(ValueError):
        _base(raw, cache, rail_touch_v=9.9)


def test_clip_longer_than_min_run_raises(tmp_path):
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    cache = _make_resp_cache(raw, out, {'S001_T1': _breathing()})
    with pytest.raises(ValueError) as exc:
        trd.BP4DPlusTIRRespDataset(
            raw_root=str(raw), clip_seconds=10.0, input_size=112,
            fps=FPS, resp_fs=100.0, resp_cache=str(cache), allow_empty=True)
    assert 'min_run_s' in str(exc.value)


def test_missing_shard_raises_with_the_builder_hint(tmp_path):
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    cache = _make_resp_cache(raw, out, {
        'S001_T1': _breathing(), 'S002_T1': _breathing()})
    os.remove(_key_root(raw, cache) / 'tasks' / 'S002_T1.npz')
    with pytest.raises(FileNotFoundError) as exc:
        _base(raw, cache)
    assert 'run_build_resp_cache.py' in str(exc.value)


def test_skipped_session_is_excluded_quietly(tmp_path):
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    cache = _make_resp_cache(raw, out, {
        'S001_T1': _breathing(), 'S002_T1': _breathing()})
    key = _key_root(raw, cache)
    os.remove(key / 'tasks' / 'S002_T1.npz')
    rb.respc.write_skip(key, 'S002_T1', 'missing_resp_volts')
    ds = _base(raw, cache)
    assert set(ds.sessions) == {'S001_T1'}
    assert ds.stats['sessions_without_resp'] == 1
    assert any(s['session'] == 'S002_T1' and 'resp_cache_skipped' in s['reason']
               for s in ds.skipped)


# --------------------------------------------------------------------------- #
# two-cache coverage intersection
# --------------------------------------------------------------------------- #
def _stub_roi_cache(raw, out, names, skipped=()):
    """A params-matching ROI cache whose shards are placeholders.

    ``_build_entries`` only checks SHARD EXISTENCE, and no clip is loaded in
    these tests, so a zero-byte shard is a faithful stand-in for the coverage
    decision.
    """
    sessions = trd.discover_sessions(str(raw))
    params = rc.build_params(str(raw), sessions, trd.TARGET_LANDMARKS,
                             trd.DEFAULT_ROI_PADDING, src_w=SRC_W,
                             src_h=SRC_H, crop_rule='square',
                             clip_seconds=CLIP_S, fps=FPS)
    root = rc.cache_root(out, rc.cache_key(params))
    rc.write_params(root, params)
    (root / rc.SUBDIR_TASKS).mkdir(parents=True, exist_ok=True)
    for name in names:
        rc.shard_path(root, name).write_bytes(b'')
    for name in skipped:
        rc.write_skip(root, name, 'missing_ir_features')
    return out                                  # the OUT ROOT, as the loader wants


def test_intersection_excludes_a_roi_skipped_session(tmp_path):
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    traces = {'S001_T1': _breathing(), 'S002_T1': _breathing()}
    cache = _make_resp_cache(raw, out, traces)
    roi = _stub_roi_cache(raw, out / 'roi', ['S001_T1'], skipped=['S002_T1'])
    ds = trd.BP4DPlusTIRRespDataset(
        raw_root=str(raw), clip_seconds=CLIP_S, input_size=112, fps=FPS,
        resp_fs=100.0, resp_cache=str(cache), roi_cache=str(roi),
        allow_empty=True)
    assert set(ds.sessions) == {'S001_T1'}
    assert ds.stats['sessions_without_roi'] == 1


def test_intersection_missing_roi_shard_raises(tmp_path):
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    cache = _make_resp_cache(raw, out, {
        'S001_T1': _breathing(), 'S002_T1': _breathing()})
    roi = _stub_roi_cache(raw, out / 'roi', ['S001_T1'])     # S002 has neither
    with pytest.raises(FileNotFoundError) as exc:
        trd.BP4DPlusTIRRespDataset(
            raw_root=str(raw), clip_seconds=CLIP_S, input_size=112, fps=FPS,
            resp_fs=100.0, resp_cache=str(cache), roi_cache=str(roi),
            allow_empty=True)
    assert 'roi_cache' in str(exc.value)


# --------------------------------------------------------------------------- #
# data path (plan G5)
# --------------------------------------------------------------------------- #
def test_stage2_data_path_shapes(tmp_path):
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    cache = _make_resp_cache(raw, out, {'S001_T1': _breathing()})
    ds = trd.TirRoiRespPretrainDataset(
        raw_root=str(raw), streams=('tir', 'resp'), fs=100.0, fps=FPS,
        clip_duration=CLIP_S, temporal_stride=2, tubelet_t=2,
        input_size=112, roi_padding=0.2, crop_rule='square',
        resp_fs=100.0, resp_cache=str(cache), allow_empty=True)
    item = ds[0]
    assert tuple(item['tir'].shape) == (3, 100, 112, 112)
    assert tuple(item['resp'].shape) == (1, 800)
    assert ds.base.stats['resp_cache_key'] == _key_root(raw, cache).name


def test_stage3_data_path_shapes(tmp_path):
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    cache = _make_resp_cache(raw, out, {
        'S001_T1': _breathing(), 'S002_T1': _breathing()})
    ds = trd.TirRoiRespFinetuneDataset(
        raw_root=str(raw), is_train=True, train_ratio=1.0, split_by='subject',
        fs=100.0, clip_duration=CLIP_S, temporal_stride=2, tubelet_t=2,
        input_size=112, roi_padding=0.2, crop_rule='square', resp_fs=100.0,
        resp_cache=str(cache), physio_norm='session')
    tir, wav = ds[0]
    assert tuple(tir.shape) == (3, 100, 112, 112)
    assert tuple(wav.shape) == (800,)


def test_resp_fs_must_equal_model_fs_with_a_cache(tmp_path):
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    cache = _make_resp_cache(raw, out, {'S001_T1': _breathing()})
    with pytest.raises(ValueError) as exc:
        trd.TirRoiRespPretrainDataset(
            raw_root=str(raw), fs=100.0, clip_duration=CLIP_S,
            resp_fs=1000.0, resp_cache=str(cache), allow_empty=True)
    assert 'resp_fs' in str(exc.value)


# --------------------------------------------------------------------------- #
# inspection tooling: seeing the mask
# --------------------------------------------------------------------------- #
def _mixed_session(tmp_path):
    """24 s session with one 3 s saturation event at t=10 s (windows 8-16 lost)."""
    x = _breathing()
    x[int(10 * RESP_FS):int(13 * RESP_FS)] = 12.0
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    cache = _make_resp_cache(raw, out, {'S001_T1': x})
    return _base(raw, cache)


def test_diagnose_window_names_the_mask_cause(tmp_path):
    """An absent window must be attributed to the MASK, not "passes everything"."""
    from runners.run_inspect_tir_resp import _diagnose_window

    ds = _mixed_session(tmp_path)
    lines, reasons = _diagnose_window(ds, int(round(8.0 * FPS)))
    assert any('respiration MASK' in ln for ln in lines), lines
    assert any(r.startswith('mask_') for r in reasons), reasons
    assert any('saturation' in ln for ln in lines), lines
    # a window that IS in the entry list is not "diagnosed" as absent
    assert _diagnose_window(ds, 0)[1] == ['present']


def test_resp_runs_lists_valid_and_invalid_stretches(tmp_path):
    """The --mask_runs overview: 2 usable runs around one saturation hole."""
    from runners.run_inspect_tir_resp import _resp_runs

    ds = _mixed_session(tmp_path)
    runs = _resp_runs(ds, 'S001_T1')
    kinds = [r[2] for r in runs]
    assert kinds.count('valid') == 2 and kinds.count('INVALID') == 1
    bad = [r for r in runs if r[2] == 'INVALID'][0]
    assert bad[3] == 'saturation'
    assert 9.5 < bad[0] < 10.5 and 12.5 < bad[1] < 13.5
    # sorted, contiguous coverage of the session
    assert runs[0][0] == 0.0
    for a, b in zip(runs, runs[1:]):
        assert a[1] == pytest.approx(b[0], abs=1e-9)


def test_check_clip_validates_the_cached_grid_not_the_raw_file(tmp_path):
    """``_check_clip`` must compare the label to the SHARD when a cache is on.

    The raw file lives on a different sample axis and in a different DC space,
    so the legacy comparison failed by construction for every cached clip.
    """
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    cache = _make_resp_cache(raw, out, {'S001_T1': _breathing()})
    ds = _base(raw, cache)
    assert ds.stats['resp_cache'] is not None
    assert trd._check_clip(ds, 0) == []
