"""Contract tests for the four-stream Stage-2 dataset.

The synthetic tree is the tri-modal fixture (TIR + RGB2D + RESP) extended with
a 1000 Hz ``BP_mmHg.txt`` and a real BP artefact, so the test exercises the
join, not a mock:

* ``tir`` / ``rgb2d`` / ``resp`` come from the existing tri-modal parent;
* ``bp`` comes from ``data.bp_artefact`` through ``bp_windows`` on the SAME
  ``t_start`` and the SAME ``seq_len``;
* an unusable BP window, a ``skips/`` marker and a missing shard each have a
  distinct, tested outcome.
"""
import argparse
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from data import bp_artefact as bpa
from data import bp_preprocess as bpp
from data import datasets as D
from data import fourstream_dataset as fsd
from data import tir_resp_dataset as trd
from data import video_io as vio
from data.universe import available_data_sets, corpus_subjects
import runners.run_build_bp_artefact as rbpa
from test_trimodal_dataset import (_Reader, build_2d_cache, make_corpus,
                                   CLIP_S, FPS, N_FRAMES, SUBJECT, TASK)

RESP_FS = 1000.0
BP_FS = 1000.0
INPUT_SIZE = 32


@pytest.fixture(autouse=True)
def _stub_video(monkeypatch):
    monkeypatch.setattr(vio, 'open_video', lambda path: _Reader(path))
    yield


def _write_physio(raw: str) -> None:
    """Overwrite RESP at 1000 Hz and add a valid 1000 Hz BP trace."""
    n = int(round(N_FRAMES / FPS * RESP_FS))
    t = np.arange(n) / RESP_FS
    phys = os.path.join(raw, 'Physiology', SUBJECT, TASK)
    os.makedirs(phys, exist_ok=True)
    resp = 3.0 + 2.0 * np.sin(2 * np.pi * 0.3 * t)
    with open(os.path.join(phys, 'Resp_Volts.txt'), 'w') as fh:
        fh.write('\n'.join(f'{v:.6f}' for v in resp) + '\n')
    bp = 100.0 + 20.0 * np.sin(2 * np.pi * 1.2 * t)
    with open(os.path.join(phys, 'BP_mmHg.txt'), 'w') as fh:
        fh.write('\n'.join(f'{v:.6f}' for v in bp) + '\n')


def build_bp_artefact(raw: str, out: str):
    """Build a synthetic 1000 Hz artefact with a 0.1 s minimum segment."""
    common = ['--raw_root', raw, '--out_root', out,
              '--fs_in', str(BP_FS), '--fs_out', str(BP_FS),
              '--min_segment_s', '0.1',
              '--array_index', '0', '--array_count', '1']
    assert rbpa.main(['--init', '--build'] + common) == 0
    sessions = bpa.discover_sessions(raw)
    params = bpa.build_params(sessions, fs_in=BP_FS, fs_out=BP_FS,
                              min_segment_s=0.1)
    root = bpa.cache_root(out, bpa.cache_key(params))
    return str(root), bpa.read_params(root)


def dataset_args(raw: str, rgb2d_cache: str, bp_out: str, bp_params: dict,
                 **kw) -> argparse.Namespace:
    return argparse.Namespace(
        data_set='tir_roi_resp_rgb2d_bp',
        raw_root=raw, streams='tir,rgb2d,bp,resp',
        fs=RESP_FS, resp_fs=RESP_FS, fps=FPS,
        clip_duration=kw.pop('clip_seconds', CLIP_S),
        clip_stride=None, temporal_stride=1,
        tubelet='2,16,16', input_size=kw.pop('input_size', INPUT_SIZE),
        roi_padding=0.10, crop_rule='square', max_sway_px=40.0,
        roi_landmarks=None, min_signal_spread=0.0, rail_touch_v=0.0,
        physio_norm=trd.TARGET_NORM,
        subjects=None, tasks=None, task_groups=None, task_set=None,
        max_clips=None, max_entries=None,
        roi_cache=None, resp_cache=None,
        rgb2d_cache=rgb2d_cache, rgb2d_cache_lru=4,
        rgb2d_store='native', rgb2d_landmarks='face',
        rgb2d_min_valid_ratio=1.0, rgb2d_max_sway_px=40.0,
        rgb2d_area_min_frac=0.0,
        bp_artefact=bp_out, bp_fs=BP_FS, bp_params=bp_params,
        bp_cache_lru=4,
        allow_empty=bool(kw.pop('allow_empty', False)),
    )


@pytest.fixture()
def built(tmp_path):
    raw = make_corpus(str(tmp_path / 'raw'))
    _write_physio(raw)
    rgb2d = str(tmp_path / 'rgb2d')
    build_2d_cache(raw, rgb2d)
    bp_out = str(tmp_path / 'bp')
    bp_root, bp_params = build_bp_artefact(raw, bp_out)
    return raw, rgb2d, bp_out, bp_params, bp_root


# --------------------------------------------------------------------------- #
# the join
# --------------------------------------------------------------------------- #
def test_sample_carries_four_streams_on_one_window(built):
    raw, rgb2d, bp_out, bp_params, _ = built
    ds = fsd.build_tir_roi_rgb2d_bp_pretrain_dataset(
        dataset_args(raw, rgb2d, bp_out, bp_params))
    assert len(ds) == 3
    item = ds[0]
    assert set(item) == {'tir', 'rgb2d', 'resp', 'bp'}
    assert tuple(item['tir'].shape) == (3, 4, INPUT_SIZE, INPUT_SIZE)
    assert item['rgb2d'].shape == item['tir'].shape
    assert item['resp'].shape == (1, 160)
    assert item['bp'].shape == (1, 160)
    assert item['bp'].dtype == torch.float32
    # the dataset owns the per-clip z-score for BOTH physio streams
    bp = item['bp'].numpy().reshape(-1)
    assert abs(float(bp.mean())) < 1e-3
    assert abs(float(bp.std()) - 1.0) < 1e-3


def test_unusable_bp_window_is_dropped(built):
    raw, rgb2d, bp_out, bp_params, bp_root = built
    session = f'{SUBJECT}_{TASK}'
    shard = bpa.read_shard(bpa.shard_path(bp_root, session))
    shard['mask'][:10] = False
    shard['cause'][:10] = bpp.CAUSE_FLATLINE
    shard['signal'][:10] = np.nan
    bpa.write_shard(bp_root, session, shard)

    ds = fsd.build_tir_roi_rgb2d_bp_pretrain_dataset(
        dataset_args(raw, rgb2d, bp_out, bp_params))
    assert len(ds) == 2
    assert ds.stats['clips_before_bp_filter'] == 3
    assert ds.stats['clips_dropped_bp_unusable'] == 1
    assert ds.stats['clips_after_bp_filter'] == 2


def test_skipped_bp_session_drops_every_clip(built):
    raw, rgb2d, bp_out, bp_params, bp_root = built
    bpa.write_skip(bp_root, f'{SUBJECT}_{TASK}', 'synthetic')
    ds = fsd.build_tir_roi_rgb2d_bp_pretrain_dataset(
        dataset_args(raw, rgb2d, bp_out, bp_params, allow_empty=True))
    assert len(ds) == 0
    assert ds.stats['clips_dropped_bp_skipped'] == 3
    assert ds.stats['sessions_bp_skipped'] == 1
    with pytest.raises(RuntimeError, match='BP gates removed EVERY'):
        fsd.build_tir_roi_rgb2d_bp_pretrain_dataset(
            dataset_args(raw, rgb2d, bp_out, bp_params, allow_empty=False))


def test_missing_bp_shard_fails_loudly(built):
    raw, rgb2d, bp_out, bp_params, bp_root = built
    os.remove(bpa.shard_path(bp_root, f'{SUBJECT}_{TASK}'))
    with pytest.raises(FileNotFoundError, match='INCOMPLETE'):
        fsd.build_tir_roi_rgb2d_bp_pretrain_dataset(
            dataset_args(raw, rgb2d, bp_out, bp_params))


def test_bp_identity_mismatch_fails(built):
    raw, rgb2d, bp_out, bp_params, _ = built
    bad = dict(bp_params)
    bad['min_segment_s'] = float(bad['min_segment_s']) + 0.1
    with pytest.raises(ValueError,
                       match='no artefact matching|DIFFERENT parameters'):
        fsd.build_tir_roi_rgb2d_bp_pretrain_dataset(
            dataset_args(raw, rgb2d, bp_out, bad))


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #
def test_registry_constant_matches_the_module():
    assert fsd.FOURSTREAM_DATA_SETS == D.FOURSTREAM_DATA_SETS


def test_dispatcher_selects_the_fourstream_builder(built):
    raw, rgb2d, bp_out, bp_params, _ = built
    ds = D._build_pretrain_source(dataset_args(raw, rgb2d, bp_out, bp_params))
    assert isinstance(ds, fsd.TirRoiResp2DBPPretrainDataset)


def test_universe_enumerates_the_fourstream_data_set(built):
    raw, _rgb2d, _bp_out, _bp_params, _bp_root = built
    assert 'tir_roi_resp_rgb2d_bp' in available_data_sets()
    assert corpus_subjects(raw_root=raw,
                           data_set='tir_roi_resp_rgb2d_bp') == [SUBJECT]


def test_fourstream_data_set_pins_target_norm():
    from runners import run_pretrain as rp
    args = argparse.Namespace(data_set='tir_roi_resp_rgb2d_bp',
                              target_norm='token')
    assert rp.pin_roi_target_norm(args) is True
    assert args.target_norm == 'none'
