"""Pipeline-wide 1-D target normalisation (`physio_norm`) + clip-level ROI box.

The TIR-ROI+RESP pipeline has exactly ONE normalisation knob, and the DATASET
owns it in BOTH stages (same flag name and values, so the mechanism carries over
unchanged to the RGB+BP extension):

* Stage 2 (``data_set: tir_roi``): ``--physio_norm {none, clip|zscore, session}``
  on ``runners/run_pretrain.py``. The dataset z-scores each clip (whole-session
  statistics for ``session``) and the MODEL never normalises the 1-D stream:
  ``run_pretrain.pin_roi_target_norm`` forces ``target_norm='none'`` for every
  ROI lineage (``tir_roi`` = TIR-ROI+RESP, ``rgb_roi`` = RGB+BP).
* Stage 3 (``run_waveform.py``): the same flag name and values, applied by
  ``TirRoiRespFinetuneDataset`` -- so one clip yields ONE identical target in
  both stages.

The TIR-ROI bounding box is CLIP-level: it is taken over the clip's own frames
(``BP4DPlusTIRRespDataset.clip_roi_box``), so two clips of the same session get
DIFFERENT boxes when the head moves.

The tests build a minimal synthetic raw tree (one session, a per-frame drifting
landmark track, a sloped respiration series) and stub the video container.
"""
import argparse
import inspect
import os
import sys

import numpy as np
import pytest
import torch
import yaml

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from core.multimae import _tiny_model                     # noqa: E402
from data import rgb_roi_dataset as rrd                   # noqa: E402
from data import tir_resp_dataset as trd                  # noqa: E402
from data import video_io as vio                          # noqa: E402
from data.datasets import build_pretraining_dataset       # noqa: E402
from runners.run_pretrain import pin_roi_target_norm      # noqa: E402

FPS = 25.0
RESP_FS = 1000.0
CLIP_S = 4.0
CLIP_FRAMES = int(CLIP_S * FPS)                           # 100
N_FRAMES = 400                                            # 16 s -> 4 clips
N_RESP = int(N_FRAMES / FPS * RESP_FS)                    # 16000
H_FRAME, W_FRAME = 160, 240

#: ROI Stage-2 configs: the normalisation knob MUST be `physio_norm` (dataset),
#: never the model-side `target_norm` (which run_pretrain pins to 'none').
ROI_PRETRAIN_CONFIGS = (
    'configs/pretrain/stage2_hpc_tir_roi_resp.yaml',
    'configs/pretrain/stage2_local_tir_roi_resp.yaml',
    'configs/pretrain/stage2_local_tir_roi_crossmae.yaml',
    'configs/pretrain/stage2_local_rgb_roi_bp.yaml',
)


class _FakeReader:
    """``data.video_io`` stand-in: container probe + blank frame decode."""

    def __init__(self, num_frames, fps, h=H_FRAME, w=W_FRAME):
        self.num_frames = num_frames
        self.fps = fps
        self._h, self._w = h, w

    def read_range(self, start, n):
        return np.zeros((n, self._h, self._w, 3), np.uint8)

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _stub_video(monkeypatch):
    monkeypatch.setattr(vio, 'open_video',
                        lambda path: _FakeReader(N_FRAMES, FPS))
    yield


def _write_raw(root) -> str:
    """One session ``S001_T1``: drifting landmarks + a sloped respiration."""
    (root / 'Thermal' / 'S001').mkdir(parents=True)
    (root / 'Thermal' / 'S001' / 'T1.wmv').write_bytes(b'\x00')
    (root / 'IRFeatures').mkdir()
    lines = []
    for i in range(N_FRAMES):
        x = 60.0 + 0.4 * i            # drifts 60..219.6 (< 240)
        y = 40.0 + 0.2 * i            # drifts 40..119.8 (< 160)
        lines.append(' '.join([f'{x:.3f}', f'{y:.3f}'] * 28))
    (root / 'IRFeatures' / 'S001_T1.txt').write_text('\n'.join(lines) + '\n')
    phys = root / 'Physiology' / 'S001' / 'T1'
    phys.mkdir(parents=True)
    y = np.linspace(-5.0, 5.0, N_RESP)      # session mean 0, but each clip's differs
    (phys / 'Resp_Volts.txt').write_text(
        '\n'.join(f'{v:.4f}' for v in y) + '\n')
    return str(root)


def _pretrain_ds(root, **kw):
    kw.setdefault('clip_duration', CLIP_S)
    kw.setdefault('clip_stride', None)
    kw.setdefault('temporal_stride', 1)
    kw.setdefault('tubelet_t', 2)
    kw.setdefault('input_size', 64)
    kw.setdefault('roi_padding', 0.2)
    return trd.TirRoiRespPretrainDataset(
        raw_root=root, streams=('tir', 'resp'), fs=100.0, fps=FPS,
        resp_fs=RESP_FS, subjects=['S001'], **kw)


def _finetune_ds(root, **kw):
    kw.setdefault('clip_duration', CLIP_S)
    kw.setdefault('clip_stride', None)
    kw.setdefault('temporal_stride', 1)
    kw.setdefault('tubelet_t', 2)
    kw.setdefault('input_size', 64)
    kw.setdefault('roi_padding', 0.2)
    kw.setdefault('subjects', ['S001'])
    kw.setdefault('is_train', True)
    kw.setdefault('train_ratio', 1.0)     # single subject -> train side
    return trd.TirRoiRespFinetuneDataset(raw_root=root, **kw)


def _args(root, **over):
    kw = dict(raw_root=root, data_path=root, data_set='tir_roi',
              pretrain_split='all', subjects=['S001'], tasks=None,
              task_set=None, task_groups=None,
              clip_duration=CLIP_S, clip_stride=None, temporal_stride=1,
              tubelet='2,16,16', input_size=64, roi_padding=0.2,
              roi_quantile=0.0, roi_landmarks='', fps=FPS, resp_fs=RESP_FS,
              fs=100.0, min_signal_spread=0.0, rail_touch_v=0.0,
              physio_norm='none',
              max_clips=None, max_entries=None)
    kw.update(over)
    return argparse.Namespace(**kw)


def _raw_on_model_grid(ds, index):
    """The RAW clip window resampled onto the model's ``[seq_len]`` grid."""
    raw = ds.base.respiration_clip(index, normalized=False)
    grid = (np.arange(ds.seq_len) / ds.fs) * ds.resp_fs
    return np.interp(grid, np.arange(raw.shape[0], dtype=np.float64),
                     raw.astype(np.float64))


# --------------------------------------------------------------------------- #
# 1. the dataset-side knob (Stage 2)
# --------------------------------------------------------------------------- #
def test_default_physio_norm_is_none(tmp_path):
    ds = _pretrain_ds(_write_raw(tmp_path))
    assert ds.physio_norm == 'none'
    assert ds.base.norm == 'none'
    assert ds.stats['norm'] == 'none'


def test_physio_norm_none_returns_raw_volts(tmp_path):
    ds = _pretrain_ds(_write_raw(tmp_path), physio_norm='none')
    got = ds[0]['resp'].numpy()[0]
    assert np.allclose(got, _raw_on_model_grid(ds, 0), atol=1e-4)


def test_physio_norm_aliases_zscore_to_clip(tmp_path):
    """'zscore' (the Stage-3 spelling) == 'clip' here."""
    root = _write_raw(tmp_path)
    a = _pretrain_ds(root, physio_norm='zscore')
    b = _pretrain_ds(root, physio_norm='clip')
    assert a.physio_norm == b.physio_norm == 'clip'
    assert a.base.norm == b.base.norm == 'clip'
    assert np.allclose(a[0]['resp'].numpy(), b[0]['resp'].numpy())


def test_rejects_unknown_physio_norm(tmp_path):
    with pytest.raises(ValueError, match='physio_norm must be'):
        _pretrain_ds(_write_raw(tmp_path), physio_norm='bogus')


def test_session_norm_uses_the_whole_session_statistics(tmp_path):
    root = _write_raw(tmp_path)
    ds = _pretrain_ds(root, physio_norm='session')
    assert ds.base.norm == 'session'
    assert ds.stats['norm'] == 'session'

    sess = ds.entries[0]['session']
    t0 = ds[0]['resp'].numpy()[0]
    t1 = ds[1]['resp'].numpy()[0]

    # (a) two clips of the SAME session do NOT share a mean -> session-level,
    #     NOT clip-level (a per-clip z-score would force both to ~0).
    assert abs(float(t0.mean()) - float(t1.mean())) > 0.5
    # (b) each clip is exactly the raw window under the SESSION affine map.
    mu_s, sd_s = ds.base._norm_cache[sess]
    expect0 = (_raw_on_model_grid(ds, 0) - mu_s) / (sd_s + 1e-8)
    expect1 = (_raw_on_model_grid(ds, 1) - mu_s) / (sd_s + 1e-8)
    assert np.allclose(t0, expect0, atol=1e-4)
    assert np.allclose(t1, expect1, atol=1e-4)


def test_session_norm_differs_from_per_clip_norm(tmp_path):
    """The two normalisations are NOT equivalent on this corpus."""
    root = _write_raw(tmp_path)
    ds = _pretrain_ds(root, physio_norm='session')
    sess = ds.entries[0]['session']
    ds.base._resp_full(sess)                 # fills _norm_cache (session stats)
    mu_s, sd_s = ds.base._norm_cache[sess]
    raw0 = _raw_on_model_grid(ds, 0)
    per_clip = (raw0 - raw0.mean()) / (raw0.std() + 1e-8)
    session = (raw0 - mu_s) / (sd_s + 1e-8)
    assert not np.allclose(per_clip, session, atol=1e-3)


# --------------------------------------------------------------------------- #
# 2. the model is pinned to identity for EVERY ROI lineage
# --------------------------------------------------------------------------- #
def test_pin_roi_target_norm_forces_identity(capsys):
    a = argparse.Namespace(data_set='tir_roi', target_norm='clip',
                           physio_norm='session')
    assert pin_roi_target_norm(a) is True
    assert a.target_norm == 'none'
    assert 'forcing' in capsys.readouterr().out


def test_pin_roi_target_norm_covers_rgb_roi():
    """RGB+BP gets the SAME treatment -- that is the extension."""
    a = argparse.Namespace(data_set='rgb_roi', target_norm='clip',
                           physio_norm='clip')
    assert pin_roi_target_norm(a) is True
    assert a.target_norm == 'none'


def test_pin_roi_target_norm_ignores_non_roi_paths():
    a = argparse.Namespace(data_set='bp4d+', target_norm='clip',
                           physio_norm='none')
    assert pin_roi_target_norm(a) is False
    assert a.target_norm == 'clip'            # untouched


# --------------------------------------------------------------------------- #
# 3. the builders thread the SAME knob for both ROI lineages
# --------------------------------------------------------------------------- #
def test_builder_threads_physio_norm(tmp_path):
    root = _write_raw(tmp_path)
    ds = build_pretraining_dataset(_args(root, physio_norm='session'))
    assert ds.physio_norm == 'session'
    assert ds.base.norm == 'session'


def test_builder_accepts_the_stage3_alias(tmp_path):
    root = _write_raw(tmp_path)
    ds = build_pretraining_dataset(_args(root, physio_norm='zscore'))
    assert ds.physio_norm == 'clip'


def test_rgb_roi_builder_threads_physio_norm():
    src = inspect.getsource(rrd.build_rgb_roi_pretrain_dataset)
    assert "getattr(args, 'physio_norm'" in src


def test_rgb_roi_dataset_validates_physio_norm():
    with pytest.raises(ValueError, match='RGBRoiPretrainDataset: physio_norm'):
        rrd.RGBRoiPretrainDataset(raw_root='/nonexistent',
                                  streams=('rgb', 'bp'),
                                  physio_norm='bogus')


# --------------------------------------------------------------------------- #
# 4. model side: target_norm='none' is an IDENTITY on the 1-D stream
# --------------------------------------------------------------------------- #
def test_target_norm_none_is_identity():
    m = _tiny_model(target_norm='none')
    B = 2
    x = torch.randn(B, 1, m.n_signal * m.sig_kernel)
    tgt = m._targets(x, 'resp')
    assert tgt.shape == (B, m.n_signal, m.sig_kernel)
    assert torch.allclose(tgt, x.reshape(B, m.n_signal, m.sig_kernel))


def test_target_norm_none_leaves_visual_streams_normalised():
    """The identity applies to the 1-D stream only -- video keeps its own
    per-token z-score (so a session-norm run does not change the visual view)."""
    m = _tiny_model(target_norm='none')
    T = m.num_frames
    S = m.input_size
    x = torch.rand(1, 3, T, S, S) * 3.0 + 1.0                # [B, C, T, H, W]
    tgt = m._targets(x, 'tir')
    per_token_mean = tgt.mean(dim=-1)
    per_token_std = tgt.std(dim=-1, unbiased=False)
    assert torch.allclose(per_token_mean, torch.zeros_like(per_token_mean),
                          atol=1e-5)
    assert torch.allclose(per_token_std, torch.ones_like(per_token_std),
                          atol=1e-4)


def test_target_norm_clip_still_z_scores_per_clip():
    m = _tiny_model(target_norm='clip')
    x = torch.randn(2, 1, m.n_signal * m.sig_kernel) * 2.0 + 3.0
    flat = m._targets(x, 'resp').reshape(2, -1)
    assert torch.allclose(flat.mean(1), torch.zeros(2), atol=1e-5)
    assert torch.allclose(flat.std(1, unbiased=False), torch.ones(2), atol=1e-4)


# --------------------------------------------------------------------------- #
# 5. the TIR-ROI box is CLIP-level, not session-level
# --------------------------------------------------------------------------- #
def test_roi_box_is_computed_per_clip(tmp_path):
    ds = _pretrain_ds(_write_raw(tmp_path))
    b0 = ds.base.clip_roi_box(0)
    b1 = ds.base.clip_roi_box(1)
    assert b0 != b1                       # the box is NOT one-per-session

    # ... and it is exactly the box of THAT clip's own frames.
    ir = ds.base._landmarks('S001_T1')
    e = ds.entries[1]
    pts = ir[e['frame_start']:e['frame_end']][:, ds.base.target_idx, :]
    expect = trd.roi_box_from_landmarks(pts, W_FRAME, H_FRAME,
                                        ds.base.roi_padding,
                                        quantile=ds.base.roi_quantile)
    assert b1 == expect


# --------------------------------------------------------------------------- #
# 7. the shipped Stage-2 configs use the ONE knob
# --------------------------------------------------------------------------- #
def test_hpc_config_uses_physio_norm_and_the_2s_hop():
    path = os.path.join(CODE_DIR, 'configs', 'pretrain',
                        'stage2_hpc_tir_roi_resp.yaml')
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    assert cfg['physio_norm'] == 'session'
    assert cfg['clip_duration'] == pytest.approx(8.0)
    assert cfg['clip_stride'] == pytest.approx(2.0)
    # the rail / dead-signal rules stay pinned
    assert cfg['rail_touch_v'] == pytest.approx(9.90)
    assert cfg['min_signal_spread'] == pytest.approx(0.1)


@pytest.mark.parametrize('rel', ROI_PRETRAIN_CONFIGS)
def test_roi_configs_never_set_the_model_side_target_norm(rel):
    """One knob: the ROI Stage-2 configs normalise via `physio_norm` (dataset)
    and must NOT set the model-side `target_norm`."""
    with open(os.path.join(CODE_DIR, rel)) as fh:
        cfg = yaml.safe_load(fh)
    assert 'target_norm' not in cfg, rel
    assert cfg['physio_norm'] in ('none', 'clip', 'zscore', 'session'), rel


# --------------------------------------------------------------------------- #
# 6. Stage 3: physio_norm='session' mirrors the Stage-2 target space
# --------------------------------------------------------------------------- #
def test_finetune_session_norm_uses_the_session_statistics(tmp_path):
    root = _write_raw(tmp_path)
    ds = _finetune_ds(root, physio_norm='session')
    assert ds.physio_norm == 'session'
    assert ds.base.norm == 'session'          # the BASE applies the z-score

    _, w = ds[1]                              # (tir, waveform)
    sess = ds.entries[1]['session']
    mu_s, sd_s = ds.base._norm_cache[sess]
    raw = ds.base.respiration_clip(1, normalized=False)
    grid = (np.arange(ds.seq_len) / ds.fs) * ds.resp_fs
    rawg = np.interp(grid, np.arange(raw.shape[0], dtype=np.float64),
                     raw.astype(np.float64))
    assert np.allclose(w.numpy(), (rawg - mu_s) / (sd_s + 1e-8), atol=1e-4)
    # session-relative: the clip offset is KEPT (a per-clip z-score would zero it)
    assert abs(float(w.mean())) > 0.3


def test_finetune_session_norm_is_not_per_clip(tmp_path):
    root = _write_raw(tmp_path)
    _, w_s = _finetune_ds(root, physio_norm='session')[0]
    _, w_c = _finetune_ds(root, physio_norm='zscore')[0]
    assert abs(float(w_c.mean())) < 1e-5      # per-clip: zero mean
    assert abs(float(w_s.mean())) > 0.3       # session: offset preserved
    assert not np.allclose(w_s.numpy(), w_c.numpy(), atol=1e-3)


def test_stage2_and_stage3_target_spaces_are_identical(tmp_path):
    """The point of the aligned pipeline: the SAME clip yields the SAME target
    in Stage 2 (``physio_norm: session`` + pinned identity) and Stage 3."""
    root = _write_raw(tmp_path)
    pre = _pretrain_ds(root, physio_norm='session')
    fin = _finetune_ds(root, physio_norm='session')
    assert len(pre) == len(fin) > 0
    for i in range(len(pre)):
        stage2 = pre[i]['resp'].numpy()[0]
        stage3 = fin[i][1].numpy()
        assert np.allclose(stage2, stage3, atol=1e-5)


def test_finetune_accepts_the_stage2_alias(tmp_path):
    ds = _finetune_ds(_write_raw(tmp_path), physio_norm='clip')
    assert ds.physio_norm == 'zscore'


def test_finetune_rejects_unknown_physio_norm(tmp_path):
    with pytest.raises(ValueError, match="physio_norm must be"):
        _finetune_ds(_write_raw(tmp_path), physio_norm='bogus')


def test_stage3_hpc_config_uses_session_physio_norm():
    path = os.path.join(CODE_DIR, 'configs', 'finetune',
                        'resp_tir_roi_hpc.yaml')
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    assert cfg['physio_norm'] == 'session'
    # the geometry that MUST still mirror the Stage-2 HPC run
    assert cfg['input_size'] == 112
    assert cfg['temporal_stride'] == 2
    assert cfg['sig_kernel'] == 16
    assert cfg['clip_duration'] == pytest.approx(8.0)


def test_run_waveform_exposes_the_session_physio_norm():
    with open(os.path.join(CODE_DIR, 'runners', 'run_waveform.py')) as fh:
        src = fh.read()
    assert "choices=['none', 'ac', 'zscore', 'session']" in src
