"""Integration test for ``data_set: tir_roi_resp_rgb2d`` (plan Step 5.2).

The point of this file is the **join**, not the two caches (each already has its
own suite): one sample must carry ``tir``/``resp`` from the thermal corpus and
``rgb2d`` from the 2D-ROI cache on the *same frames*, and a 2D artefact that
cannot serve a clip must drop that clip rather than silently mis-pair it.

Synthetic tree, one subject / one task:

* ``2D+3D/<S>/<T>/0000.jpg..``    16x20 BGR, 12 frames @ 25 fps  -> 2D cache
* ``2DFeatures/<S>_<T>.mat``      49 'face' landmarks, in frame
* ``Thermal/<S>/<T>.wmv``         stubbed reader, same 16x20 / 12 frames
* ``IRFeatures/<S>_<T>.txt``      28 TIR landmarks, a small in-frame ring
* ``Physiology/<S>/<T>/Resp_Volts.txt``  breathing trace at 100 Hz

With ``clip_seconds=0.16`` (4 frames) the session yields 3 clips; with
``temporal_stride=1`` and ``tubelet_t=2`` the patch grid is ``grid_t=2`` and a
``sig_kernel=8`` waveform gives ``n_signal=2`` -- so the 3-stream alignment
invariant (``grid_t == n_signal``) is exercised for real.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pytest
import torch

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from data import rgb2d_cache as c2d
from data import tir_resp_dataset as trd
from data import trimodal_dataset as td
from data import video_io as vio
import runners.run_build_2d_roi_cache as r2d

FPS = 25.0
RESP_FS = 100.0
CLIP_S = 0.16                       # 4 frames per clip
N_FRAMES = 12
SRC_HW = (16, 20)                   # (h, w) -- tiny frames keep the suite fast
SUBJECT = 'F900'
TASK = 'T1'
INPUT_SIZE = 32
SIG_KERNEL = 8


# --------------------------------------------------------------------------- #
# synthetic corpus
# --------------------------------------------------------------------------- #
class _Reader:
    """Stand-in for ``video_io``: a deterministic 16x20 clip of 12 frames."""

    def __init__(self, path):
        self.num_frames = N_FRAMES
        self.fps = FPS

    def read_range(self, start: int, count: int) -> np.ndarray:
        n = max(0, min(count, self.num_frames - start))
        h, w = SRC_HW
        y, x = np.mgrid[0:h, 0:w]
        frame = np.stack([x, y, (x + y) % 255], axis=-1).astype(np.uint8)
        return np.repeat(frame[None], n, axis=0)

    def read_all(self) -> np.ndarray:
        return self.read_range(0, self.num_frames)

    def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _stub_video(monkeypatch):
    monkeypatch.setattr(vio, 'open_video', lambda path: _Reader(path))
    yield


def _face_landmarks() -> np.ndarray:
    """``[N_FRAMES, 49, 2]`` in-frame 'face' points for a 16x20 source."""
    base = np.zeros((49, 2), np.float64)
    for k in range(49):
        base[k, 0] = 4.0 + 2.0 * (k % 7)        # x 4..16
        base[k, 1] = 4.0 + 1.33 * (k // 7)      # y 4..12
    return np.repeat(base[None], N_FRAMES, axis=0)


def _write_mat(path: str, pts: np.ndarray) -> None:
    from scipy.io import savemat
    fit = np.zeros((1, pts.shape[0]),
                   dtype=[('frame', 'O'), ('pts_2d', 'O'), ('headPose', 'O')])
    for i in range(pts.shape[0]):
        fit[0, i]['frame'] = np.array([[i + 1]], np.uint16)
        fit[0, i]['pts_2d'] = pts[i]
        fit[0, i]['headPose'] = np.zeros((3, 1), np.float32)
    savemat(path, {'fit': fit})


def _write_jpeg(path: str, i: int) -> None:
    import cv2
    h, w = SRC_HW
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = np.linspace(0, 200, w, dtype=np.uint8)[None, :]
    img[..., 1] = np.linspace(0, 200, h, dtype=np.uint8)[:, None]
    img[..., 2] = (i * 10) % 256
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    cv2.imwrite(path, img)


def make_corpus(root: str) -> str:
    """Write the 2D + TIR + RESP tree and return ``raw_root``."""
    raw = str(root)
    os.makedirs(os.path.join(raw, '2D+3D'), exist_ok=True)
    os.makedirs(os.path.join(raw, '2DFeatures'), exist_ok=True)
    _write_mat(os.path.join(raw, '2DFeatures', f'{SUBJECT}_{TASK}.mat'),
               _face_landmarks())
    d2d = os.path.join(raw, '2D+3D', SUBJECT, TASK)
    os.makedirs(d2d, exist_ok=True)
    for i in range(N_FRAMES):
        _write_jpeg(os.path.join(d2d, f'{i:04d}.jpg'), i)

    os.makedirs(os.path.join(raw, 'Thermal', SUBJECT), exist_ok=True)
    with open(os.path.join(raw, 'Thermal', SUBJECT, f'{TASK}.wmv'), 'wb') as fh:
        fh.write(b'\x00')
    os.makedirs(os.path.join(raw, 'IRFeatures'), exist_ok=True)
    ring = []
    for i in range(28):                        # small ring inside the frame
        ang = 2 * np.pi * i / 28.0
        ring += [f'{9.0 + 4.0 * np.cos(ang):.2f}', f'{8.0 + 3.0 * np.sin(ang):.2f}']
    line = ' '.join(ring)
    with open(os.path.join(raw, 'IRFeatures', f'{SUBJECT}_{TASK}.txt'),
              'w') as fh:
        fh.write('\n'.join([line] * N_FRAMES) + '\n')

    phys = os.path.join(raw, 'Physiology', SUBJECT, TASK)
    os.makedirs(phys, exist_ok=True)
    t = np.arange(int(round(N_FRAMES / FPS * RESP_FS))) / RESP_FS
    trace = 3.0 + 2.0 * np.sin(2 * np.pi * 0.3 * t)
    with open(os.path.join(phys, 'Resp_Volts.txt'), 'w') as fh:
        fh.write('\n'.join(f'{v:.5f}' for v in trace) + '\n')
    return raw


def build_2d_cache(raw: str, out: str, **kw) -> str:
    """Two invocations, exactly as the sbatch wrapper does (init then build)."""
    common = ['--raw_root', raw, '--out_root', out,
              '--clip_seconds', str(kw.pop('clip_seconds', CLIP_S)),
              '--fps', str(FPS),
              '--store_resolution', str(kw.pop('store', 'native'))]
    assert r2d.main(['--init'] + common) == 0
    assert r2d.main(['--build'] + common) == 0
    return out


def dataset_args(raw: str, cache: str, **kw) -> argparse.Namespace:
    return argparse.Namespace(
        raw_root=raw, streams='tir,rgb2d,resp', fs=RESP_FS, resp_fs=RESP_FS,
        fps=FPS, clip_duration=kw.pop('clip_seconds', CLIP_S),
        clip_stride=None, temporal_stride=kw.pop('temporal_stride', 1),
        tubelet=f'{kw.pop("tubelet_t", 2)},16,16',
        input_size=kw.pop('input_size', INPUT_SIZE),
        roi_padding=0.10, crop_rule='square',
        max_sway_px=kw.pop('max_sway_px', 40.0),
        roi_landmarks=None, min_signal_spread=0.0, rail_touch_v=0.0,
        physio_norm=trd.TARGET_NORM, resp_fs_=RESP_FS,
        subjects=None, tasks=None, task_groups=None, task_set=None,
        max_clips=None, max_entries=None,
        roi_cache=None, resp_cache=None,
        rgb2d_cache=cache, rgb2d_cache_lru=kw.pop('lru', 4),
        rgb2d_store=kw.pop('store', c2d.STORE_RESOLUTION),
        rgb2d_landmarks='face',
        rgb2d_min_valid_ratio=kw.pop('min_valid_ratio', 1.0),
        rgb2d_max_sway_px=kw.pop('max_sway_px_2d', 40.0),
        rgb2d_area_min_frac=kw.pop('area_min_frac', 0.0),
        sig_kernel=kw.pop('sig_kernel', SIG_KERNEL),
    )


@pytest.fixture()
def built(tmp_path):
    raw = make_corpus(str(tmp_path / 'raw'))
    out = str(tmp_path / 'cache')
    build_2d_cache(raw, out)
    return raw, out


# --------------------------------------------------------------------------- #
# the join
# --------------------------------------------------------------------------- #
def test_one_cache_serves_any_input_size(built):
    """Store native, rescale at load: ``input_size`` is a load-time decision."""
    raw, cache = built
    shapes = {}
    for size in (INPUT_SIZE, 16):
        ds = td.build_tir_roi_rgb2d_pretrain_dataset(
            dataset_args(raw, cache, input_size=size, sig_kernel=SIG_KERNEL))
        item = ds[0]
        shapes[size] = tuple(item['rgb2d'].shape)
        # identical geometry on every declared stream
        assert item['rgb2d'].shape[1] == item['tir'].shape[1]
        assert item['rgb2d'].shape[0] == 3
        assert item['rgb2d'].dtype == torch.float32
    assert shapes[INPUT_SIZE] == (3, 4, INPUT_SIZE, INPUT_SIZE)
    assert shapes[16] == (3, 4, 16, 16)


def test_sample_carries_three_streams_on_the_same_frames(built):
    raw, cache = built
    ds = td.build_tir_roi_rgb2d_pretrain_dataset(dataset_args(raw, cache))
    item = ds[0]
    assert set(item) >= {'tir', 'resp', 'rgb2d'}
    t = item['tir'].shape[1]
    assert item['rgb2d'].shape[1] == t == int(round(CLIP_S * FPS))
    assert item['resp'].shape[-1] == int(round(CLIP_S * RESP_FS))
    assert item['rgb2d'].min() >= 0.0 and item['rgb2d'].max() <= 1.0


def test_pixel_store_serves_the_same_clips_and_is_kept_apart(tmp_path):
    """A ``px112`` store produces the same sample shape as the native one."""
    raw = make_corpus(str(tmp_path / 'raw'))
    out = str(tmp_path / 'cache')
    build_2d_cache(raw, out, store=112)
    ds = td.build_tir_roi_rgb2d_pretrain_dataset(
        dataset_args(raw, out, store=112, sig_kernel=SIG_KERNEL))
    assert 'px112' in ds.rgb2d_cache_root
    item = ds[0]
    assert tuple(item['rgb2d'].shape) == (3, 4, INPUT_SIZE, INPUT_SIZE)
    assert item['rgb2d'].shape[1] == item['tir'].shape[1]
    assert item['rgb2d'].min() >= 0.0 and item['rgb2d'].max() <= 1.0
    # asking for the NATIVE store against a px112-only tree fails loudly
    # instead of silently attaching the wrong pixels
    with pytest.raises(ValueError, match='Closest existing cache'):
        td.build_tir_roi_rgb2d_pretrain_dataset(
            dataset_args(raw, out, store='native', sig_kernel=SIG_KERNEL))


def test_temporal_stride_shortens_both_visual_streams(built):
    raw, cache = built
    ds = td.build_tir_roi_rgb2d_pretrain_dataset(
        dataset_args(raw, cache, temporal_stride=2, tubelet_t=2,
                     sig_kernel=4))
    item = ds[0]
    assert item['rgb2d'].shape[1] == item['tir'].shape[1] == 2


def test_a_skipped_2d_session_is_excluded_and_never_silently_empty(built):
    raw, cache = built
    ds = td.build_tir_roi_rgb2d_pretrain_dataset(dataset_args(raw, cache))
    assert len(ds) > 0
    # Re-declare the only session as skipped, as the builder does on failure.
    c2d.write_skip(ds.rgb2d_cache_root, f'{SUBJECT}_{TASK}', 'synthetic failure')
    with pytest.raises(RuntimeError, match='sessions_excluded'):
        td.build_tir_roi_rgb2d_pretrain_dataset(dataset_args(raw, cache))


def test_a_missing_shard_excludes_the_session_and_is_counted(built):
    """Coverage gaps exclude; they do not abort the run.

    A session the cache does not hold yet (its archive was still being copied
    when the cache was built) is excluded and counted here, not fatal — and a
    marker is never written for it, because the copy finishes later.
    """
    raw, cache = built
    ds = td.build_tir_roi_rgb2d_pretrain_dataset(dataset_args(raw, cache))
    assert ds.stats['sessions_2d_missing'] == 0
    os.remove(c2d.frames_path(ds.rgb2d_cache_root, f'{SUBJECT}_{TASK}'))
    with pytest.raises(RuntimeError, match='missing=3'):
        td.build_tir_roi_rgb2d_pretrain_dataset(dataset_args(raw, cache))


def test_an_envelope_gate_can_empty_the_dataset(built):
    """``area_min_frac`` is a downstream decision the cache does not encode."""
    raw, cache = built
    with pytest.raises(RuntimeError, match='envelope=3'):
        td.build_tir_roi_rgb2d_pretrain_dataset(
            dataset_args(raw, cache, area_min_frac=1.0))


def test_declared_streams_must_include_rgb2d(built):
    raw, cache = built
    args = dataset_args(raw, cache)
    args.streams = 'tir,resp'
    with pytest.raises((ValueError, AssertionError)):
        td.build_tir_roi_rgb2d_pretrain_dataset(args)


# --------------------------------------------------------------------------- #
# the motion gate's two numbers live in two places -> guard the drift
# --------------------------------------------------------------------------- #
def test_motion_gates_are_the_measured_pair():
    """``rgb2d`` 110 px is the conversion of TIR's 40 px, not a copy.

    Both are measured (`code/notes/eval/dataset_inspection/rgb2d/sway_retention.md`,
    `tir_rgb2d_sway_ratio.py`). The library default and the shipped config must
    agree, otherwise a change to one silently leaves the other stale -- the
    config overrides the default, so the two never contradict each other at
    runtime and nothing else would notice.
    """
    import yaml
    assert td.DEFAULT_RGB2D_MAX_SWAY_PX == 110.0
    assert trd.DEFAULT_MAX_SWAY_PX == 40.0
    cfg_path = os.path.join(CODE_DIR, 'configs', 'pretrain',
                            'stage2_budget_tir_roi_resp_rgb2d.yaml')
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    assert cfg['max_sway_px'] == trd.DEFAULT_MAX_SWAY_PX
    assert cfg['rgb2d_max_sway_px'] == td.DEFAULT_RGB2D_MAX_SWAY_PX
    # a px gate is modality-relative: copying TIR's number here would reject a
    # third of the 2D clips
    assert cfg['rgb2d_max_sway_px'] != cfg['max_sway_px']
