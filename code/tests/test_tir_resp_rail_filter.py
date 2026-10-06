"""Unit tests for the TIR-ROI/RESP respiration rail-touch clip filter.

The filter lives in ``data/tir_resp_dataset.BP4DPlusTIRRespDataset``:

* ``rail_touch_v`` (the cleaning knob) drops a clip whose respiration window
  contains ANY sample that TOUCHED the recorder rail, i.e. ``abs(x) >=
  rail_touch_v`` volts. The recorder clamps at ``+/-10 V`` (the negative end is
  :data:`data.tir_resp_dataset.RESP_RAIL_V`), so the shipped ``9.90`` rejects
  every window whose label holds a rail-valued reading -- a dead/pinned sensor
  AND a genuinely clipped flat-topped trough;
* ``min_signal_spread`` (degenerate-window guard) drops a (near-)constant
  window whatever its cause -- they are complementary, not nested.

These tests build a MINIMAL synthetic raw tree (one session, one IR track, one
``Resp_Volts.txt``) and stub only the video container probe: the build path
reads ``num_frames``/``fps`` and never asks ``cv2`` to decode a frame.
"""
import os
import sys

import numpy as np
import pytest

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from data import tir_resp_dataset as trd
from data import video_io as vio

FPS = 25.0
RESP_FS = 1000.0
N_FRAMES = 400                            # 16 s -> two 8 s windows
CLIP_S = 8.0
N_RESP = int(N_FRAMES / FPS * RESP_FS)    # 16000 samples


class _FakeReader:
    """Stand-in for ``data.video_io``'s reader (container-probe fields only)."""

    def __init__(self, num_frames, fps):
        self.num_frames = num_frames
        self.fps = fps

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _stub_video(monkeypatch):
    monkeypatch.setattr(vio, 'open_video',
                        lambda path: _FakeReader(N_FRAMES, FPS))
    yield


def _write_raw(root, resp: np.ndarray) -> str:
    """Minimal raw tree: one session ``S001_T1`` with IR + Resp_Volts."""
    (root / 'Thermal' / 'S001').mkdir(parents=True)
    (root / 'Thermal' / 'S001' / 'T1.wmv').write_bytes(b'\x00')
    (root / 'IRFeatures').mkdir()
    # 28 landmarks x 2 coords = 56 non-zero values -> no (0,0) sentinel
    line = ' '.join(['100.0', '100.0'] * 28)
    (root / 'IRFeatures' / 'S001_T1.txt').write_text(
        '\n'.join([line] * N_FRAMES) + '\n')
    phys = root / 'Physiology' / 'S001' / 'T1'
    phys.mkdir(parents=True)
    (phys / 'Resp_Volts.txt').write_text(
        '\n'.join(f'{v:.4f}' for v in resp) + '\n')
    return str(root)


def _build(root, **kw):
    kw.setdefault('min_signal_spread', 0.0)
    return trd.BP4DPlusTIRRespDataset(
        raw_root=root, clip_seconds=CLIP_S, input_size=64, roi_padding=0.2,
        fps=FPS, resp_fs=RESP_FS, allow_empty=True, **kw)


def _breathing(n: int, amp: float = 4.0) -> np.ndarray:
    """Healthy respiration in [-8, 0] V -- never touches the +/-10 V rail."""
    t = np.arange(n) / RESP_FS
    return -4.0 + amp * np.sin(2 * np.pi * 0.3 * t)


def _clipped_troughs(run_s: float) -> np.ndarray:
    """Healthy breathing whose trough is flat-topped AT the rail for ``run_s``
    seconds at the start of each 8 s window (the genuine-clipping signature)."""
    y = _breathing(N_RESP)
    for start in (0, N_RESP // 2):
        y[start:start + int(round(run_s * RESP_FS))] = trd.RESP_RAIL_V
    return y


def _pinned_baseline(run_s: float) -> np.ndarray:
    """Channel pinned at the rail between rare 0.2 s spikes (sensor failure)."""
    y = np.full(N_RESP, trd.RESP_RAIL_V)
    step = int(round(run_s * RESP_FS))
    for start in range(0, N_RESP, step):
        y[start:start + 200] = -5.0
    return y


def _spike(sample: int, value: float) -> np.ndarray:
    """Healthy breathing with ONE sample set to ``value``: index 100 falls in
    the FIRST 8 s window, index 8100 in the second."""
    y = _breathing(N_RESP)
    y[sample] = value
    return y


# --------------------------------------------------------------------------- #
# dead / fully railed channel -> dropped
# --------------------------------------------------------------------------- #
def test_fully_railed_clip_is_dropped(tmp_path):
    root = _write_raw(tmp_path, np.full(N_RESP, trd.RESP_RAIL_V))
    ds = _build(root, rail_touch_v=9.90)
    assert len(ds) == 0
    assert ds.stats['clips_dropped_rail'] == 2
    assert ds.stats['clips_dropped_flat'] == 0


def test_fully_railed_clip_kept_when_the_rule_is_off(tmp_path):
    root = _write_raw(tmp_path, np.full(N_RESP, trd.RESP_RAIL_V))
    assert len(_build(root)) == 2                        # rail_touch_v=0.0


def test_pinned_baseline_is_dropped(tmp_path):
    """Baseline pinned at the rail between rare spikes (sensor failure)."""
    root = _write_raw(tmp_path, _pinned_baseline(6.0))
    ds = _build(root, rail_touch_v=9.90)
    assert len(ds) == 0
    assert ds.stats['clips_dropped_rail'] == 2


# --------------------------------------------------------------------------- #
# a clipped trough is dropped TOO -- the deliberate point of the strict rule   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('run_s', [0.5, 1.0, 2.5, 4.0])
def test_clipped_trough_is_dropped(tmp_path, run_s):
    """A genuine extreme expiration drives the trough into the recorder clamp,
    so the LABEL itself holds unrepresentable rail-valued samples: the clip goes.
    The rule keys on the sample VALUE, not on how much of the window is railed,
    so even a half-second clipped trough is enough."""
    root = _write_raw(tmp_path, _clipped_troughs(run_s))
    ds = _build(root, rail_touch_v=9.90)
    assert len(ds) == 0
    assert ds.stats['clips_dropped_rail'] == 2


def test_clipped_trough_is_kept_when_the_rule_is_off(tmp_path):
    root = _write_raw(tmp_path, _clipped_troughs(1.0))
    assert len(_build(root)) == 2                        # rail_touch_v=0.0


def test_threshold_beyond_the_recorder_range_keeps_everything(tmp_path):
    """The recorder clamps at +/-10 V, so a threshold above it is never touched."""
    root = _write_raw(tmp_path, _clipped_troughs(1.0))
    ds = _build(root, rail_touch_v=10.5)
    assert len(ds) == 2
    assert ds.stats['clips_dropped_rail'] == 0


# --------------------------------------------------------------------------- #
# it is a TOUCH rule: ONE rail-valued sample is enough, and it is two-sided    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('value', [-10.0, -9.95, 9.95, 10.0])
def test_one_rail_sample_drops_its_window(tmp_path, value):
    """Sample 100 lives in the FIRST 8 s window -> exactly 1 of the 2 clips
    goes; the second window is untouched and survives."""
    root = _write_raw(tmp_path, _spike(100, value))
    ds = _build(root, rail_touch_v=9.90)
    assert len(ds) == 1
    assert ds.stats['clips_dropped_rail'] == 1
    assert ds.entries[0]['frame_start'] == N_FRAMES // 2


def test_near_rail_but_inside_is_kept(tmp_path):
    """A deep excursion that never REACHED the rail is kept."""
    root = _write_raw(tmp_path, _spike(100, -9.0))
    ds = _build(root, rail_touch_v=9.90)
    assert len(ds) == 2
    assert ds.stats['clips_dropped_rail'] == 0


def test_threshold_is_compared_in_float32(tmp_path):
    """A sample valued EXACTLY at the threshold must drop the clip. The data
    are float32 (``-9.9000`` -> ``-9.8999996``), so a raw float64
    ``abs(x) >= 9.9`` test would silently MISS it -- the threshold is therefore
    quantised to float32 once in ``__init__`` (the same representability trap
    that hides ``-9.999`` from a ``<= -9.999`` test on float32 data)."""
    root = _write_raw(tmp_path, _spike(100, -9.90))
    ds = _build(root, rail_touch_v=9.90)
    assert len(ds) == 1
    assert ds.stats['clips_dropped_rail'] == 1


def test_just_inside_the_threshold_is_kept(tmp_path):
    root = _write_raw(tmp_path, _spike(100, -9.89))
    ds = _build(root, rail_touch_v=9.90)
    assert len(ds) == 2
    assert ds.stats['clips_dropped_rail'] == 0


# --------------------------------------------------------------------------- #
# the two rules are complementary + attribution order
# --------------------------------------------------------------------------- #
def test_constant_non_rail_window_is_a_spread_problem_not_a_rail_one(tmp_path):
    """A dead channel stuck at -5 V: min_signal_spread catches it, the rail
    touch rule is blind to it."""
    root = _write_raw(tmp_path, np.full(N_RESP, -5.0))
    assert len(_build(root, rail_touch_v=9.90)) == 2
    ds = _build(root, min_signal_spread=0.01)
    assert len(ds) == 0
    assert ds.stats['clips_dropped_flat'] == 2
    assert ds.stats['clips_dropped_rail'] == 0


def test_rail_touch_rule_is_checked_first(tmp_path):
    """A fully railed window is ALSO 'flat'; the rail test must claim it so
    clips_dropped_rail is not undercounted."""
    root = _write_raw(tmp_path, np.full(N_RESP, trd.RESP_RAIL_V))
    ds = _build(root, min_signal_spread=0.01, rail_touch_v=9.90)
    assert ds.stats['clips_dropped_rail'] == 2
    assert ds.stats['clips_dropped_flat'] == 0


# --------------------------------------------------------------------------- #
# validation / healthy data
# --------------------------------------------------------------------------- #
def test_rail_touch_v_must_be_non_negative(tmp_path):
    root = _write_raw(tmp_path, _breathing(N_RESP))
    with pytest.raises(ValueError, match='rail_touch_v'):
        _build(root, rail_touch_v=-0.1)


def test_healthy_clip_survives_both_rules(tmp_path):
    root = _write_raw(tmp_path, _breathing(N_RESP))
    ds = _build(root, min_signal_spread=0.01, rail_touch_v=9.90)
    assert len(ds) == 2
    assert ds.stats['clips_dropped_rail'] == 0
    assert ds.stats['clips_dropped_flat'] == 0
    assert ds.stats['clips_dropped_flat'] == 0
