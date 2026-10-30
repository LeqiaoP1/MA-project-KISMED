"""The motion gate -- ``max_sway_px``, invariant I10 of the ROI-crop contract.

A clip-static box only means something if the head does not leave it. The gate
drops a window whose target-landmark centroid travels further than
``max_sway_px`` SOURCE pixels from its own within-clip median, so the box is not
inflated into a mostly-background crop:

    sway = max_t || c_t - median_tau(c_tau) ||_2,   c_t = mean of the target points

Two properties are pinned here, because both are easy to get wrong:

* the statistic reads the TRACK only (no decode, no cache) and ignores
  non-target landmarks -- ``crop_rule``/``max_sway_px`` decide WHICH CLIPS EXIST,
  so a drift between Stage 2 and Stage 3 would silently change the corpus;
* the gate is a SELECTION filter: it removes whole windows and is counted, but it
  never touches a stored pixel (which is why it is not in the cache key).

The synthetic tree mirrors ``tests/test_tir_resp_sentinel.py``: one session, one
IR track, one ``Resp_Volts.txt`` and a stubbed container probe (the build path
reads ``num_frames``/``fps``, never a decoded frame).
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
CLIP_S = 8.0
STRIDE_S = 2.0
N_FRAMES = 400                          # 16 s -> 5 overlapping 8 s windows
CLIP_FRAMES = int(CLIP_S * FPS)
N_RESP = int(N_FRAMES / FPS * RESP_FS)
IDX = trd.TARGET_LANDMARK_IDX


class _FakeReader:
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


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _static_track(n=N_FRAMES):
    """A perfectly still head: every target point at (350, 240)."""
    ir = np.zeros((n, trd.NUM_LANDMARKS, 2), np.float32)
    ir[:, IDX, :] = (350.0, 240.0)
    return ir


def _swept_track(n=N_FRAMES, first=50, shift=60.0):
    """The head sits ``shift`` px to the right for the FIRST ``first`` frames."""
    ir = _static_track(n)
    ir[:first, IDX, 0] += shift
    return ir


def _ir_text(ir):
    return '\n'.join(' '.join(f'{v:.2f}' for v in row.ravel())
                     for row in ir) + '\n'


def _breathing(n=N_RESP, amp=4.0):
    t = np.arange(n) / RESP_FS
    return -4.0 + amp * np.sin(2 * np.pi * 0.3 * t)


def _write_tree(root, ir_text):
    (root / 'Thermal' / 'S001').mkdir(parents=True)
    (root / 'Thermal' / 'S001' / 'T1.wmv').write_bytes(b'\x00')
    (root / 'IRFeatures').mkdir()
    (root / 'IRFeatures' / 'S001_T1.txt').write_text(ir_text)
    phys = root / 'Physiology' / 'S001' / 'T1'
    phys.mkdir(parents=True)
    (phys / 'Resp_Volts.txt').write_text(
        '\n'.join(f'{v:.4f}' for v in _breathing()) + '\n')
    return str(root)


def _build(root, **kw):
    kw.setdefault('min_signal_spread', 0.0)
    kw.setdefault('rail_touch_v', 0.0)
    kw.setdefault('clip_stride', STRIDE_S)
    return trd.BP4DPlusTIRRespDataset(
        raw_root=root, clip_seconds=CLIP_S,
        input_size=64, roi_padding=0.2, fps=FPS, resp_fs=RESP_FS,
        allow_empty=True, **kw)


# --------------------------------------------------------------------------- #
# the statistic
# --------------------------------------------------------------------------- #
def test_sway_of_a_still_head_is_zero():
    assert trd.landmark_sway(_static_track(8), IDX) == 0.0


def test_sway_is_the_largest_centroid_excursion():
    ir = _static_track(3)
    ir[1, IDX, 0] += 45.0
    assert trd.landmark_sway(ir, IDX) == pytest.approx(45.0)

    ir2 = _static_track(3)
    ir2[2, IDX, 0] += 30.0
    ir2[2, IDX, 1] += 40.0                      # 3-4-5: the norm, not max axis
    assert trd.landmark_sway(ir2, IDX) == pytest.approx(50.0)


def test_sway_ignores_non_target_landmarks():
    ir = _static_track(5)
    ir[2, 0, :] += 500.0                        # label 1: not in the ROI set
    assert trd.landmark_sway(ir, IDX) == 0.0


def test_sway_is_infinite_for_a_nonfinite_target():
    ir = _static_track(5)
    ir[3, IDX[4], 0] = np.nan                   # the sentinel gate drops these
    assert not np.isfinite(trd.landmark_sway(ir, IDX))


def test_sway_of_one_frame_is_zero():
    assert trd.landmark_sway(_static_track(1), IDX) == 0.0


def test_sway_rejects_a_wrong_shape():
    with pytest.raises(ValueError, match='expected'):
        trd.landmark_sway(np.zeros((4, 3)), IDX)


# --------------------------------------------------------------------------- #
# the gate, in the dataset
# --------------------------------------------------------------------------- #
def test_sway_above_the_threshold_drops_the_window(tmp_path):
    """60 px of head travel: the windows overlapping it are dropped."""
    root = _write_tree(tmp_path, _ir_text(_swept_track(first=50, shift=60.0)))
    ds = _build(root, max_sway_px=40.0)
    assert ds.stats['clips_dropped_motion'] == 1
    assert len(ds.entries) == 4                  # 5 windows - the swept one
    assert ds.stats['clips'] == 4
    assert 'max_sway_px' in ds.describe()
    assert '1 for head sway' in ds.describe()


def test_sway_below_the_threshold_keeps_the_window(tmp_path):
    root = _write_tree(tmp_path, _ir_text(_swept_track(first=50, shift=35.0)))
    ds = _build(root, max_sway_px=40.0)
    assert ds.stats['clips_dropped_motion'] == 0
    assert len(ds.entries) == 5


def test_zero_disables_the_gate(tmp_path):
    root = _write_tree(tmp_path, _ir_text(_swept_track(first=50, shift=60.0)))
    ds = _build(root, max_sway_px=0.0)
    assert ds.stats['clips_dropped_motion'] == 0
    assert len(ds.entries) == 5


def test_the_gate_never_removes_a_whole_session(tmp_path):
    """A dropped window is a clip filter, not a session filter."""
    root = _write_tree(tmp_path, _ir_text(_swept_track(first=50, shift=200.0)))
    ds = _build(root, max_sway_px=40.0)
    assert ds.stats['sessions_used'] == 1
    assert len(ds.entries) >= 1


def test_no_kept_window_exceeds_the_threshold(tmp_path):
    """The invariant, checked on every emitted entry."""
    root = _write_tree(tmp_path, _ir_text(_swept_track(first=50, shift=60.0)))
    ds = _build(root, max_sway_px=40.0)
    for e in ds.entries:
        ir = ds._landmarks(e['session'])[e['frame_start']:e['frame_end']]
        assert trd.landmark_sway(ir, ds.target_idx) <= 40.0


def test_a_negative_threshold_is_rejected():
    with pytest.raises(ValueError, match='max_sway_px'):
        trd.BP4DPlusTIRRespDataset(raw_root='/nonexistent', max_sway_px=-1.0)


# --------------------------------------------------------------------------- #
# the inspector's diagnosis: a dropped window is NOT addressable by index
# --------------------------------------------------------------------------- #
def test_inspector_diagnoses_the_motion_gate(tmp_path):
    """``--t_start`` on a gated window must name the gate, not say PASS."""
    from runners.run_inspect_tir_resp import _diagnose_window

    root = _write_tree(tmp_path, _ir_text(_swept_track(first=50, shift=60.0)))
    ds = _build(root, max_sway_px=40.0, clip_stride=None)   # hop == clip
    assert [int(e['frame_start']) for e in ds.entries] == [200]

    lines, reasons = _diagnose_window(ds, 0)
    assert reasons == ['motion']
    assert any('MOTION' in ln and '60.0 px' in ln for ln in lines)


def test_inspector_recognises_a_present_window(tmp_path):
    from runners.run_inspect_tir_resp import _diagnose_window

    root = _write_tree(tmp_path, _ir_text(_static_track()))
    ds = _build(root, max_sway_px=40.0, clip_stride=None)
    lines, reasons = _diagnose_window(ds, 200)
    assert reasons == ['present']
    assert 'nothing to diagnose' in lines[0]


def test_inspector_diagnoses_a_sentinel_window(tmp_path):
    from runners.run_inspect_tir_resp import _diagnose_window

    ir = _static_track()
    ir[10] = 0.0                                   # whole-row sentinel
    root = _write_tree(tmp_path, _ir_text(ir))
    ds = _build(root, max_sway_px=40.0, clip_stride=None)
    lines, reasons = _diagnose_window(ds, 0)
    assert reasons == ['sentinel']
    assert any('SENTINEL' in ln for ln in lines)


def test_inspector_coverage_metrics_use_the_target_landmarks_only():
    """``envelope_frac_of_patch`` must be <= 1: the target envelope is IN the box.

    It was computed over all 28 landmarks while the box comes from the 12 target
    points, which reported 1.36 on a real session (impossible) and silently
    disabled the ``envelope_lt_25pct_patch`` flag.
    """
    from runners.run_inspect_tir_resp import _coverage

    lm = np.zeros((4, trd.NUM_LANDMARKS, 2), np.float32)
    lm[:, IDX, :] = (300.0, 240.0)                 # target cloud: a point
    lm[:, IDX[0], :] = (340.0, 280.0)              # ... 40 x 40 px
    lm[:, 0, :] = (0.0, 0.0)                       # a NON-target far away
    box = (280, 380, 220, 320)                     # 100 x 100, contains them
    cov = _coverage(box, lm, IDX, 480, 726)
    assert cov['is_square'] and cov['side_px'] == 100
    assert cov['targets_inside']
    assert cov['envelope_frac_of_patch'] == pytest.approx((41 * 41) / 100 ** 2,
                                                          rel=0.05)
    assert cov['envelope_frac_of_patch'] <= 1.0
    # a box that misses the cloud is flagged, not silently accepted
    cov2 = _coverage((10, 110, 10, 110), lm, IDX, 480, 726)
    assert not cov2['targets_inside']
