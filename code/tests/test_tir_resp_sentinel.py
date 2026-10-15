"""Landmark-lost ("sentinel") filtering -- the dataset's Requirement 2.

``data.tir_resp_dataset.missing_frame_mask`` marks a thermal frame unusable when
it is

* an all-``(0,0)`` line (the tracker's missing-data sentinel),
* unparseable (an all-NaN row out of ``parse_ir_features``), or
* MISSING A TARGET LANDMARK at ``(0,0)`` -- the corner-stretch guard: a single
  such pair would blow a min/max ROI box out to the image corner.

``BP4DPlusTIRRespDataset._build_entries`` then drops EVERY clip whose window
overlaps any such frame. This file pins both halves: the mask's exact rule, and
the window-level drop the real Stage-3 run depends on (on real data a tracker
dropout at t=5.6-6.9 s of ``M030_T4`` left exactly one surviving clip, starting
at t=8 s).

The synthetic tree mirrors ``tests/test_tir_resp_rail_filter.py``: one session,
one IR track, one ``Resp_Volts.txt``, and a stubbed video CONTAINER probe (the
build path only reads ``num_frames``/``fps``, never a decoded frame).
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
N_FRAMES = 400                            # 16 s -> 5 overlapping 8 s windows
CLIP_FRAMES = int(CLIP_S * FPS)
N_RESP = int(N_FRAMES / FPS * RESP_FS)
LABELS = trd.TARGET_LANDMARKS
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
def _track(n=N_FRAMES, seed=0):
    """``(n, 28, 2)`` of plausible, fully usable landmark rows."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(300, 400, size=(n, trd.NUM_LANDMARKS)),
                     rng.uniform(180, 300, size=(n, trd.NUM_LANDMARKS))],
                    axis=-1).astype(np.float32)


def _ir_text(ir):
    return '\n'.join(' '.join(f'{v:.2f}' for v in row.ravel())
                     for row in ir) + '\n'


def _breathing(n=N_RESP, amp=4.0):
    """Healthy respiration in [-8, 0] V -- never trips the rail/spread rules."""
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
    """Overlapping windows; the volt rules OFF so only the sentinel can fire."""
    kw.setdefault('min_signal_spread', 0.0)
    kw.setdefault('rail_touch_v', 0.0)
    return trd.BP4DPlusTIRRespDataset(
        raw_root=root, clip_seconds=CLIP_S, clip_stride=STRIDE_S,
        input_size=64, roi_padding=0.2, fps=FPS, resp_fs=RESP_FS,
        allow_empty=True, **kw)


def _sentinel_frames(frames):
    """Frame indices the mask rejects -- the ground truth the drop must honour."""
    return set(np.flatnonzero(trd.missing_frame_mask(frames, IDX)).tolist())


# --------------------------------------------------------------------------- #
# missing_frame_mask -- the exact rule
# --------------------------------------------------------------------------- #
def test_mask_accepts_a_fully_usable_track():
    assert not trd.missing_frame_mask(_track(12), IDX).any()


def test_mask_flags_a_whole_row_sentinel():
    ir = _track(6)
    ir[3] = 0.0
    np.testing.assert_array_equal(
        trd.missing_frame_mask(ir, IDX), [False, False, False, True, False, False])


def test_mask_flags_an_unparseable_row():
    """A blank / non-numeric line becomes an all-NaN row -- also unusable."""
    ir = _track(4)
    ir[1] = np.nan
    assert trd.missing_frame_mask(ir, IDX)[1]


def test_mask_flags_a_row_with_ANY_target_landmark_at_the_origin():
    """ONE degenerate target point is enough -- the corner-stretch guard."""
    ir = _track(4)
    ir[2, IDX[5]] = 0.0
    assert trd.missing_frame_mask(ir, IDX)[2]
    assert trd.missing_frame_mask(ir, IDX).sum() == 1


def test_mask_ignores_a_non_target_landmark_at_the_origin():
    """The guard is scoped to the ROI's own landmarks, not all 28."""
    ir = _track(4)
    ir[1, 0] = 0.0                        # label 1 is in no ROI preset
    assert not trd.missing_frame_mask(ir, IDX).any()


def test_mask_only_guards_the_requested_landmark_subset():
    """A config guarding (10,21) must not police the mouth corners."""
    ir = _track(4)
    ir[1, 10] = 0.0                       # label 11 = mouth corner R
    assert trd.missing_frame_mask(ir, IDX)[1]
    assert not trd.missing_frame_mask(ir, (9, 20))[1]


def test_mask_rejects_a_wrong_shape():
    with pytest.raises(ValueError, match='expected'):
        trd.missing_frame_mask(np.zeros((5, 10, 2), np.float32), IDX)


# --------------------------------------------------------------------------- #
# parse_ir_features -- what feeds the mask
# --------------------------------------------------------------------------- #
def test_parser_turns_a_blank_line_into_a_nan_row(tmp_path):
    p = tmp_path / 'IR.txt'
    p.write_text(_ir_text(_track(2)).replace('\n', '\n\n', 1))
    ir = trd.parse_ir_features(str(p))
    assert ir.shape == (3, trd.NUM_LANDMARKS, 2)
    assert np.isnan(ir[1]).all(), 'a blank line must NaN-fill, keeping line==frame'
    assert trd.missing_frame_mask(ir, IDX)[1]


def test_parser_refuses_a_wrong_token_count(tmp_path):
    p = tmp_path / 'IR.txt'
    p.write_text(_ir_text(_track(2)) + '1.0 2.0\n')
    with pytest.raises(ValueError, match='refusing to guess'):
        trd.parse_ir_features(str(p))


# --------------------------------------------------------------------------- #
# dataset level: the WINDOW drop
# --------------------------------------------------------------------------- #
def test_a_sentinel_frame_drops_every_overlapping_window(tmp_path):
    """A dropout at frame 100 kills windows starting 0/50/100, not 150/200."""
    ir = _track()
    ir[100] = 0.0
    ds = _build(_write_tree(tmp_path, _ir_text(ir)))

    assert ds.stats['clips_dropped_sentinel'] == 3
    assert [e['frame_start'] for e in ds.entries] == [150, 200]


def test_no_kept_clip_ever_overlaps_a_sentinel_frame(tmp_path):
    """The property the whole rule exists for -- checked on a scatter of rows.

    Both branches of the rule are planted: a WHOLE-ROW sentinel and a
    single-degenerate-TARGET frame. With 8 s clips at a 2 s hop they land inside
    windows [0:200), [50:250) and [100:300), leaving [150:350) and [200:400)
    clean -- so there is at least something kept to check.
    """
    ir = _track(seed=7)
    ir[40] = 0.0                          # whole-row sentinel
    ir[130, IDX[5]] = 0.0                 # ONE target landmark at the origin
    ds = _build(_write_tree(tmp_path, _ir_text(ir)))

    bad = _sentinel_frames(ir)
    assert bad == {40, 130}, 'the fixture must actually contain both kinds'
    assert [e['frame_start'] for e in ds.entries] == [150, 200]
    for e in ds.entries:
        covered = set(range(e['frame_start'], e['frame_end']))
        assert not (covered & bad), f'{e["session"]} kept an overlapping clip'


def test_landmark_dropout_leaves_only_the_clean_window(tmp_path):
    """The real M030_T4 shape: a 1.3 s dropout at t=5.6-6.9 s.

    Frames 141-173 fall inside the first four windows, so only start=200
    (t = 8.0 s) survives -- exactly what produced the single-clip session whose
    assembled figure began with an 8 s gap.
    """
    ir = _track()
    ir[141:174] = 0.0
    ds = _build(_write_tree(tmp_path, _ir_text(ir)))

    assert ds.stats['clips_dropped_sentinel'] == 4
    assert len(ds) == 1
    assert ds.entries[0]['frame_start'] == 200
    assert ds.entries[0]['t_start'] == pytest.approx(8.0)


def test_every_window_dropped_skips_the_session(tmp_path):
    ir = _track()
    ir[:] = 0.0
    ds = _build(_write_tree(tmp_path, _ir_text(ir)))

    assert len(ds) == 0
    assert ds.stats['sessions_used'] == 0
    assert ds.stats['sessions_skipped'] == 1
    reason = ds.skipped[0]['reason']
    assert reason.startswith('all_clips_dropped')
    assert 'sentinel' in reason


def test_the_sentinel_rule_is_independent_of_the_volt_rules(tmp_path):
    """A clean track keeps every window once the volt rules are off."""
    ds = _build(_write_tree(tmp_path, _ir_text(_track())))
    assert ds.stats['clips_dropped_sentinel'] == 0
    assert [e['frame_start'] for e in ds.entries] == [0, 50, 100, 150, 200]


def test_describe_reports_the_sentinel_drop(tmp_path):
    ir = _track()
    ir[100] = 0.0
    ds = _build(_write_tree(tmp_path, _ir_text(ir)))
    assert '3 window(s) dropped for a (0,0) sentinel line' in ds.describe()
