"""The SQUARE crop rule -- invariants I6/I7 of the ROI-crop contract.

``roi_box_from_landmarks(..., crop_rule='square')`` must return a box whose two
half-open sides are EXACTLY equal, which is what lets the loader apply one
isotropic scale (``sx == sy``) instead of squashing the anatomy:

* the short axis is GROWN about the padded box centre, never stretched, so a
  near-collinear landmark set (``nostrils``) becomes a square rather than an 8x
  vertical stretch;
* the box is TRANSLATED to fit the frame, never shrunk;
* a box that cannot be placed without cutting the landmark envelope -- or that is
  larger than the frame -- raises :class:`RoiBoxOutOfFrame`, so the clip is
  dropped explicitly instead of being silently clipped.

``crop_rule='box'`` (the function's own default) must keep the legacy per-axis
geometry byte for byte: the RGB lineage and the A/B arm still run it.
"""
import math
import os
import sys

import numpy as np
import pytest

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from data import tir_resp_dataset as trd

W, H = 726, 480


def _rect(x0, x1, y0, y1):
    """A 4-point cloud spanning exactly ``[x0, x1] x [y0, y1]``."""
    return np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], dtype=np.float64)


def _square(pts, padding=0.0, w=W, h=H):
    return trd.roi_box_from_landmarks(pts, w, h, padding,
                                      crop_rule=trd.CROP_RULE_SQUARE)


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #
def test_taller_than_wide_grows_the_short_axis_about_the_centre():
    """80 x 120 px -> a 121 px square that grew 20 px on EACH side."""
    box = _square(_rect(100, 180, 100, 220))
    assert box == (80, 201, 100, 221)
    assert box[1] - box[0] == box[3] - box[2] == 121


def test_an_already_square_extent_keeps_its_size():
    box = _square(_rect(100, 200, 100, 200))
    assert box[1] - box[0] == box[3] - box[2] == 101     # inclusive bounds


def test_collinear_set_is_grown_instead_of_stretched():
    """A thin slab (100 x 5) becomes 100 x 100 -- the `min_extent` job, done."""
    box = _square(_rect(100, 200, 100, 105))
    assert box[1] - box[0] == box[3] - box[2] == 101
    # the strip is CENTRED in the new height, not pinned to an edge
    assert abs((box[2] + box[3]) / 2 - 102.5) <= 1.0


def test_frame_edge_clamp_translates_without_dropping():
    """Envelope at +5 px with a margin that overflows: clamp, do NOT drop."""
    box = _square(_rect(5, 105, 100, 200), padding=0.1)
    assert box[0] == 0
    assert box[1] - box[0] == box[3] - box[2]
    assert box[0] <= 5 and box[1] > 105 and box[2] <= 100 and box[3] > 200


def test_envelope_outside_the_frame_is_dropped():
    with pytest.raises(trd.RoiBoxOutOfFrame, match='would cut'):
        _square(_rect(-2, 98, 100, 200), padding=0.1)


def test_a_square_larger_than_the_frame_is_dropped():
    with pytest.raises(trd.RoiBoxOutOfFrame, match='does not fit'):
        _square(_rect(10, 130, 10, 130), w=100, h=480)


def test_min_size_applies_to_a_degenerate_cloud():
    box = _square(np.array([[300.0, 200.0]]))
    assert box[1] - box[0] == box[3] - box[2] >= 2


def test_unknown_crop_rule_is_rejected():
    with pytest.raises(ValueError, match='crop_rule must be one of'):
        trd.roi_box_from_landmarks(_rect(0, 10, 0, 10), W, H,
                                   crop_rule='round')


# --------------------------------------------------------------------------- #
# integer geometry: equal sides must survive fractional centres
# --------------------------------------------------------------------------- #
def test_equal_sides_on_random_fractional_clouds():
    """200 random clouds: the integer sides must be EQUAL every time.

    Rounding each axis separately (``floor``/``ceil + 1``) leaves them 1-2 px
    apart, which is a per-axis scale at the resize -- the exact regression this
    rule exists to prevent.
    """
    rng = np.random.default_rng(7)
    for _ in range(200):
        x0, y0 = rng.uniform(120, 480), rng.uniform(80, 300)
        w, h = rng.uniform(10, 180), rng.uniform(10, 180)
        pts = _rect(x0, x0 + w, y0, y0 + h)
        box = _square(pts, padding=0.2)
        assert box[1] - box[0] == box[3] - box[2], (pts.tolist(), box)
        # and the envelope is still inside it
        assert box[0] <= math.floor(x0) and box[1] >= math.ceil(x0 + w) + 1
        assert box[2] <= math.floor(y0) and box[3] >= math.ceil(y0 + h) + 1


# --------------------------------------------------------------------------- #
# the legacy rule is untouched (RGB lineage + the A/B arm)
# --------------------------------------------------------------------------- #
def _legacy(pts, padding, w, h):
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    p = p[np.isfinite(p).all(axis=1)]
    x0, x1 = p[:, 0].min(), p[:, 0].max()
    y0, y1 = p[:, 1].min(), p[:, 1].max()
    return (math.floor(x0 - padding * (x1 - x0)),
            math.ceil(x1 + padding * (x1 - x0)) + 1,
            math.floor(y0 - padding * (y1 - y0)),
            math.ceil(y1 + padding * (y1 - y0)) + 1)


def test_box_rule_reproduces_the_legacy_geometry():
    rng = np.random.default_rng(11)
    for _ in range(100):
        # kept well inside the frame so the legacy clamp never fires
        pts = rng.uniform(150, 500, size=(12, 2))
        pts[:, 1] = rng.uniform(120, 320, size=12)
        assert (trd.roi_box_from_landmarks(pts, W, H, 0.2)
                == trd.roi_box_from_landmarks(pts, W, H, 0.2, crop_rule='box')
                == _legacy(pts, 0.2, W, H))


def test_the_default_rule_is_the_legacy_one():
    """The FUNCTION default stays 'box': that is what keeps RGB unchanged."""
    pts = _rect(100, 180, 100, 220)
    assert (trd.roi_box_from_landmarks(pts, W, H, 0.0)
            != trd.roi_box_from_landmarks(pts, W, H, 0.0, crop_rule='square'))


def test_the_dataset_default_is_square():
    assert trd.DEFAULT_CROP_RULE == trd.CROP_RULE_SQUARE
