"""``data.roi_cache``: box geometry, cache identity and shard I/O.

The cache is a ~91 GB FROZEN artifact, so two things in it are contracts:

* its **key** -- a wrong key silently reuses another build's pixels, which is the
  one failure mode that invalidates every downstream comparison while looking
  healthy (hence the fail-loudly gates);
* its **box rule** -- the stored crop must CONTAIN every kept clip's box, or the
  loader's sub-crop clips the ROI.

Both are pure functions of the landmark track and the frame size, so every test
here runs with no video decode and no cache on disk. The regression that
motivated the module: ``valid_rows`` used to exclude a frame only when ALL
target landmarks were ``(0,0)``, while ``missing_frame_mask`` (the dataset's
drop rule) fires when ANY is -- one partially-sentinel frame then stretched the
union box to the image corner.
"""
import json

import numpy as np
import pytest

from data import roi_cache as rc
from data import tir_resp_dataset as trd

LABELS = trd.TARGET_LANDMARKS                 # 1-indexed, the 12-pt default
IDX = trd.TARGET_LANDMARK_IDX                 # 0-indexed numpy indices
RAW = '/fake/raw'
SESSIONS = [{'session': 'S001_T1'}, {'session': 'S002_T1'}]
W, H, PAD = 726, 480, 0.2


def _cloud(n=6, seed=0):
    """A plausible nose+mouth cloud: all 28 landmarks in a face-sized box."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(300, 400, size=(n, trd.NUM_LANDMARKS)),
                     rng.uniform(180, 300, size=(n, trd.NUM_LANDMARKS))],
                    axis=-1).astype(np.float32)


# --------------------------------------------------------------------------- #
# label -> index conversion (the 1.8x box bug)
# --------------------------------------------------------------------------- #
def test_landmark_indices_are_one_indexed():
    assert rc.landmark_indices((10, 21)) == (9, 20)
    assert rc.landmark_indices(LABELS) == IDX


def test_landmark_indices_rejects_out_of_range():
    with pytest.raises(ValueError, match='outside 1..28'):
        rc.landmark_indices((29,))


def test_landmark_indices_rejects_an_empty_set():
    with pytest.raises(ValueError, match='empty landmark set'):
        rc.landmark_indices(())


def test_landmark_set_name_recognises_the_presets():
    assert rc.landmark_set_name(LABELS) == 'nose_mouth'
    assert rc.landmark_set_name((10, 21)) == 'nostrils'
    assert rc.landmark_set_name((11, 12, 13)) == 'custom'


# --------------------------------------------------------------------------- #
# corpus identity
# --------------------------------------------------------------------------- #
def test_corpus_digest_is_order_independent_and_counts_sessions():
    a = rc.corpus_digest([{'session': 'B_T1'}, {'session': 'A_T1'}])
    b = rc.corpus_digest([{'session': 'A_T1'}, {'session': 'B_T1'}])
    assert a == b
    assert a.endswith('n2')


def test_corpus_digest_changes_with_the_corpus():
    one = rc.corpus_digest([{'session': 'A_T1'}])
    two = rc.corpus_digest([{'session': 'A_T1'}, {'session': 'A_T2'}])
    assert one != two


# --------------------------------------------------------------------------- #
# valid_rows -- MUST mirror missing_frame_mask
# --------------------------------------------------------------------------- #
def test_valid_rows_drops_a_whole_row_sentinel():
    pts = _cloud()
    pts[3] = 0.0
    keep = rc.valid_rows(pts, LABELS)
    assert keep.shape == (5, len(LABELS), 2)
    assert not np.isclose(keep, 0.0).all(axis=(1, 2)).any()


def test_valid_rows_drops_a_row_with_any_non_finite_landmark():
    pts = _cloud()
    pts[2, IDX[3], :] = np.nan
    assert rc.valid_rows(pts, LABELS).shape[0] == 5


def test_valid_rows_drops_a_row_with_ANY_target_landmark_at_the_origin():
    """THE regression: one partially-sentinel frame must drop the whole row.

    ``missing_frame_mask`` fires on ``.any()``; the old copy here used
    ``.all()``, which kept the row and stretched the union box to the corner.
    """
    pts = _cloud()
    pts[4, IDX[5], :] = 0.0                   # ONE of twelve target points
    keep = rc.valid_rows(pts, LABELS)
    assert keep.shape[0] == 5, 'a single (0,0) target landmark must drop the row'


def test_valid_rows_keeps_a_partially_degenerate_frame_out_of_the_box():
    """One partially-sentinel frame must not drag the box to the corner.

    The PRE-FIX rule kept the frame, so its ``(0,0)`` point entered the min/max
    and the box reached the origin. Here the box must equal the one built from
    the frames the dataset actually keeps (the dirty frame excluded).
    """
    clean = _cloud()
    dirty = clean.copy()
    dirty[2, IDX[1], :] = 0.0
    x0, x1, y0, y1 = rc.union_box(dirty, LABELS, PAD, W, H)
    assert x0 > 0 and y0 > 0, 'a (0,0) target must not reach the image corner'
    keep = [i for i in range(len(clean)) if i != 2]
    expect = trd.roi_box_from_landmarks(
        clean[keep][:, IDX, :].reshape(-1, 2), W, H, PAD)
    assert (x0, x1, y0, y1) == expect


def test_valid_rows_matches_missing_frame_mask_row_for_row():
    """Parity BY CONSTRUCTION -- guard against a future re-implementation.

    ``valid_rows`` delegates to ``missing_frame_mask``; the explicit count is
    what keeps this test from being purely tautological.
    """
    pts = _cloud(n=12)
    pts[0] = 0.0                              # whole-row sentinel
    pts[3, IDX[7], :] = 0.0                   # one target at the origin
    pts[5, 0, :] = np.nan                     # a NON-target landmark unusable
    pts[7, IDX[2], :] = np.inf
    keep = rc.valid_rows(pts, LABELS)
    assert keep.shape[0] == 8                 # 4 of 12 frames unusable
    assert keep.shape[0] == int((~trd.missing_frame_mask(pts, IDX)).sum())


def test_valid_rows_only_guards_the_requested_landmark_set():
    """A (0,0) OUTSIDE the run's landmark set is not a reason to drop a row."""
    pts = _cloud()
    pts[1, 0, :] = 0.0                        # label 1: not in ANY preset
    assert rc.valid_rows(pts, LABELS).shape[0] == 6
    # but the same frame IS fatal once its label enters the set
    pts[1, 10, :] = 0.0                       # label 11 (mouth corner R)
    assert rc.valid_rows(pts, LABELS).shape[0] == 5
    assert rc.valid_rows(pts, (10, 21)).shape[0] == 6   # nostrils unaffected


def test_missing_frame_mask_raises_on_a_wrong_shape():
    with pytest.raises(ValueError, match='expected'):
        trd.missing_frame_mask(np.zeros((3, 10, 2), np.float32), IDX)


# --------------------------------------------------------------------------- #
# union_box
# --------------------------------------------------------------------------- #
def test_union_box_is_the_padded_minmax_of_the_valid_rows():
    pts = _cloud(n=4)
    pts[0] = 0.0                              # excluded from the box
    box = rc.union_box(pts, LABELS, PAD, W, H)
    expect = trd.roi_box_from_landmarks(
        rc.valid_rows(pts, LABELS).reshape(-1, 2), W, H, PAD)
    assert box == expect


def test_union_box_hardcoded_small_case():
    """Two distinct target points -> a box of exactly 100x100 px, padded 0.2."""
    pts = np.zeros((1, trd.NUM_LANDMARKS, 2), np.float32)
    pts[0, IDX[0], :] = (100.0, 200.0)
    pts[0, IDX[1], :] = (200.0, 300.0)
    pts[0, IDX[2], :] = (200.0, 300.0)
    # every other target sits on the same corner so the cloud is those two
    for i in IDX[3:]:
        pts[0, i, :] = (200.0, 300.0)
    assert rc.union_box(pts, LABELS, 0.2, W, H) == (80, 221, 180, 321)


def test_union_box_is_none_when_every_row_is_unusable():
    pts = _cloud(n=3)
    pts[:] = 0.0
    assert rc.union_box(pts, LABELS, PAD, W, H) is None


def test_union_box_contains_every_kept_clip_box():
    """The invariant the whole cache rests on: a sub-crop cannot clip the ROI."""
    pts = _cloud(n=20, seed=3)
    pts[6] = 0.0
    pts[11, IDX[2], :] = 0.0
    ub = rc.union_box(pts, LABELS, PAD, W, H)
    ds_keep = ~trd.missing_frame_mask(pts, IDX)
    for start in range(0, 20, 4):             # every 4-frame clip window
        win = pts[start:start + 4][ds_keep[start:start + 4]]
        if win.shape[0] == 0:
            continue
        cb = trd.roi_box_from_landmarks(win[:, IDX, :].reshape(-1, 2), W, H, PAD)
        assert rc.box_contains(ub, cb)


def test_box_contains_and_shift_box():
    assert rc.box_contains((0, 10, 0, 10), (2, 8, 2, 8))
    assert not rc.box_contains((0, 10, 0, 10), (-1, 8, 2, 8))
    assert rc.shift_box((10, 20, 30, 40), (5, 7)) == (5, 15, 23, 33)


# --------------------------------------------------------------------------- #
# cache identity
# --------------------------------------------------------------------------- #
def test_build_params_is_stable_and_names_the_landmark_set():
    p1 = rc.build_params(RAW, SESSIONS, LABELS, PAD, W, H)
    p2 = rc.build_params(RAW, SESSIONS, LABELS, PAD, W, H)
    assert p1 == p2
    assert p1['roi_landmarks_name'] == 'nose_mouth'
    assert p1['format_version'] == rc.CACHE_FORMAT_VERSION
    assert p1['valid_frame_rule'] == rc.VALID_FRAME_RULE


def test_cache_key_changes_when_a_frozen_field_changes():
    base = rc.cache_key(rc.build_params(RAW, SESSIONS, LABELS, PAD, W, H))
    assert base.startswith('tirroi_nose_mouth_pad0.20_minmax_src726x480_native_v2_')
    variants = [
        rc.build_params(RAW, SESSIONS, LABELS, 0.1, W, H),         # padding
        rc.build_params(RAW, SESSIONS, (10, 21), PAD, W, H),       # landmarks
        rc.build_params(RAW, SESSIONS, LABELS, PAD, 640, H),       # src width
        rc.build_params('/other/raw', SESSIONS, LABELS, PAD, W, H),  # corpus
    ]
    assert all(rc.cache_key(v) != base for v in variants)


def test_loader_params_covers_exactly_the_comparable_keys():
    want = rc.loader_params(RAW, SESSIONS, LABELS, PAD)
    assert set(want) == set(rc.COMPARE_KEYS)
    full = rc.build_params(RAW, SESSIONS, LABELS, PAD, 0, 0)
    assert want == {k: full[k] for k in rc.COMPARE_KEYS}


def test_loader_params_ignores_the_build_subset_but_not_the_corpus():
    """The key is a property of the CORPUS, so --subjects must not change it."""
    assert (rc.loader_params(RAW, SESSIONS, LABELS, PAD)
            == rc.loader_params(RAW, SESSIONS, LABELS, PAD))
    subset = rc.loader_params(RAW, SESSIONS[:1], LABELS, PAD)
    assert subset['corpus'] != rc.loader_params(RAW, SESSIONS, LABELS, PAD)['corpus']


def test_write_and_read_params_round_trip(tmp_path):
    p = rc.build_params(RAW, SESSIONS, LABELS, PAD, W, H)
    rc.write_params(tmp_path, p)
    assert rc.read_params(tmp_path) == p


def test_find_cache_returns_the_single_match(tmp_path):
    p = rc.build_params(RAW, SESSIONS, LABELS, PAD, W, H)
    d = tmp_path / 'c1'
    rc.write_params(d, p)
    assert rc.find_cache(tmp_path, rc.loader_params(RAW, SESSIONS, LABELS, PAD)) == d
    rc.check_params_compatible(d, p)                      # must not raise


def test_find_cache_raises_a_field_diff_when_nothing_matches(tmp_path):
    rc.write_params(tmp_path / 'c1',
                    rc.build_params(RAW, SESSIONS, LABELS, 0.2, W, H))
    with pytest.raises(ValueError) as exc:
        rc.find_cache(tmp_path, rc.loader_params(RAW, SESSIONS, LABELS, 0.1))
    msg = str(exc.value)
    assert 'roi_padding' in msg and 'Closest existing cache' in msg


def test_find_cache_refuses_an_ambiguous_match(tmp_path):
    want = rc.loader_params(RAW, SESSIONS, LABELS, PAD)
    p = rc.build_params(RAW, SESSIONS, LABELS, PAD, W, H)
    rc.write_params(tmp_path / 'c1', p)
    rc.write_params(tmp_path / 'c2', p)
    with pytest.raises(ValueError, match='refusing to guess'):
        rc.find_cache(tmp_path, want)


def test_find_cache_rejects_a_missing_out_root(tmp_path):
    with pytest.raises(FileNotFoundError, match='out_root does not exist'):
        rc.find_cache(tmp_path / 'nope', rc.loader_params(RAW, SESSIONS, LABELS, PAD))


def test_check_params_compatible_raises_on_a_diff(tmp_path):
    rc.write_params(tmp_path, rc.build_params(RAW, SESSIONS, LABELS, PAD, W, H))
    with pytest.raises(ValueError, match='built with DIFFERENT parameters'):
        rc.check_params_compatible(
            tmp_path, rc.build_params(RAW, SESSIONS, LABELS, 0.1, W, H))


# --------------------------------------------------------------------------- #
# shard I/O
# --------------------------------------------------------------------------- #
def _payload(n_frames=4, side=6, with_landmarks=True):
    p = {'frames': np.zeros((n_frames, side, side, 3), np.uint8),
         'box': np.asarray([10, 10 + side, 20, 20 + side], np.int32),
         'session': 'S001_T1',
         'video': '/fake/T1.wmv',
         'n_frames': n_frames}
    if with_landmarks:
        p['landmarks'] = np.zeros((n_frames, trd.NUM_LANDMARKS, 2), np.float32)
    return p


def test_shard_write_then_read_round_trip(tmp_path):
    path = rc.write_shard(tmp_path, 'S001_T1', _payload())
    assert path == rc.shard_path(tmp_path, 'S001_T1')
    assert path.is_file()
    sh = rc.read_shard(path)
    assert sh['session'] == 'S001_T1'
    assert sh['n_frames'] == 4
    assert sh['box'] == (10, 16, 20, 26)
    assert sh['frames'].shape == (4, 6, 6, 3)
    assert sh['landmarks'].shape == (4, trd.NUM_LANDMARKS, 2)
    # the PID-staged temp file must not survive the os.replace
    assert list((tmp_path / rc.SUBDIR_TMP).glob('*.npz')) == []


def test_read_shard_reports_missing_landmarks_as_none(tmp_path):
    path = rc.write_shard(tmp_path, 'S002_T1', _payload(with_landmarks=False))
    assert rc.read_shard(path)['landmarks'] is None


def test_scan_counts_cached_skipped_and_missing(tmp_path):
    rc.write_shard(tmp_path, 'S001_T1', _payload())
    rc.write_skip(tmp_path, 'S002_T1', 'missing_video')
    out = rc.scan(tmp_path, [{'session': 'S001_T1'}, {'session': 'S002_T1'},
                             {'session': 'S003_T1'}])
    assert (out['cached'], out['skipped'], out['missing']) == (1, 1, 1)
    assert out['missing_names'] == ['S003_T1']
    assert out['bytes'] > 0


def test_write_skip_records_the_reason(tmp_path):
    path = rc.write_skip(tmp_path, 'S009_T1', 'undecodable_video')
    assert json.loads(path.read_text()) == {'session': 'S009_T1',
                                            'reason': 'undecodable_video'}


def test_build_manifest_summarises_the_directory(tmp_path):
    p = rc.build_params(RAW, SESSIONS, LABELS, PAD, W, H)
    rc.write_shard(tmp_path, 'S001_T1', _payload(with_landmarks=False))
    rc.write_skip(tmp_path, 'S002_T1', 'missing_ir_features')
    m = rc.build_manifest(tmp_path, SESSIONS, p)
    assert (m['n_ok'], m['n_skipped'], m['n_missing']) == (1, 1, 0)
    assert m['key'] == rc.cache_key(p)
    assert (tmp_path / rc.MANIFEST_NAME).is_file()
    by_name = {e['session']: e for e in m['entries']}
    assert by_name['S001_T1']['has_landmarks'] is False
    assert by_name['S002_T1']['reason'] == 'missing_ir_features'
