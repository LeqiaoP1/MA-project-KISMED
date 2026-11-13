"""``data.rgb2d_cache`` -- key contract, geometry, shard IO, coordinate frame.

Every case is synthetic (tiny JPEGs + a real ``.mat`` via ``scipy.io.savemat``,
the ``tests/test_rgb_zip_source.py`` pattern): no raw corpus, no GPU.

The load-bearing case is :func:`test_coordinate_frame_has_no_pad_offset`. The
stored array is sized to the UNCLAMPED hull, so its origin IS the hull origin and
``pad`` is descriptive only. An earlier review proposed adding ``pad_top`` /
``pad_left`` when translating a clip box; that double-shifts, and this test fails
loudly if anyone implements it.
"""
import json
import os
import sys
import zipfile

import numpy as np
import pytest

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from data import rgb2d_cache as c2d

FPS = 25.0
CLIP_S = 0.16                       # 4 frames per window
N_FRAMES = 12
SUBJECT = 'F900'
SRC_HW = (16, 20)                   # (h, w): tiny frames keep the suite fast
LABELS = tuple(range(1, 50))        # 'face'


# --------------------------------------------------------------------------- #
# synthetic tree
# --------------------------------------------------------------------------- #
def _frame(i: int, hw=SRC_HW) -> np.ndarray:
    """Deterministic BGR frame: a gradient makes a mis-decode obvious."""
    h, w = hw
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = np.linspace(0, 200, w, dtype=np.uint8)[None, :]
    img[..., 1] = np.linspace(0, 200, h, dtype=np.uint8)[:, None]
    img[..., 2] = (i * 10) % 256
    return img


def _jpeg_bytes(i: int) -> bytes:
    import cv2
    ok, buf = cv2.imencode('.jpg', _frame(i),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    return buf.tobytes()


def _landmarks(n: int = N_FRAMES, inside: bool = True) -> np.ndarray:
    """``[n, 49, 2]`` (x, y) source pixels, a 7x7 face-ish patch.

    ``inside=True`` spans x 4..16 / y 4..12 (hull fully in frame);
    ``inside=False`` hugs the top-left corner so the padded hull leaves the
    frame by a pixel or two -- the case the black-padding rule exists for.
    """
    h, w = SRC_HW
    base = np.zeros((49, 2), np.float64)
    for k in range(49):
        if inside:
            base[k, 0] = 4.0 + 2.0 * (k % 7)
            base[k, 1] = 4.0 + 1.33 * (k // 7)
        else:
            base[k, 0] = 0.0 + 0.4 * (k % 7)
            base[k, 1] = 0.0 + 0.4 * (k // 7)
    return np.repeat(base[None], n, axis=0)


def _write_mat(path: str, pts: np.ndarray,
               frame_offset: int = 0, drop_last: bool = False) -> None:
    from scipy.io import savemat
    n = pts.shape[0] - (1 if drop_last else 0)
    fit = np.zeros((1, n), dtype=[('frame', 'O'), ('pts_2d', 'O'),
                                  ('headPose', 'O')])
    for i in range(n):
        fit[0, i]['frame'] = np.array([[i + 1 + frame_offset]], np.uint16)
        fit[0, i]['pts_2d'] = pts[i]
        fit[0, i]['headPose'] = np.zeros((3, 1), np.float32)
    savemat(path, {'fit': fit})


def make_tree(root: str, task: str = 'T1', inside: bool = True,
              archived: bool = False, extracted: bool = True,
              drop_last: bool = False) -> str:
    """Write a minimal raw tree; returns ``raw_root``."""
    raw = str(root)
    os.makedirs(os.path.join(raw, '2D+3D'), exist_ok=True)
    os.makedirs(os.path.join(raw, '2DFeatures'), exist_ok=True)
    _write_mat(os.path.join(raw, '2DFeatures', f'{SUBJECT}_{task}.mat'),
               _landmarks(inside=inside), drop_last=drop_last)
    if extracted:
        d = os.path.join(raw, '2D+3D', SUBJECT, task)
        os.makedirs(d, exist_ok=True)
        for i in range(N_FRAMES):
            with open(os.path.join(d, f'{i:04d}.jpg'), 'wb') as fh:
                fh.write(_jpeg_bytes(i))
    if archived:
        zip_path = os.path.join(raw, '2D+3D', f'{SUBJECT}.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
            for i in range(N_FRAMES):
                zf.writestr(f'{SUBJECT}/{task}/{i:04d}.jpg', _jpeg_bytes(i))
        if not extracted:
            # A zip-only tree must not need an extracted directory.
            pass
    return raw


def sessions(raw: str):
    return c2d.discover_sessions(raw)


def params_for(raw: str, **kw):
    return c2d.build_params(raw, sessions(raw), kw.pop('labels', LABELS),
                            kw.pop('padding', 0.10), kw.pop('src_w', SRC_HW[1]),
                            kw.pop('src_h', SRC_HW[0]), **kw)


# --------------------------------------------------------------------------- #
# identity / key
# --------------------------------------------------------------------------- #
def test_slug_prefix_is_readable_and_size_rule_only(tmp_path):
    key = c2d.cache_key(params_for(make_tree(tmp_path)))
    assert key.startswith(
        f'rgb2droi_face_pad0.10_square_window-hull_clip8s_'
        f'src{SRC_HW[1]}x{SRC_HW[0]}_native_v1_')
    assert key.count('_') >= 8 and not key.endswith('_')


def test_key_changes_on_pixel_fields_and_ignores_load_time_knobs(tmp_path):
    raw = make_tree(tmp_path)
    base = c2d.cache_key(params_for(raw))
    # IN the key
    assert c2d.cache_key(params_for(raw, padding=0.20)) != base
    assert c2d.cache_key(params_for(raw, crop_rule='box')) != base
    assert c2d.cache_key(params_for(raw, clip_seconds=4.0)) != base
    assert c2d.cache_key(params_for(raw, fps=30.0)) != base
    assert c2d.cache_key(params_for(raw, src_w=100)) != base
    assert c2d.cache_key(params_for(raw, labels=(1, 2, 3))) != base
    assert c2d.cache_key(params_for(raw, source_policy='dir-then-zip')) == base
    # OUT of the key -- they cannot change a stored byte
    for k in c2d.COMPARE_KEYS:
        assert k in params_for(raw)
    for k in ('input_size', 'clip_stride', 'max_sway_px', 'min_valid_ratio',
              'temporal_stride', 'task_set'):
        assert k not in params_for(raw), k


def test_store_resolution_grammar_and_key(tmp_path):
    """Native is the default; a pixel store is a DIFFERENT key."""
    raw = make_tree(tmp_path)
    assert c2d.parse_store_resolution(None) == 'native'
    assert c2d.parse_store_resolution('') == 'native'
    assert c2d.parse_store_resolution('native') == 'native'
    assert c2d.parse_store_resolution(112) == 'px112'
    assert c2d.parse_store_resolution('112') == 'px112'
    assert c2d.parse_store_resolution('px112') == 'px112'
    assert c2d.store_size_of('native') is None
    assert c2d.store_size_of('px112') == 112
    for bad in ('huge', 0, 8, -1):
        with pytest.raises(ValueError):
            c2d.parse_store_resolution(bad)
    # the SAME pixels rule, but a different stored artefact -> different key
    native = c2d.cache_key(params_for(raw))
    px = c2d.cache_key(params_for(raw, store_resolution=112))
    assert '_native_' in native and '_px112_' in px
    assert native != px
    # ... and a loader asking for one never matches the other
    assert c2d.loader_params(
        raw, sessions(raw), LABELS, 0.10, store_resolution=112
    )['store_resolution'] == 'px112'


def test_pixel_store_needs_a_square_crop_rule(tmp_path):
    raw = make_tree(tmp_path)
    with pytest.raises(ValueError, match='square'):
        params_for(raw, crop_rule='box', store_resolution=112)


def test_find_cache_matches_and_explains_a_miss(tmp_path):
    raw = make_tree(tmp_path)
    out = tmp_path / 'cache'
    p = params_for(raw)
    root = c2d.cache_root(out, c2d.cache_key(p))
    c2d.write_params(root, p)
    assert c2d.find_cache(out, c2d.loader_params(
        raw, sessions(raw), LABELS, 0.10)) == root
    with pytest.raises(ValueError, match='Closest existing cache'):
        c2d.find_cache(out, c2d.loader_params(
            raw, sessions(raw), LABELS, 0.20))


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def test_valid_rows_2d_drops_a_frame_with_one_bad_target():
    pts = _landmarks(n=4)
    assert c2d.valid_rows_2d(pts, LABELS).shape == (4, 49, 2)
    pts[2, 17] = np.nan                      # ONE target point of ONE frame
    rows = c2d.valid_rows_2d(pts, LABELS)
    assert rows.shape == (3, 49, 2)
    assert np.isfinite(rows).all()


def test_square_span_is_square_and_unclamped():
    pts = _landmarks(n=4, inside=False)      # hugs the top-left corner
    box = c2d.square_span_2d(pts, 0.10, 'square')
    x0, x1, y0, y1 = box
    assert (x1 - x0) == (y1 - y0), 'one integer side for both axes (I6)'
    assert x0 < 0 or y0 < 0, 'the span must NOT be clamped to the frame'


def test_in_frame_share_bounds():
    assert c2d.in_frame_share((0, 10, 0, 10), 20, 16) == 1.0
    assert c2d.in_frame_share((-50, -10, 0, 10), 20, 16) == 0.0
    assert c2d.in_frame_share((-5, 5, 0, 10), 20, 16) == pytest.approx(0.5)
    assert c2d.in_frame_share((15, 25, 0, 10), 20, 16) == pytest.approx(0.5)


def test_hull_contains_every_stride1_window_box():
    pts = _landmarks(n=N_FRAMES)
    hull = c2d.hull_box_2d(pts, LABELS, 0.10, 'square', CLIP_S, FPS)
    assert hull is not None
    cf = int(round(CLIP_S * FPS))
    for start in range(0, N_FRAMES - cf + 1):
        box = c2d.square_span_2d(pts[start:start + cf], 0.10, 'square')
        assert hull[0] <= box[0] and box[1] <= hull[1]
        assert hull[2] <= box[2] and box[3] <= hull[3]


# --------------------------------------------------------------------------- #
# build / commit / open
# --------------------------------------------------------------------------- #
@pytest.fixture()
def built(tmp_path):
    raw = make_tree(tmp_path)
    p = c2d.build_params(
        raw, sessions(raw), LABELS, 0.10, SRC_HW[1], SRC_HW[0],
        clip_seconds=CLIP_S, fps=FPS)
    root = c2d.cache_root(tmp_path / 'cache', c2d.cache_key(p))
    c2d.write_params(root, p)
    spec = sessions(raw)[0]
    stage = root / c2d.SUBDIR_TMP / 'stage'
    meta, reason = c2d.build_task(spec, LABELS, 0.10, stage,
                                  clip_seconds=CLIP_S, fps=FPS)
    assert reason is None, reason
    c2d.commit_shard(root, str(spec['session']), meta, stage)
    return raw, root, meta, spec


def test_shard_round_trip_keeps_native_resolution(built):
    raw, root, meta, spec = built
    with c2d.open_shard(root, str(spec['session'])) as h:
        assert h.frames.shape[0] == N_FRAMES
        assert h.frames.dtype == np.uint8
        assert h.frames.shape[1] == h.frames.shape[2] == meta['hull_hw'][0]
        assert h.landmarks.shape == (N_FRAMES, 49, 2)
        assert h.box == tuple(meta['box'])
        assert h.src_hw == SRC_HW
        # native: NOT resampled to a model size
        assert h.frames.shape[1] != 96
        # reads a VIEW, not a copy (decision 9's whole claim)
        s = h.frames[1:3]
        assert s.base is not None and s.flags.owndata is False
        assert s.flags.writeable is False
        assert np.shares_memory(s, h.frames)


def test_pixel_store_resizes_at_build_and_reports_the_scale(built, tmp_path):
    """A ``px<N>`` store: the HULL stays in source px, the ARRAY is resized."""
    raw, root, native, spec = built
    short = min(native['hull_hw'])
    store = max(16, short // 2)          # a real downscale of THIS synthetic hull
    assert store < short, 'the synthetic hull must be larger than the store'
    stage = root / c2d.SUBDIR_TMP / 'stage_px'
    meta, reason = c2d.build_task(spec, LABELS, 0.10, stage,
                                  clip_seconds=CLIP_S, fps=FPS,
                                  store_resolution=store)
    assert reason is None, reason
    assert meta['store_resolution'] == f'px{store}'
    assert meta['store_size'] == store
    assert meta['hull_hw'] == native['hull_hw'], 'SOURCE hull is unchanged'
    sh, sw = meta['frames']['shape'][1:3]
    assert min(sh, sw) == store, 'the SHORT side is the store size'
    assert max(sh, sw) >= store, 'isotropic: the long side keeps the aspect'
    assert (sh, sw) != tuple(native['hull_hw'][:2])
    assert meta['frames']['nbytes'] == N_FRAMES * sh * sw * 3
    c2d.commit_shard(root, str(spec['session']), meta, stage)
    with c2d.open_shard(root, str(spec['session'])) as h:
        assert h.store_size == store
        # source -> stored mapping the dataset applies (one isotropic scale)
        assert h.store_scale == pytest.approx(store / short, rel=1e-3)
        assert h.box == tuple(meta['box']), 'hull box stays in SOURCE px'


def test_pixel_store_never_upsamples(built):
    """A hull already smaller than the store size keeps source pixels."""
    raw, root, native, spec = built
    huge = 512
    stage = root / c2d.SUBDIR_TMP / 'stage_up'
    meta, reason = c2d.build_task(spec, LABELS, 0.10, stage,
                                  clip_seconds=CLIP_S, fps=FPS,
                                  store_resolution=huge)
    assert reason is None, reason
    assert meta['store_size'] is None, 'upsampling would invent detail'
    assert meta['frames']['shape'][1:3] == native['hull_hw'][:2]


def test_commit_marker_and_truncation_are_detected(built):
    raw, root, meta, spec = built
    session = str(spec['session'])
    rep = c2d.scan(root, sessions(raw))
    assert rep['cached'] == 1 and rep['bytes'] > 0
    # remove the commit marker -> incomplete, and open_shard refuses
    mp = c2d.meta_path(root, session)
    saved = mp.read_text()
    mp.unlink()
    assert c2d.scan(root, sessions(raw))['incomplete'] == 1
    with pytest.raises(FileNotFoundError, match='INCOMPLETE'):
        c2d.open_shard(root, session)
    mp.write_text(saved)
    # truncate an array -> detected by scan and by open_shard
    fp = c2d.frames_path(root, session)
    with open(fp, 'r+b') as fh:
        fh.truncate(fp.stat().st_size - 10)
    assert c2d.scan(root, sessions(raw))['truncated'] == 1
    with pytest.raises(ValueError, match='TRUNCATED'):
        c2d.open_shard(root, session)


def test_pad_is_recorded_and_matches_the_hull_identity(tmp_path):
    """``(x1-x0) == (cx1-cx0) + pad_left + pad_right`` (same in y)."""
    raw = make_tree(tmp_path)
    p = c2d.build_params(raw, sessions(raw), LABELS, 0.10, SRC_HW[1], SRC_HW[0],
                         clip_seconds=CLIP_S, fps=FPS)
    root = c2d.cache_root(tmp_path / 'cache', c2d.cache_key(p))
    c2d.write_params(root, p)
    spec = sessions(raw)[0]
    stage = root / c2d.SUBDIR_TMP / 'stage'
    meta, reason = c2d.build_task(spec, LABELS, 0.10, stage,
                                  clip_seconds=CLIP_S, fps=FPS)
    assert reason is None, reason
    x0, x1, y0, y1 = meta['box']
    pad_t, pad_b, pad_l, pad_r = meta['pad']
    fh, fw = meta['src_hw']
    cx0, cx1 = max(0, x0), min(fw, x1)
    cy0, cy1 = max(0, y0), min(fh, y1)
    assert (x1 - x0) == (cx1 - cx0) + pad_l + pad_r
    assert (y1 - y0) == (cy1 - cy0) + pad_t + pad_b


def test_coordinate_frame_has_no_pad_offset(tmp_path):
    """stored_index = source - hull_origin; ``pad`` is NEVER an offset.

    The hull leaves the frame, so ``pad_left``/``pad_top`` are non-zero. A clip
    box inside the frame must map to ``source - hull_origin`` -- adding the pad
    reads the wrong pixel (and eventually runs off the array).
    """
    raw = make_tree(tmp_path, inside=False)      # corner-hugging landmarks
    p = c2d.build_params(raw, sessions(raw), LABELS, 0.10, SRC_HW[1], SRC_HW[0],
                         clip_seconds=CLIP_S, fps=FPS)
    root = c2d.cache_root(tmp_path / 'cache', c2d.cache_key(p))
    c2d.write_params(root, p)
    spec = sessions(raw)[0]
    stage = root / c2d.SUBDIR_TMP / 'stage'
    meta, reason = c2d.build_task(spec, LABELS, 0.10, stage,
                                  clip_seconds=CLIP_S, fps=FPS)
    assert reason is None, reason
    c2d.commit_shard(root, str(spec['session']), meta, stage)
    pad_t, _pad_b, pad_l, _pad_r = meta['pad']
    assert pad_t > 0 and pad_l > 0, 'fixture must exercise the padded margin'

    import cv2

    from data import rgb_roi_dataset as rrd
    hx0, hx1, hy0, hy1 = meta['box']
    with c2d.open_shard(root, str(spec['session'])) as h:
        cx0, cy0 = max(0, hx0), max(0, hy0)
        # The DECODED source frame (JPEG is lossy: never compare to the pre-
        # encode array), as RGB, which is what the builder stores.
        src = cv2.imread(os.path.join(str(spec['rgb_dir']), '0000.jpg'),
                         cv2.IMREAD_COLOR)
        src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        row, col = (cy0 + 1) - hy0, (cx0 + 1) - hx0
        assert tuple(int(v) for v in h.frames[0, row, col]) == \
            tuple(int(v) for v in src_rgb[cy0 + 1, cx0 + 1])
        # The pad margin is BLACK -- not a shifted copy of the frame.
        assert np.all(np.asarray(h.frames[0, 0, 0]) == 0)
        assert np.all(np.asarray(h.frames[0, :pad_t, :]) == 0)
        assert np.all(np.asarray(h.frames[0, :, :pad_l]) == 0)
        # ... and the "+pad" variant the review proposed reads a DIFFERENT pixel
        # (or runs off the array), which is exactly why it is not applied.
        r2, c2 = row + pad_t, col + pad_l
        shifted_is_wrong = (
            r2 >= h.frames.shape[1] or c2 >= h.frames.shape[2]
            or tuple(int(v) for v in h.frames[0, r2, c2])
            != tuple(int(v) for v in src_rgb[cy0 + 1, cx0 + 1]))
        assert shifted_is_wrong


def test_image_count_mismatch_is_fatal(tmp_path):
    raw = make_tree(tmp_path, drop_last=True)      # 11 mat rows vs 12 jpgs
    spec = sessions(raw)[0]
    stage = tmp_path / 'stage'
    meta, reason = c2d.build_task(spec, LABELS, 0.10, stage,
                                  clip_seconds=CLIP_S, fps=FPS)
    assert meta is None
    assert reason.startswith('image_count_mismatch')


def test_hull_mostly_out_of_frame_is_skipped(tmp_path):
    """Landmarks translated far outside the frame -> ``box_out_of_frame``."""
    raw = make_tree(tmp_path)
    pts = _landmarks(inside=True)
    pts[..., 0] += 10_000.0                        # way off the right edge
    _write_mat(os.path.join(raw, '2DFeatures', f'{SUBJECT}_T1.mat'), pts)
    spec = sessions(raw)[0]
    meta, reason = c2d.build_task(spec, LABELS, 0.10, tmp_path / 'stage',
                                  clip_seconds=CLIP_S, fps=FPS)
    assert meta is None and reason.startswith('box_out_of_frame')


def test_zip_only_source_builds_without_creating_a_directory(tmp_path):
    raw = make_tree(tmp_path, extracted=False, archived=True)
    spec = sessions(raw)[0]
    assert spec['source_kind'] == 'zip'
    meta, reason = c2d.build_task(spec, LABELS, 0.10, tmp_path / 'stage',
                                  clip_seconds=CLIP_S, fps=FPS)
    assert reason is None, reason
    assert meta['frames']['shape'][0] == N_FRAMES
    # the archive stays an archive: nothing was unpacked next to it
    assert not os.path.isdir(os.path.join(raw, '2D+3D', SUBJECT))


def test_skip_file_creates_its_own_staging_dir(tmp_path):
    root = tmp_path / 'cache'
    p = c2d.write_skip(root, 'F900_T9', 'missing_2d_features')
    assert json.loads(p.read_text())['reason'] == 'missing_2d_features'
    rep = c2d.scan(root, [{'session': 'F900_T9'}])
    assert rep['skipped'] == 1 and rep['missing'] == 0


def test_commit_clears_a_stale_skip_marker(built):
    """A rebuilt session must not stay excluded by an old refusal.

    ``too_large_for_memory`` and ``build_error`` are re-attemptable; the loader
    checks ``skipped/`` BEFORE the shard, so a leftover marker would exclude a
    session whose pixels are on disk.
    """
    raw, root, meta, spec = built
    session = str(spec['session'])
    marker = c2d.write_skip(root, session, 'too_large_for_memory: 13.5 GB')
    assert marker.is_file()
    stage = root / c2d.SUBDIR_TMP / 'stage2'
    meta2, reason = c2d.build_task(spec, LABELS, 0.10, stage,
                                   clip_seconds=CLIP_S, fps=FPS)
    assert reason is None, reason
    c2d.commit_shard(root, session, meta2, stage)
    assert not marker.is_file(), 'stale skip marker survived a successful commit'
    assert c2d.shard_path(root, session).is_file()
