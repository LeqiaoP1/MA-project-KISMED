"""Offline 2D-texture face-ROI pixel cache (`rgb2d`): key, hull, shard IO.

Contract: ``notes/design/ADR-012-rgb2d-native-memmap-roi-cache.md`` (v2).
Build order: ``notes/plan/features/ADR-012-rgb2d-native-memmap-roi-cache-plan.md`` (Phase 1).

**The cache is ONE STEP of the pipeline and encodes no downstream decision** --
the same doctrine as :mod:`data.roi_cache`, which this module deliberately
mirrors. It answers "which pixels" faster than decoding 1392x1040 JPEGs, and
nothing else:

* :func:`cache_key` covers ONLY what changes the STORED PIXELS -- the ROI
  contract (landmark set, padding, crop rule), the box rule, the valid-frame
  rule, the source frame size, the pixel format, the store-resolution RULE
  (``'native'`` or ``'px<N>'``), the source policy and the corpus identity.
* Everything that SELECTS WINDOWS (``clip_stride``, ``max_sway_px``,
  ``min_valid_ratio``, ``task_set``) or RESIZES to the MODEL input
  (``input_size``) is absent from the key AND from the writer, because both
  happen at load. The native store therefore serves any hop, any fold and any
  ``input_size`` (64/96/112); a ``px<N>`` store serves any hop/fold and any
  ``input_size <= N`` (see ``STORE_RESOLUTION`` for why it exists).
* The gate is GEOMETRY-ONLY: a session is skipped only when there is no pixel to
  crop, never because a clip is slow-moving or mostly out of frame -- those are
  dataset-level rules with their own counters.

Differences from the TIR cache (and why):

* **Raw ``npy``-style ``.dat`` + JSON instead of ``.npz``.** A task-level native
  hull is ~0.19 MB/frame (~300 MB/session), so a materialising reader would
  re-read and re-allocate hundreds of MB for every clip under DataLoader
  shuffling. ``np.memmap(mode='r')`` gives zero-copy slicing; ``npz`` members
  cannot be mapped. Uncompressed, so no storage regression versus ``np.savez``.
* **No ``read_shard``.** Only :func:`open_shard` (a :class:`ShardHandle` over two
  mappings) exists, so no later caller can reintroduce a materialising read.
* **``valid_frame_rule='finite-excluded'``.** A lost 2D frame is an empty ``[]``
  / NaN field, not the thermal ``(0,0)`` sentinel (``rgb_features`` docstring:
  *"Do not reuse missing_frame_mask here"*). The difference is in the key, so it
  can never be mixed up with a TIR shard.

Layout::

    <out_root>/<CACHE_KEY>/
        params.json              # the frozen parameters (--init)
        tasks/<S>_<T>.dat        # hull pixels, uint8 [T, H, W, 3] RGB, native
        tasks/<S>_<T>.lm.dat     # the 49-point track, float32 [T, 49, 2]
        tasks/<S>_<T>.json       # scalars + shapes; THE COMMIT MARKER
        skipped/<S>_<T>.json     # {"session", "reason"} for the geometry-fatal
        manifest.json            # DERIVED index (--index)

The ``.json`` is written LAST and is the commit marker: a ``.dat`` without it is
``incomplete`` (never read), and a ``.dat`` whose size disagrees with its
``.json`` shape is ``truncated``. Both are reported by :func:`scan`.

**Coordinate frame (binding).** The stored array's origin IS the hull origin
``(x0, y0)``, padding included: ``frames[i, r, c]`` holds source pixel
``(x0 + c, y0 + r)``, so the load-side mapping is
``stored_index = (source_coordinate - hull_origin) * store_scale`` with **no**
pad term. ``pad`` only records where the black bars sit inside the stored
rectangle, and ``store_scale`` is ``1.0`` for the native store (which is what
makes that the identity mapping). ``meta['hull_hw']`` is ALWAYS the SOURCE hull
size, so the scale survives a resized store.
"""
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import rgb_features as rf                              # noqa: E402
from data import rgb_roi_dataset as rrd                          # noqa: E402
from data import roi_cache as rc                                 # noqa: E402
from data import tir_resp_dataset as trd                         # noqa: E402

Box = Tuple[int, int, int, int]

#: Bump when a change alters the STORED PIXELS or the shard schema. Independent
#: of ``roi_cache.CACHE_FORMAT_VERSION`` and of the resp cache's version on
#: purpose: three artefacts, three lifecycles.
CACHE_FORMAT_VERSION = 1

#: Slug prefix. Distinct from TIR's ``tirroi_`` so the two never look alike.
CACHE_PREFIX = 'rgb2droi'

#: The STORAGE rule: the stored rectangle is the bounding box over the box of
#: every valid stride-1 window (see :func:`hull_box_2d`). It is EXACT -- the hull
#: is the union of exactly those boxes -- so containment (I2) holds by
#: construction and no safety margin is needed.
BOX_RULE = 'window-hull'

#: A frame is INVALID when any TARGET landmark is non-finite. NOT the thermal
#: sentinel rule; the string is part of the key so the distinction is enforced.
VALID_FRAME_RULE = 'finite-excluded'

PIXEL_FORMAT = 'rgb8'

#: The resolution RULE. ``'native'`` means the builder performs NO resampling:
#: the stored pixel is a source pixel, which is what keeps ``input_size`` a free
#: load-time knob (and out of the key).
#:
#: ``'px<N>'`` (e.g. ``'px112'``) instead stores the hull RESIZED so its
#: SHORTER side is N px (isotropic: the longer side keeps the aspect ratio).
#: The native store costs ~316 MB of reads per 8-s clip (measured 2026-11-12:
#: median hull 725 px x 200 frames), i.e. ~4.9 TB per fold-0 epoch, which made
#: the tri-modal Stage-2 run I/O-bound (22.6 s/it). Storing at 112 px is
#: 7.5 MB/clip (42x less). The price is that N becomes part of the KEY: one
#: cache then serves only ``input_size <= N`` (docs: design §13/§14).
STORE_RESOLUTION = 'native'

#: Frame source resolution: extracted ``2D+3D/<S>/<T>/`` first, else the members
#: of ``2D+3D/<S>.zip`` decoded in memory. ``2D+3D`` is NEVER extracted to disk.
SOURCE_POLICY = 'dir-then-zip'

#: Frames decoded per write block. A native hull frame is ~0.19 MB at 250 px, so
#: 32 frames is ~6 MB resident -- independent of task length.
CHUNK_FRAMES = 32

SUBDIR_TASKS = 'tasks'
SUBDIR_SKIPPED = 'skipped'
SUBDIR_TMP = '.tmp'
FRAMES_SUFFIX = '.dat'
LANDMARKS_SUFFIX = '.lm.dat'
META_SUFFIX = '.json'
PARAMS_NAME = 'params.json'
MANIFEST_NAME = 'manifest.json'

#: The fields a LOADER can verify without touching a pixel. ``src_w``/``src_h``
#: and ``raw_root`` stay in the hash (so a foreign build lands in a different
#: directory) but are not compared: the loader never decodes, so it cannot prove
#: them. ``input_size``, ``clip_stride``, ``max_sway_px``, ``min_valid_ratio``
#: and the resize kernel are absent entirely -- they cannot change a byte.
COMPARE_KEYS = ('format_version', 'corpus', 'roi_landmarks_name',
                'roi_landmarks', 'roi_padding', 'crop_rule', 'box_rule',
                'clip_seconds', 'fps', 'valid_frame_rule', 'pixel_format',
                'store_resolution', 'source_policy')


# --------------------------------------------------------------------------- #
# cache identity
# --------------------------------------------------------------------------- #
def landmark_set_name_2d(target_landmarks: Sequence[int]) -> str:
    """Readable name of the 1-indexed label set, for the key slug.

    ``'face'`` for the full 1..49 set (the shipped default), a preset name from
    :data:`data.rgb_features.ROI_LANDMARKS_2D` when it matches exactly, else
    ``'custom'``.
    """
    want = tuple(int(i) for i in target_landmarks)
    if want == tuple(range(1, rf.NUM_LANDMARKS_2D + 1)):
        return 'face'
    for name, pts in rf.ROI_LANDMARKS_2D.items():
        if tuple(int(i) for i in pts) == want:
            return name
    return 'custom'


def landmark_indices_2d(target_labels: Sequence[int]) -> Tuple[int, ...]:
    """1-indexed labels -> 0-indexed numpy indices, validated.

    One conversion point, exactly as in ``roi_cache.landmark_indices``: passing
    labels straight to numpy is off by one and silently selects a DIFFERENT
    point set (and a hull from the wrong set is not guaranteed to contain the
    right set's clip boxes).
    """
    out = tuple(int(l) - 1 for l in target_labels)
    if not out:
        raise ValueError('rgb2d_cache: empty landmark set')
    for i in out:
        if not 0 <= i < rf.NUM_LANDMARKS_2D:
            raise ValueError(
                f'rgb2d_cache: landmark label {i + 1} outside '
                f'1..{rf.NUM_LANDMARKS_2D}')
    return out


def valid_mask_2d(landmarks: np.ndarray,
                  target_labels: Sequence[int]) -> np.ndarray:
    """``bool [T]`` -- the DATASET's frame predicate, called not re-implemented.

    A frame is invalid when ANY target landmark of it is non-finite, i.e. the
    empty-``[]``/NaN representation ``rgb_features.parse_2d_features`` produces
    for a lost track. Same "ANY target bad drops the frame" semantics as
    ``tir_resp_dataset.missing_frame_mask``; a different predicate.
    """
    pts = np.asarray(landmarks, dtype=np.float64)
    if pts.ndim != 3 or pts.shape[1:] != (rf.NUM_LANDMARKS_2D, 2):
        raise ValueError(
            f'expected (frames, {rf.NUM_LANDMARKS_2D}, 2), got {pts.shape}')
    idx = np.asarray(landmark_indices_2d(target_labels), dtype=np.int64)
    return ~np.isfinite(pts[:, idx, :]).all(axis=(1, 2))


def valid_rows_2d(landmarks: np.ndarray,
                  target_labels: Sequence[int]) -> np.ndarray:
    """Target-landmark rows of the frames the dataset would keep (``[n, L, 2]``)."""
    pts = np.asarray(landmarks)
    idx = np.asarray(landmark_indices_2d(target_labels), dtype=np.int64)
    keep = ~valid_mask_2d(pts, target_labels)
    return pts[keep][:, idx, :]


def parse_store_resolution(value) -> str:
    """Normalise a store-resolution request to ``'native'`` or ``'px<N>'``.

    ``'native'`` (the default) keeps source pixels and a free ``input_size``.
    ``112`` / ``'112'`` / ``'px112'`` store the hull resized to 112 x 112 at
    BUILD time, which removes the native store's 40-70x read amplification but
    pins ``N`` into the cache key. ``None``/``''`` mean ``'native'``.
    """
    if value is None:
        return STORE_RESOLUTION
    if isinstance(value, bool):
        raise ValueError(f'store_resolution must be a string or an int, '
                         f'got {value!r}')
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ('', 'native', 'none'):
            return STORE_RESOLUTION
        if text.startswith('px'):
            text = text[2:]
    else:
        text = str(int(value))
    try:
        size = int(text)
    except (TypeError, ValueError):
        raise ValueError(
            f"store_resolution must be 'native' or a pixel size "
            f"(112 / '112' / 'px112'); got {value!r}")
    if size < 16:
        raise ValueError(
            f'store_resolution: {size} px is too small to be a face ROI; use '
            f"'native' or >= 16")
    return f'px{size}'


def store_size_of(resolution) -> Optional[int]:
    """``None`` for the native store, else the stored SHORT side in px."""
    res = parse_store_resolution(resolution)
    return None if res == STORE_RESOLUTION else int(res[2:])


def build_params(raw_root: str, sessions: Sequence[dict],
                 target_landmarks: Sequence[int], roi_padding: float,
                 src_w: int, src_h: int, crop_rule: str = 'square',
                 clip_seconds: float = 8.0, fps: float = 25.0,
                 store_resolution: str = STORE_RESOLUTION,
                 source_policy: str = SOURCE_POLICY) -> Dict:
    """The FROZEN parameter set. Everything here changes stored pixels."""
    store = parse_store_resolution(store_resolution)
    if store != STORE_RESOLUTION and crop_rule != 'square':
        raise ValueError(
            f'rgb2d_cache: store_resolution={store!r} needs crop_rule '
            f"'square' (one isotropic scale); got crop_rule={crop_rule!r}. "
            f'An anisotropic store would change the aspect ratio.')
    if crop_rule not in trd.CROP_RULES:
        raise ValueError(f'crop_rule must be one of {list(trd.CROP_RULES)}, '
                         f'got {crop_rule!r}')
    return {
        'format_version': CACHE_FORMAT_VERSION,
        'raw_root': str(raw_root),
        'corpus': rc.corpus_digest(sessions),
        'n_sessions_discovered': len(sessions),
        'roi_landmarks_name': landmark_set_name_2d(target_landmarks),
        'roi_landmarks': [int(i) for i in target_landmarks],
        'roi_padding': float(roi_padding),
        'crop_rule': str(crop_rule),
        'box_rule': BOX_RULE,
        'clip_seconds': float(clip_seconds),
        'fps': float(fps),
        'valid_frame_rule': VALID_FRAME_RULE,
        'src_w': int(src_w),
        'src_h': int(src_h),
        'pixel_format': PIXEL_FORMAT,
        'store_resolution': store,
        'source_policy': str(source_policy),
    }


def cache_key(params: Dict) -> str:
    """Readable slug + content hash. The hash is over EVERY frozen field."""
    body = json.dumps(params, sort_keys=True, default=str)
    slug = (f'{CACHE_PREFIX}_{params["roi_landmarks_name"]}'
            f'_pad{float(params["roi_padding"]):.2f}'
            f'_{params["crop_rule"]}'
            f'_{params["box_rule"]}'
            f'_clip{float(params["clip_seconds"]):g}s'
            f'_src{params["src_w"]}x{params["src_h"]}'
            f'_{params["store_resolution"]}'
            f'_v{params["format_version"]}')
    return f'{slug}_{rc._sha1(body)[:10]}'


def loader_params(raw_root: str, corpus_sessions: Sequence[dict],
                  target_landmarks: Sequence[int], roi_padding: float,
                  crop_rule: str = 'square', clip_seconds: float = 8.0,
                  fps: float = 25.0,
                  store_resolution: str = STORE_RESOLUTION,
                  source_policy: str = SOURCE_POLICY) -> Dict:
    """The subset of the frozen parameters a loader can compute on its own.

    ``corpus_sessions`` MUST be the UNFILTERED discovery: the key is a property
    of the CORPUS, not of the subset a run trains on.
    """
    full = build_params(raw_root, corpus_sessions, target_landmarks,
                        roi_padding, 0, 0, crop_rule=crop_rule,
                        clip_seconds=clip_seconds, fps=fps,
                        store_resolution=store_resolution,
                        source_policy=source_policy)
    return {k: full[k] for k in COMPARE_KEYS}


def check_params_compatible(root: Path, params: Dict) -> None:
    """FAIL LOUDLY if the on-disk cache was built with different parameters."""
    have = read_params(root)
    if have == params:
        return
    diffs = [f'{k}: cache={have.get(k)!r} run={params.get(k)!r}'
             for k in sorted(set(have) | set(params))
             if have.get(k) != params.get(k)]
    raise ValueError(
        f'rgb2d_cache: {root} was built with DIFFERENT parameters:\n  '
        + '\n  '.join(diffs)
        + '\nRefusing to mix shards. Use a different --out_root or delete it.')


# Re-exported from ``roi_cache`` unchanged: same layout, same atomicity.
cache_root = rc.cache_root
write_params = rc.write_params
read_params = rc.read_params


def find_cache(out_root, want: Dict) -> Path:
    """Locate the 2D cache whose parameters match ``want`` (fail-loudly).

    Delegates to :func:`data.roi_cache.find_cache` with this module's
    ``COMPARE_KEYS``, so the "Closest existing cache(es)" diff diagnostics are
    shared instead of duplicated.
    """
    return rc.find_cache(out_root, want, keys=COMPARE_KEYS)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def square_span_2d(pts, padding: float = 0.10,
                   crop_rule: str = 'square',
                   min_size: int = 2) -> Box:
    """Integer box around one landmark cloud, **NOT clamped to the frame**.

    Half-open ``(x0, x1, y0, y1)``, one integer side for both axes when
    ``crop_rule='square'`` (the short axis grows about the padded centre), which
    is what gives the loader ONE isotropic scale (I6) and never cuts the
    envelope (I7).

    Being unclamped is the point: the hull unions these spans, and the builder
    black-pads wherever they leave the image, so a subject near the border keeps
    a square crop instead of being translated (design §3.4). ONE implementation
    is used by the build side (:func:`hull_box_2d`) and by the dataset's
    clip-level gate, so the two cannot disagree about a clip's box.
    """
    if crop_rule not in trd.CROP_RULES:
        raise ValueError(f'crop_rule must be one of {list(trd.CROP_RULES)}, '
                         f'got {crop_rule!r}')
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    p = p[np.isfinite(p).all(axis=1)]
    if p.size == 0:
        raise ValueError('no finite landmark coordinates in this window')
    ex0, ex1 = float(p[:, 0].min()), float(p[:, 0].max())
    ey0, ey1 = float(p[:, 1].min()), float(p[:, 1].max())
    w, h = ex1 - ex0, ey1 - ey0
    x0, x1 = ex0 - padding * w, ex1 + padding * w
    y0, y1 = ey0 - padding * h, ey1 + padding * h
    # Legacy integer convention: [floor(low), ceil(high) + 1), so a padded
    # envelope of exactly 100..220 needs rows 100..220 INCLUSIVE.
    ix_lo, ix_hi = int(math.floor(x0)), int(math.ceil(x1)) + 1
    iy_lo, iy_hi = int(math.floor(y0)), int(math.ceil(y1)) + 1
    w_b, h_b = ix_hi - ix_lo, iy_hi - iy_lo
    if crop_rule != 'square':
        return (ix_lo, max(ix_lo + min_size, ix_hi),
                iy_lo, max(iy_lo + min_size, iy_hi))
    n = max(int(min_size), w_b, h_b)
    ix0 = ix_lo - (n - w_b) // 2
    iy0 = iy_lo - (n - h_b) // 2
    return (ix0, ix0 + n, iy0, iy0 + n)


def in_frame_share(box: Box, src_w: int, src_h: int) -> float:
    """Fraction of ``box`` that lies inside the source frame, in ``[0, 1]``.

    ``0.0`` when the box is entirely outside, ``1.0`` when entirely inside. Used
    by the build-side task gate AND the dataset's per-clip gate, so one rule
    serves both levels.
    """
    x0, x1, y0, y1 = (int(v) for v in box)
    area = max(1, (x1 - x0) * (y1 - y0))
    inter_w = max(0, min(x1, int(src_w)) - max(x0, 0))
    inter_h = max(0, min(y1, int(src_h)) - max(y0, 0))
    return float(inter_w * inter_h) / float(area)


def hull_box_2d(landmarks: np.ndarray, target_labels: Sequence[int],
                padding: float, crop_rule: str = 'square',
                clip_seconds: float = 8.0, fps: float = 25.0
                ) -> Optional[Box]:
    """The STORAGE box: bounding box over every valid window's clip box.

    Union over **stride-1** ``clip_seconds``-long windows of that window's
    ``crop_rule`` box, so the hull contains the box of EVERY clip the loader can
    emit, for any hop and any ``task_set`` (I2). Windows overlapping an invalid
    (non-finite) row are skipped -- the dataset would drop them -- via
    :func:`valid_mask_2d`, the same predicate.

    Unlike ``roi_cache.hull_box`` an out-of-frame window is KEPT: its unclamped
    square span joins the union and the builder black-pads the margin. Returns
    ``None`` when there is no usable row at all (a task shorter than one clip
    falls back to its whole valid track).
    """
    pts = np.asarray(landmarks)
    idx = np.asarray(landmark_indices_2d(target_labels), dtype=np.int64)
    bad = valid_mask_2d(pts, target_labels)
    tgt = pts[:, idx, :]
    n = int(tgt.shape[0])
    cf = max(1, int(round(float(clip_seconds) * float(fps))))

    boxes: List[Box] = []
    for start in range(0, max(0, n - cf) + 1):
        if bad[start:start + cf].any():
            continue
        try:
            boxes.append(square_span_2d(tgt[start:start + cf], padding,
                                        crop_rule))
        except ValueError:
            continue
    if not boxes:
        rows = valid_rows_2d(landmarks, target_labels)
        if rows.size == 0:
            return None
        return square_span_2d(rows, padding, crop_rule)
    return (min(b[0] for b in boxes), max(b[1] for b in boxes),
            min(b[2] for b in boxes), max(b[3] for b in boxes))


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def meta_path(root, session: str) -> Path:
    return Path(root) / SUBDIR_TASKS / f'{session}{META_SUFFIX}'


def frames_path(root, session: str) -> Path:
    return Path(root) / SUBDIR_TASKS / f'{session}{FRAMES_SUFFIX}'


def landmarks_path(root, session: str) -> Path:
    return Path(root) / SUBDIR_TASKS / f'{session}{LANDMARKS_SUFFIX}'


def shard_path(root, session: str) -> Path:
    """The shard's COMMIT MARKER (the ``.json``); a shard exists iff it does."""
    return meta_path(root, session)


# --------------------------------------------------------------------------- #
# build one task
# --------------------------------------------------------------------------- #
def _load_track_on_jpg_grid(feat_file: str, n_jpg: int) -> np.ndarray:
    """``[n_jpg, 49, 2]`` track, NaN where the ``.mat`` leaves a jpg uncovered."""
    pts, _pose, frame, valid = rf.parse_2d_features(feat_file)
    out = np.full((int(n_jpg), rf.NUM_LANDMARKS_2D, 2), np.nan, np.float64)
    for i in range(int(pts.shape[0])):
        if not bool(valid[i]):
            continue
        j = rf.jpg_index_of(int(frame[i]))
        if 0 <= j < int(n_jpg):
            out[j] = pts[i]
    return out


def probe_frame_size(spec: dict) -> Tuple[int, int, int]:
    """Decode the FIRST frame of a session -> ``(h, w, c)``. Used by ``--init``."""
    files = rrd._list_rgb_files(str(spec['rgb_dir']))
    if not files:
        raise ValueError(f'{spec["session"]}: no image frames')
    img = rrd._imread(os.path.join(str(spec['rgb_dir']), files[0]),
                      rrd._cv2().IMREAD_COLOR)
    return int(img.shape[0]), int(img.shape[1]), int(img.shape[2])


def build_task(spec: dict, target_labels: Sequence[int], padding: float,
               stage_dir, crop_rule: str = 'square',
               clip_seconds: float = 8.0, fps: float = 25.0,
               max_task_bytes: int = 0,
               store_resolution: str = STORE_RESOLUTION
               ) -> Tuple[Optional[Dict], Optional[str]]:
    """Write ONE session's ``.dat``/``.lm.dat`` into ``stage_dir`` -> metadata.

    Returns ``(meta, None)`` on success or ``(None, reason)`` when the task is
    GEOMETRY-FATAL -- i.e. when there is no pixel to crop. A label-side or
    selection-side problem (sway, low validity, mostly-out-of-frame clips) must
    NOT reach here and must NOT cause a skip: those are dataset rules with their
    own counters, and skipping on them would make the cache depend on a knob it
    is supposed to be independent of.
    """
    session = str(spec['session'])
    feat_file = spec.get('feat_file')
    rgb_dir = spec.get('rgb_dir')
    if not feat_file or not os.path.isfile(str(feat_file)):
        return None, 'missing_2d_features'
    if not rgb_dir:
        return None, 'missing_image_seq'
    cv2 = rrd._cv2()

    # --- 1. frame list + landmark track on the JPG index grid --------------
    files = rrd._list_rgb_files(str(rgb_dir))
    n_jpg = len(files)
    if n_jpg <= 0:
        return None, 'missing_image_seq'
    try:
        track = _load_track_on_jpg_grid(str(feat_file), n_jpg)
    except Exception as exc:
        return None, f'invalid_2d_features: {exc}'
    n_mat = int(rf.parse_2d_features(str(feat_file))[0].shape[0])
    if n_mat != n_jpg:
        # The v1 RGB rule: a count mismatch makes frame <-> landmark alignment
        # unprovable, so the session is unusable rather than silently shifted.
        return None, f'image_count_mismatch: n_jpg={n_jpg} n_mat={n_mat}'
    if valid_rows_2d(track, target_labels).size == 0:
        return None, 'no_valid_landmark_rows'

    # --- 2. probe one frame: size + hull (a pure function of the track) ----
    try:
        fh, fw = probe_frame_size(spec)[:2]
    except Exception as exc:
        return None, f'undecodable_images: {exc}'
    hull = hull_box_2d(track, target_labels, padding, crop_rule,
                       clip_seconds, fps)
    if hull is None:
        return None, 'no_valid_landmark_rows'
    x0, x1, y0, y1 = hull
    share = in_frame_share(hull, fw, fh)
    if share < 0.5:
        return None, f'box_out_of_frame: {share:.3f}'

    # Black padding: the ONLY place the bars come from (design §3.4).
    pad_top, pad_left = max(0, -y0), max(0, -x0)
    pad_bottom, pad_right = max(0, y1 - fh), max(0, x1 - fw)
    cy0, cy1 = max(0, y0), min(fh, y1)
    cx0, cx1 = max(0, x0), min(fw, x1)
    assert (x1 - x0) == (cx1 - cx0) + pad_left + pad_right, \
        f'{session}: hull width {x1 - x0} != clamped {cx1 - cx0} + pads'
    assert (y1 - y0) == (cy1 - cy0) + pad_top + pad_bottom, \
        f'{session}: hull height {y1 - y0} != clamped {cy1 - cy0} + pads'

    h, w = int(y1 - y0), int(x1 - x0)
    store = parse_store_resolution(store_resolution)
    s_store = store_size_of(store)
    # A pixel store is an ISOTROPIC resize of the hull RECTANGLE whose SHORTER
    # side becomes ``s_store`` (the hull is the union of SQUARE clip boxes at
    # different offsets, so it is a rectangle -- 769x795 on F001_T1 -- and
    # squashing it to SxS would apply a per-axis scale). Using the short side as
    # the reference is also what keeps every square clip box in range: a clip
    # side is <= min(hull_w, hull_h).
    if s_store is not None and min(h, w) <= s_store:
        # The hull is already at or below the store size: downscaling is the
        # ONLY thing a pixel store is for, so keep source pixels (upsampling
        # invents detail). ``meta['store_size']`` records the per-session truth
        # and the reader derives its scale from the shapes either way.
        s_store = None
    out_h, out_w = (h, w)
    if s_store is not None:
        scale = float(s_store) / float(min(h, w))
        out_h = max(1, int(round(h * scale)))
        out_w = max(1, int(round(w * scale)))
    nbytes = int(n_jpg) * out_h * out_w * 3
    if max_task_bytes and nbytes > max_task_bytes:
        return None, (f'too_large_for_memory: the artefact alone would be '
                      f'{nbytes / 1e9:.2f} GB > {max_task_bytes / 1e9:.2f} GB '
                      f'budget ({n_jpg} frames at {out_w}x{out_h})')

    # --- 3. stream the frames straight into a staging memmap ---------------
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    s_frames = stage_dir / f'{session}{FRAMES_SUFFIX}'
    s_lm = stage_dir / f'{session}{LANDMARKS_SUFFIX}'
    mm_frames = np.memmap(str(s_frames), dtype=np.uint8, mode='w+',
                          shape=(int(n_jpg), out_h, out_w, 3))
    mm_lm = np.memmap(str(s_lm), dtype=np.float32, mode='w+',
                      shape=(int(n_jpg), rf.NUM_LANDMARKS_2D, 2))
    try:
        mm_lm[:] = track.astype(np.float32)
        for start in range(0, int(n_jpg), CHUNK_FRAMES):
            stop = min(start + CHUNK_FRAMES, int(n_jpg))
            for i in range(start, stop):
                img = rrd._imread(os.path.join(str(rgb_dir), files[i]),
                                  cv2.IMREAD_COLOR)
                if img.shape[0] != fh or img.shape[1] != fw:
                    return None, (f'undecodable_images: frame {i} is '
                                  f'{img.shape[0]}x{img.shape[1]}, expected '
                                  f'{fh}x{fw}')
                crop = img[cy0:cy1, cx0:cx1]
                pad = cv2.copyMakeBorder(crop, pad_top, pad_bottom,
                                         pad_left, pad_right,
                                         cv2.BORDER_CONSTANT, value=[0, 0, 0])
                rgb = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB)
                if s_store is not None:
                    # ONE downsample, at BUILD time, isotropic (INTER_AREA).
                    rgb = cv2.resize(rgb, (out_w, out_h),
                                     interpolation=cv2.INTER_AREA)
                mm_frames[i] = rgb
            mm_frames.flush()
            mm_lm.flush()
    except Exception as exc:
        return None, f'undecodable_images: {exc}'
    finally:
        # Drop the mappings BEFORE the caller renames the files: a still-dirty
        # mapping can publish a partial array that the .json would then bless.
        mm_frames.flush()
        mm_lm.flush()
        del mm_frames
        del mm_lm

    meta = {
        'session': session,
        'source': str(rgb_dir),
        'n_frames': int(n_jpg),
        'box': [int(v) for v in hull],
        'pad': [int(pad_top), int(pad_bottom), int(pad_left), int(pad_right)],
        'src_hw': [int(fh), int(fw)],
        'hull_hw': [int(h), int(w)],          # SOURCE px, always
        'store_resolution': store,
        'store_size': (int(s_store) if s_store is not None else None),
        'area_in_frac': float(share),
        'frames': {'shape': [int(n_jpg), out_h, out_w, 3], 'dtype': 'uint8',
                   'nbytes': nbytes},
        'landmarks': {'shape': [int(n_jpg), rf.NUM_LANDMARKS_2D, 2],
                      'dtype': 'float32',
                      'nbytes': int(n_jpg) * rf.NUM_LANDMARKS_2D * 2 * 4},
    }
    return meta, None


def commit_shard(root, session: str, meta: Dict, stage_dir) -> Path:
    """Publish a staged shard: arrays first, then the ``.json`` COMMIT MARKER.

    A publication also CLEARS any earlier ``skipped/<session>.json``: a session
    can be refused for a transient reason (``too_large_for_memory`` against a
    budget that is later raised, or a ``build_error``) and rebuilt later, and a
    stale marker would make the dataset exclude a session whose shard is
    perfectly good -- the loader consults the marker BEFORE the shard.
    """
    root = Path(root)
    (root / SUBDIR_TASKS).mkdir(parents=True, exist_ok=True)
    stage_dir = Path(stage_dir)
    dst_tasks = root / SUBDIR_TASKS
    os.replace(stage_dir / f'{session}{FRAMES_SUFFIX}',
               dst_tasks / f'{session}{FRAMES_SUFFIX}')
    os.replace(stage_dir / f'{session}{LANDMARKS_SUFFIX}',
               dst_tasks / f'{session}{LANDMARKS_SUFFIX}')
    tmp = dst_tasks / f'{session}.{os.getpid()}{META_SUFFIX}'
    with open(tmp, 'w') as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    os.replace(tmp, dst_tasks / f'{session}{META_SUFFIX}')
    stale = root / SUBDIR_SKIPPED / f'{session}.json'
    if stale.is_file():
        try:
            stale.unlink()
        except OSError:
            pass
    return shard_path(root, session)


def write_skip(root, session: str, reason: str) -> Path:
    """Record WHY a session was not cached (same layout as the TIR cache)."""
    return rc.write_skip(root, session, reason)


# --------------------------------------------------------------------------- #
# read one shard
# --------------------------------------------------------------------------- #
class ShardHandle:
    """Two read-only memory maps + the parsed metadata for ONE session.

    The dataset keeps a small LRU of these (``rgb2d_cache_lru``), so the knob
    counts OPEN HANDLES, not materialised shards: resident memory is ~0 and each
    handle costs two file descriptors. ``close()`` is idempotent and drops the
    references, which releases the fds once the last view dies.

    **Never** call ``._mmap.close()`` directly: ``np.memmap._mmap`` IS the
    ``mmap.mmap``, and closing it behind a live numpy view raises ``BufferError``
    at best and leaves a dangling buffer at worst.
    """

    def __init__(self, session: str, meta: Dict, root):
        self.session = str(session)
        self.meta = dict(meta)
        self.root = Path(root)
        self.frames = np.memmap(str(frames_path(root, session)),
                                dtype=np.dtype(meta['frames']['dtype']),
                                mode='r',
                                shape=tuple(int(v) for v in
                                            meta['frames']['shape']))
        self.landmarks = np.memmap(
            str(landmarks_path(root, session)),
            dtype=np.dtype(meta['landmarks']['dtype']), mode='r',
            shape=tuple(int(v) for v in meta['landmarks']['shape']))

    # convenience mirrors of the metadata the dataset uses every clip
    @property
    def box(self) -> Box:
        return tuple(int(v) for v in self.meta['box'])

    @property
    def pad(self) -> Tuple[int, int, int, int]:
        return tuple(int(v) for v in self.meta['pad'])

    @property
    def n_frames(self) -> int:
        return int(self.meta['n_frames'])

    @property
    def src_hw(self) -> Tuple[int, int]:
        return tuple(int(v) for v in self.meta['src_hw'])

    @property
    def store_scale(self) -> float:
        """Stored pixels per SOURCE pixel (``1.0`` for the native store).

        The stored array's origin IS the hull origin and it covers the SAME
        source rectangle either way, so a source-space offset ``d`` maps to
        stored index ``d * store_scale``. Deriving it from the shapes (rather
        than a stored scalar) means a legacy ``.json`` without ``store_size``
        still yields exactly 1.0.
        """
        shape = (self.meta.get('frames') or {}).get('shape') or []
        hull = self.meta.get('hull_hw') or []
        if len(shape) < 2 or not hull or not int(hull[0]):
            return 1.0
        return float(shape[1]) / float(int(hull[0]))

    @property
    def store_size(self) -> Optional[int]:
        """Effective SHORT side of the stored hull, or ``None`` if native."""
        return self.meta.get('store_size')

    def __enter__(self) -> 'ShardHandle':
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        for name in ('frames', 'landmarks'):
            m = getattr(self, name, None)
            if m is None:
                continue
            if getattr(m, 'flags', None) is not None and m.flags.writeable:
                m.flush()          # mode='r' maps have nothing to flush
            setattr(self, name, None)


def open_shard(root, session: str) -> ShardHandle:
    """Open one shard, or raise if it is missing/incomplete/truncated."""
    root = Path(root)
    mp = meta_path(root, session)
    if not mp.is_file():
        if frames_path(root, session).is_file():
            raise FileNotFoundError(
                f'rgb2d_cache: {session}: the .dat exists but {mp.name} does not '
                f'-- the shard is INCOMPLETE (the .json is the commit marker). '
                f'Rebuild it.')
        raise FileNotFoundError(
            f'rgb2d_cache: {session}: no shard in {root}. Run '
            f'`runners/run_build_2d_roi_cache.py --status` then `--build`.')
    with open(mp) as fh:
        meta = json.load(fh)
    for key, path in (('frames', frames_path(root, session)),
                      ('landmarks', landmarks_path(root, session))):
        if not Path(path).is_file():
            raise FileNotFoundError(
                f'rgb2d_cache: {session}: {path.name} is missing (incomplete).')
        want = int(meta[key]['nbytes'])
        have = Path(path).stat().st_size
        if have != want:
            raise ValueError(
                f'rgb2d_cache: {session}: {path.name} is TRUNCATED '
                f'({have} B on disk vs {want} B declared by the .json).')
    return ShardHandle(session, meta, root)


def shard_metadata(root, sessions: Sequence[dict]) -> Dict[str, Dict]:
    """Per-session ``{box, pad, n_frames, hull_hw, src_hw, store_size}``."""
    out: Dict[str, Dict] = {}
    for s in sessions:
        name = str(s['session'])
        mp = meta_path(root, name)
        if not mp.is_file():
            continue
        try:
            with open(mp) as fh:
                meta = json.load(fh)
        except Exception:
            continue
        out[name] = {'box': tuple(int(v) for v in meta['box']),
                     'pad': tuple(int(v) for v in meta['pad']),
                     'n_frames': int(meta['n_frames']),
                     'hull_hw': tuple(int(v) for v in meta['hull_hw']),
                     'src_hw': tuple(int(v) for v in meta['src_hw']),
                     'store_size': meta.get('store_size')}
    return out


def scan(root, sessions: Sequence[dict]) -> Dict:
    """Filesystem truth: cached / skipped / missing / incomplete / truncated."""
    root = Path(root)
    cached: List[str] = []
    skipped: List[str] = []
    missing: List[str] = []
    incomplete: List[str] = []
    truncated: List[str] = []
    n_bytes = 0
    for s in sessions:
        name = str(s['session'])
        mp = meta_path(root, name)
        fp = frames_path(root, name)
        lp = landmarks_path(root, name)
        if mp.is_file():
            if not (fp.is_file() and lp.is_file()):
                incomplete.append(name)
                continue
            try:
                with open(mp) as fh:
                    meta = json.load(fh)
            except Exception:
                incomplete.append(name)
                continue
            if (fp.stat().st_size != int(meta['frames']['nbytes'])
                    or lp.stat().st_size != int(meta['landmarks']['nbytes'])):
                truncated.append(name)
                continue
            cached.append(name)
            n_bytes += fp.stat().st_size + lp.stat().st_size
            continue
        if (root / SUBDIR_SKIPPED / f'{name}.json').is_file():
            skipped.append(name)
            continue
        if fp.is_file() or lp.is_file():
            incomplete.append(name)      # arrays published, marker not (killed)
            continue
        missing.append(name)
    return {'cached': len(cached), 'skipped': len(skipped),
            'missing': len(missing), 'incomplete': len(incomplete),
            'truncated': len(truncated), 'bytes': n_bytes,
            'cached_names': cached, 'skipped_names': skipped,
            'missing_names': missing, 'incomplete_names': incomplete,
            'truncated_names': truncated}


def build_manifest(root, sessions: Sequence[dict], params: Dict) -> Dict:
    """Derived index (``--index``). Shards, not this file, are authoritative."""
    root = Path(root)
    entries: List[Dict] = []
    for s in sessions:
        name = str(s['session'])
        mp = meta_path(root, name)
        if mp.is_file():
            try:
                with open(mp) as fh:
                    meta = json.load(fh)
                entries.append({
                    'session': name, 'status': 'ok',
                    'bytes': (int(meta['frames']['nbytes'])
                              + int(meta['landmarks']['nbytes'])),
                    'box': [int(v) for v in meta['box']],
                    'pad': [int(v) for v in meta['pad']],
                    'n_frames': int(meta['n_frames']),
                    'shape': [int(v) for v in meta['frames']['shape']],
                    'area_in_frac': float(meta.get('area_in_frac', 1.0)),
                    'source': meta.get('source'),
                })
                continue
            except Exception:
                entries.append({'session': name, 'status': 'incomplete'})
                continue
        sp = root / SUBDIR_SKIPPED / f'{name}.json'
        if sp.is_file():
            with open(sp) as fh:
                entries.append(dict(json.load(fh), status='skipped'))
        else:
            entries.append({'session': name, 'status': 'missing'})
    manifest = {
        'key': cache_key(params),
        'params': params,
        'n_sessions': len(sessions),
        'n_ok': sum(1 for e in entries if e['status'] == 'ok'),
        'n_skipped': sum(1 for e in entries if e['status'] == 'skipped'),
        'n_missing': sum(1 for e in entries if e['status'] == 'missing'),
        'total_bytes': sum(int(e.get('bytes') or 0) for e in entries),
        'entries': entries,
    }
    tmp = root / (MANIFEST_NAME + '.tmp')
    with open(tmp, 'w') as fh:
        json.dump(manifest, fh, indent=2)
    os.replace(tmp, root / MANIFEST_NAME)
    return manifest


def discover_sessions(raw_root: str, subjects=None, tasks=None) -> List[dict]:
    """Every ``(subject, task)`` with a frame source, usable or not.

    Wraps ``rgb_roi_dataset.discover_rgb_sessions`` (which already decides the
    jpg/mat count mismatch and resolves dir-or-zip) so the corpus identity is
    computed from ONE discovery, unfiltered by the run.
    """
    return rrd.discover_rgb_sessions(raw_root, subjects=subjects, tasks=tasks,
                                     attach_landmarks=False)
