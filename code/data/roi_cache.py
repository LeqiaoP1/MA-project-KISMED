"""Offline TIR-ROI pixel cache: cache key, geometry gate, shard IO, verification.

Spec: ``notes/eval/tir_resp/hpc_precomputation.md`` section 9. The probe that
validates the design is ``notes/eval/tir_resp/cache_parity_probe.py``.

**The cache is ONE STEP of the pipeline and encodes no downstream decision.**
It answers "which pixels" faster than a decoder can, and nothing else. That
principle produces every rule here:

* :func:`cache_key` covers ONLY what changes the STORED PIXELS -- the ROI
  contract (landmark set, padding), the box rule, the valid-frame rule, the
  source frame size, the pixel format, the store resolution, the decoder, and the
  corpus identity.
* Everything that only SELECTS WINDOWS (``clip_stride``, ``task_set``/split,
  ``min_signal_spread``, ``rail_touch_v``) or RESIZES (``input_size``) is
  deliberately absent from the key AND from the writer, because both happen at
  load. That is exactly what lets ONE cache serve Stage 2 (2.0 s hop) and Stage 3
  (1.0 s hop), and both the 112 px and 64 px lineages.
* The gate is GEOMETRY-ONLY (:func:`build_task`): a session is skipped only when
  there is no pixel to crop, never because its label is unusable.
* Pixels are stored at NATIVE resolution and integer-sliced -- no scaling, no
  interpolation, no colour conversion beyond the one the decoder already does.
  Crop + resize belong to the loader (``_roi_patches``).

Layout::

    <out_root>/<CACHE_KEY>/
        params.json           # the frozen parameters (--init)
        tasks/F001_T1.npz     # ~66 MB mean; THE SOURCE OF TRUTH
        skipped/F011_T1.json  # {"session", "reason"} for the geometry-fatal few
        manifest.json         # DERIVED index (--index), never written concurrently

Shards, not the manifest, are authoritative: an array job writes only its own
disjoint set of shards, and ``--index`` rebuilds the manifest by scanning. Every
shard appears via ``os.replace``, so a killed job cannot leave a half-written one
behind -- which is also what makes the artifact resumable after a purge.

Shard contents (``np.savez``):

* ``frames``     uint8 ``[T, H_union, W_union, 3]`` RGB, native, the union crop
* ``box``        int32 ``[x0, x1, y0, y1]`` in SOURCE pixels (half-open)
* ``landmarks``  float32 ``[T, 28, 2]`` the source track this shard was built
                 from, so the shard is self-verifying without ``IRFeatures``
* ``session`` / ``video`` / ``n_frames`` -- provenance
"""
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import tir_resp_dataset as trd                          # noqa: E402
from data import video_io as vio                                  # noqa: E402

Box = Tuple[int, int, int, int]

#: Bump when a change alters the STORED PIXELS or the shard schema.
#:
#: v2 -> v3: the stored crop is no longer the padded-envelope union box of the
#: whole task but the SQUARE HULL over the sentinel-valid clip windows, and the
#: hull depends on the clip LENGTH and on ``crop_rule``. The rectangles change
#: SHAPE, so older shards are unusable rather than merely differently padded.
#:
#: The key is a hash of PARAMETERS, not of the code, so the version field is what
#: makes a stale cache impossible to reuse silently; ``--check`` is what catches
#: a mistake that got past it.
CACHE_FORMAT_VERSION = 3

#: The STORAGE rule: the stored rectangle is the bounding box over the box of
#: every sentinel-valid stride-1 window (see :func:`hull_box`). It is EXACT --
#: the hull is the union of exactly those boxes -- so containment (I2) holds by
#: construction and no safety margin is needed. ``crop_rule`` is a separate field
#: because it decides the SHAPE of each window box.
BOX_RULE = 'window-hull'

#: Tracker ``(0,0)`` sentinel rows are EXCLUDED from the union box. Safe because
#: a clip overlapping any sentinel/missing target landmark is dropped entirely by
#: ``missing_frame_mask``, so no kept clip's frames are removed here. Materially
#: cheaper too: including them inflates the p90 task box from 174 px to 385 px.
#:
#: 2026-10-15: the exclusion now CALLS ``missing_frame_mask`` (see
#: :func:`valid_rows`) instead of re-implementing it. The old hand-rolled copy
#: excluded a row only when ALL target landmarks were ``(0,0)`` (``.all()``)
#: while the dataset drops a frame when ANY is (``.any()``). The string is
#: unchanged because it is part of the cache key, and CACHE_FORMAT_VERSION is
#: deliberately NOT bumped: the change is provably a NO-OP on the shipped
#: corpus -- all 5760 sentinel frames across its 1385 tasks are WHOLE-ROW
#: sentinels (0 partial-sentinel, 0 all-NaN rows), so both rules agree on every
#: single frame and a rebuild would reproduce the existing shards byte for byte.
#: Bumping would invalidate ~91 GB for nothing.
VALID_FRAME_RULE = 'sentinel-excluded'

PIXEL_FORMAT = 'rgb8'
STORE_RESOLUTION = 'native'

#: Frames decoded per read. The writer must NOT materialise a whole task: a
#: 726x480x3 frame is 1.05 MB, the corpus' longest task is 5149 frames (5.4 GB),
#: and ``read_all`` builds a Python list of every frame AND stacks it, so its
#: peak was MEASURED at 11.1 GB RSS -- that is what OOM-killed 8 of the first 32
#: array elements. 256 frames = 0.27 GB per chunk, independent of task length.
CHUNK_FRAMES = 256

SUBDIR_TASKS = 'tasks'
SUBDIR_SKIPPED = 'skipped'
SUBDIR_TMP = '.tmp'
SHARD_SUFFIX = '.npz'
PARAMS_NAME = 'params.json'
MANIFEST_NAME = 'manifest.json'


# --------------------------------------------------------------------------- #
# cache identity
# --------------------------------------------------------------------------- #
def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def decoder_id() -> str:
    """Identity of the decode stack -- because the cache FREEZES its output.

    The shards hold DECODED pixels, so a decoder change (OpenCV build, FFmpeg
    version, ``VIDEO_BACKEND``) silently invalidates every one of them. Hashing
    this into the key is what turns that silent drift into a key mismatch.
    """
    parts = [f'backend={os.environ.get("VIDEO_BACKEND", "auto")}']
    try:
        import cv2
        parts.append(f'cv2={cv2.__version__}')
        for line in str(cv2.getBuildInformation()).splitlines():
            if 'FFMPEG' in line.upper():
                parts.append(' '.join(line.split())[:100])
                break
    except Exception as exc:                       # pragma: no cover - defensive
        parts.append(f'cv2=unavailable({type(exc).__name__})')
    try:
        import decord
        parts.append(f'decord={getattr(decord, "version", "?")}')
    except Exception:
        pass
    return '|'.join(parts)


def corpus_digest(sessions: Sequence[dict]) -> str:
    """Cheap, cluster-independent identity of the discovered corpus.

    Uses the sorted session NAMES only (not paths and not frame counts), so the
    writer and the loader can both compute it in milliseconds. Deep verification
    -- per-task frame counts and mtimes -- lives in the manifest instead.
    """
    names = sorted(str(s['session']) for s in sessions)
    return f'{_sha1(chr(10).join(names))[:16]}n{len(names)}'


def landmark_set_name(target_landmarks: Sequence[int]) -> str:
    """Name of the landmark set, for the readable part of the key.

    ``target_landmarks`` are 1-indexed labels, matching ``ROI_LANDMARK_PRESETS``.
    """
    want = tuple(int(i) for i in target_landmarks)
    for name, pts in trd.ROI_LANDMARK_PRESETS.items():
        if tuple(int(i) for i in pts) == want:
            return name
    return 'custom'


def landmark_indices(target_labels: Sequence[int]) -> Tuple[int, ...]:
    """1-indexed labels -> 0-indexed numpy indices.

    ``resolve_roi_landmarks`` returns 1-INDEXED labels (its documented contract)
    and the dataset converts them with ``l - 1`` (``BP4DPlusTIRRespDataset
    .target_idx``). Everything in this module funnels through HERE so the
    conversion cannot be forgotten at one call site.

    This is not hypothetical: the first version of the writer passed the labels
    straight to numpy, which selected landmarks 9..13/20..26 instead of
    8..12/19..25 and produced a union box 1.8x too large. A box from the WRONG
    set is not guaranteed to contain the right set's clip boxes, so it was not a
    harmless waste either -- it was caught by the size disagreement and by
    ``--check``.
    """
    out = tuple(int(l) - 1 for l in target_labels)
    for i in out:
        if not 0 <= i < trd.NUM_LANDMARKS:
            raise ValueError(
                f'roi_cache: landmark label {i + 1} outside 1..{trd.NUM_LANDMARKS}')
    if not out:
        raise ValueError('roi_cache: empty landmark set')
    return out


def build_params(raw_root: str, sessions: Sequence[dict],
                 target_landmarks: Sequence[int], roi_padding: float,
                 src_w: int, src_h: int,
                 crop_rule: str = trd.DEFAULT_CROP_RULE,
                 clip_seconds: float = trd.DEFAULT_CLIP_SECONDS,
                 fps: float = trd.DEFAULT_FPS) -> Dict:
    """The FROZEN parameter set. Everything here changes stored pixels.

    ``target_landmarks`` are 1-INDEXED labels, i.e. exactly what
    ``resolve_roi_landmarks`` returns and what the configs carry.

    ``crop_rule``, ``clip_seconds`` and ``fps`` join the set because the stored
    crop is now the hull over CLIP-SIZED windows: the geometry of each window box
    (``crop_rule``) and the length of those windows (``clip_seconds * fps``)
    decide which pixels the hull spans. ``input_size``, ``clip_stride``,
    ``task_set``, ``temporal_stride`` and the cleaning filters stay OUT -- they
    select or rescale, they never change a stored pixel (I3).
    """
    return {
        'format_version': CACHE_FORMAT_VERSION,
        'raw_root': str(raw_root),
        'corpus': corpus_digest(sessions),
        'n_sessions_discovered': len(sessions),
        'roi_landmarks_name': landmark_set_name(target_landmarks),
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
        'store_resolution': STORE_RESOLUTION,
        'decoder': decoder_id(),
    }


def cache_key(params: Dict) -> str:
    """Readable slug + content hash. The hash is over EVERY frozen field."""
    body = json.dumps(params, sort_keys=True, default=str)
    slug = (f'tirroi_{params["roi_landmarks_name"]}'
            f'_pad{float(params["roi_padding"]):.2f}'
            f'_{params["crop_rule"]}'
            f'_{params["box_rule"]}'
            f'_clip{float(params["clip_seconds"]):g}s'
            f'_src{params["src_w"]}x{params["src_h"]}'
            f'_{params["store_resolution"]}'
            f'_v{params["format_version"]}')
    return f'{slug}_{_sha1(body)[:10]}'


def cache_root(out_root: str, key: str) -> Path:
    return Path(out_root) / key


def write_params(root: Path, params: Dict) -> None:
    """Write ``params.json`` atomically. Concurrent identical writes are safe."""
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / (PARAMS_NAME + '.tmp')
    with open(tmp, 'w') as fh:
        json.dump(params, fh, indent=2, sort_keys=True)
    os.replace(tmp, root / PARAMS_NAME)


def read_params(root: Path) -> Dict:
    with open(Path(root) / PARAMS_NAME) as fh:
        return json.load(fh)


#: The fields a LOADER can verify without decoding anything. ``src_w``/``src_h``
#: are excluded because the loader never decodes (that is the point of the
#: cache) and ``decoder`` because it does not decode either; both stay in the
#: key, so a cache built by a different decoder still lands in a different
#: directory, but the loader cannot prove it was not tampered with.
COMPARE_KEYS = ('format_version', 'corpus', 'roi_landmarks_name', 'roi_landmarks',
                'roi_padding', 'crop_rule', 'box_rule', 'clip_seconds', 'fps',
                'valid_frame_rule', 'pixel_format', 'store_resolution')


def loader_params(raw_root: str, corpus_sessions: Sequence[dict],
                  target_landmarks: Sequence[int], roi_padding: float,
                  crop_rule: str = trd.DEFAULT_CROP_RULE,
                  clip_seconds: float = trd.DEFAULT_CLIP_SECONDS,
                  fps: float = trd.DEFAULT_FPS) -> Dict:
    """The subset of the frozen parameters a loader can compute on its own.

    ``corpus_sessions`` MUST be the UNFILTERED discovery: the cache key is a
    property of the CORPUS, not of which subset a given run trains on. That is
    what lets ``--subjects``/``--limit`` restrict a BUILD without creating a
    second cache identity.
    """
    full = build_params(raw_root, corpus_sessions, target_landmarks,
                        roi_padding, 0, 0, crop_rule=crop_rule,
                        clip_seconds=clip_seconds, fps=fps)
    return {k: full[k] for k in COMPARE_KEYS}


def find_cache(out_root, want: Dict) -> Path:
    """Locate the cache directory whose parameters match ``want``.

    THIS IS THE FAIL-LOUDLY GATE (spec section 9.2). A run whose ``roi_*`` keys
    differ must REFUSE the cache rather than fall back to slightly wrong pixels:
    silent data drift is the one failure mode that would invalidate every
    downstream comparison while looking perfectly healthy.

    Raises with a field-by-field diff when nothing matches, and refuses an
    ambiguous match rather than picking one.
    """
    out_root = Path(out_root)
    if not out_root.is_dir():
        raise FileNotFoundError(
            f'roi_cache: out_root does not exist: {out_root}. Build it with '
            f'`runners/run_build_roi_cache.py --init --build`.')
    matches: List[Path] = []
    near: List[Tuple[Path, Dict]] = []
    for d in sorted(out_root.iterdir()):
        pf = d / PARAMS_NAME
        if not (d.is_dir() and pf.is_file()):
            continue
        try:
            with open(pf) as fh:
                have = json.load(fh)
        except Exception:
            continue
        diffs = {k: (have.get(k), want.get(k)) for k in COMPARE_KEYS
                 if have.get(k) != want.get(k)}
        if diffs:
            near.append((d, diffs))
        else:
            matches.append(d)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        lines = [f'roi_cache: no cache under {out_root} matches this run. '
                 f'Wanted (key fields):']
        lines += [f'    {k} = {want[k]!r}' for k in COMPARE_KEYS]
        if near:
            lines.append('  Closest existing cache(es):')
            for d, diffs in near[:3]:
                lines.append(f'    {d.name}')
                for k, (h, w) in diffs.items():
                    lines.append(f'        {k}: cache={h!r} run={w!r}')
        else:
            lines.append('  (the directory holds no cache at all)')
        raise ValueError('\n'.join(lines))
    raise ValueError(
        f'roi_cache: {len(matches)} caches under {out_root} match this run '
        f'({", ".join(m.name for m in matches)}); refusing to guess.')


def shard_metadata(root: Path, sessions: Sequence[dict]) -> Dict[str, Dict]:
    """Per-session ``{box, n_frames, shape}`` read from the shards.

    Touches only the SMALL members: an ``npz`` member is decompressed on access,
    so pulling ``box``/``n_frames``/``shape`` costs nothing like ``frames``.
    """
    out: Dict[str, Dict] = {}
    for s in sessions:
        name = str(s['session'])
        p = shard_path(root, name)
        if not p.is_file():
            continue
        with np.load(str(p), allow_pickle=False) as z:
            out[name] = {'box': tuple(int(v) for v in z['box']),
                         'n_frames': int(z['n_frames']),
                         'shape': tuple(int(v) for v in z['frames'].shape)}
    return out


def check_params_compatible(root: Path, params: Dict) -> None:
    """FAIL LOUDLY if the on-disk cache was built with different parameters.

    Spec section 9.2: silently falling back to a slightly wrong pixel set is the
    one failure mode that would invalidate every downstream comparison while
    looking healthy. So this raises instead of warning.
    """
    have = read_params(root)
    if have == params:
        return
    diffs = [f'{k}: cache={have.get(k)!r} run={params.get(k)!r}'
             for k in sorted(set(have) | set(params))
             if have.get(k) != params.get(k)]
    raise ValueError(
        f'roi_cache: {root} was built with DIFFERENT parameters:\n  '
        + '\n  '.join(diffs)
        + f'\nRefusing to mix shards. Use a different --out_root or delete it.')


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def valid_rows(landmarks: np.ndarray, target_labels: Sequence[int]) -> np.ndarray:
    """Target-landmark rows of the frames the DATASET would keep for a clip.

    ``target_labels`` are **1-INDEXED landmark labels**, exactly as
    ``tir_resp_dataset.resolve_roi_landmarks`` returns them; the conversion to
    numpy indices happens in :func:`landmark_indices`. Passing labels directly
    to numpy would be off by one and silently select a DIFFERENT set of points.

    The keep predicate IS ``~tir_resp_dataset.missing_frame_mask`` -- the
    DATASET'S OWN function, CALLED rather than re-implemented, so "the frames
    the cache unions over" and "the frames the dataset keeps" cannot drift
    apart. That is what makes excluding them from the union box SAFE: a clip
    overlapping any such frame is dropped entirely, so every KEPT clip's
    landmarks survive here.

    NOT equivalent to what this function used to do. The hand-rolled version
    excluded a frame only when ALL target landmarks were ``(0,0)``
    (``.all()``), while ``missing_frame_mask`` marks a frame bad when ANY target
    landmark is (``.any()``). The corpus never exposed it -- every one of its
    5760 sentinel frames (1385 tasks) is a WHOLE-ROW sentinel, 0 partial -- but
    ONE partial frame would have stretched the union box toward the image
    corner (measured: box area x6.4), which is exactly the failure
    :func:`tir_resp_dataset.roi_box_from_landmarks` documents as prevented
    "upstream rather than here".
    """
    pts = np.asarray(landmarks)
    idx = np.asarray(landmark_indices(target_labels), dtype=np.int64)
    keep = ~trd.missing_frame_mask(pts, idx)
    return pts[keep][:, idx, :]


def hull_box(landmarks: np.ndarray, target_labels: Sequence[int],
             padding: float, w: int, h: int,
             crop_rule: str = trd.DEFAULT_CROP_RULE,
             clip_seconds: float = trd.DEFAULT_CLIP_SECONDS,
             fps: float = trd.DEFAULT_FPS) -> Optional[Box]:
    """The STORAGE box: bounding box over every valid window's CLIP box.

    ``target_labels`` are 1-INDEXED labels (see :func:`valid_rows`).

    The stored crop must contain the box of EVERY clip the loader can emit, for
    any hop and any ``task_set``. Because each clip box is built from a
    ``clip_seconds``-long window (and, with ``crop_rule='square'``, is centred on
    that window's own padded envelope), the union over the WINDOWS at stride 1 is
    the exact answer: it is the union of the very boxes the loader will ask for.
    Merging the whole track instead (one padded box over every valid frame) is
    both larger and not guaranteed to contain a window box near a session edge.

    Windows overlapping a sentinel/missing frame are SKIPPED, exactly as the
    dataset would drop them -- the keep predicate is the dataset's own
    :func:`tir_resp_dataset.missing_frame_mask`, not a copy of it. Windows whose
    box cannot be placed (``RoiBoxOutOfFrame``) are skipped for the same reason:
    the loader can never ask for them.

    :returns: ``(x0, x1, y0, y1)`` in SOURCE pixels, or ``None`` when the task has
        no usable landmark row at all. A task shorter than one clip falls back to
        its whole valid track (it emits no clip, so anything covering the track is
        a superset of the empty request set).
    """
    pts = np.asarray(landmarks)
    idx = np.asarray(landmark_indices(target_labels), dtype=np.int64)
    tgt = pts[:, idx, :]
    bad = trd.missing_frame_mask(pts, idx)
    n = int(tgt.shape[0])
    cf = max(1, int(round(float(clip_seconds) * float(fps))))

    boxes: List[Box] = []
    for start in range(0, max(0, n - cf) + 1):
        if bad[start:start + cf].any():
            continue
        try:
            boxes.append(trd.roi_box_from_landmarks(
                tgt[start:start + cf], int(w), int(h), float(padding),
                crop_rule=crop_rule))
        except trd.RoiBoxOutOfFrame:
            continue
    if not boxes:
        rows = valid_rows(landmarks, target_labels)
        if rows.size == 0:
            return None
        try:
            return trd.roi_box_from_landmarks(rows, int(w), int(h),
                                              float(padding),
                                              crop_rule=crop_rule)
        except trd.RoiBoxOutOfFrame:
            return None
    return (min(b[0] for b in boxes), max(b[1] for b in boxes),
            min(b[2] for b in boxes), max(b[3] for b in boxes))


def box_contains(outer: Box, inner: Box) -> bool:
    ox0, ox1, oy0, oy1 = outer
    ix0, ix1, iy0, iy1 = inner
    return bool(ox0 <= ix0 and ix1 <= ox1 and oy0 <= iy0 and iy1 <= oy1)


def shift_box(box: Box, origin_xy: Tuple[int, int]) -> Box:
    """Translate a SOURCE box into the union crop's coordinate frame."""
    ox, oy = int(origin_xy[0]), int(origin_xy[1])
    x0, x1, y0, y1 = box
    return (int(x0) - ox, int(x1) - ox, int(y0) - oy, int(y1) - oy)


# --------------------------------------------------------------------------- #
# build one task
# --------------------------------------------------------------------------- #
def build_task(spec: dict, target_labels: Sequence[int], padding: float,
               with_landmarks: bool = True, max_task_bytes: int = 0,
               crop_rule: str = trd.DEFAULT_CROP_RULE,
               clip_seconds: float = trd.DEFAULT_CLIP_SECONDS,
               fps: float = trd.DEFAULT_FPS
               ) -> Tuple[Optional[Dict], Optional[str]]:
    """Decode ONE subject-task and return its shard payload.

    Returns ``(payload, None)`` on success, or ``(None, reason)`` when the task
    is GEOMETRY-FATAL -- i.e. when there is no pixel to crop.

    A LABEL-side problem (no respiration file, a flat or railed signal, too few
    usable windows) must NOT reach here and must NOT cause a skip: those sessions
    have perfectly good pixels, and their filters are deliberately outside the
    cache key. Skipping them would make the cache depend on a knob it is supposed
    to be independent of.

    The decode is CHUNKED and seek-based, not ``read_all``: the whole task never
    exists in RAM. That keeps the peak at one chunk plus the output crop
    regardless of task length, and it uses the same access pattern as the live
    dataset's per-clip ``read_range`` -- which the parity probe verified is
    byte-identical to a sequential decode (8000/8000 frames, section 10.1).
    """
    session = str(spec['session'])
    ir_file = spec.get('ir_file')
    video = spec.get('video')

    # --- 1. landmark track (cheap; this is the up-front part of the gate)
    if not ir_file or not os.path.isfile(ir_file):
        return None, 'missing_ir_features'
    try:
        ir = trd.parse_ir_features(ir_file)
    except Exception as exc:
        return None, f'invalid_ir_features: {exc}'
    if valid_rows(ir, target_labels).size == 0:
        return None, 'no_valid_landmark_rows'
    if not video or not os.path.isfile(video):
        return None, 'missing_video'

    # --- 2. one-frame probe: frame size, length, and the storage box ------
    # The box is a pure function of the landmark track and the frame size, so it
    # can be computed BEFORE any pixel is read and the output crop allocated
    # once. That is what lets the decode below be chunked.
    try:
        probe = vio.open_video(video)
        try:
            n_vid = int(probe.num_frames or 0)
            f0 = probe.read_range(0, 1)[0]
        finally:
            probe.close()
    except Exception as exc:
        return None, f'undecodable_video: {exc}'
    fh, fw = int(f0.shape[0]), int(f0.shape[1])

    # Truncate to the frames the dataset can use: min(video, IRFeatures).
    # NOT CAP_PROP_FRAME_COUNT alone -- the container over-reported by 12 frames
    # on M005_T3 (1905 claimed against 1893 decodable, matching its 1893 rows).
    n_frames = int(min(ir.shape[0], n_vid if n_vid > 0 else ir.shape[0]))
    if n_frames <= 0:
        return None, 'no_common_frames'
    ir = ir[:n_frames]
    box = hull_box(ir, target_labels, padding, fw, fh,
                   crop_rule=crop_rule, clip_seconds=clip_seconds, fps=fps)
    if box is None:
        return None, 'no_valid_landmark_rows'
    x0, x1, y0, y1 = box
    if x0 < 0 or y0 < 0 or x1 > fw or y1 > fh:
        return None, f'box {box} outside the decoded frame {fw}x{fh}'

    crop_bytes = n_frames * (y1 - y0) * (x1 - x0) * 3
    if max_task_bytes and crop_bytes > max_task_bytes:
        return None, (f'too_large_for_memory: the crop alone would be '
                      f'{crop_bytes / 1e9:.1f} GB > '
                      f'{max_task_bytes / 1e9:.1f} GB budget '
                      f'({n_frames} frames at {x1 - x0}x{y1 - y0})')

    # --- 3. decode in BOUNDED CHUNKS -------------------------------------
    # The chunks use read_range(start, k) -- one seek per chunk. That is the
    # same access pattern the LIVE dataset uses per clip, and section 10.1
    # measured seek == sequential byte-for-byte (8000/8000 frames), so the
    # pixels are unchanged; only the memory profile is.
    crop = np.empty((n_frames, y1 - y0, x1 - x0, 3), np.uint8)
    n_got = 0
    try:
        reader = vio.open_video(video)
        try:
            for start in range(0, n_frames, CHUNK_FRAMES):
                k = min(CHUNK_FRAMES, n_frames - start)
                chunk = np.asarray(reader.read_range(start, k))
                if chunk.ndim == 2:                        # gray fallback
                    chunk = np.repeat(chunk[..., None], 3, axis=2)
                if chunk.ndim != 4:
                    return None, f'malformed_frames: shape {chunk.shape}'
                if chunk.shape[3] == 1:
                    chunk = np.repeat(chunk, 3, axis=3)
                crop[start:start + k] = chunk[:, y0:y1, x0:x1]
                n_got = start + k
                del chunk
        finally:
            reader.close()
    except Exception as exc:
        # A container that over-reports its frame count makes read_range RAISE
        # on the short tail instead of returning fewer frames (section 10.2).
        # read_all() stops at EOF without demanding an exact count, so it
        # recovers those frames -- at the cost of the memory the chunking exists
        # to bound, which is why --max_task_gb still guards this path.
        try:
            reader = vio.open_video(video)
            try:
                frames = np.asarray(reader.read_all())
            finally:
                reader.close()
            if frames.ndim == 2:
                frames = np.repeat(frames[..., None], 3, axis=2)
            if frames.ndim != 4:
                return None, f'malformed_frames: shape {frames.shape}'
            if frames.shape[3] == 1:
                frames = np.repeat(frames, 3, axis=3)
            n_got = int(min(frames.shape[0], n_frames))
            crop = np.ascontiguousarray(frames[:n_got, y0:y1, x0:x1])
        except Exception as exc2:
            return None, f'undecodable_video: {exc!r}; fallback {exc2!r}'
    if n_got <= 0:
        return None, 'no_decodable_frames'

    payload = {
        'frames': np.ascontiguousarray(crop[:n_got]),
        'box': np.asarray(box, dtype=np.int32),
        'session': session,
        'video': str(video),
        'n_frames': int(n_got),
        'src_hw': (fh, fw),
    }
    if with_landmarks:
        payload['landmarks'] = np.asarray(ir[:n_got], dtype=np.float32)
    return payload, None


# --------------------------------------------------------------------------- #
# shard IO
# --------------------------------------------------------------------------- #
def write_shard(root: Path, session: str, payload: Dict,
                compress: bool = False) -> Path:
    """Write one shard atomically (staging dir + ``os.replace``).

    The staging name carries the PID, so two processes can never share it: with
    a fixed ``<session>.npz`` staging path, concurrent writers of the SAME task
    would truncate each other's partial file and then one ``os.replace`` would
    fail (or publish the other's bytes).
    """
    tasks_dir = Path(root) / SUBDIR_TASKS
    tmp_dir = Path(root) / SUBDIR_TMP
    tasks_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    final = tasks_dir / f'{session}{SHARD_SUFFIX}'
    tmp = tmp_dir / f'{session}.{os.getpid()}{SHARD_SUFFIX}'
    saver = np.savez_compressed if compress else np.savez
    with open(tmp, 'wb') as fh:
        saver(fh, **payload)
    os.replace(tmp, final)
    return final


def write_skip(root: Path, session: str, reason: str) -> Path:
    """Record WHY a session was not cached.

    Without the reason a resumed run cannot tell "geometry-fatal" from "not
    written yet", and would either redo work forever or leave a silent hole.

    Creates ``.tmp`` itself, like :func:`write_shard`: it must NOT depend on an
    earlier successful shard having created the staging directory. The first
    task of the first array element can legitimately be a SKIP (e.g.
    ``missing_ir_features``, 15 of the real corpus), and the old version raised
    ``FileNotFoundError`` in exactly that case -- it only worked when another
    task had already written a shard into the same root.
    """
    d = Path(root) / SUBDIR_SKIPPED
    tmp_dir = Path(root) / SUBDIR_TMP
    d.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = d / f'{session}.json'
    tmp = tmp_dir / f'{session}.{os.getpid()}.json'
    with open(tmp, 'w') as fh:
        json.dump({'session': session, 'reason': reason}, fh, indent=2)
    os.replace(tmp, path)
    return path


def read_shard(path) -> Dict:
    """Load one shard. Arrays are materialised before the archive closes."""
    with np.load(str(path), allow_pickle=False) as z:
        out = {
            'frames': z['frames'],
            'box': tuple(int(v) for v in z['box']),
            'session': str(z['session']),
            'video': str(z['video']),
            'n_frames': int(z['n_frames']),
            'landmarks': z['landmarks'] if 'landmarks' in z.files else None,
        }
    return out


def shard_path(root: Path, session: str) -> Path:
    return Path(root) / SUBDIR_TASKS / f'{session}{SHARD_SUFFIX}'


def scan(root: Path, sessions: Sequence[dict]) -> Dict:
    """Filesystem truth: which sessions are cached, skipped, or still missing.

    Deliberately does NOT trust ``manifest.json`` -- the manifest is derived and
    may be stale while an array job is running, whereas the directory listing is
    always current.
    """
    root = Path(root)
    cached, skipped, missing, n_bytes = [], [], [], 0
    for s in sessions:
        name = str(s['session'])
        p = shard_path(root, name)
        if p.is_file():
            cached.append(name)
            n_bytes += p.stat().st_size
            continue
        sp = root / SUBDIR_SKIPPED / f'{name}.json'
        if sp.is_file():
            skipped.append(name)
            continue
        missing.append(name)
    return {'cached': len(cached), 'skipped': len(skipped),
            'missing': len(missing), 'bytes': n_bytes,
            'cached_names': cached, 'skipped_names': skipped,
            'missing_names': missing}


def build_manifest(root: Path, sessions: Sequence[dict],
                   params: Dict) -> Dict:
    """Derived index. Cheap for cached shards, so it does not read pixels."""
    root = Path(root)
    entries: List[Dict] = []
    for s in sessions:
        name = str(s['session'])
        p = shard_path(root, name)
        if p.is_file():
            with np.load(str(p), allow_pickle=False) as z:
                entries.append({
                    'session': name, 'status': 'ok',
                    'bytes': p.stat().st_size,
                    'box': [int(v) for v in z['box']],
                    'n_frames': int(z['n_frames']),
                    'shape': [int(v) for v in z['frames'].shape],
                    'has_landmarks': 'landmarks' in z.files,
                })
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
