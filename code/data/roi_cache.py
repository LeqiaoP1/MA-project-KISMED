"""Offline TIR-ROI pixel cache: cache key, geometry gate, shard IO, verification.

Spec: ``analysis/tir_resp/hpc_precomputation.md`` section 9. The probe that
validates the design is ``analysis/tir_resp/cache_parity_probe.py``.

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
#: v1 -> v2 (2026-10-08): the union box was computed from 1-indexed landmark
#: LABELS used as numpy indices, selecting 9..13/20..26 instead of 8..12/19..25.
#: The key is a hash of PARAMETERS, not of the code, so a stale v1 cache would
#: otherwise collide with a correct v2 build and be silently reused. Bumping is
#: what makes the collision impossible; ``--check`` is what would have caught it.
CACHE_FORMAT_VERSION = 2

#: The box rule. ``roi_quantile`` was removed 2026-10-08, so min/max is the only
#: rule in production; it stays a named field because it is part of the key.
BOX_RULE = 'minmax'

#: Tracker ``(0,0)`` sentinel rows are EXCLUDED from the union box. Safe because
#: a clip overlapping any sentinel/missing target landmark is dropped entirely by
#: ``missing_frame_mask``, so no kept clip's frames are removed here. Materially
#: cheaper too: including them inflates the p90 task box from 174 px to 385 px.
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
                 src_w: int, src_h: int) -> Dict:
    """The FROZEN parameter set. Everything here changes stored pixels.

    ``target_landmarks`` are 1-INDEXED labels, i.e. exactly what
    ``resolve_roi_landmarks`` returns and what the configs carry.
    """
    return {
        'format_version': CACHE_FORMAT_VERSION,
        'raw_root': str(raw_root),
        'corpus': corpus_digest(sessions),
        'n_sessions_discovered': len(sessions),
        'roi_landmarks_name': landmark_set_name(target_landmarks),
        'roi_landmarks': [int(i) for i in target_landmarks],
        'roi_padding': float(roi_padding),
        'box_rule': BOX_RULE,
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
            f'_{params["box_rule"]}'
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
                'roi_padding', 'box_rule', 'valid_frame_rule', 'pixel_format',
                'store_resolution')


def loader_params(raw_root: str, corpus_sessions: Sequence[dict],
                  target_landmarks: Sequence[int], roi_padding: float) -> Dict:
    """The subset of the frozen parameters a loader can compute on its own.

    ``corpus_sessions`` MUST be the UNFILTERED discovery: the cache key is a
    property of the CORPUS, not of which subset a given run trains on. That is
    what lets ``--subjects``/``--limit`` restrict a BUILD without creating a
    second cache identity.
    """
    full = build_params(raw_root, corpus_sessions, target_landmarks,
                        roi_padding, 0, 0)
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
    """Target-landmark rows with non-finite and ``(0,0)`` sentinel rows removed.

    ``target_labels`` are **1-INDEXED landmark labels**, exactly as
    ``tir_resp_dataset.resolve_roi_landmarks`` returns them; the conversion to
    numpy indices happens here. Passing labels directly to numpy would be off by
    one and silently select a DIFFERENT set of points.

    Mirrors ``missing_frame_mask``'s notion of unusable, which is what makes
    excluding these rows from the union box SAFE: a clip is dropped entirely if
    it overlaps any such frame, so every KEPT clip's landmarks survive here.
    """
    pts = np.asarray(landmarks)[:, list(landmark_indices(target_labels)), :]
    finite = np.isfinite(pts).all(axis=2)
    sentinel = (np.abs(pts) < 1e-9).all(axis=2)
    keep = finite.all(axis=1) & ~sentinel.all(axis=1)
    return pts[keep]


def union_box(landmarks: np.ndarray, target_labels: Sequence[int],
              padding: float, w: int, h: int) -> Optional[Box]:
    """The STORAGE box: min/max over a task's valid rows, + padding, clamped.

    ``target_labels`` are 1-INDEXED labels (see :func:`valid_rows`).

    This is the smallest fixed rectangle that provably contains every clip box
    any future config could ask for (any hop, any ``task_set``, any cleaning
    setting), which is what makes the cache hop-independent. It never reaches the
    encoder: the loader recomputes the per-clip box and sub-crops from here.
    """
    rows = valid_rows(landmarks, target_labels)
    if rows.size == 0:
        return None
    return trd.roi_box_from_landmarks(rows, int(w), int(h), float(padding))


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
               with_landmarks: bool = True, max_task_bytes: int = 0
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
    box = union_box(ir, target_labels, padding, fw, fh)
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
    """
    d = Path(root) / SUBDIR_SKIPPED
    d.mkdir(parents=True, exist_ok=True)
    path = d / f'{session}.json'
    tmp = Path(root) / SUBDIR_TMP / f'{session}.{os.getpid()}.json'
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
