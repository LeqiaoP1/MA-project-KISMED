"""Inspect the RAW BP4D ``Thermal/`` videos (format info + first decoded frames).

Reads the single thermal video of a session straight out of the raw tree::

    <raw_root>/Thermal/<subject>/<task>.wmv

(note: unlike ``Physiology/``, which nests one directory per task, ``Thermal/``
is a FLAT per-subject directory of ``<task>.wmv`` files.)

and reports, per session:

* **file** -- size and md5. The md5 is not decoration: ImplementationPlan.md
  S1.1 claims the raw file and the canonical ``tir.wmv`` are byte-identical, and
  the quoted hash (``ae2d0247c921662a4b322e62f040143f`` for F001/T1, verified
  2026-09-16) can be reproduced here.
* **container / stream** -- container format, codec, profile, pixel format,
  resolution, frame rate, time base, bit rate and the container's frame count,
  from PyAV *and* from OpenCV, side by side: the two decoders do not always
  agree and ``.wmv`` containers carry a notoriously unreliable frame count
  (PyAV reports ``nb_frames=0`` for this corpus while OpenCV's
  ``CAP_PROP_FRAME_COUNT`` is right).
* **decode** -- a full pass reporting how many frames really decoded, how many
  failed, the duration and the MEASURED frame rate from presentation
  timestamps. The measured rate, not the nominal one, is what the alignment
  layer depends on (BP4D thermal measures exactly 25.0000 fps over 64.44 s).
* **chroma** -- the check that the stream really is the documented FALSE-COLOUR
  (rainbow) thermal RENDERING and not a gray image. THE NUMBER DEPENDS ON HOW
  YOU MEASURE IT and this was measured, not assumed: the DECODED yuv444p planes
  give ``|U-128| 25.1 / |V-128| 17.2`` (inside the documented 23-27 / 16-20),
  while an RGB -> cv2.COLOR_RGB2YUV round trip gives ``23.9 / 22.3`` -- U
  agrees but **V is inflated ~5 codes**, because the saturated rainbow palette
  is clipped by the YUV -> RGB -> YUV trip. The plane number is therefore
  PRIMARY (it is what the README quotes) and the RGB number is a cross-check.
  When the decoder cannot expose planes the RGB number is all there is, and it
  is NOT comparable to the documented range -- the report says so.
* **luma / auto-range** -- per-frame luma statistics plus a scan for global
  auto-range STEP changes (``ImplementationPlan.md`` S4.5 asks for this scan).
  The test is deliberately conservative: a step must be a sustained shift of
  the per-frame luma percentiles over a ~1 s window, in the same direction at
  both ends, and larger than both a 4-code floor and 10x the MAD of the frame
  to-frame increments -- a naive per-frame threshold flags ~5% of frames on
  ordinary content motion, which is useless.
* **geometry** -- where the ``resize_center_crop`` window lands in SOURCE
  pixels and whether the burned-in degC legend on the right edge survives it.
  The README claims it is cropped away; this verifies it per session.  The box
  is computed arithmetically and then CHECKED against
  ``data.video_io.resize_center_crop`` by array equality.  (An earlier version
  of this check ran the real function on a coordinate ramp
  ``np.arange(H*W).reshape(H,W)``; that is BROKEN here -- on a 726x480 float64
  ramp INTER_AREA returns values compressed toward the middle, 558.6/347920.4
  instead of 122/348355 -- so the ramp probe was replaced.  The arithmetic
  mirror itself is exact: array equality against the real function, maxdiff 0.)

Artifacts (same ``<subject>_<task>`` directory as the physiology inspector, so
one session keeps all of its artifacts together; no file name collides with the
physiology outputs)::

    <output_dir>/<subject>_<task>/
        TIR_frames.png     # the first N decoded frames (default 6)
        TIR_overview.png   # luma / chroma / column-std / 224 px crop / hist / palette
        TIR.json           # everything above, machine-readable
    <output_dir>/thermal_index.json   # one row per session inspected so far

Usage (from ``code/``)::

    # one session: format info + the first 6 frames
    python runners/run_inspect_thermal.py --subject F001 --task T1

    # every task of two subjects, 16 frames, JSON only
    python runners/run_inspect_thermal.py --subject F001,F002 --task all \
        --frames 16 --no-plot

    # what is on disk?
    python runners/run_inspect_thermal.py --list

DECODER NOTE. The pipeline reads TIR through ``video_io.open_video`` (OpenCV
first, then decord). In THIS environment decord is NOT installed
(2026-09-16: av 18.1.0 present, decord missing, cv2 5.0.0), so OpenCV is doing
the decoding and PyAV -- not decord -- is the usable fallback. This inspector
prefers PyAV because a single pass then yields the presentation timestamps and
the raw YUV planes as well as the frames; ``--decoder opencv`` reproduces what
training sees (minus the timestamps and the planes).

Paths default to ``$RAW_DATA_PATH`` (raw BP4D root) and ``$OUTPUT_DIR`` (see
``scripts/env_local.sh``); without the env profile they fall back to the
in-repo ``data/raw/BP4D`` and ``<repo>/output``. ``--output_dir`` is the
``inspect_data`` ROOT -- the ``<subject>_<task>`` subdirectory is appended.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data import video_io as vio

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_DIR = os.path.dirname(_CODE_DIR)

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
RAW_TREE = 'Thermal'
VIDEO_EXTS = ('.wmv', '.avi', '.mp4', '.mkv', '.mov')
FIG_NAME = 'TIR'                       # TIR_frames.png / TIR_overview.png / TIR.json

# Pixel stride for the per-frame statistics. A BP4D TIR frame is 726x480 =
# 348 k pixels (RGB is the 1392x1040 one -- do not confuse them), and
# np.percentile/np.unique over the full frame cost more than the decode itself.
# Stride 4 keeps ~22 k samples, ample for means, percentiles and colour counts.
# min/max are still taken over the FULL frame: they are O(N) and cheap, and
# they are what the auto-range scan keys on.
STAT_STRIDE = 4

# A run of columns whose TEMPORAL std is below this fraction of the frame-wide
# maximum is called "static": the heuristic that locates the burned-in degC
# legend / colour bar on the right edge (measured: x >= 684, 42 px of 726).
STATIC_COL_FRAC = 0.05
STATIC_MIN_WIDTH_FRAC = 0.005

# Auto-range step detector: a step must be a SUSTAINED shift of the luma
# percentiles over this window (seconds) and exceed both a floor in luma codes
# and a multiple of the MAD of the frame-to-frame increments.
AUTORANGE_WINDOW_S = 1.0
AUTORANGE_MIN_CODES = 4.0
AUTORANGE_K_MAD = 10.0

CROP_CHECK_TOL = 1.5                   # px, arithmetic vs real resize_center_crop


# --------------------------------------------------------------------------- #
# path defaults
# --------------------------------------------------------------------------- #
def _env(key: str, default: str = '') -> str:
    return os.environ.get(key, default)


def default_raw_root() -> str:
    """``$RAW_DATA_PATH`` when set, else the in-repo raw BP4D root."""
    return _env('RAW_DATA_PATH', '') or os.path.join(_REPO_DIR, 'data', 'raw', 'BP4D')


def default_output_dir() -> str:
    """``$OUTPUT_DIR/inspect_data``, else ``<repo>/output/inspect_data``.

    Absolute on purpose (same reasoning as the physiology inspector): a bare
    ``./output/...`` would land inside ``code/`` when the env profile was not
    sourced.
    """
    root = _env('OUTPUT_DIR', '') or os.path.join(_REPO_DIR, 'output')
    return os.path.join(root, 'inspect_data')


def _fmt_bytes(n: float) -> str:
    n = float(n)
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or unit == 'GB':
            return f'{n:.0f} {unit}' if unit == 'B' else f'{n:.1f} {unit}'
        n /= 1024.0
    return f'{n:.1f} GB'


def _opt(v, fmt: str = '%s') -> str:
    """Format a possibly-``None`` value without crashing the printout."""
    return 'n/a' if v is None else (fmt % v)


# --------------------------------------------------------------------------- #
# raw data access
# --------------------------------------------------------------------------- #
def find_video(raw_root: str, subject: str, task: str) -> str:
    """Path of a session's thermal video, or ``''`` when there is none."""
    base = os.path.join(raw_root, RAW_TREE, subject, task)
    for ext in VIDEO_EXTS:
        if os.path.isfile(base + ext):
            return base + ext
    hits = [p for p in sorted(glob.glob(base + '.*'))
            if os.path.splitext(p)[1].lower() in VIDEO_EXTS]
    return hits[0] if hits else ''


def discover_sessions(raw_root: str):
    """Sorted ``[(subject, task), ...]`` with a video under ``Thermal/``.

    Scans FILES (the layout is flat) and filters by extension, which also skips
    the ``Thumbs.db`` that Windows leaves in these directories.
    """
    root = os.path.join(raw_root, RAW_TREE)
    out = []
    if not os.path.isdir(root):
        return out
    for subj in sorted(os.listdir(root)):
        sdir = os.path.join(root, subj)
        if not os.path.isdir(sdir):
            continue
        for name in sorted(os.listdir(sdir)):
            if os.path.splitext(name)[1].lower() in VIDEO_EXTS:
                out.append((subj, os.path.splitext(name)[0]))
    return out


def file_info(path: str) -> dict:
    """``{'bytes', 'md5'}`` -- the md5 makes raw-vs-canonical comparable."""
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return {'bytes': os.path.getsize(path), 'md5': h.hexdigest()}


# --------------------------------------------------------------------------- #
# format probes (no decoding)
# --------------------------------------------------------------------------- #
def _first(obj, *names, default=None):
    """First non-``None`` attribute of ``obj`` among ``names``."""
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return default


def _as_float(v):
    """Best-effort float (``Fraction``/``Decimal``/``str`` -> float, else None)."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def probe_pyav(path: str):
    """Decoder-level container/stream info from PyAV, or ``None``.

    PyAV is the only probe that exposes the codec profile, the true pixel
    format and the presentation timestamps. Every value is converted to a
    JSON-safe scalar here -- the native types are ``Fraction``/``VideoFormat``
    objects that ``json`` cannot serialise.
    """
    try:
        import av
    except Exception:
        return None
    info = {}
    try:
        with av.open(path) as container:
            fmt = container.format
            info['format_name'] = _first(fmt, 'name')
            info['format_long_name'] = _first(fmt, 'long_name')
            info['duration_s'] = (_as_float(container.duration) / 1e6
                                  if container.duration else None)
            info['bit_rate'] = _as_float(_first(container, 'bit_rate'))
            info['nb_streams'] = len(container.streams)
            if not container.streams.video:
                return info
            stream = container.streams.video[0]
            cc = stream.codec_context
            pix_fmt = _first(cc, 'pix_fmt')
            if pix_fmt is None:                 # PyAV >= 10: cc.format.name
                pix_fmt = _first(getattr(cc, 'format', None), 'name')
            tb = _first(stream, 'time_base')
            dur = _as_float(_first(stream, 'duration'))
            info.update({
                'codec': _first(cc, 'name'),
                'codec_long_name': _first(getattr(cc, 'codec', None), 'long_name'),
                'pix_fmt': pix_fmt,
                'profile': _first(cc, 'profile'),
                'level': _first(cc, 'level'),
                'width': _first(cc, 'width'),
                'height': _first(cc, 'height'),
                'coded_width': _first(cc, 'coded_width'),
                'coded_height': _first(cc, 'coded_height'),
                'sample_aspect_ratio': str(_first(stream, 'sample_aspect_ratio')),
                'display_aspect_ratio': str(_first(stream, 'display_aspect_ratio')),
                'avg_frame_rate': _as_float(_first(stream, 'average_rate')),
                'base_rate': _as_float(_first(stream, 'base_rate')),
                'guessed_rate': _as_float(_first(stream, 'guessed_rate')),
                'time_base': str(tb) if tb is not None else None,
                'stream_duration_s': dur * _as_float(tb) if (dur and tb) else None,
                'stream_bit_rate': _as_float(_first(cc, 'bit_rate')),
                'nb_frames': _first(stream, 'frames'),
                'has_b_frames': _first(cc, 'has_b_frames'),
            })
    except Exception as exc:
        info['error'] = f'{type(exc).__name__}: {exc}'
    return info


def probe_opencv(path: str):
    """What the decoder the PIPELINE uses reports about the same file.

    A second, independent opinion -- and the numbers the training loader
    actually sees (``CV2ClipReader`` reads exactly these properties).
    """
    try:
        import cv2
    except Exception:
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return None
    info = {}
    try:
        getter = getattr(cap, 'getBackendName', None)
        if callable(getter):
            try:
                info['backend'] = str(getter())
            except Exception:
                info['backend'] = None
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        info['fourcc'] = (''.join(chr((fourcc >> (8 * i)) & 0xFF)
                                  for i in range(4)) if fourcc else None)
        info['fps'] = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        info['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        prop = getattr(cv2, 'CAP_PROP_BITRATE', None)
        info['bitrate_kbps'] = (float(cap.get(prop)) if prop is not None else None)
    finally:
        cap.release()
    return info


def probe_ffprobe(path: str):
    """Raw ``ffprobe`` JSON when the binary is on PATH, else ``None``.

    Optional (``--ffprobe``): a third opinion, useful when PyAV and OpenCV
    disagree about the codec or the frame count.
    """
    exe = shutil.which('ffprobe')
    if not exe:
        return None
    cmd = [exe, '-v', 'error', '-print_format', 'json',
           '-show_format', '-show_streams', path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            return {'error': out.stderr.strip()[:400]}
        return json.loads(out.stdout)
    except Exception as exc:
        return {'error': f'{type(exc).__name__}: {exc}'}


# --------------------------------------------------------------------------- #
# streaming decode + statistics
# --------------------------------------------------------------------------- #
class _TirAccum:
    """Per-frame statistics and the few frames kept for the figures.

    Frames are NEVER all held in memory: one 726x480 RGB frame is ~1.0 MB, so a
    1612-frame clip would need ~1.7 GB. Only the first ``keep_first`` frames
    (the frame grid) plus running accumulators are retained, which is why this
    streams instead of using ``video_io``'s ``read_all()``.
    """

    def __init__(self, height: int, width: int, keep_first: int, max_frames: int):
        self.height = int(height)
        self.width = int(width)
        self.keep_first = max(0, int(keep_first))
        self.max_frames = int(max_frames or 0)
        self.kept = []
        self.kept_times = []
        self.planes = []
        self.decoder = None
        self.n_decoded = 0
        self.n_failed = 0
        self.times = []
        self.mean = []
        self.min = []
        self.max = []
        self.p1 = []
        self.p99 = []
        self.abs_u = []
        self.abs_v = []
        self.colors = []
        self.col_sum = np.zeros(max(0, self.width), dtype=np.float64)
        self.col_sq = np.zeros(max(0, self.width), dtype=np.float64)
        self.col_n = 0

    # -- accumulation ------------------------------------------------------- #
    def add_failure(self):
        self.n_failed += 1

    def full(self) -> bool:
        return bool(self.max_frames) and (self.n_decoded + self.n_failed) >= self.max_frames

    def add(self, t, rgb):
        """Absorb one decoded RGB frame (uint8 ``[H, W, 3]``)."""
        import cv2
        if rgb is None or getattr(rgb, 'ndim', 0) != 3:
            self.add_failure()
            return
        yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
        y = yuv[:, :, 0]
        rs = rgb[::STAT_STRIDE, ::STAT_STRIDE]
        ys = y[::STAT_STRIDE, ::STAT_STRIDE]
        us = yuv[::STAT_STRIDE, ::STAT_STRIDE, 1].astype(np.int16)
        vs = yuv[::STAT_STRIDE, ::STAT_STRIDE, 2].astype(np.int16)

        self.times.append(float(t))
        self.mean.append(float(ys.mean()))
        self.min.append(float(y.min()))            # full frame: O(N) and cheap
        self.max.append(float(y.max()))
        self.p1.append(float(np.percentile(ys, 1.0)))
        self.p99.append(float(np.percentile(ys, 99.0)))
        self.abs_u.append(float(np.abs(us - 128).mean()))
        self.abs_v.append(float(np.abs(vs - 128).mean()))
        packed = ((rs[:, :, 0].astype(np.int32) << 16)
                  | (rs[:, :, 1].astype(np.int32) << 8)
                  | rs[:, :, 2].astype(np.int32))
        self.colors.append(int(np.unique(packed).size))

        yc = y[::STAT_STRIDE, :].astype(np.float64)     # full columns
        if yc.shape[0]:
            self.col_sum += yc.sum(axis=0)
            self.col_sq += (yc ** 2).sum(axis=0)
            self.col_n += yc.shape[0]

        if len(self.kept) < self.keep_first:
            self.kept.append(np.ascontiguousarray(rgb))
            self.kept_times.append(float(t))
        self.n_decoded += 1

    # -- derived ------------------------------------------------------------ #
    def col_std(self):
        """Temporal std of luma per COLUMN (the static-band / legend detector)."""
        if self.col_n <= 0 or self.col_sum.size == 0:
            return np.zeros(0, dtype=np.float64)
        m = self.col_sum / float(self.col_n)
        var = np.maximum(self.col_sq / float(self.col_n) - m * m, 0.0)
        return np.sqrt(var)


def _plane_chroma(frame):
    """``(mean|U-128|, mean|V-128|)`` from the DECODED yuv444p planes.

    THE comparable number: this is what the README quotes (23-27 / 16-20 on
    this corpus) and what the RGB round trip cannot reproduce (it inflates V by
    ~5 codes through gamut clipping of the saturated palette).
    """
    try:
        arr = frame.to_ndarray(format='yuv444p')
    except Exception:
        return None
    if getattr(arr, 'ndim', 0) != 3 or arr.shape[0] < 3:
        return None
    u = arr[1].astype(np.int16) - 128
    v = arr[2].astype(np.int16) - 128
    return float(np.abs(u).mean()), float(np.abs(v).mean())


def _nominal_fps(stream, fallback: float = 25.0) -> float:
    f = _as_float(_first(stream, 'average_rate', 'base_rate', 'guessed_rate'))
    return f if f and f > 0 else float(fallback)


def _stream_pyav(path: str, acc: _TirAccum, plane_samples: int = 4):
    """Decode with PyAV: frames + presentation timestamps + raw planes, one pass."""
    import av
    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = _nominal_fps(stream)
        for frame in container.decode(stream):
            if acc.full():
                break
            t = _as_float(getattr(frame, 'time', None))
            if t is None:
                t = acc.n_decoded / fps
            try:
                rgb = frame.to_ndarray(format='rgb24')
            except Exception:
                acc.add_failure()
                continue
            acc.add(t, rgb)
            if len(acc.planes) < plane_samples:
                p = _plane_chroma(frame)
                if p is not None:
                    acc.planes.append(p)


def _stream_opencv(path: str, acc: _TirAccum, fps_hint: float = 25.0):
    """Decode with OpenCV -- the decoder the training loader uses.

    OpenCV exposes no presentation timestamps, so ``t`` is synthesised from the
    container fps; that is recorded in ``decode.times_source``.
    """
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise IOError(f'OpenCV could not open {path}')
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or float(fps_hint)
    try:
        while not acc.full():
            ok, frame = cap.read()
            if not ok:
                break
            acc.add(acc.n_decoded / fps, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()


def _stream_decord(path: str, acc: _TirAccum):
    """decord (ffmpeg) fallback, matching ``video_io.open_video``'s chain."""
    import decord
    vr = decord.VideoReader(path)
    fps = float(vr.get_avg_fps() or 0.0) or 25.0
    for i in range(len(vr)):
        if acc.full():
            break
        try:
            rgb = vr[i].asnumpy()
        except Exception:
            acc.add_failure()
            continue
        acc.add(i / fps, rgb)


def stream_video(path: str, decoder: str, keep_first: int, max_frames: int,
                 height: int = 0, width: int = 0):
    """Decode the whole clip once; returns ``(accumulator, errors)``.

    ``decoder`` is ``auto`` (PyAV -> OpenCV -> decord), or a forced name. A
    decoder that fails AFTER producing frames is not retried -- the partial
    result is reported instead, because a partial decode is itself a finding.
    """
    import cv2  # noqa: F401 -- the frame statistics need it; fail early if absent
    acc = _TirAccum(height, width, keep_first, max_frames)
    order = {'auto': ('pyav', 'opencv', 'decord'),
             'pyav': ('pyav',), 'opencv': ('opencv',), 'decord': ('decord',)}[decoder]
    errors = []
    for name in order:
        try:
            if name == 'pyav':
                _stream_pyav(path, acc)
            elif name == 'opencv':
                _stream_opencv(path, acc)
            else:
                _stream_decord(path, acc)
        except Exception as exc:
            errors.append(f'{name}: {type(exc).__name__}: {exc}')
            if acc.n_decoded == 0:
                continue                    # nothing absorbed -> try the next one
            acc.decoder = name              # partial result: keep and report it
            return acc, errors
        acc.decoder = name
        return acc, errors
    raise IOError(f'no usable decoder for {path} [{"; ".join(errors)}]')


# --------------------------------------------------------------------------- #
# geometry + quality checks
# --------------------------------------------------------------------------- #
def crop_source_box(height: int, width: int, size: int):
    """Source-pixel ``(x0, x1, y0, y1)`` of the ``vio.resize_center_crop`` window.

    Mirrors ``data.video_io.resize_center_crop`` arithmetically (shorter side
    scaled to ``size``, then a centred crop). For 726x480 at size 224 this is
    ``nh == size``, so ``top == 0``: the crop keeps the FULL height and only
    trims the width. VERIFIED pixel-exact against the real function by
    ``crop_check`` -- do not replace that check with a coordinate ramp.
    """
    if min(height, width) <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    scale = size / float(min(height, width))
    nh, nw = height, width
    if scale < 1.0:
        nh, nw = int(height * scale), int(width * scale)
    top, left = (nh - size) // 2, (nw - size) // 2
    return (left / scale, (left + size) / scale, top / scale, (top + size) / scale)


def crop_check(frame, size: int):
    """``(equal, maxdiff, box)``: does the arithmetic mirror match the real fn?

    Rebuilds the crop with the same integer arithmetic as ``crop_source_box``
    and compares it to ``vio.resize_center_crop`` by ARRAY EQUALITY. This is
    the check that keeps the reported box honest, and it is a real test: an
    earlier coordinate-ramp probe silently returned a box with ``x0 > x1``.
    """
    import cv2
    h, w = frame.shape[:2]
    scale = size / float(min(h, w))
    nh, nw = h, w
    if scale < 1.0:
        nh, nw = int(h * scale), int(w * scale)
    top, left = (nh - size) // 2, (nw - size) // 2
    tmp = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    mirror = tmp[top:top + size, left:left + size]
    real = vio.resize_center_crop(frame, size)
    if real.shape != mirror.shape:
        return False, None, crop_source_box(h, w, size)
    dmax = int(np.abs(real.astype(np.int32) - mirror.astype(np.int32)).max())
    return bool(np.array_equal(real, mirror)), dmax, crop_source_box(h, w, size)


def static_right_band(col_std, frac: float = STATIC_COL_FRAC,
                      min_width_frac: float = STATIC_MIN_WIDTH_FRAC):
    """``(x0, width)`` of the static right-hand band, or ``None``.

    Heuristic: columns whose TEMPORAL std is below ``frac`` x the frame-wide
    maximum, scanning inwards from the right edge. The burned-in degC legend /
    colour bar is a static graphic, so it shows up here -- but so would a
    static border or a letterbox bar, which is why the result is only turned
    into a claim about the legend together with the crop geometry.
    """
    col_std = np.asarray(col_std, dtype=np.float64)
    if col_std.size == 0:
        return None
    thr = float(frac) * float(np.max(col_std))
    below = col_std < thr
    if not bool(below[-1]):
        return None
    i = int(col_std.size) - 1
    while i >= 0 and bool(below[i]):
        i -= 1
    x0 = i + 1
    width = int(col_std.size) - x0
    if width < max(1, int(min_width_frac * col_std.size)):
        return None
    return x0, width


def autorange_steps(p1, p99, fps: float, window_s: float = AUTORANGE_WINDOW_S,
                    min_codes: float = AUTORANGE_MIN_CODES,
                    k_mad: float = AUTORANGE_K_MAD) -> dict:
    """Sustained shifts of the luma percentiles = auto-range step candidates.

    The camera's auto-range re-scales the palette, which shows up as a step in
    the global luma extremes rather than in the content
    (``ImplementationPlan.md`` S4.5 asks for exactly this scan).

    Deliberately conservative, because the naive version is useless: a
    per-frame threshold of ``max(2 codes, 8x median increment)`` fires on 79 of
    1612 frames (4.9%) of ordinary content motion. A step must therefore be
    (a) a shift between the medians of the ``window_s`` windows before and
    after the candidate, (b) larger than a ``min_codes`` floor, (c) larger than
    ``k_mad`` x the MAD of the frame-to-frame increments, and (d) the SAME
    direction at both percentiles (a palette re-scale moves both ends alike).
    Adjacent candidates within one window are suppressed.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p99 = np.asarray(p99, dtype=np.float64)
    n = p1.size
    w = max(2, int(round(float(window_s) * float(fps or 25.0))))
    out = {'window_s': float(window_s), 'window_frames': w,
           'threshold_codes': None, 'n_candidates_raw': 0,
           'step_frames': [], 'step_times': [], 'n_steps': 0,
           'method': ('sustained median shift over window_s at both luma '
                      'percentiles, same sign, > floor and > k*MAD of the '
                      'frame increments; index = first frame of the NEW level')}
    if n < 3 * w:
        return out
    d = np.abs(np.diff(p1))
    mad = float(np.median(np.abs(d - np.median(d))))
    thr = max(float(min_codes), float(k_mad) * mad)
    out['threshold_codes'] = thr
    cands = []
    for i in range(w, n - w):
        j1 = float(np.median(p1[i:i + w]) - np.median(p1[i - w:i]))
        j99 = float(np.median(p99[i:i + w]) - np.median(p99[i - w:i]))
        if j1 == 0.0 or j99 == 0.0 or (j1 > 0) != (j99 > 0):
            continue
        mag = max(abs(j1), abs(j99))
        if mag > thr:
            cands.append((i, mag))
    out['n_candidates_raw'] = len(cands)
    accepted = []
    for i, mag in sorted(cands, key=lambda c: -c[1]):
        if all(abs(i - j) > w for j, _ in accepted):
            accepted.append((i, mag))
    accepted.sort()
    out['step_frames'] = [int(i + 1) for i, _ in accepted]
    out['step_times'] = [None] * len(accepted)    # filled by the caller (needs t)
    out['n_steps'] = len(accepted)
    return out


def quality_warnings(rep: dict) -> list:
    """Self-diagnosis of everything the report above measured.

    Same philosophy as the physiology inspector: a number that turned out not
    to be trustworthy must say so, rather than being printed as if it were.
    """
    w = []
    dec = rep.get('decode') or {}
    ov = rep.get('opencv') or {}
    st = rep.get('stream') or {}
    ch = rep.get('chroma') or {}
    geo = rep.get('geometry') or {}

    if not dec.get('n_decoded'):
        w.append('no frames decoded')
        return w
    if dec.get('n_failed'):
        w.append(f"{dec['n_failed']} frame(s) failed to decode mid-stream")
    if dec.get('partial'):
        w.append('partial decode: a decoder raised before the end of the clip, '
                 'so every count above is a LOWER bound')
    fc = ov.get('frame_count') or 0
    if fc and abs(int(fc) - int(dec['n_decoded'])) > 1:
        w.append(f"OpenCV reports {fc} frames but {dec['n_decoded']} decoded "
                 f"(the container frame count is unreliable for wmv)")
    nbf = st.get('nb_frames')
    if not nbf:
        w.append('container nb_frames absent/0 (typical for wmv) -- the DECODED '
                 'count is the authoritative one')
    elif abs(int(nbf) - int(dec['n_decoded'])) > 1:
        w.append(f"container nb_frames {nbf} != {dec['n_decoded']} decoded")
    fm, fn = dec.get('fps_measured'), dec.get('fps_nominal')
    if fm and fn:
        dev = abs(fm / fn - 1.0)
        if dev > 0.02:
            w.append(f'measured {fm:.4f} fps vs nominal {fn:.4f} '
                     f'({dev * 100:.1f}% off) -- alignment must use the MEASURED rate')
    if (ch.get('mean_abs_u') is not None and ch.get('mean_abs_v') is not None
            and ch['mean_abs_u'] < 2.0 and ch['mean_abs_v'] < 2.0):
        w.append('chroma is nearly flat (|U-128|, |V-128| < 2): this stream is '
                 'effectively GRAY, so tir_channels 1 is the honest setting for it')
    if ch.get('planes_missing'):
        w.append('the decoder could not expose the raw chroma planes, so the '
                 'reported chroma is RGB-derived and NOT comparable to the '
                 'documented 23-27 / 16-20 range (it inflates |V-128| by ~5 codes)')
    n_steps = (rep.get('autorange') or {}).get('n_steps') or 0
    if n_steps:
        w.append(f'{n_steps} auto-range step candidate(s): the palette may '
                 f're-scale mid-clip, which makes thermal amplitudes '
                 f'incomparable across frames')
    if geo.get('legend_in_crop'):
        w.append(f"the static right band (x >= {geo.get('static_band_x0')}) falls "
                 f"INSIDE the {geo.get('crop_size')} px crop: the model would see "
                 f"the degC legend")
    if geo.get('check_equal') is False:
        w.append(f"crop-box arithmetic does NOT reproduce "
                 f"video_io.resize_center_crop (maxdiff {geo.get('check_dmax')}) "
                 f"-- the reported box is unreliable")
    if rep.get('session_has_physio') is False:
        w.append('no matching Physiology/<subject>/<task>/ directory: this session '
                 'cannot be aligned against 1-D signals')
    return w


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def _import_pyplot():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        print(f'[plot] matplotlib unavailable, skipping figures ({exc})')
        return None


def _grid_shape(n: int, max_cols: int = 5):
    """``(rows, cols)`` for an ``n``-panel grid, preferring balanced rows.

    A fixed ``cols = min(5, n)`` puts 6 panels on a 5+1 grid, which wastes most
    of the second row; preferring a 2-row (then 3-row) layout gives 6 -> 2x3
    and 10 -> 2x5.
    """
    for rows in (2, 3):
        if n % rows == 0 and n // rows <= max_cols:
            return rows, n // rows
    cols = min(max_cols, n)
    return int(math.ceil(n / float(cols))), cols


def plot_frames(frames, times, out_path: str, header: str, crop_box=None) -> str:
    """The first N decoded frames in a grid; returns the path ('' when skipped)."""
    plt = _import_pyplot()
    if plt is None or not frames:
        return ''
    n = len(frames)
    rows, cols = _grid_shape(n)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.6 * rows),
                             layout='constrained')
    axes = np.atleast_1d(axes).ravel()
    for k, ax in enumerate(axes):
        if k >= n:
            ax.axis('off')
            continue
        ax.imshow(frames[k])                      # false-colour render: no colorbar
        ax.set_title(f'#{k}  t={times[k]:.3f} s', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if k == 0 and crop_box and crop_box[1] > crop_box[0]:
            x0, x1, y0, y1 = crop_box
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       edgecolor='w', lw=1.0, ls='--'))
    fig.suptitle(header, fontsize=10)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_overview(acc: _TirAccum, rep: dict, out_path: str, crop_size: int) -> str:
    """Temporal + rendering diagnostics; returns the path ('' when skipped)."""
    import cv2
    plt = _import_pyplot()
    if plt is None or not acc.times:
        return ''
    t = np.asarray(acc.times, dtype=np.float64)
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.0), layout='constrained')

    # (a) luma level + extremes -> the auto-range detector
    ax = axes[0][0]
    n_steps = (rep.get('autorange') or {}).get('n_steps') or 0
    ax.fill_between(t, acc.p1, acc.p99, color='tab:blue', alpha=0.15)
    ax.plot(t, acc.p99, color='tab:blue', lw=0.7, ls='--', label='luma p99')
    ax.plot(t, acc.p1, color='tab:blue', lw=0.7, ls=':', label='luma p1')
    ax.plot(t, acc.mean, color='tab:blue', lw=1.0, label='luma mean')
    for f in (rep.get('autorange') or {}).get('step_frames', []):
        if 0 <= f < t.size:
            ax.axvline(t[f], color='tab:red', lw=0.7, alpha=0.6)
    ax.set_title(f'luma level (red = auto-range candidates; {n_steps} found)',
                 fontsize=8.5)
    ax.set_ylim(0, 260)
    ax.set_xlabel('t (s)')
    ax.set_ylabel('luma code')
    ax.legend(fontsize=7, loc='best')

    # (b) chroma -> is the false-colour render stable?
    ax = axes[0][1]
    ax.plot(t, acc.abs_u, color='tab:orange', lw=0.8, label='mean |U-128| (RGB path)')
    ax.plot(t, acc.abs_v, color='tab:green', lw=0.8, label='mean |V-128| (RGB path)')
    planes = (rep.get('chroma') or {}).get('raw_planes') or {}
    if planes:
        ax.axhline(planes['mean_abs_u'], color='tab:orange', ls='--', lw=0.9,
                   label='planes |U-128|')
        ax.axhline(planes['mean_abs_v'], color='tab:green', ls='--', lw=0.9,
                   label='planes |V-128|')
    ax.set_title('chroma: solid = RGB path, dashed = decoded planes '
                 '(the comparable number)', fontsize=8.5)
    ax.set_xlabel('t (s)')
    ax.set_ylabel('codes from neutral')
    ax.legend(fontsize=7, loc='best')

    # (c) column-wise temporal std -> where is the static legend?
    ax = axes[0][2]
    cs = acc.col_std()
    if cs.size:
        ax.plot(np.arange(cs.size), cs, color='tab:purple', lw=0.8)
        box = (rep.get('geometry') or {}).get('crop_box_source_px') or {}
        for key in ('x0', 'x1'):
            if box.get(key) is not None:
                ax.axvline(box[key], color='tab:orange', ls='--', lw=0.9)
        x0 = (rep.get('geometry') or {}).get('static_band_x0')
        if x0 is not None:
            ax.axvspan(x0, cs.size, color='tab:red', alpha=0.15)
    ax.set_title('temporal std per column (orange = crop box, red = legend)',
                 fontsize=8.5)
    ax.set_xlabel('source x (px)')
    ax.set_ylabel('std of luma over time')

    # (d) exactly what the model sees: the size px centre crop of frame 0
    ax = axes[1][0]
    if acc.kept:
        ax.imshow(vio.resize_center_crop(acc.kept[0], crop_size))
        ax.set_title(f'frame 0 through resize_center_crop({crop_size})',
                     fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # (e) luma histogram of frame 0
    ax = axes[1][1]
    if acc.kept:
        y = cv2.cvtColor(acc.kept[0], cv2.COLOR_RGB2YUV)[:, :, 0]
        ax.hist(y.ravel()[::8], bins=64, range=(0, 255), color='tab:gray')
        ax.set_title('frame 0 luma histogram (railing check)', fontsize=8.5)
        ax.set_xlabel('luma code')
        ax.set_ylabel('pixels (1/8 subsample)')

    # (f) distinct colours -> interpolated gradient + lossy compression
    ax = axes[1][2]
    if acc.colors:
        ax.plot(t[:len(acc.colors)], acc.colors, color='tab:brown', lw=0.9)
        ax.set_title(f'distinct RGB colours per frame (stride {STAT_STRIDE})',
                     fontsize=8.5)
        ax.set_xlabel('t (s)')
        ax.set_ylabel('distinct colours (a 256-entry palette would sit at 256)')

    fig.suptitle(f"{rep['session']}  {rep['raw_file_name']}  --  TIR temporal "
                 f"diagnostics ({acc.n_decoded} frames)", fontsize=10)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# per-session inspection
# --------------------------------------------------------------------------- #
def _json_default(o):
    """JSON fallback for numpy scalars/arrays (PyAV objects are pre-converted)."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _decimate(*series, max_points: int = 400):
    """Decimate equal-length series for the JSON (figures keep every sample)."""
    n = len(series[0]) if series else 0
    if n == 0:
        return []
    step = max(1, int(math.ceil(n / float(max_points))))
    return [step] + [np.asarray(s)[::step].tolist() for s in series]


def inspect_session(raw_root: str, subject: str, task: str, output_dir: str,
                    decoder: str = 'auto', frames_n: int = 6,
                    crop_size: int = 224, max_frames: int = 0,
                    plot: bool = True, md5: bool = True,
                    use_ffprobe: bool = False) -> dict:
    """Inspect one session; writes figures + ``TIR.json``; returns the report."""
    path = find_video(raw_root, subject, task)
    if not path:
        raise FileNotFoundError(
            f'no thermal video under {os.path.join(raw_root, RAW_TREE, subject)} '
            f'for task {task} (looked for {", ".join(VIDEO_EXTS)})')
    session = f'{subject}_{task}'
    out_dir = os.path.join(output_dir, session)
    os.makedirs(out_dir, exist_ok=True)          # BEFORE any figure is written

    rep = {'session': session, 'modality': 'tir', 'subject': subject,
           'task': task, 'video': path,
           'raw_file_name': os.path.basename(path),
           'session_has_physio': os.path.isdir(
               os.path.join(raw_root, 'Physiology', subject, task))}
    fi = file_info(path) if md5 else {'bytes': os.path.getsize(path), 'md5': None}
    rep['file_bytes'] = fi['bytes']
    rep['md5'] = fi['md5']

    # --- format info: independent views, no decoding ----------------------- #
    rep['pyav'] = probe_pyav(path)
    rep['opencv'] = probe_opencv(path)
    rep['ffprobe'] = probe_ffprobe(path) if use_ffprobe else None
    stream = rep['pyav'] or {}
    rep['stream'] = stream
    height = int((rep['opencv'] or {}).get('height') or stream.get('height') or 0)
    width = int((rep['opencv'] or {}).get('width') or stream.get('width') or 0)

    # --- one full decode pass --------------------------------------------- #
    acc, errors = stream_video(path, decoder, frames_n, max_frames, height, width)
    rep['decode_errors'] = errors
    times = np.asarray(acc.times, dtype=np.float64)
    fps_nom = (_as_float(stream.get('avg_frame_rate'))
               or (rep['opencv'] or {}).get('fps') or 25.0)
    fps_meas = None
    span = None
    if times.size >= 2 and (times[-1] - times[0]) > 0:
        span = float(times[-1] - times[0])
        fps_meas = (times.size - 1) / span
    fps_used = fps_meas or fps_nom
    rep['decode'] = {
        'decoder': acc.decoder,
        'n_decoded': acc.n_decoded,
        'n_failed': acc.n_failed,
        'partial': bool(errors),
        'width': width, 'height': height,
        'fps_nominal': fps_nom,
        'fps_measured': fps_meas,
        'span_s': span,
        'duration_s': float(acc.n_decoded / fps_used) if fps_used else None,
        'times_source': ('container pts' if acc.decoder == 'pyav'
                         else 'synthesised from the container fps'),
        'max_frames_cap': max_frames or None,
    }

    # --- rendering checks -------------------------------------------------- #
    planes = [p for p in acc.planes if p]
    rgb_u = float(np.mean(acc.abs_u)) if acc.abs_u else None
    rgb_v = float(np.mean(acc.abs_v)) if acc.abs_v else None
    rep['chroma'] = {
        # PRIMARY: the decoded planes -- the number the README quotes
        'mean_abs_u': float(np.mean([p[0] for p in planes])) if planes else rgb_u,
        'mean_abs_v': float(np.mean([p[1] for p in planes])) if planes else rgb_v,
        'primary': 'decoded yuv444p planes' if planes else 'rgb round-trip',
        'planes_missing': not planes,
        'raw_planes': ({
            'mean_abs_u': float(np.mean([p[0] for p in planes])),
            'mean_abs_v': float(np.mean([p[1] for p in planes])),
            'n_frames': len(planes),
            'method': 'PyAV yuv444p planes (no RGB round-trip) -- comparable to '
                      'the documented 23-27 / 16-20 range',
        } if planes else None),
        'rgb_roundtrip': {
            'mean_abs_u': rgb_u, 'mean_abs_v': rgb_v,
            'method': ('decoded RGB -> cv2.COLOR_RGB2YUV on a '
                       f'1/{STAT_STRIDE} subsample; |V-128| is inflated by ~5 '
                       'codes through gamut clipping of the saturated palette, '
                       'so this is a cross-check, not the headline'),
        },
        'unique_colors_frame0': acc.colors[0] if acc.colors else None,
        'unique_colors_mean': float(np.mean(acc.colors)) if acc.colors else None,
        'color_stride': STAT_STRIDE,
        'palette_note': ('~5k distinct colours per frame on this corpus => an '
                         'INTERPOLATED palette gradient plus lossy wmv3 chroma, '
                         'NOT a literal 8-bit palette lookup'),
    }
    rep['luma'] = {
        'mean': float(np.mean(acc.mean)) if acc.mean else None,
        'p1': float(np.mean(acc.p1)) if acc.p1 else None,
        'p99': float(np.mean(acc.p99)) if acc.p99 else None,
        'min': float(np.min(acc.min)) if acc.min else None,
        'max': float(np.max(acc.max)) if acc.max else None,
        'yuv_range': '0-255 as decoded (this corpus reaches 0 and 254)',
    }
    ar = autorange_steps(acc.p1, acc.p99, fps_used)
    ar['step_times'] = [float(times[i]) for i in ar['step_frames']
                        if 0 <= i < times.size]
    rep['autorange'] = ar

    # --- crop geometry + legend ------------------------------------------- #
    if height and width and acc.kept:
        equal, dmax, box = crop_check(acc.kept[0], crop_size)
        band = static_right_band(acc.col_std())
        rep['geometry'] = {
            'crop_size': int(crop_size),
            'crop_box_source_px': {'x0': box[0], 'x1': box[1],
                                   'y0': box[2], 'y1': box[3]},
            'check_equal': equal,
            'check_dmax': dmax,
            'check_method': ('arithmetic mirror vs the REAL '
                             'video_io.resize_center_crop, compared by '
                             'np.array_equal on frame 0'),
            'static_band_x0': band[0] if band else None,
            'static_band_width': band[1] if band else None,
            'legend_in_crop': bool(band and band[0] < box[1]),
            'legend_cropped': bool(band and band[0] >= box[1]),
            'note': ('static_band_x0 is a heuristic (columns with near-zero '
                     'temporal std); a static border would look the same. '
                     'When nh == size the crop keeps the FULL height.'),
        }
    else:
        rep['geometry'] = {'crop_size': int(crop_size),
                           'error': 'no frame kept or no width/height'}

    # --- per-frame table + decimated time series --------------------------- #
    rep['frames'] = [
        {'index': k, 't_s': float(acc.kept_times[k]),
         'luma_mean': acc.mean[k], 'luma_min': acc.min[k], 'luma_max': acc.max[k],
         'abs_u_rgb': acc.abs_u[k], 'abs_v_rgb': acc.abs_v[k],
         'unique_colors': acc.colors[k]}
        for k in range(len(acc.kept))]
    step, *cols = _decimate(acc.times, acc.mean, acc.p1, acc.p99, acc.abs_u,
                            acc.abs_v, acc.colors)
    rep['temporal'] = {'decimating_stride': step,
                       't_s': cols[0], 'luma_mean': cols[1], 'luma_p1': cols[2],
                       'luma_p99': cols[3], 'abs_u_rgb': cols[4],
                       'abs_v_rgb': cols[5], 'unique_colors': cols[6]}

    # --- figures ----------------------------------------------------------- #
    figs = []
    if plot:
        box = rep['geometry'].get('crop_box_source_px') or {}
        crop_box = ((box['x0'], box['x1'], box['y0'], box['y1'])
                    if box.get('x0') is not None else None)
        header = (f"{session}  raw TIR frames  {width}x{height}  "
                  f"{fps_used:.3f} fps  ({rep['raw_file_name']}, first "
                  f"{len(acc.kept)} decoded)\nfalse-colour thermal RENDERING: the "
                  f"colours are a palette, not temperature; dashed box = "
                  f"{crop_size} px crop (keeps the full height)")
        p = plot_frames(acc.kept, acc.kept_times,
                        os.path.join(out_dir, f'{FIG_NAME}_frames.png'), header,
                        crop_box=crop_box)
        if p:
            figs.append(p)
        p = plot_overview(acc, rep, os.path.join(out_dir, f'{FIG_NAME}_overview.png'),
                          crop_size)
        if p:
            figs.append(p)
    rep['figures'] = figs

    rep['warnings'] = quality_warnings(rep)
    with open(os.path.join(out_dir, f'{FIG_NAME}.json'), 'w') as fh:
        json.dump(rep, fh, indent=2, default=_json_default)
    rep['output_dir'] = out_dir
    rep['ok'] = acc.n_decoded > 0
    return rep


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_list(spec: str):
    return [s.strip() for s in (spec or '').replace(';', ',').split(',') if s.strip()]


def get_args(argv=None):
    p = argparse.ArgumentParser(
        'BP4D raw thermal-video inspection', add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('--subject', default='',
                   help='subject id(s), comma-separated, e.g. F001 or F001,F002')
    p.add_argument('--task', default='', help='task id(s), comma-separated, or "all"')
    p.add_argument('--raw_root', default=default_raw_root(),
                   help='raw BP4D root (default: $RAW_DATA_PATH, else the in-repo '
                        'data/raw/BP4D)')
    p.add_argument('--output_dir', default=default_output_dir(),
                   help='inspect_data ROOT; <output_dir>/<subject>_<task>/ is '
                        'created (default: $OUTPUT_DIR/inspect_data, else '
                        '<repo>/output/inspect_data)')
    p.add_argument('--decoder', default='auto',
                   choices=('auto', 'pyav', 'opencv', 'decord'),
                   help='streaming decoder: auto (default; pyav -> opencv -> '
                        'decord), pyav (gives the pts and the raw YUV planes in '
                        'one pass), opencv (what the training loader uses), decord')
    p.add_argument('--frames', default=6, type=int,
                   help='how many leading frames to draw (default 6)')
    p.add_argument('--max_frames', default=0, type=int,
                   help='cap the decoded frames per session (0 = decode all, the '
                        'default: the DECODED count is the authoritative one)')
    p.add_argument('--crop_size', default=224, type=int,
                   help='crop size whose source-pixel window is reported and '
                        'drawn (default 224, the Stage-2/3 input size)')
    p.add_argument('--ffprobe', dest='ffprobe', action='store_true', default=False,
                   help='also store the raw ffprobe JSON (a third opinion)')
    p.add_argument('--no-md5', dest='md5', action='store_false', default=True,
                   help='skip the md5 (faster; drops the raw-vs-canonical check)')
    p.add_argument('--no-plot', dest='plot', action='store_false', default=True,
                   help='write the JSON only, skip the figures')
    p.add_argument('--list', action='store_true',
                   help='list the (subject, task) sessions found in Thermal/ and exit')
    return p.parse_args(argv)


def _print_report(rep: dict) -> None:
    dec = rep['decode']
    stream = rep.get('stream') or {}
    ov = rep.get('opencv') or {}
    ch = rep['chroma']
    geo = rep['geometry']
    lu = rep['luma']

    def kv(label, value):
        print(f'  {label:<10}: {value}')

    kv('file', f"{rep['raw_file_name']}  ({_fmt_bytes(rep['file_bytes'])}"
               + (f", md5 {rep['md5']}" if rep.get('md5') else '') + ')')
    kv('container', f"{stream.get('format_name')} ({stream.get('format_long_name')})"
                    f"  duration {_opt(stream.get('duration_s'), '%.2f s')}"
                    f"  bit_rate {_opt(stream.get('bit_rate'), '%.0f b/s')}"
                    f"  streams {stream.get('nb_streams')}")
    kv('stream 0', f"{stream.get('codec')} "
                   f"({_opt(stream.get('codec_long_name'), '%s')})"
                   f"  {_opt(stream.get('pix_fmt'), '%s')}"
                   f"  {stream.get('width')}x{stream.get('height')}"
                   f"  sar {_opt(stream.get('sample_aspect_ratio'), '%s')}"
                   f"  profile {_opt(stream.get('profile'), '%s')}")
    kv('', f"avg_rate {_opt(stream.get('avg_frame_rate'), '%.4f')}"
           f"  base_rate {_opt(stream.get('base_rate'), '%.4f')}"
           f"  time_base {_opt(stream.get('time_base'), '%s')}"
           f"  stream bit_rate {_opt(stream.get('stream_bit_rate'), '%.0f b/s')}"
           f"  nb_frames {stream.get('nb_frames')}"
           f"  has_b_frames {stream.get('has_b_frames')}")
    kv('opencv', f"{_opt(ov.get('backend'), '%s')}"
                 f"  fps {_opt(ov.get('fps'), '%.4f')}"
                 f"  frame_count {ov.get('frame_count')}"
                 f"  {ov.get('width')}x{ov.get('height')}"
                 f"  fourcc {_opt(ov.get('fourcc'), '%s')}")
    kv('decode', f"{dec['decoder']}: {dec['n_decoded']} frames, "
                 f"{dec['n_failed']} failed, "
                 f"{_opt(dec['duration_s'], '%.2f s')}, "
                 f"fps nominal {_opt(dec['fps_nominal'], '%.4f')} / "
                 f"measured {_opt(dec['fps_measured'], '%.4f')}"
                 + ('  [PARTIAL]' if dec['partial'] else ''))
    planes = ch.get('raw_planes')
    if planes:
        kv('chroma', f"decoded planes |U-128| {planes['mean_abs_u']:.2f}  "
                     f"|V-128| {planes['mean_abs_v']:.2f}"
                     f"   <- comparable to the documented 23-27 / 16-20")
        kv('', f"RGB->YUV check |U-128| {_opt(ch['rgb_roundtrip']['mean_abs_u'], '%.2f')}  "
               f"|V-128| {_opt(ch['rgb_roundtrip']['mean_abs_v'], '%.2f')}"
               f"   (|V-128| inflated ~5 codes by gamut clipping)")
    else:
        kv('chroma', f"RGB->YUV |U-128| {_opt(ch['mean_abs_u'], '%.2f')}  "
                     f"|V-128| {_opt(ch['mean_abs_v'], '%.2f')}"
                     f"   <- planes unavailable: NOT comparable to 23-27 / 16-20")
    kv('', f"distinct colours (frame 0, stride {ch['color_stride']}) "
           f"{ch['unique_colors_frame0']}"
           f"  mean {_opt(ch['unique_colors_mean'], '%.1f')}")
    kv('luma', f"mean {_opt(lu['mean'], '%.1f')}  p1 {_opt(lu['p1'], '%.0f')}  "
               f"p99 {_opt(lu['p99'], '%.0f')}  min {_opt(lu['min'], '%.0f')}"
               f"  max {_opt(lu['max'], '%.0f')}"
               f"   auto-range steps {rep['autorange']['n_steps']}"
               f" (of {rep['autorange']['n_candidates_raw']} raw candidates, "
               f"thr {_opt(rep['autorange']['threshold_codes'], '%.1f')} codes)")
    box = geo.get('crop_box_source_px') or {}
    if box.get('x0') is not None:
        kv('geometry', f"{geo['crop_size']} px crop -> source x {box['x0']:.0f}"
                       f"..{box['x1']:.0f}, y {box['y0']:.0f}..{box['y1']:.0f}"
                       f"   (mirror == real fn: {geo.get('check_equal')}"
                       f", maxdiff {geo.get('check_dmax')})")
        kv('', f"static right band x >= {geo.get('static_band_x0')} "
               f"({geo.get('static_band_width')} px) -> legend "
               + ('INSIDE the crop' if geo.get('legend_in_crop')
                  else 'outside the crop'
                  if geo.get('static_band_x0') is not None else 'not detected'))
    print('  frames    :')
    for fr in rep['frames']:
        print(f"      #{fr['index']:<3} t={fr['t_s']:7.3f} s  "
              f"luma {fr['luma_mean']:6.1f} [{fr['luma_min']:.0f},{fr['luma_max']:.0f}]  "
              f"|U-128| {fr['abs_u_rgb']:5.2f}  |V-128| {fr['abs_v_rgb']:5.2f}  "
              f"colours {fr['unique_colors']}")
    kv('figures', ', '.join(os.path.basename(f) for f in rep['figures']) or '(none)')
    kv('warnings', '; '.join(rep['warnings']) if rep['warnings'] else 'none')


def main() -> int:
    args = get_args()
    raw_root = args.raw_root
    available = discover_sessions(raw_root)
    if not os.path.isdir(raw_root):
        raise SystemExit(f'raw root not found: {raw_root}\n'
                         f'set $RAW_DATA_PATH or pass --raw_root')
    if args.list:
        print(f'raw root: {raw_root}')
        print(f'{len(available)} session(s) under {RAW_TREE}/:')
        for s in sorted({s for s, _ in available}):
            print(f'  {s}: {" ".join(t for ss, t in available if ss == s)}')
        if not available:
            print('  (none - is this the right --raw_root?)')
        return 0

    if not args.subject:
        raise SystemExit('--subject is required (or use --list); available: '
                         + ', '.join(sorted({s for s, _ in available})))
    if not args.task:
        raise SystemExit('--task is required (or "all", or use --list)')

    subjects = _parse_list(args.subject)
    tasks = _parse_list(args.task)
    known = {}
    for s, t in available:
        known.setdefault(s, []).append(t)

    wanted = []
    for s in subjects:
        if s not in known:
            raise SystemExit(f'subject {s!r} not found under '
                             f'{os.path.join(raw_root, RAW_TREE)}; available: '
                             + ', '.join(sorted(known)))
        ts = known[s] if any(t.lower() == 'all' for t in tasks) else tasks
        for t in ts:
            if t not in known[s]:
                raise SystemExit(f'task {t!r} not found for {s!r}; available: '
                                 + ', '.join(known[s]))
            wanted.append((s, t))

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'raw root   : {raw_root}')
    print(f'output root: {args.output_dir}')
    print(f'modality   : TIR ({RAW_TREE}/<subject>/<task>.wmv)')
    print(f'decoder    : {args.decoder}')
    print(f'sessions   : {", ".join(f"{s}_{t}" for s, t in wanted)}')
    print()

    summaries, n_ok = [], 0
    for subject, task in wanted:
        print(f'=== {subject}_{task} ===')
        try:
            rep = inspect_session(
                raw_root=raw_root, subject=subject, task=task,
                output_dir=args.output_dir, decoder=args.decoder,
                frames_n=args.frames, crop_size=args.crop_size,
                max_frames=args.max_frames, plot=args.plot, md5=args.md5,
                use_ffprobe=args.ffprobe)
        except Exception as exc:
            print(f'  [FAIL] {subject}/{task}: {type(exc).__name__}: {exc}')
            summaries.append({'session': f'{subject}_{task}', 'ok': False,
                              'error': f'{type(exc).__name__}: {exc}'})
            continue

        _print_report(rep)
        print(f"  -> {os.path.relpath(rep['output_dir'], args.output_dir)}/   "
              f"({len(rep['figures'])} figure(s), {len(rep['warnings'])} warning(s))")
        print()
        n_ok += 1 if rep['ok'] else 0
        summaries.append({
            'session': rep['session'], 'ok': rep['ok'],
            'video': rep['raw_file_name'], 'md5': rep['md5'],
            'bytes': rep['file_bytes'], 'width': rep['decode']['width'],
            'height': rep['decode']['height'],
            'n_decoded': rep['decode']['n_decoded'],
            'fps_nominal': rep['decode']['fps_nominal'],
            'fps_measured': rep['decode']['fps_measured'],
            'duration_s': rep['decode']['duration_s'],
            'codec': (rep.get('stream') or {}).get('codec'),
            'pix_fmt': (rep.get('stream') or {}).get('pix_fmt'),
            'chroma_primary': rep['chroma']['primary'],
            'mean_abs_u': rep['chroma']['mean_abs_u'],
            'mean_abs_v': rep['chroma']['mean_abs_v'],
            'unique_colors_frame0': rep['chroma']['unique_colors_frame0'],
            'autorange_steps': rep['autorange']['n_steps'],
            'legend_in_crop': rep['geometry'].get('legend_in_crop'),
            'crop_check_equal': rep['geometry'].get('check_equal'),
            'warnings': rep['warnings'],
            'output_dir': rep['output_dir'],
        })

    index_path = os.path.join(args.output_dir, 'thermal_index.json')
    payload = {'raw_root': raw_root, 'decoder': args.decoder,
               'n_sessions': len(summaries), 'n_ok': n_ok,
               'sessions': summaries}
    with open(index_path, 'w') as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    print(f'{n_ok}/{len(summaries)} session(s) inspected; index: {index_path}')
    return 0 if n_ok == len(summaries) else 1


if __name__ == '__main__':
    sys.exit(main())
