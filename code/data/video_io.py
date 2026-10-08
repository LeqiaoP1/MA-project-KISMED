"""Per-modality frame readers for the asymmetric recording layout.

* RGB -- an ordered directory of ``jpg`` frames (``data/raw/<session>/rgb/``).
* TIR -- a single ``.wmv`` video container (~60 s) (``data/raw/<session>/tir.wmv``).
  NOTE: the BP4D thermal stream is a FALSE-COLOUR (rainbow) thermal *rendering*
  with a burned-in degC legend -- NOT a gray thermal image. Verified with two
  independent decoders (OpenCV and PyAV): ``wmv3``/``yuv420p``, 3 planes, mean
  |U-128| ~ 23-27 on every session, and the chroma plane follows the scene.
  ``read_all`` therefore defaults to ``gray=False`` (RGB, 3 channels); pass
  ``gray=True`` only for the legacy luma-only surrogate.

Both nominal 25 fps -- probe at runtime. WMV3/VC-1 decoding is **not** present
in every OpenCV build, so ``open_video`` falls back to decord (ffmpeg-based);
if neither can decode a container an informative error is raised.

RGB jpgs are decoded at reduced DCT scale when ``target_size`` allows it (see
``_imread_flag``): the source frames are 1392x1040, so decoding them in full and
then cropping to 224 discards ~97% of the libjpeg work, and measurement shows
decode CPU -- not I/O (0.6 ms/frame) -- dominates the cost of a clip.
"""
import os
import time
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    'IMAGE_EXTS', 'list_image_files', 'read_image', 'read_image_range',
    'resize_center_crop', 'open_video', 'CV2ClipReader', 'DecordClipReader',
    'OpenPolicy', 'resolve_open_policy',
]

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# JPEG SOF markers carrying the frame size (C0-CF minus DHT/JPG/DAC)
_JPEG_SOF_MARKERS = frozenset(
    (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
     0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _jpeg_size(path: str):
    """``(height, width)`` from the JPEG SOF marker, without decoding.

    Returns ``None`` for anything that is not a parseable JPEG, so callers fall
    back to a normal full decode rather than guessing.
    """
    try:
        with open(path, 'rb') as fh:
            if fh.read(2) != b'\xff\xd8':            # not SOI
                return None
            while True:
                b = fh.read(1)
                while b and b != b'\xff':            # scan to the next marker
                    b = fh.read(1)
                if not b:
                    return None
                marker = fh.read(1)
                while marker == b'\xff':             # fill bytes
                    marker = fh.read(1)
                if not marker:
                    return None
                m = marker[0]
                if m == 0xDA:                        # start of scan: no SOF left
                    return None
                if m == 0x01 or 0xD0 <= m <= 0xD9:   # standalone, no length
                    continue
                ln = fh.read(2)
                if len(ln) < 2:
                    return None
                seg = int.from_bytes(ln, 'big')
                if m in _JPEG_SOF_MARKERS:
                    data = fh.read(5)
                    if len(data) < 5:
                        return None
                    return (int.from_bytes(data[1:3], 'big'),
                            int.from_bytes(data[3:5], 'big'))
                fh.seek(seg - 2, os.SEEK_CUR)
    except OSError:
        return None


def _imread_flag(path: str, target_size: Optional[int], gray: bool, cv2):
    """Cheapest ``imread`` flag that still yields >= ``target_size`` pixels.

    libjpeg can decode at 1/2, 1/4 or 1/8 DCT scale, so ask for the coarsest
    scale whose short side still covers the target (never upscaling). Only JPEGs
    are probed; other formats keep the previous full-decode behaviour.
    """
    if not target_size:
        return cv2.IMREAD_UNCHANGED
    size = _jpeg_size(path)
    if size is None:
        return cv2.IMREAD_UNCHANGED
    short = min(size)
    for factor in (8, 4, 2):
        if -(-short // factor) >= target_size:       # ceil(short / factor)
            kind = 'GRAYSCALE' if gray else 'COLOR'
            return getattr(cv2, f'IMREAD_REDUCED_{kind}_{factor}')
    return cv2.IMREAD_UNCHANGED
def resize_center_crop(img, size: int):
    """Resize the shorter side then center-crop to a square of ``size``.

    Accepts ``[H, W]`` or ``[H, W, C]`` uint8 and returns the same layout.
    """
    import cv2
    gray = img.ndim == 2
    h, w = img.shape[:2]
    if h == size and w == size:
        return img
    scale = size / float(min(h, w))
    if scale < 1.0:          # downscale a large frame first (faster crop)
        nh, nw = int(h * scale), int(w * scale)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        h, w = nh, nw
    top = (h - size) // 2
    left = (w - size) // 2
    out = img[top:top + size, left:left + size]
    if gray:
        return out
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    return out


# --------------------------------------------------------------------------- #
# RGB image sequence
# --------------------------------------------------------------------------- #
def list_image_files(directory: str, exts=IMAGE_EXTS) -> List[str]:
    files = sorted(
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in exts)
    if not files:
        raise FileNotFoundError(f'No image files found in {directory}')
    return [os.path.join(directory, f) for f in files]


def read_image(path: str, target_size: Optional[int] = None,
               gray: bool = False):
    """Read one image as uint8 ``[H, W, C]`` (or ``[H, W]`` if ``gray``).

    When ``target_size`` is set, a JPEG is decoded directly at the coarsest DCT
    scale that still stays at or above the target (:func:`_imread_flag`), which
    is ~1.6x cheaper than a full decode followed by a downscale (mean |diff|
    0.56/255 at 224 px) and never upscales. Consequence of that path: a
    grayscale-*encoded* jpg read with ``gray=False`` is returned 3-channel,
    which matches the documented ``[H, W, C]`` contract.
    """
    import cv2
    img = cv2.imread(path, _imread_flag(path, target_size, gray, cv2))
    if img is None:
        raise IOError(f'Failed to read image {path}')
    if img.ndim == 3 and img.shape[2] == 4:      # drop alpha
        img = img[:, :, :3]
    if gray:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_size:
        img = resize_center_crop(img, target_size)
    return img


def read_image_range(image_dir: str, start: int = 0, n: Optional[int] = None,
                     target_size: Optional[int] = None, gray: bool = False):
    """Read ``[start, start+n)`` frames of a jpg sequence into one array.

    Returns uint8 ``[T, H, W, C]`` (or ``[T, H, W]`` when ``gray``). Indices
    are clamped to the available frames.
    """
    files = list_image_files(image_dir)
    end = len(files) if n is None else min(len(files), start + n)
    start = max(0, min(start, len(files)))
    if end <= start:
        raise ValueError(
            f'Empty read range [{start}:{start + n}] over {len(files)} frames')
    frames = [read_image(p, target_size=target_size, gray=gray)
              for p in files[start:end]]
    return np.stack(frames, axis=0)


# --------------------------------------------------------------------------- #
# TIR video (.wmv)
# --------------------------------------------------------------------------- #
#: Attempts (per backend) before ``open_video`` gives up on a container.
DEFAULT_OPEN_ATTEMPTS = 3
#: Seconds to wait before re-trying a failed open (doubles per attempt).
DEFAULT_OPEN_BACKOFF_S = 1.0
#: Upper bound OpenCV may spend inside a single ffmpeg open(), milliseconds.
DEFAULT_OPEN_TIMEOUT_MS = 15000
#: Upper bound OpenCV may spend inside a single ffmpeg read(), milliseconds.
DEFAULT_READ_TIMEOUT_MS = 15000


class OpenPolicy:
    """Retry / timeout policy for :func:`open_video` (env-overridable).

    Rationale -- the TIR ``.wmv`` corpus sits on a parallel file system that
    occasionally stalls a single read for minutes. Without an explicit ffmpeg
    timeout, OpenCV blocks inside ``open()``/``read()`` for ~4.5 min before its
    own interrupt callback fires (five workers logged
    ``cap_ffmpeg_impl.hpp ... Stream timeout triggered after 265852 ms`` on
    Lichtenberg). In a DDP run that stall is long enough for the other ranks to
    trip the 600 s NCCL watchdog, so ONE slow container killed the whole
    4-GPU job instead of one clip. Capping the wait at a few seconds and
    re-trying turns the same event into a short hiccup.

    Environment overrides: ``VIDEO_OPEN_TIMEOUT_MS``, ``VIDEO_READ_TIMEOUT_MS``
    (0 or negative disables the cap), ``VIDEO_OPEN_ATTEMPTS``,
    ``VIDEO_OPEN_BACKOFF_S`` (seconds, doubles per attempt), ``VIDEO_BACKEND``
    (``auto`` | ``cv2`` | ``decord``).
    """

    __slots__ = ('attempts', 'backoff_s', 'open_timeout_ms', 'read_timeout_ms',
                 'backends')

    def __init__(self, attempts: int = DEFAULT_OPEN_ATTEMPTS,
                 backoff_s: float = DEFAULT_OPEN_BACKOFF_S,
                 open_timeout_ms: int = DEFAULT_OPEN_TIMEOUT_MS,
                 read_timeout_ms: int = DEFAULT_READ_TIMEOUT_MS,
                 backends: Tuple[str, ...] = ('cv2', 'decord')):
        self.attempts = max(1, int(attempts))
        self.backoff_s = float(backoff_s)
        self.open_timeout_ms = int(open_timeout_ms)
        self.read_timeout_ms = int(read_timeout_ms)
        self.backends = tuple(backends)

    def sleep_before(self, attempt: int) -> None:
        """Back off before retry ``attempt`` (0-based); 0 sleeps nothing."""
        if attempt > 0 and self.backoff_s > 0:
            time.sleep(self.backoff_s * (2 ** (attempt - 1)))


def resolve_open_policy(backend: Optional[str] = None,
                        policy: Optional[OpenPolicy] = None) -> OpenPolicy:
    """Build the effective :class:`OpenPolicy` from ``backend`` + the env.

    Precedence: an explicit ``policy`` argument wins outright; otherwise a
    ``backend`` argument picks the order, else ``$VIDEO_BACKEND``; the numbers
    always come from the environment so a job script can widen the timeouts
    without touching the call site.
    """
    if policy is not None:
        return policy
    name = (backend or os.environ.get('VIDEO_BACKEND') or 'auto').strip().lower()
    if name in ('auto', ''):
        backends: Tuple[str, ...] = ('cv2', 'decord')
    elif name in _BACKEND_FACTORIES:
        backends = (name,)
    else:
        raise ValueError(f'unknown VIDEO_BACKEND {name!r}: expected one of '
                         f"{', '.join(sorted(_BACKEND_FACTORIES))} or 'auto'")
    return OpenPolicy(
        attempts=int(os.environ.get('VIDEO_OPEN_ATTEMPTS',
                                    DEFAULT_OPEN_ATTEMPTS)),
        backoff_s=float(os.environ.get('VIDEO_OPEN_BACKOFF_S',
                                       DEFAULT_OPEN_BACKOFF_S)),
        open_timeout_ms=int(os.environ.get('VIDEO_OPEN_TIMEOUT_MS',
                                           DEFAULT_OPEN_TIMEOUT_MS)),
        read_timeout_ms=int(os.environ.get('VIDEO_READ_TIMEOUT_MS',
                                           DEFAULT_READ_TIMEOUT_MS)),
        backends=backends)


class CV2ClipReader:
    """OpenCV (VideoCapture) based reader for a single video file."""

    backend = 'cv2'

    def __init__(self, path: str, open_timeout_ms: int = DEFAULT_OPEN_TIMEOUT_MS,
                 read_timeout_ms: int = DEFAULT_READ_TIMEOUT_MS):
        import cv2
        if not os.path.exists(path):     # retrying cannot help; fail fast
            raise FileNotFoundError(f'No such video file: {path}')
        self.cv2 = cv2
        self.path = path
        self._cap = self._open(cv2, path, open_timeout_ms, read_timeout_ms)
        if not self._cap.isOpened():
            self._cap.release()
            raise IOError(f'OpenCV could not open {path}')
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.num_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if self.fps <= 0:
            self._cap.release()
            raise IOError(f'Could not determine fps of {path}')

    @staticmethod
    def _open(cv2, path: str, open_timeout_ms: int, read_timeout_ms: int):
        """``VideoCapture`` with the ffmpeg wait caps applied when supported.

        OpenCV only honours ``CAP_PROP_OPEN_TIMEOUT_MSEC`` /
        ``CAP_PROP_READ_TIMEOUT_MSEC`` when they are passed to the FFMPEG
        backend at construction time (values must be ints; floats raise).
        Builds without those properties fall back to the plain constructor.
        """
        params = []
        for prop, value in ((getattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC', None),
                             open_timeout_ms),
                            (getattr(cv2, 'CAP_PROP_READ_TIMEOUT_MSEC', None),
                             read_timeout_ms)):
            if prop is not None and value and value > 0:
                params += [int(prop), int(value)]
        if params:
            try:
                cap = cv2.VideoCapture(path, int(cv2.CAP_FFMPEG), params)
            except Exception:                 # no FFMPEG preference in this build
                cap = None
            if cap is not None and cap.isOpened():
                return cap
            if cap is not None:
                cap.release()
        return cv2.VideoCapture(path)

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps

    def read_all(self, gray: bool = False, target_size: Optional[int] = None):
        frames = []
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            if gray and frame.ndim == 3:
                frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
            elif frame.ndim == 3:
                frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
            if target_size:
                frame = resize_center_crop(frame, target_size)
            frames.append(frame)
        self._cap.release()
        if not frames:
            raise IOError(f'No frames decoded from {self.path} -- WMV codec '
                          f'may be missing from this OpenCV build.')
        return np.stack(frames, axis=0)          # [T, H, W(, C)]

    def read_range(self, start: int = 0, n: Optional[int] = None,
                   gray: bool = False):
        """Decode ``[start, start+n)`` frames -> uint8 ``[T, H, W(, C)]``.

        Unlike :meth:`read_all` this SEEKs (``CAP_PROP_POS_FRAMES``) instead of
        decoding from the beginning, so a dataset can pull one clip per access
        without paying for the preceding frames. ``n=None`` reads to the end;
        both bounds are clamped to ``[0, num_frames)`` and a short read (a
        truncated container or a decoder that cannot seek) raises instead of
        silently returning fewer frames.
        NOTE: ``start`` is absolute only on a FRESH reader. When ``start == 0``
        no seek is issued (that is the safest call for exotic decoders), so a
        reader that was already advanced continues from its current position --
        the same behaviour as :meth:`read_all`. Open a reader per range, or
        pass ``start > 0``, when the absolute index matters.        """
        total = self.num_frames if self.num_frames > 0 else None
        start = max(0, int(start))
        if total is not None:
            start = min(start, total)
        if n is None:
            count = None if total is None else max(0, total - start)
        else:
            count = max(0, int(n))
            if total is not None:
                count = min(count, total - start)
        if count == 0:
            raise ValueError(
                f'Empty read range start={start} n={n} over {total} frames of '
                f'{self.path}')

        if start > 0 and not self._cap.set(self.cv2.CAP_PROP_POS_FRAMES, start):
            raise IOError(f'Could not seek to frame {start} of {self.path}')
        frames = []
        while count is None or len(frames) < count:
            ok, frame = self._cap.read()
            if not ok:
                break
            if gray and frame.ndim == 3:
                frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
            elif frame.ndim == 3:
                frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
            frames.append(frame)
        if not frames:
            raise IOError(f'No frames decoded from {self.path} at '
                          f'start={start}')
        if count is not None and len(frames) != count:
            raise IOError(
                f'Short read on {self.path}: asked for {count} frames at '
                f'start={start}, decoded {len(frames)}')
        return np.stack(frames, axis=0)          # [T, H, W(, C)]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._cap.release()


class DecordClipReader:
    """decord (ffmpeg) fallback reader for containers OpenCV cannot decode."""

    backend = 'decord'

    def __init__(self, path: str):
        import decord
        if not os.path.exists(path):     # retrying cannot help; fail fast
            raise FileNotFoundError(f'No such video file: {path}')
        self.decord = decord
        self.path = path
        self._vr = decord.VideoReader(path)
        self.fps = float(self._vr.get_avg_fps() or 0.0)
        self.num_frames = len(self._vr)
        if self.fps <= 0:
            raise IOError(f'Could not determine fps of {path}')

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps

    def read_all(self, gray: bool = False, target_size: Optional[int] = None):
        import numpy as _np
        frames = self._vr.get_batch(list(range(self.num_frames))).asnumpy()
        if frames.ndim == 3:
            frames = frames[..., None]
        if gray and frames.shape[-1] != 1:
            import cv2
            frames = _np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
                                for f in frames])
        if target_size:
            frames = _np.stack([resize_center_crop(f, target_size)
                                for f in frames])
        return frames

    def read_range(self, start: int = 0, n: Optional[int] = None,
                   gray: bool = False):
        """Decode ``[start, start+n)`` frames -> uint8 ``[T, H, W(, C)]``.

        Same contract as :meth:`CV2ClipReader.read_range`: bounds are clamped
        to ``[0, num_frames)`` and a short read raises.
        """
        import numpy as _np
        total = self.num_frames
        start = max(0, min(int(start), total))
        end = total if n is None else min(total, start + max(0, int(n)))
        if end <= start:
            raise ValueError(
                f'Empty read range start={start} n={n} over {total} frames of '
                f'{self.path}')
        frames = self._vr.get_batch(list(range(start, end))).asnumpy()
        if frames.ndim == 3:
            frames = frames[..., None]
        if gray and frames.shape[-1] != 1:
            import cv2
            frames = _np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
                                for f in frames])
        if len(frames) != end - start:
            raise IOError(
                f'Short read on {self.path}: asked for {end - start} frames '
                f'at start={start}, decoded {len(frames)}')
        return frames

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        pass


#: Backend name -> reader factory (filled in once both classes exist).
_BACKEND_FACTORIES = {'cv2': CV2ClipReader, 'decord': DecordClipReader}


def open_video(path: str, policy: Optional[OpenPolicy] = None,
               backend: Optional[str] = None):
    """Open ``path``, trying the configured decoder backends in order.

    Default order (``VIDEO_BACKEND=auto``) is OpenCV -> decord: OpenCV is the
    fast path for the corpus (measured against both decoders -- see the module
    docstring) and decord is the ffmpeg fallback for builds without a WMV3/
    VC-1 decoder. A failed open is retried per :class:`OpenPolicy` so a
    transient stall or a brief file-system blip does not abort the run; every
    attempt is bounded by the ffmpeg open/read timeouts.

    :param policy: explicit :class:`OpenPolicy`; defaults to the env-derived one.
    :param backend: ``'cv2'`` | ``'decord'`` | ``'auto'``, overriding
        ``$VIDEO_BACKEND`` for this call.
    :returns: a reader with ``.fps``, ``.num_frames``, ``.duration``,
        ``.read_all(gray, target_size)`` and ``.read_range(start, n, gray)``.
    """
    pol = resolve_open_policy(backend=backend, policy=policy)
    errors: List[str] = []
    for attempt in range(pol.attempts):
        pol.sleep_before(attempt)
        missing = False
        for name in pol.backends:
            try:
                if name == 'cv2':
                    return CV2ClipReader(path, pol.open_timeout_ms,
                                         pol.read_timeout_ms)
                return DecordClipReader(path)
            except FileNotFoundError as exc:  # nothing to retry
                errors.append(f'{name}: {exc}')
                missing = True
            except ImportError as exc:       # backend simply not installed
                errors.append(f'{name}: not installed ({exc})')
            except Exception as exc:         # unopenable / timed out
                errors.append(f'{name}: {type(exc).__name__}: {exc}')
        if missing:
            break
    detail = ' | '.join(errors[-len(pol.backends):]) or 'no backend tried'
    raise IOError(
        f'No usable video decoder for {path} after {pol.attempts} attempt(s) '
        f'[{detail}]. Install ffmpeg-based decord (`pip install decord`) or a '
        f'WMV-capable OpenCV/ffmpeg; on a slow parallel file system raise '
        f'VIDEO_OPEN_TIMEOUT_MS / VIDEO_READ_TIMEOUT_MS.')
