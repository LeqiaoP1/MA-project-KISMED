"""Per-modality frame readers for the asymmetric recording layout.

* RGB -- an ordered directory of ``jpg`` frames (``data/raw/<session>/rgb/``).
* TIR -- a single ``.wmv`` video container (~60 s) (``data/raw/<session>/tir.wmv``).

Both nominal 25 fps -- probe at runtime. WMV3/VC-1 decoding is **not** present
in every OpenCV build, so ``open_video`` falls back to decord (ffmpeg-based);
if neither can decode a container an informative error is raised.
"""
import os
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    'IMAGE_EXTS', 'list_image_files', 'read_image', 'read_image_range',
    'resize_center_crop', 'open_video', 'CV2ClipReader', 'DecordClipReader',
]

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
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
    """Read one image as uint8 ``[H, W, C]`` (or ``[H, W]`` if ``gray``)."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
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
class CV2ClipReader:
    """OpenCV (VideoCapture) based reader for a single video file."""

    def __init__(self, path: str):
        import cv2
        self.cv2 = cv2
        self.path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f'OpenCV could not open {path}')
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.num_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if self.fps <= 0:
            raise IOError(f'Could not determine fps of {path}')

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps

    def read_all(self, gray: bool = True, target_size: Optional[int] = None):
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

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._cap.release()


class DecordClipReader:
    """decord (ffmpeg) fallback reader for containers OpenCV cannot decode."""

    def __init__(self, path: str):
        import decord
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

    def read_all(self, gray: bool = True, target_size: Optional[int] = None):
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

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        pass


def open_video(path: str):
    """Open ``path`` preferring OpenCV, falling back to decord.

    :returns: a reader with ``.fps``, ``.num_frames``, ``.duration`` and
        ``.read_all(gray, target_size)``.
    """
    try:
        return CV2ClipReader(path)
    except Exception:
        try:
            return DecordClipReader(path)
        except Exception:
            raise IOError(
                f'No usable video decoder for {path}. Install ffmpeg-based '
                f'decord (`pip install decord`) or a WMV-capable OpenCV/ffmpeg.')
