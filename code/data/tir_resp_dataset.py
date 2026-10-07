"""BP4D+ thermal ``.wmv`` -> 1000 Hz respiration (``Resp_Volts.txt``) dataset.

ADD-ON dataset (does not touch the Stage-1/2/3 datasets or their builders).
One sample is one thermal CLIP plus the respiration waveform recorded over
exactly the same time span::

    'tir_video'    float32 [3, T, H, W]  T = clip_seconds * 25 fps (8 s -> 200)
    'resp_signal'  float32 [L]           L = clip_seconds * 1000 Hz (8 s -> 8000)
    'subject_task' str                   e.g. 'F001_T1'

Raw layout consumed (the RAW tree -- no ``prepare_bp4d.py`` copy is needed)::

    <raw_root>/Thermal/<S>/<T>.wmv                 # 25 fps, 726x480, false colour
    <raw_root>/IRFeatures/<S>_<T>.txt              # 28 (x, y) pairs per frame
    <raw_root>/Physiology/<S>/<T>/Resp_Volts.txt   # 1000 Hz, one value per line

``find_thermal_video`` also accepts ``Thermal/<S>/<S>_<T>.wmv`` and a flat
``Thermal/<S>_<T>.wmv``; the nested ``<T>.wmv`` form is what the BP4D+ corpus
actually ships (verified locally: ``Thermal/F001/T1.wmv``).

Format facts (verified on disk + in ``data/raw/BP4D/BP4D+UserGuide_v0.2.pdf``)
-----------------------------------------------------------------------------
* ``IRFeatures/<S>_<T>.txt``: ONE line per thermal frame, 56 plain floats =
  28 ``(x, y)`` pairs, RAW PIXEL coordinates in the 726x480 thermal frame
  (origin top-left). **Line ``n`` == frame ``n`` (1-based)**; F001_T1 has 1612
  lines and the video decodes 1612 frames at 25.000 fps.
* ``(0, 0)`` is the undocumented MISSING-DATA sentinel, written for whole
  frames when the tracker cannot find the frontal fiducials (the head is turned
  away). It is all-or-nothing per frame: the per-frame count of zero pairs is
  only ever 0 or 28. F001_T8 carries 112/227 such lines in three contiguous
  blocks. **Any clip whose frame range touches such a line is DROPPED**
  (requirement 2) -- the sentinel would otherwise drag the ROI box to the image
  corner. A line that cannot be parsed (blank / non-numeric) is treated the same
  way, and a line with the wrong FIELD COUNT rejects the whole file, because
  then line index == frame index is no longer guaranteed.
* A session with NO ``IRFeatures`` file is skipped gracefully (requirement 1);
  the BP4D+ guide ships 15 such untracked, glasses-wearing sequences
  (F016_T2..T4, F054_T10, M045_T2, M049_T1..T10).
* The thermal stream is a FALSE-COLOUR (rainbow) rendering with a burned-in
  degC legend, not a gray image (see ``video_io``), so frames are decoded as
  3-channel RGB and the tensor is ``[3, T, H, W]``.

ROI strategy -- CLIP-LEVEL STATIC BOX (requirement: no spatial warping)
-----------------------------------------------------------------------
For the 12 target landmarks (mouth + nose) of a clip *[1-indexed, user-guide
Figure 3]* ``[9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26]``:

1. collect their ``(x, y)`` over all ``T`` frames of the clip;
2. take the global ``(x_min, x_max, y_min, y_max)`` over frames AND points;
3. extend each side by ``roi_padding * extent`` (0.2 -> the box grows by 40 %
   in total) and clamp to the frame bounds;
4. crop ALL ``T`` frames with that ONE box and ``cv2.resize`` each patch to
   ``input_size x input_size``.

Because the box is computed once per clip (not per frame), the mouth/nose patch
cannot jitter spatially between frames -- the price is that a moving head stays
inside a slightly loose box, which is why the padding is applied.

Alignment
---------
Both streams are addressed by TIME, not by a hand-tuned offset: a clip starting
at frame ``f`` covers ``[f/fps, (f + T)/fps)`` seconds, and the respiration
slice is ``np.interp`` of that window onto the ``[L] = clip_seconds * resp_fs``
grid. With the corpus' nominal rates (25 fps / 1000 Hz, one respiration sample
per 40 frames) this is an exact integer slice; the interpolation only matters if
a session is resampled. Verified: ``Resp_Volts.txt`` holds 64597 samples over
the 64.48 s of F001_T1 = 1001.8 Hz, i.e. the 1000 Hz nominal rate.

Clips are non-overlapping by default (``clip_stride=None`` -> one clip per
``clip_seconds``); pass ``clip_stride`` (seconds) for a hop. Sessions are never
split here -- subject-disjoint splitting is the caller's job.
"""
import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:                                          # `python -m data.tir_resp_dataset`
    from . import video_io as vio
    from . import task_groups as tgrp
except ImportError:                           # `python data/tir_resp_dataset.py`
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data import video_io as vio
    from data import task_groups as tgrp

__all__ = ['NUM_LANDMARKS', 'TARGET_LANDMARKS', 'TARGET_LANDMARK_IDX',
           'TARGET_LANDMARK_NAMES', 'DEFAULT_CLIP_SECONDS', 'DEFAULT_FPS',
           'DEFAULT_RESP_FS', 'parse_ir_features', 'missing_frame_mask',
           'roi_box_from_landmarks', 'discover_sessions', 'find_thermal_video',
           'find_ir_features', 'find_resp_file', 'default_raw_root',
           'ROI_LANDMARK_PRESETS', 'resolve_roi_landmarks',
           'DEFAULT_ROI_QUANTILE', 'RESP_RAIL_V',
           'BP4DPlusTIRRespDataset', 'TirRoiRespPretrainDataset',
           'TirRoiRespFinetuneDataset',
           'build_tir_roi_pretrain_dataset', 'build_tir_roi_finetune_dataset',
           'check_view_parity']

#: 28 landmarks per frame in the IRFeatures track (user-guide Figure 3).
NUM_LANDMARKS = 28
IR_VALUES_PER_FRAME = 2 * NUM_LANDMARKS

#: 1-INDEXED labels (user-guide Figure 3) of the mouth + nose ROI: nose-bridge
#: sides (9/20), nostril wings (10/21), mouth corners (11/22), upper (12/23) and
#: lower (13/24) lips, upper- (25) and lower-lip (26) centres. 12 points.
TARGET_LANDMARKS = (9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26)
TARGET_LANDMARK_IDX = tuple(i - 1 for i in TARGET_LANDMARKS)
TARGET_LANDMARK_NAMES = ('9 nose bridge R', '10 nostril R', '11 mouth corner R',
                         '12 upper lip R', '13 lower lip R',
                         '20 nose bridge L', '21 nostril L',
                         '22 mouth corner L', '23 upper lip L',
                         '24 lower lip L', '25 upper-lip centre',
                         '26 lower-lip centre')
assert len(TARGET_LANDMARK_IDX) == 12, 'the ROI is a 12-point landmark set'
assert all(0 <= i < NUM_LANDMARKS for i in TARGET_LANDMARK_IDX)

DEFAULT_CLIP_SECONDS = 8.0
DEFAULT_FPS = 25.0
DEFAULT_RESP_FS = 1000.0
DEFAULT_INPUT_SIZE = 64
DEFAULT_ROI_PADDING = 0.1

#: 0.0 reproduces the historical box EXACTLY (min/max over all frames of the
#: clip). A value in (0, 0.5) clips each side to the ``q`` / ``1-q`` PERCENTILE
#: of the landmark cloud, so a few motion outliers can no longer inflate it.
#: Motivation (measured 2026-09-26): the min/max union box grows with head
#: motion; sessions with a SMALL box + low landmark jitter reconstruct at
#: Pearson ~0.7 while large/jerky ones sit at ~0.0 (possible because the
#: inflated crop is mostly STATIC BACKGROUND, so the mean-pooled tokens encode
#: pose rather than nostril temperature).
DEFAULT_ROI_QUANTILE = 0.0

#: Named 1-indexed landmark sets for the ROI box (``--roi_landmarks``).
#: ``nose_mouth`` is the historical default; ``nostrils`` is the thermally
#: informative region for respiration and is a much smaller box, so head motion
#: sweeps less background through it.
ROI_LANDMARK_PRESETS = {
    'nose_mouth': TARGET_LANDMARKS,      # historical 12-point default
    'nostrils': (10, 21),                # nostril wings only (2 points)
    'nostril_mouth': (10, 11, 21, 22),   # nostrils + mouth corners
    'nose_tip': (9, 10, 20, 21),         # nose-bridge sides + nostrils
}


def resolve_roi_landmarks(spec) -> Tuple[int, ...]:
    """``'nostrils'`` | ``'10,21'`` | ``(10, 21)`` -> a 1-INDEXED tuple.

    ``None`` / ``''`` -> :data:`TARGET_LANDMARKS`, i.e. unchanged behaviour.
    """
    if spec is None:
        return tuple(TARGET_LANDMARKS)
    if isinstance(spec, str):
        s = spec.strip()
        if not s:
            return tuple(TARGET_LANDMARKS)
        if s.lower() in ROI_LANDMARK_PRESETS:
            return tuple(ROI_LANDMARK_PRESETS[s.lower()])
        try:
            vals = tuple(int(x) for x in s.replace(' ', '').split(',') if x)
        except ValueError:
            raise ValueError(
                f'roi_landmarks {spec!r}: expected a preset name from '
                f'{sorted(ROI_LANDMARK_PRESETS)} or a comma-separated list of '
                f'1-indexed landmark labels, e.g. "10,21".')
        if not vals:
            return tuple(TARGET_LANDMARKS)
    else:
        vals = tuple(int(x) for x in spec)
    bad = [v for v in vals if not 1 <= v <= NUM_LANDMARKS]
    if bad:
        raise ValueError(f'roi_landmarks {spec!r}: labels out of range '
                         f'1..{NUM_LANDMARKS}: {bad}')
    return vals


#: raw tree names
RAW_TREE = 'Thermal'
IR_TREE = 'IRFeatures'
PHYS_TREE = 'Physiology'
RESP_FILE = 'Resp_Volts.txt'
VIDEO_EXTS = ('.wmv', '.avi', '.mp4', '.mkv', '.mov')

#: a landmark pair at exactly (0, 0) is the missing-data sentinel
IR_SENTINEL = 0.0

#: the corpus' ``Resp_Volts.txt`` DEAD-CHANNEL floor in volts, signed as
#: stored: a saturated channel reads exactly ``-10.0``. This is the reference
#: the RAIL DIAGNOSTICS measure against (``analysis/tir_resp/rail_forensics.py``
#: and ``runners/run_survey_tir_resp.py``) -- it is NOT the clip filter. The
#: filter is :paramref:`BP4DPlusTIRRespDataset.rail_touch_v`, a symmetric
#: MAGNITUDE, because the rail is 2-sided (the recorder clamps at both ends).
#:
#: NOTE ``-10.0`` is EXACTLY representable in float32 -- the dtype ``_load_1d``
#: returns -- while ``-9.999`` is NOT: it rounds to ``-9.9989996``, so a
#: ``<= -9.999`` test on float32 data silently misses every railed sample.
#: The same trap applies to the FILTER threshold (see :meth:`__init__`).
RESP_RAIL_V = -10.0

_TASK_RE = re.compile(r'^[Tt](\d+)$')


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def _repo_root() -> str:
    """``.../MA-project-KISMED`` from ``.../code/data/tir_resp_dataset.py``."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def default_raw_root() -> str:
    """``$RAW_DATA_PATH`` when set, else the in-repo raw BP4D root."""
    return os.environ.get('RAW_DATA_PATH') or os.path.join(
        _repo_root(), 'data', 'raw', 'BP4D')


def _task_key(task: str):
    m = _TASK_RE.match(str(task))
    return (0, int(m.group(1))) if m else (1, str(task))


def find_thermal_video(raw_root: str, subject: str, task: str) -> Optional[str]:
    """Resolve ``Thermal/<S>/<T>.wmv`` (and the ``<S>_<T>`` / flat variants)."""
    subj_dir = os.path.join(raw_root, RAW_TREE, subject)
    bases = [os.path.join(subj_dir, str(task)),                    # Thermal/F001/T1.wmv
             os.path.join(subj_dir, f'{subject}_{task}'),          # Thermal/F001/F001_T1.wmv
             os.path.join(raw_root, RAW_TREE, f'{subject}_{task}')]  # Thermal/F001_T1.wmv
    for base in bases:
        if os.path.isfile(base) and os.path.splitext(base)[1]:
            return base
        for ext in VIDEO_EXTS:
            cand = base + ext
            if os.path.isfile(cand):
                return cand
    return None


def find_ir_features(raw_root: str, subject: str, task: str) -> Optional[str]:
    """``<raw_root>/IRFeatures/<S>_<T>.txt`` (flat, like ``Thermal/``)."""
    path = os.path.join(raw_root, IR_TREE, f'{subject}_{task}.txt')
    return path if os.path.isfile(path) else None


def find_resp_file(raw_root: str, subject: str, task: str) -> Optional[str]:
    """``<raw_root>/Physiology/<S>/<T>/Resp_Volts.txt``."""
    path = os.path.join(raw_root, PHYS_TREE, subject, str(task), RESP_FILE)
    return path if os.path.isfile(path) else None


def discover_sessions(raw_root: str, subjects: Optional[Sequence[str]] = None,
                      tasks: Optional[Sequence[str]] = None) -> List[dict]:
    """Every ``(subject, task)`` that has a thermal video, with resolved paths.

    Discovery keys on the VIDEO (that is the modality the dataset predicts
    from); a missing ``IRFeatures``/``Resp_Volts`` file is not filtered here --
    the dataset records it as a skip so ``--list`` can show why.
    """
    if not os.path.isdir(raw_root):
        raise FileNotFoundError(f'raw_root does not exist: {raw_root}')
    th = os.path.join(raw_root, RAW_TREE)
    if not os.path.isdir(th):
        raise FileNotFoundError(f'no {RAW_TREE}/ tree under {raw_root}')
    want_subj = {str(s) for s in subjects} if subjects else None
    want_task = {str(t) for t in tasks} if tasks else None

    out = []
    for subj in sorted(os.listdir(th)):
        subj_dir = os.path.join(th, subj)
        if not os.path.isdir(subj_dir) or (want_subj and subj not in want_subj):
            continue
        videos: Dict[str, str] = {}
        for name in sorted(os.listdir(subj_dir)):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in VIDEO_EXTS:
                continue
            task = stem[len(subj) + 1:] if stem.startswith(subj + '_') else stem
            if want_task and task not in want_task:
                continue
            videos.setdefault(task, os.path.join(subj_dir, name))
        for task in sorted(videos, key=_task_key):
            out.append({
                'session': f'{subj}_{task}',
                'subject': subj,
                'task': task,
                'video': videos[task],
                'ir_file': find_ir_features(raw_root, subj, task),
                'resp_file': find_resp_file(raw_root, subj, task),
            })
    if not out:
        raise FileNotFoundError(
            f'No thermal videos under {th}. Check --raw_root / $RAW_DATA_PATH.')
    return out


# --------------------------------------------------------------------------- #
# IRFeatures parsing
# --------------------------------------------------------------------------- #
def parse_ir_features(path: str, num_landmarks: int = NUM_LANDMARKS,
                      strict: bool = True) -> np.ndarray:
    """Parse an ``IRFeatures/<S>_<T>.txt`` track -> ``(num_frames, 28, 2)``.

    One line per thermal frame; line index is preserved EXACTLY, because the
    dataset relies on ``line n == frame n`` (1-based).

    * a blank line, or a line whose 56 tokens are not all numeric, becomes an
      all-NaN row -- the frame is "present but unusable", so clips covering it
      are dropped instead of the whole file being lost;
    * a line with any OTHER token count means the line/frame correspondence can
      no longer be trusted -> ``ValueError`` (the same call the thermal
      inspector makes).

    :param strict: when ``False``, wrong token counts NaN-fill as well (only for
        diagnostics -- never for training data).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f'IRFeatures file not found: {path}')
    rows: List[np.ndarray] = []
    with open(path, 'r') as fh:
        for lineno, raw in enumerate(fh, start=1):
            parts = raw.split()
            if not parts:
                rows.append(np.full((num_landmarks, 2), np.nan, np.float32))
                continue
            if len(parts) != 2 * num_landmarks:
                if strict:
                    raise ValueError(
                        f'{path}: line {lineno} holds {len(parts)} values, '
                        f'expected {2 * num_landmarks} '
                        f'({num_landmarks} (x, y) pairs) -- refusing to guess '
                        f'which frame it belongs to')
                rows.append(np.full((num_landmarks, 2), np.nan, np.float32))
                continue
            try:
                vals = np.asarray([float(p) for p in parts], dtype=np.float32)
            except ValueError:
                rows.append(np.full((num_landmarks, 2), np.nan, np.float32))
                continue
            rows.append(vals.reshape(num_landmarks, 2))
    if not rows:
        raise ValueError(f'{path}: no frames (empty file)')
    return np.stack(rows, axis=0).astype(np.float32)


def missing_frame_mask(ir: np.ndarray,
                       target_idx: Sequence[int] = TARGET_LANDMARK_IDX) -> np.ndarray:
    """Per-frame "this frame cannot be used" mask -> ``bool[num_frames]``.

    True when the frame is

    * an all-``(0, 0)`` line (the missing-data sentinel), or
    * unparseable (an all-NaN row from :func:`parse_ir_features`), or
    * missing a TARGET landmark at ``(0, 0)`` -- a safety net: a single such
      pair would stretch the clip's min/max ROI box to the image corner.

    Requirement: a clip overlapping ANY of these is dropped entirely.
    """
    ir = np.asarray(ir, dtype=np.float32)
    if ir.ndim != 3 or ir.shape[1:] != (NUM_LANDMARKS, 2):
        raise ValueError(f'expected (frames, {NUM_LANDMARKS}, 2), got {ir.shape}')
    finite = np.isfinite(ir).all(axis=(1, 2))
    all_zero = np.all(ir == IR_SENTINEL, axis=(1, 2))
    idx = np.asarray(list(target_idx), dtype=np.int64)
    tgt = ir[:, idx, :]
    tgt_sentinel = np.all(tgt == IR_SENTINEL, axis=2).any(axis=1)
    return ~finite | all_zero | tgt_sentinel


def _clamp_box(a: int, b: int, limit: int, min_size: int = 2) -> Tuple[int, int]:
    """Clamp half-open ``[a, b)`` into ``[0, limit)`` with a minimum size."""
    a = max(0, min(int(a), limit - 1))
    b = max(a + 1, min(int(b), limit))
    if b - a < min_size:
        grow = min_size - (b - a)
        a = max(0, a - (grow + 1) // 2)
        b = min(limit, a + min_size)
        a = max(0, b - min_size)
    return a, b


def roi_box_from_landmarks(pts: np.ndarray, width: int, height: int,
                           padding: float = DEFAULT_ROI_PADDING,
                           min_size: int = 2,
                           quantile: float = DEFAULT_ROI_QUANTILE
                           ) -> Tuple[int, int, int, int]:
    """Clip-level STATIC ROI box ``(x0, x1, y0, y1)`` (half-open slice bounds).

    :param pts: ``(..., 2)`` landmark coordinates in SOURCE pixels (the frames
        of ONE clip; the caller passes the target points of every frame).
    :param width, height: source frame size the box is clamped to.
    :param padding: fraction of the box EXTENT added on EACH side, so
        ``0.2`` grows the box by 40 % overall (``0.1`` would give exactly the
        1.2x reading of "expand by 20 %").
    :param quantile: ``0.0`` (default) = min/max over all frames, i.e. the
        historical behaviour, bit-for-bit. A value in ``(0, 0.5)`` instead
        clips each side to the ``quantile`` / ``1 - quantile`` PERCENTILE of
        the landmark cloud, which makes the box robust to head-motion outliers
        (see :data:`DEFAULT_ROI_QUANTILE` for the measured motivation).
    """
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    p = p[np.isfinite(p).all(axis=1)]
    if p.size == 0:
        raise ValueError('no finite landmark coordinates in this clip')
    q = float(min(max(quantile, 0.0), 0.49))
    if q > 0.0:
        lo, hi = 100.0 * q, 100.0 * (1.0 - q)
        x0, x1 = [float(v) for v in np.percentile(p[:, 0], [lo, hi])]
        y0, y1 = [float(v) for v in np.percentile(p[:, 1], [lo, hi])]
    else:
        x0, x1 = float(p[:, 0].min()), float(p[:, 0].max())
        y0, y1 = float(p[:, 1].min()), float(p[:, 1].max())
    w, h = x1 - x0, y1 - y0
    x0 -= padding * w
    x1 += padding * w
    y0 -= padding * h
    y1 += padding * h
    ix0, ix1 = _clamp_box(math.floor(x0), math.ceil(x1) + 1, int(width), min_size)
    iy0, iy1 = _clamp_box(math.floor(y0), math.ceil(y1) + 1, int(height), min_size)
    return ix0, ix1, iy0, iy1


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _count_samples(path: str) -> int:
    n = 0
    with open(path, 'r') as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def _load_1d(path: str) -> np.ndarray:
    """Load a one-value-per-line physiology file as a float32 array."""
    vals = []
    with open(path, 'r') as fh:
        for line in fh:
            tok = line.strip()
            if not tok:
                continue
            try:
                vals.append(float(tok.split()[0]))
            except ValueError:
                vals.append(np.nan)
    return np.asarray(vals, dtype=np.float32)


def _resp_start_limit(n_samples: int, resp_fs: float, fps: float,
                      resp_len: int) -> int:
    """Largest clip start FRAME whose ``resp_len`` samples are fully covered.

    ``-1`` means not even a clip at frame 0 fits (signal too short / too slow).
    """
    if n_samples <= 0:
        return -1
    t_last = (resp_len - 1) / float(resp_fs)
    t_max = (n_samples - 1) / float(resp_fs)
    return int(math.floor((t_max - t_last) * float(fps) + 1e-9))


def _cv2():
    import cv2
    return cv2


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class BP4DPlusTIRRespDataset(Dataset):
    """Thermal clip -> respiration waveform, over the raw BP4D+ tree.

    :param raw_root: raw BP4D root (default :func:`default_raw_root`, i.e.
        ``$RAW_DATA_PATH`` or ``<repo>/data/raw/BP4D``).
    :param subjects, tasks: optional subset (e.g. ``['F001']``, ``['T1','T2']``).
    :param task_groups: named task GROUPS ("distortion levels") as a mapping
        ``{'low': ['T1', 'T2'], ...}`` or the ``'low=T1|T2;high=T7'`` string
        form. No mapping is hardcoded -- the definitions come from the config
        (see :mod:`data.task_groups`).
    :param task_set: the group name(s) to SELECT (``'low'`` / ``'low,high'`` /
        ``['low','high']``); ``None``/``''``/``'all'`` = no group filter. The
        final task list is the UNION of the resolved groups and ``tasks``.
    :param clip_seconds: clip length in seconds (8 s -> 200 frames / 8000
        samples); ``clip_frames = round(clip_seconds * fps)``.
    :param fps: NOMINAL frame rate used for the frame<->time mapping (25 for
        BP4D thermal). The measured rate of each file is recorded in
        ``self.session_fps`` and a deviation > 5 % is reported in
        ``self.warnings``.
    :param resp_fs: respiration sample rate (1000 Hz nominal).
    :param input_size: ROI patch side after ``cv2.resize``.
    :param clip_stride: clip hop in SECONDS (``None``/``0`` -> non-overlapping,
        i.e. ``clip_stride = clip_seconds``).
    :param roi_padding: per-side ROI margin, see :func:`roi_box_from_landmarks`.
    :param target_landmarks: 1-indexed landmark labels forming the ROI.
    :param min_signal_spread: DROP a clip whose respiration window is
        (near-)constant, i.e. ``max(window) - min(window) < min_signal_spread``
        in raw VOLTS (``0.0`` = disabled). **The shipped ``0.1`` is the
        DEAD-SIGNAL guard**: measured on the corpus, every window with a
        spread below 0.1 V sits with a median level of ``+9.1 V`` -- i.e. the
        channel is PINNED just inside the ``+/-10 V`` clamp with a
        0.05-0.09 V ripple (a disconnected belt / loose contact). Such a
        window is not a breathing target: under ``norm='clip'`` / the Stage-3
        ``physio_norm: zscore`` it is rescaled to UNIT variance, so it enters
        training as a full-amplitude label that is pure artifact. Genuine
        breathing is >= 0.3 V p-p (corpus p10 ~0.75 V), so the guard does not
        reach real (even shallow) respiration. NOTE this rule is per WINDOW and
        is applied AFTER :paramref:`rail_touch_v`: the two rules are
        COMPLEMENTARY -- the rail test catches what touches a clamp, this one
        what is pinned near it without touching (see
        ``analysis/tir_resp/resp_data_quality.md`` §3.4 and
        ``analysis/tir_resp/resp_rail_touch_filter_results.md`` §5).
    :param rail_touch_v: DROP a clip whose respiration window contains ANY
        sample that has TOUCHED THE RAIL, i.e. ``abs(sample) >= rail_touch_v``
        volts (``0.0`` = disabled, the default). **This is the cleaning knob for
        the TIR-ROI/RESP study** and it is deliberately STRICT: the recorder's
        full-scale clamp is ``+/-10.0 V`` (see :data:`RESP_RAIL_V`), so the
        shipped ``9.90`` rejects a window as soon as a single reading reached
        within 0.1 V of either end of the range.

        Why a MAGNITUDE and why a TOUCH test -- the ``-10 V`` rail has two
        different causes (see ``analysis/tir_resp/resp_data_quality.md`` §3.4
        and ``analysis/tir_resp/rail_forensics.py``):

        * a SENSOR FAILURE (dead / disconnected belt, a stuck DAQ, or a DC
          offset beyond the recorder's range) PINS the channel at the rail;
        * a GENUINE clipped expiration drives the trough into the clamp, so the
          waveform is flat-topped at the rail until the breath recovers.

        BOTH leave a rail-valued sample in the label, which is what this rule
        keys on:

        * untouched window -> kept  (the normal case);
        * any sample at ``|x| >= rail_touch_v`` -> dropped (clipped OR dead).

        The cost is explicit and intended: clips that merely graze the clamp
        are removed together with the dead-channel ones, so the retained corpus
        contains no rail-valued label at all. The test is applied to the SAME
        interpolated window that becomes the target (see ``_build_entries``),
        so "touched" means "the label contains a rail-valued sample".
        Must match between the Stage-2 and Stage-3 runs.
    :param norm: ``'clip'`` (per-clip z-score, default), ``'session'`` (z-score
        with the whole session's statistics) or ``'none'`` (raw volts).
    :param max_clips_per_session, max_entries: dev caps (smoke runs).
    :param preload: materialise every clip tensor at init (small -- a 64 px
        clip is 2.5 MB -- and makes repeated epochs cheap). Off by default.
    :param allow_empty: allow a dataset with zero clips (diagnostics only).
    """

    def __init__(self, raw_root: Optional[str] = None,
                 subjects: Optional[Sequence[str]] = None,
                 tasks: Optional[Sequence[str]] = None,
                 task_groups=None, task_set=None,
                 clip_seconds: float = DEFAULT_CLIP_SECONDS,
                 fps: float = DEFAULT_FPS,
                 resp_fs: float = DEFAULT_RESP_FS,
                 input_size: int = DEFAULT_INPUT_SIZE,
                 clip_stride: Optional[float] = None,
                 roi_padding: float = DEFAULT_ROI_PADDING,
                 roi_quantile: float = DEFAULT_ROI_QUANTILE,
                 target_landmarks: Sequence[int] = TARGET_LANDMARKS,
                 min_signal_spread: float = 0.0,
                 rail_touch_v: float = 0.0,
                 norm: str = 'clip',
                 max_clips_per_session: Optional[int] = None,
                 max_entries: Optional[int] = None,
                 preload: bool = False,
                 allow_empty: bool = False,
                 verbose: bool = False):
        if norm not in ('clip', 'session', 'none'):
            raise ValueError(f"norm must be 'clip', 'session' or 'none', got {norm!r}")
        if clip_seconds <= 0 or fps <= 0 or resp_fs <= 0:
            raise ValueError('clip_seconds, fps and resp_fs must be positive')
        self.raw_root = raw_root or default_raw_root()
        self.clip_seconds = float(clip_seconds)
        self.fps = float(fps)
        self.resp_fs = float(resp_fs)
        self.clip_frames = max(1, int(round(self.clip_seconds * self.fps)))
        self.resp_len = max(1, int(round(self.clip_seconds * self.resp_fs)))
        self.stride_frames = (self.clip_frames if not clip_stride
                              else max(1, int(round(float(clip_stride) * self.fps))))
        self.input_size = int(input_size)
        self.roi_padding = float(roi_padding)
        self.roi_quantile = float(roi_quantile)
        if not 0.0 <= self.roi_quantile < 0.5:
            raise ValueError(f'roi_quantile must be in [0, 0.5), got '
                             f'{roi_quantile!r} (0 = min/max, the historical '
                             f'behaviour; 0.05 = 5th/95th percentile)')
        self.target_landmarks = tuple(int(l) for l in target_landmarks)
        self.target_idx = np.asarray([l - 1 for l in self.target_landmarks],
                                    dtype=np.int64)
        self.norm = norm
        self.min_signal_spread = float(min_signal_spread)
        if self.min_signal_spread < 0.0:
            raise ValueError(f'min_signal_spread must be >= 0 (volts); got '
                             f'{min_signal_spread!r}')
        # The threshold is quantised to float32 ONCE, because the samples it is
        # compared against are float32-exact (``_load_1d`` -> float32): a
        # decimal literal such as 9.9 is 9.90000000000000036 as a float64, so a
        # window sample stored as 9.9000 (float32 9.89999961853027) would NOT
        # satisfy ``>= 9.9`` -- exactly the trap that makes ``-9.999``
        # invisible to a ``<= -9.999`` test on float32 data. Quantising the
        # threshold makes ``>=`` match a sample OF the threshold's own value.
        self.rail_touch_v = float(np.float32(rail_touch_v))
        if self.rail_touch_v < 0.0:
            raise ValueError(f'rail_touch_v must be >= 0 (volts); got '
                             f'{rail_touch_v!r} (0.0 = disabled)')
        # sample offsets of one clip on the respiration grid (flat-window test)
        self._resp_offsets = np.arange(self.resp_len)
        self.preload = bool(preload)
        self.verbose = bool(verbose)

        # Named task GROUPS ("distortion levels") -> explicit task labels. The
        # DEFINITIONS come from the YAML config (nothing is hardcoded); the
        # selection is UNIONED with the explicit ``tasks`` list and the result
        # feeds the SAME ``discover_sessions`` filter used before, so nothing
        # downstream has to know about groups.
        self.task_selection = tgrp.resolve_task_selection(
            tasks=tasks, task_set=task_set, task_groups=task_groups)

        self.entries: List[dict] = []
        self.sessions: Dict[str, dict] = {}
        self.skipped: List[dict] = []
        self.warnings: List[str] = []
        self.session_fps: Dict[str, float] = {}
        self.stats: Dict[str, object] = {}

        self._ir_cache: Dict[str, np.ndarray] = {}
        self._resp_cache: Dict[str, np.ndarray] = {}
        self._norm_cache: Dict[str, Tuple[float, float]] = {}
        self._preloaded: List[dict] = []

        discovered = discover_sessions(self.raw_root, subjects=subjects,
                                       tasks=self.task_selection.tasks)
        self._build_entries(discovered, max_clips_per_session, max_entries)
        if not self.entries and not allow_empty:
            reasons = '; '.join(f"{s['session']}: {s['reason']}" for s in self.skipped)
            raise RuntimeError(
                f'No usable clips under {self.raw_root} for '
                f'clip_seconds={self.clip_seconds} ({self.clip_frames} frames). '
                f'Skipped: {reasons or "none"}')
        if self.preload:
            self._preload()

    # ------------------------------------------------------------------ build
    def _build_entries(self, discovered: List[dict],
                       max_clips_per_session: Optional[int],
                       max_entries: Optional[int]) -> None:
        n_dropped_zero = 0
        n_dropped_flat = 0
        n_dropped_rail = 0
        for spec in discovered:
            sess, subj, task = spec['session'], spec['subject'], spec['task']

            # --- 1. IRFeatures gate (requirement 1: skip untracked sequences)
            if spec['ir_file'] is None:
                self._skip(spec, 'missing_ir_features')
                continue
            try:
                ir = parse_ir_features(spec['ir_file'])
            except Exception as exc:                       # malformed track
                self._skip(spec, f'invalid_ir_features: {exc}')
                continue
            n_ir = int(ir.shape[0])

            # --- 2. respiration stream
            if spec['resp_file'] is None:
                self._skip(spec, 'missing_resp_volts')
                continue
            n_sig = _count_samples(spec['resp_file'])
            start_limit = _resp_start_limit(n_sig, self.resp_fs, self.fps,
                                            self.resp_len)
            if start_limit < 0:
                self._skip(spec, f'resp_too_short: {n_sig} samples < '
                                 f'{self.resp_len} needed')
                continue

            # --- 3. thermal video (container probe only; frames stay lazy)
            try:
                reader = vio.open_video(spec['video'])
                try:
                    n_vid, fps_meas = int(reader.num_frames), float(reader.fps)
                finally:
                    reader.close()
            except Exception as exc:
                self._skip(spec, f'undecodable_video: {exc}')
                continue
            self.session_fps[sess] = fps_meas
            if fps_meas > 0 and abs(fps_meas - self.fps) / self.fps > 0.05:
                self.warnings.append(
                    f'{sess}: measured fps {fps_meas:.4f} deviates from the '
                    f'nominal {self.fps:.4f} used for slicing')

            n_common = min(n_vid, n_ir)
            if n_vid != n_ir:
                self.warnings.append(
                    f'{sess}: thermal video has {n_vid} frames but IRFeatures '
                    f'has {n_ir} lines -> using the first {n_common}')
            n_avail = min(n_common, start_limit + self.clip_frames)
            if n_avail < self.clip_frames:
                self._skip(spec, f'too_short: {n_common} common frames < '
                                 f'{self.clip_frames} needed')
                continue

            bad = missing_frame_mask(ir, self.target_idx)
            # read the raw series when EITHER window-content rule is active
            needs_series = (self.min_signal_spread > 0.0
                            or self.rail_touch_v > 0.0)
            y_raw = _load_1d(spec['resp_file']) if needs_series else None

            # --- 4. slice clips (requirement 2: any sentinel line drops the clip)
            kept, dropped, dropped_flat, dropped_rail, starts = 0, 0, 0, 0, []
            for start in range(0, n_avail - self.clip_frames + 1,
                               self.stride_frames):
                if bad[start:start + self.clip_frames].any():
                    dropped += 1
                    continue
                if y_raw is not None:
                    # raw window on the respiration grid, exactly as
                    # _resp_clip() would build it (interp over the file)
                    s0 = (start / self.fps) * self.resp_fs
                    w = np.interp(s0 + self._resp_offsets,
                                  np.arange(y_raw.shape[0], dtype=np.float64),
                                  y_raw.astype(np.float64))
                    # RAIL FIRST: a fully railed window is also "flat", so the
                    # probe order decides which rule gets the credit -- else
                    # clips_dropped_rail would undercount by exactly those.
                    # TOUCH TEST: ONE sample at +/-rail_touch_v or beyond
                    # drops the clip, so neither a dead channel nor a clipped
                    # flat-topped trough survives.
                    if self.rail_touch_v > 0.0 and w.size and \
                            bool(np.any(np.abs(w) >= self.rail_touch_v)):
                        dropped_rail += 1
                        continue
                    if self.min_signal_spread > 0.0 and \
                            float(w.max() - w.min()) < self.min_signal_spread:
                        dropped_flat += 1
                        continue
                starts.append(start)
                kept += 1
                if max_clips_per_session and kept >= max_clips_per_session:
                    break
            n_dropped_zero += dropped
            n_dropped_flat += dropped_flat
            n_dropped_rail += dropped_rail
            if not starts:
                why = [f'{dropped} window(s) contain a (0,0) sentinel line']
                if dropped_flat:
                    why.append(f'{dropped_flat} have a (near-)constant '
                               f'respiration window (< {self.min_signal_spread:g} V)')
                if dropped_rail:
                    why.append(f'{dropped_rail} touch the rail '
                               f'(|x| >= {self.rail_touch_v:g} V)')
                self._skip(spec, 'all_clips_dropped: ' + '; '.join(why))
                continue

            self.sessions[sess] = dict(spec, n_frames=n_common, n_ir=n_ir,
                                       n_vid=n_vid, n_resp=n_sig,
                                       resp_start_limit=start_limit,
                                       clips_dropped_sentinel=dropped,
                                       clips_dropped_flat=dropped_flat,
                                       clips_dropped_rail=dropped_rail)
            for start in starts:
                self.entries.append(self._make_entry(spec, start))
                if max_entries and len(self.entries) >= max_entries:
                    break
            if max_entries and len(self.entries) >= max_entries:
                break

        # per-level roll-up + a warning when a SELECTED task produced no clips
        # (a partial corpus is normal, so this warns instead of failing)
        clips_per_level: Dict[str, int] = {}
        sessions_per_level: Dict[str, int] = {}
        for entry in self.entries:
            for lv in entry.get('task_level', ()):
                clips_per_level[lv] = clips_per_level.get(lv, 0) + 1
        for spec in self.sessions.values():
            for lv in self.task_selection.levels_of(spec['task']):
                sessions_per_level[lv] = sessions_per_level.get(lv, 0) + 1
        if self.task_selection.tasks and not max_entries:
            present = {e['task'] for e in self.entries}
            missing = [t for t in self.task_selection.tasks if t not in present]
            if missing:
                self.warnings.append(
                    f'selected task(s) {missing} produced 0 clips under '
                    f'{self.raw_root} (not downloaded, all windows dropped, or '
                    f'too short) -- the run uses the remaining task(s) only')

        self.stats = {
            'raw_root': self.raw_root,
            'sessions_discovered': len(discovered),
            'sessions_used': len(self.sessions),
            'sessions_skipped': len(self.skipped),
            'clips': len(self.entries),
            'clips_dropped_sentinel': n_dropped_zero,
            'clips_dropped_flat': n_dropped_flat,
            'clips_dropped_rail': n_dropped_rail,
            'min_signal_spread': self.min_signal_spread,
            'rail_touch_v': self.rail_touch_v,
            'rail_v': RESP_RAIL_V,
            'clip_seconds': self.clip_seconds,
            'clip_frames': self.clip_frames,
            'clip_stride_seconds': self.stride_frames / self.fps,
            'clip_stride_frames': self.stride_frames,
            'fps': self.fps,
            'resp_fs': self.resp_fs,
            'resp_len': self.resp_len,
            'input_size': self.input_size,
            'roi_padding': self.roi_padding,
            'roi_quantile': self.roi_quantile,
            'target_landmarks': list(self.target_landmarks),
            'norm': self.norm,
            'tasks': (list(self.task_selection.tasks)
                      if self.task_selection.tasks else None),
            'task_set': list(self.task_selection.requested),
            'task_groups': {k: list(v) for k, v in
                            self.task_selection.levels.items()},
            'clips_per_level': clips_per_level,
            'sessions_per_level': sessions_per_level,
            'warnings': list(self.warnings),
            'skipped': list(self.skipped),
        }

    def _make_entry(self, spec: dict, start: int) -> dict:
        t0 = start / self.fps
        return {
            'session': spec['session'],
            'subject': spec['subject'],
            'task': spec['task'],
            'frame_start': start,
            'frame_end': start + self.clip_frames,
            't_start': t0,
            't_end': t0 + self.clip_seconds,
            'resp_start': t0 * self.resp_fs,
            'video': spec['video'],
            'ir_file': spec['ir_file'],
            'resp_file': spec['resp_file'],
            # the group name(s) ("distortion level") this task belongs to, e.g.
            # ['low']; [] when no defined group mentions the task.
            'task_level': list(self.task_selection.levels_of(spec['task'])),
        }

    def _skip(self, spec: dict, reason: str) -> None:
        self.skipped.append({'session': spec['session'],
                             'subject': spec['subject'],
                             'task': spec['task'],
                             'reason': reason,
                             'video': spec['video'],
                             'ir_file': spec['ir_file'],
                             'resp_file': spec['resp_file']})
        if self.verbose:
            print(f'[tir_resp] skip {spec["session"]}: {reason}')

    # ------------------------------------------------------------------ caches
    def _landmarks(self, session: str) -> np.ndarray:
        if session not in self._ir_cache:
            self._ir_cache[session] = parse_ir_features(
                self.sessions[session]['ir_file'])
        return self._ir_cache[session]

    def _resp_full(self, session: str) -> np.ndarray:
        if session not in self._resp_cache:
            y = _load_1d(self.sessions[session]['resp_file'])
            if self.norm == 'session':
                self._norm_cache[session] = (float(np.nanmean(y)),
                                             float(np.nanstd(y)))
            self._resp_cache[session] = y
        return self._resp_cache[session]

    def _resp_clip(self, entry: dict) -> np.ndarray:
        """Resample the clip's respiration window onto the ``[L]`` target grid."""
        y = self._resp_full(entry['session'])
        src_fs = float(self.resp_fs)          # one sample per 1/1000 s, nominal
        idx = (entry['t_start'] + np.arange(self.resp_len) / self.resp_fs) * src_fs
        return np.interp(idx, np.arange(y.shape[0], dtype=np.float64),
                         y.astype(np.float64)).astype(np.float32)

    def _normalize(self, y: np.ndarray, session: str) -> np.ndarray:
        if self.norm == 'none':
            return y.astype(np.float32)
        if self.norm == 'session':
            mu, sd = self._norm_cache.get(session, (float(np.nanmean(y)),
                                                    float(np.nanstd(y))))
        else:                                  # per-clip (requirement)
            mu, sd = float(np.nanmean(y)), float(np.nanstd(y))
        return ((y - mu) / (sd + 1e-8)).astype(np.float32)

    def _frames(self, entry: dict) -> np.ndarray:
        """Decode the clip's frames as uint8 RGB ``[T, H, W, 3]``."""
        reader = vio.open_video(entry['video'])
        try:
            frames = reader.read_range(entry['frame_start'], self.clip_frames)
        finally:
            reader.close()
        frames = np.asarray(frames)
        if frames.ndim == 2:                                  # gray fallback
            frames = np.repeat(frames[..., None], 3, axis=2)
        if frames.shape[0] != self.clip_frames:
            raise IOError(f'{entry["session"]} frame {entry["frame_start"]}: '
                          f'decoded {frames.shape[0]} of '
                          f'{self.clip_frames} frames')
        return frames

    # ------------------------------------------------------------------ roi
    def clip_roi_box(self, index: int) -> Tuple[int, int, int, int]:
        """The static ROI box of entry ``index``, in SOURCE pixels."""
        entry = self.entries[index]
        ir = self._landmarks(entry['session'])
        pts = ir[entry['frame_start']:entry['frame_end']][:, self.target_idx, :]
        frames = self._frames(entry)
        h, w = frames.shape[1], frames.shape[2]
        return roi_box_from_landmarks(pts, w, h, self.roi_padding,
                                      quantile=self.roi_quantile)

    def clip_landmarks(self, index: int) -> np.ndarray:
        """All 28 landmarks of the clip's frames -> ``[T, 28, 2]`` (source px)."""
        entry = self.entries[index]
        ir = self._landmarks(entry['session'])
        return ir[entry['frame_start']:entry['frame_end']]

    def respiration_clip(self, index: int, normalized: bool = True) -> np.ndarray:
        """The clip's respiration window (``[L]``); raw volts when
        ``normalized=False`` -- what the ROI/response plots need."""
        entry = self.entries[index]
        y = self._resp_clip(entry)
        return self._normalize(y, entry['session']) if normalized else y

    def _roi_patches(self, entry: dict, frames: np.ndarray,
                     box: Tuple[int, int, int, int]) -> np.ndarray:
        cv2 = _cv2()
        x0, x1, y0, y1 = box
        s = self.input_size
        interp = (cv2.INTER_AREA if s <= min(x1 - x0, y1 - y0)
                  else cv2.INTER_LINEAR)
        out = np.empty((frames.shape[0], s, s, 3), np.uint8)
        for i in range(frames.shape[0]):
            out[i] = cv2.resize(frames[i, y0:y1, x0:x1], (s, s),
                                interpolation=interp)
        return out

    # ------------------------------------------------------------------ item
    def _load(self, index: int) -> dict:
        entry = self.entries[index]
        session = entry['session']
        ir = self._landmarks(session)
        pts = ir[entry['frame_start']:entry['frame_end']][:, self.target_idx, :]

        frames = self._frames(entry)
        h, w = frames.shape[1], frames.shape[2]
        box = roi_box_from_landmarks(pts, w, h, self.roi_padding,
                                     quantile=self.roi_quantile)
        patches = self._roi_patches(entry, frames, box)          # [T, s, s, 3]

        # [T, H, W, C] uint8 RGB -> [C, T, H, W] float in [0, 1]
        tir = torch.from_numpy(np.ascontiguousarray(patches.transpose(3, 0, 1, 2)))
        tir = tir.to(torch.float32).div_(255.0)

        resp = self.respiration_clip(index)
        return {'tir_video': tir,
                'resp_signal': torch.from_numpy(resp),
                'subject_task': session,
                '_entry': entry, '_roi_box': box}

    def _preload(self) -> None:
        self._preloaded = [self._load(i) for i in range(len(self.entries))]

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self.entries)
        if not 0 <= index < len(self.entries):
            raise IndexError(index)
        if self._preloaded:
            item = self._preloaded[index]
            return {'tir_video': item['tir_video'].clone(),
                    'resp_signal': item['resp_signal'].clone(),
                    'subject_task': item['subject_task']}
        item = self._load(index)
        return {'tir_video': item['tir_video'],
                'resp_signal': item['resp_signal'],
                'subject_task': item['subject_task']}

    def __len__(self) -> int:
        return len(self.entries)

    # ------------------------------------------------------------------ report
    def describe(self) -> str:
        """Human-readable summary (also the source of the JSON report fields)."""
        s = self.stats
        lines = [
            f'raw_root        : {s["raw_root"]}',
            f'sessions        : {s["sessions_used"]} used / '
            f'{s["sessions_discovered"]} found / {s["sessions_skipped"]} skipped',
            f'clips           : {s["clips"]} '
            f'({s["clips_dropped_sentinel"]} window(s) dropped for a (0,0) '
            f'sentinel line, {s["clips_dropped_flat"]} for a (near-)constant '
            f'respiration window at min_signal_spread={s["min_signal_spread"]:g} V, '
            f'{s["clips_dropped_rail"]} for touching the rail '
            f'(|x| >= {s["rail_touch_v"]:g} V; the recorder clamps at '
            f'{s["rail_v"]:g} V)',
            f'clip            : {s["clip_seconds"]:g} s = {s["clip_frames"]} '
            f'frames @ {s["fps"]:g} fps, stride {s["clip_stride_frames"]} '
            f'frames; resp {s["resp_len"]} samples @ {s["resp_fs"]:g} Hz',
            f'roi             : landmarks {list(self.target_landmarks)} '
            f'padding {s["roi_padding"]:g} -> {self.input_size}x{self.input_size}',
            f'norm            : {s["norm"]}',
            f'task selection  : {self.task_selection.describe()}',
        ]
        if s['clips_per_level']:
            lines.append('clips per level : ' + ', '.join(
                f'{k}={s["clips_per_level"][k]} clip(s)/'
                f'{s["sessions_per_level"].get(k, 0)} session(s)'
                for k in sorted(s['clips_per_level'])))
        if self.session_fps:
            fps_vals = ', '.join(f'{k}={v:.4f}' for k, v in
                                 sorted(self.session_fps.items())[:4])
            lines.append(f'measured fps    : {fps_vals}'
                         f'{" ..." if len(self.session_fps) > 4 else ""}')
        for w in self.warnings:
            lines.append(f'warning         : {w}')
        for sk in self.skipped:
            lines.append(f'skipped         : {sk["session"]} ({sk["reason"]})')
        return '\n'.join(lines)


# --------------------------------------------------------------------------- #
# Stage-2 (masked pre-training) view of the same clips
# --------------------------------------------------------------------------- #
def _as_list(value) -> Optional[List[str]]:
    """``None``/``''`` -> ``None``; ``'a,b'`` or ``['a','b']`` -> ``['a','b']``."""
    if value is None:
        return None
    if isinstance(value, str):
        items = [v.strip() for v in value.split(',') if v.strip()]
        return items or None
    items = [str(v) for v in value]
    return items or None


class TirRoiRespPretrainDataset(Dataset):
    """Stage-2 masked-pretraining dataset: thermal ROI -> respiration.

    Wraps :class:`BP4DPlusTIRRespDataset` (clip enumeration, ROI crop, raw
    respiration) and returns exactly the stream dict ``MultiModalMAE`` consumes::

        {'tir':  float32 [3, T, input_size, input_size]   ROI crop, [0, 1]
         'resp': float32 [1, L]                           RAW volts}

    The 1-D target is normalised HERE, under ``physio_norm`` -- the ONE knob of
    the whole pipeline (same flag name and values in Stage 2 and Stage 3, for
    TIR-ROI+RESP and RGB+BP alike). The MODEL never normalises a 1-D stream:
    ``run_pretrain.py`` pins ``target_norm='none'`` for every ROI lineage.

    * ``'none'`` -- raw values (debug only: nothing else normalises either).
    * ``'clip'`` (alias ``'zscore'``) -- per-clip z-score, one mean/std per
      clip, applied by ``BP4DPlusTIRRespDataset(norm='clip')``.
    * ``'session'`` -- z-scored with the WHOLE SESSION's mean/std (population
      statistics over the entire ``Resp_Volts.txt``, via
      ``BP4DPlusTIRRespDataset(norm='session')``), so a clip keeps its
      SESSION-relative amplitude and offset instead of being flattened to unit
      variance on its own. This is the target space the study uses.

    Geometry mirrors ``core.multimae.build_pretraining_model`` exactly::

        n = round(clip_duration * fps / temporal_stride)
        T = n rounded DOWN to a multiple of ``tubelet_t``  (>= tubelet_t)
        L = round((T * temporal_stride / fps) * fs)

    so ``grid_t = T / tubelet_t`` equals ``n_signal = L / sig_kernel`` and the
    model's hard space-time alignment check passes. ``tubelet_t`` MUST be the
    first component of the run's ``--tubelet``.

    This is an ADD-ON path: it does not touch ``PairedPretrainDataset``, and
    every existing ``rgb``/``bp`` config keeps using that class.
    """

    def __init__(self, raw_root: Optional[str] = None,
                 streams: Sequence[str] = ('tir', 'resp'),
                 fs: float = 100.0, fps: float = DEFAULT_FPS,
                 clip_duration: float = 4.0,
                 clip_stride: Optional[float] = None,
                 temporal_stride: int = 1,
                 tubelet_t: int = 2,
                 input_size: int = DEFAULT_INPUT_SIZE,
                 roi_padding: float = DEFAULT_ROI_PADDING,
                 roi_quantile: float = DEFAULT_ROI_QUANTILE,
                 target_landmarks: Sequence[int] = TARGET_LANDMARKS,
                 min_signal_spread: float = 0.0,
                 rail_touch_v: float = 0.0,
                 physio_norm: str = 'none',
                 resp_fs: float = DEFAULT_RESP_FS,
                 subjects=None, tasks=None,
                 task_groups=None, task_set=None,
                 max_clips_per_session: Optional[int] = None,
                 max_entries: Optional[int] = None,
                 verbose: bool = False):
        self.streams = tuple(str(s).strip() for s in streams if str(s).strip())
        if not self.streams:
            raise ValueError('TirRoiRespPretrainDataset: empty streams list')
        unknown = [s for s in self.streams if s not in ('tir', 'resp')]
        if unknown:
            raise ValueError(
                f'TirRoiRespPretrainDataset: unknown stream(s) {unknown}; this '
                f'path serves exactly (tir, resp).')
        if 'tir' not in self.streams or 'resp' not in self.streams:
            raise ValueError(
                'TirRoiRespPretrainDataset: Stage 2 needs BOTH streams -- the '
                f'thermal ROI visual + the respiration waveform; got '
                f'{self.streams}.')

        # WHERE the 1-D target is normalised -- the single knob of the pipeline
        # (see the class docstring). 'zscore' is accepted as an ALIAS of 'clip'
        # so the SAME value means the same thing in Stage 2 and Stage 3.
        self.physio_norm = str(physio_norm or 'none').strip().lower()
        if self.physio_norm == 'zscore':
            self.physio_norm = 'clip'
        if self.physio_norm not in ('none', 'clip', 'session'):
            raise ValueError(
                "TirRoiRespPretrainDataset: physio_norm must be 'none', '"
                f"clip\" ('zscore') or 'session'; got {physio_norm!r}")

        self.fs = float(fs)
        self.fps = float(fps)
        self.clip_duration = float(clip_duration)
        self.temporal_stride = max(1, int(temporal_stride))
        self.tubelet_t = max(1, int(tubelet_t))
        self.resp_fs = float(resp_fs)
        self.input_size = int(input_size)
        self.roi_padding = float(roi_padding)
        self.roi_quantile = float(roi_quantile)

        # ---- geometry, identical to build_pretraining_model ----------------
        n = max(1, int(round(self.clip_duration * self.fps / self.temporal_stride)))
        if n % self.tubelet_t:
            n -= n % self.tubelet_t
        self.num_frames = max(self.tubelet_t, n)
        kept_seconds = self.num_frames * self.temporal_stride / self.fps
        self.seq_len = max(1, int(round(kept_seconds * self.fs)))

        # ---- clips: the ROI + respiration base dataset (RAW values) -------
        self.base = BP4DPlusTIRRespDataset(
            raw_root=raw_root, subjects=_as_list(subjects),
            tasks=tasks, task_groups=task_groups, task_set=task_set,
            clip_seconds=self.clip_duration,
            fps=self.fps, resp_fs=self.resp_fs, input_size=self.input_size,
            clip_stride=clip_stride, roi_padding=self.roi_padding,
            roi_quantile=self.roi_quantile,
            target_landmarks=target_landmarks, norm=self.physio_norm,
            min_signal_spread=min_signal_spread,
            rail_touch_v=rail_touch_v,
            max_clips_per_session=max_clips_per_session,
            max_entries=max_entries, verbose=verbose)
        self.entries = self.base.entries
        self.stats = self.base.stats
        self.task_selection = self.base.task_selection

        # the respiration grid of one clip is fixed -> precompute the mapping
        self._resp_idx = (np.arange(self.seq_len) / self.fs) * self.resp_fs

    # ------------------------------------------------------------------ item
    def __getitem__(self, index: int) -> dict:
        item = self.base[index]                       # tir [3,nf,S,S] + volts
        tir = item['tir_video']
        if self.temporal_stride > 1:
            tir = tir[:, ::self.temporal_stride]      # decimate INSIDE the window
        tir = tir[:, :self.num_frames]                # drop the tubelet tail
        if tir.shape[1] != self.num_frames:
            raise RuntimeError(
                f'{self.entries[index]["session"]}: ROI clip has '
                f'{tir.shape[1]} frames, geometry needs {self.num_frames}')

        raw = item['resp_signal'].numpy()             # [nf_resp] raw volts
        resp = np.interp(self._resp_idx, np.arange(raw.shape[0], dtype=np.float64),
                         raw.astype(np.float64)).astype(np.float32)

        out = {}
        if 'tir' in self.streams:
            out['tir'] = tir.contiguous()
        if 'resp' in self.streams:
            out['resp'] = torch.from_numpy(resp).unsqueeze(0)      # [1, L]
        return out

    def __len__(self) -> int:
        return len(self.entries)

    def describe(self) -> str:
        return (f'TirRoiRespPretrainDataset streams={list(self.streams)}\n'
                f'  clips {len(self)} from {len(self.base.sessions)} session(s), '
                f'ROI {self.input_size}px padding {self.roi_padding:g}\n'
                f'  geometry T={self.num_frames} frames '
                f'({self.num_frames * self.temporal_stride / self.fps:.4f} s), '
                f'L={self.seq_len} @ {self.fs:g} Hz, '
                f'temporal_stride={self.temporal_stride}\n' + self.base.describe())


class TirRoiRespFinetuneDataset(TirRoiRespPretrainDataset):
    """Stage-3 view: RAW TIR-ROI clip (input) -> respiration window (target).

    Stage 3 is the simulated COMPLETE sensor failure: ONLY the thermal ROI
    enters the model (no 1-D input, and no masking anywhere -- see
    ``core/waveform_model.MultiModalWaveformRegressor``), and the whole window
    is regressed at once. This class is the Stage-3 counterpart of
    :class:`TirRoiRespPretrainDataset`, so a Stage-2 TIR-ROI checkpoint sees
    exactly the input it was pre-trained on:

    * the ROI crop comes from the same :class:`BP4DPlusTIRRespDataset` (same
      12-landmark set, same per-clip STATIC box, same ``roi_padding``, same
      ``cv2.resize``), and the SAME exclusion rules apply -- a session without
      ``IRFeatures`` is skipped, and any clip whose frame span overlaps an
      all-``(0,0)`` sentinel line is dropped (``self.skipped`` and
      ``self.stats['clips_dropped_sentinel']`` record it);
    * the geometry uses the same expressions as the Stage-2 dataset and
      ``core.waveform_model.build_waveform_model``
      (``n = round(clip_duration*fps/temporal_stride)`` rounded down to
      ``tubelet_t``, ``L = round(kept_seconds*fs)``), so ``T``/``L`` match the
      Stage-2 run and ``_fit_time`` / ``samples_per_token == sig_kernel`` pass.

    Stage-3-specific differences, all deliberate:

    * a **train/val split** (``is_train`` / ``train_ratio`` / ``split_by``),
      SUBJECT-disjoint by default -- the same policy and the same arithmetic as
      ``data.paired_dataset.PairedSessionDataset`` (``'subject'``: whole
      subjects per side, so the evaluated subject is never trained on;
      ``'session'``: whole sessions per side, no frame-level leakage);
    * an explicit ``val_subject`` OVERRIDE for a leave-one-subject-out sweep
      (train = every other subject, val = exactly that one); the 4-subject local
      corpus needs it, because a fixed 0.8 ratio can only ever produce one fold;
    * the item is ``(tir, waveform)`` -- the tensor contract
      ``runners/run_waveform.py`` and ``engines/waveform.py`` consume -- rather
      than the Stage-2 stream dict;
    * the target is normalised HERE (``physio_norm``), because Stage 3 scores the
      FINAL waveform against the label:

      - ``'zscore'`` (default) / ``'ac'``: per CLIP -- the twin of the Stage-2
        model's internal ``target_norm: clip`` (the base class returns RAW
        volts, so the two never stack);
      - ``'session'``: the WHOLE session's mean/std. The base dataset is then
        built with ``physio_norm='session'`` and performs the z-score itself, so
        this class leaves the window untouched -- the twin of the Stage-2
        ``physio_norm: session`` target space (where the model is pinned to
        identity). Use it when the
        checkpoint was pre-trained that way (the HPC TIR-ROI run) so both
        stages share ONE target space;
      - ``'none'``: raw volts.
    """

    def __init__(self, raw_root: Optional[str] = None,
                 target: str = 'resp',
                 is_train: bool = True,
                 train_ratio: float = 0.8,
                 split_by: str = 'subject',
                 val_subject: Optional[str] = None,
                 fs: float = 100.0,
                 clip_duration: float = 8.0,
                 clip_stride: Optional[float] = None,
                 temporal_stride: int = 1,
                 tubelet_t: int = 2,
                 input_size: int = DEFAULT_INPUT_SIZE,
                 roi_padding: float = DEFAULT_ROI_PADDING,
                 roi_quantile: float = DEFAULT_ROI_QUANTILE,
                 resp_fs: float = DEFAULT_RESP_FS,
                 fps: float = DEFAULT_FPS,
                 target_landmarks: Sequence[int] = TARGET_LANDMARKS,
                 min_signal_spread: float = 0.0,
                 rail_touch_v: float = 0.0,
                 physio_norm: str = 'zscore',
                 subjects: Optional[Sequence[str]] = None,
                 tasks: Optional[Sequence[str]] = None,
                 task_groups=None, task_set=None,
                 max_clips_per_session: Optional[int] = None,
                 max_entries: Optional[int] = None,
                 verbose: bool = False):
        if target != 'resp':
            raise ValueError(
                f'TirRoiRespFinetuneDataset serves the respiration waveform '
                f'only (the ROI box is anchored on the mouth+nose landmarks); '
                f'got target={target!r}.')
        physio_norm = str(physio_norm or 'zscore').strip().lower()
        if physio_norm == 'clip':                       # Stage-2 name
            physio_norm = 'zscore'
        if physio_norm not in ('none', 'ac', 'zscore', 'session'):
            raise ValueError(
                f"physio_norm must be 'none', 'ac', 'zscore' ('clip') or "
                f"'session'; got {physio_norm!r}")
        if split_by not in ('session', 'subject'):
            raise ValueError(
                f"split_by must be 'session' or 'subject'; got {split_by!r}")
        if not 0.0 < float(train_ratio) <= 1.0:
            raise ValueError(
                f'train_ratio must be in (0, 1]; got {train_ratio!r}')

        self.target = target
        self.physio_norm = physio_norm
        self.split_by = split_by
        self.is_train = bool(is_train)
        self.train_ratio = float(train_ratio)
        self.val_subject = (str(val_subject).strip() or None
                            if val_subject is not None else None)

        # resolve the group selection BEFORE the split, so the split and the
        # data are computed on exactly the same task set (see data.task_groups)
        self.task_selection = tgrp.resolve_task_selection(
            tasks=tasks, task_set=task_set, task_groups=task_groups)

        # ---- the split, computed on USABLE sessions, before any clip work ---
        # "usable" = has BOTH the landmark file and the respiration file, so a
        # subject whose sessions are all untracked cannot silently sit on one
        # side of the split.
        root = raw_root or default_raw_root()
        usable = [s for s in discover_sessions(root,
                                              subjects=_as_list(subjects),
                                              tasks=self.task_selection.tasks)
                  if s['ir_file'] and s['resp_file']]
        if not usable:
            raise RuntimeError(
                f'TirRoiRespFinetuneDataset: no session under {root} has both '
                f'{IR_TREE} and {RESP_FILE}; nothing to split.')
        if self.val_subject:
            subs = sorted({s['subject'] for s in usable})
            if self.val_subject not in subs:
                raise ValueError(
                    f'val_subject={self.val_subject!r} is not among the usable '
                    f'subjects {subs}.')
            keep = ({self.val_subject} if not self.is_train
                    else set(subs) - {self.val_subject})
            if not keep:
                raise ValueError(
                    f'val_subject={self.val_subject!r} leaves the train split '
                    f'empty (it is the only usable subject).')
            self.split_keys = sorted(keep)
            self.split_key_kind = 'subject'
        else:
            keys = sorted({s['subject'] if split_by == 'subject'
                           else s['session'] for s in usable})
            n_keep = max(1, int(round(len(keys) * self.train_ratio)))
            keep = set(keys[:n_keep]) if self.is_train else set(keys[n_keep:])
            if not keep:
                raise ValueError(
                    f"split_by={split_by!r} with train_ratio {self.train_ratio} "
                    f'leaves the {"train" if self.is_train else "val"} split '
                    f'empty: {len(keys)} {split_by}(s) -> {n_keep} train '
                    f'side(s). Lower train_ratio, or pass val_subject.')
            self.split_keys = sorted(keep)
            self.split_key_kind = split_by

        print(f'[data] tir_roi_resp split_by={self.split_key_kind}: '
              f'{"train" if self.is_train else "val"} = {self.split_keys} '
              f'({len(usable)} usable session(s), '
              f'{len({s["subject"] for s in usable})} subject(s))')
        print(f'[data] tir_roi_resp {self.task_selection.describe()}')

        keep_subjects = ([k.rsplit('_', 1)[0] for k in self.split_keys]
                         if self.split_key_kind == 'session'
                         else self.split_keys)

        super().__init__(
            raw_root=root, streams=('tir', 'resp'), fs=fs, fps=fps,
            clip_duration=clip_duration, clip_stride=clip_stride,
            temporal_stride=temporal_stride, tubelet_t=tubelet_t,
            input_size=input_size, roi_padding=roi_padding,
            roi_quantile=roi_quantile,
            target_landmarks=target_landmarks, resp_fs=resp_fs,
            min_signal_spread=min_signal_spread,
            rail_touch_v=rail_touch_v,
            # 'session'/'clip' are implemented by the BASE dataset (whole-
            # session or per-clip statistics); this class then leaves the
            # window alone -- see _normalize_target.
            physio_norm=('session' if self.physio_norm == 'session' else 'none'),
            subjects=_as_list(keep_subjects), tasks=self.task_selection.tasks,
            task_groups=task_groups, task_set=task_set,
            max_clips_per_session=max_clips_per_session,
            max_entries=max_entries, verbose=verbose)

        # The parent stores ITS canonical dataset norm under the same attribute
        # name (it is the knob driving the base dataset's ``norm``), so restore
        # this class's STAGE-3 value here: ``_normalize_target`` and
        # ``describe`` must see the Stage-3 vocabulary ('zscore'/'ac'), not the
        # Stage-2 one ('clip'/'none').
        self.physio_norm = physio_norm

        # a SESSION-level split still needs the per-session filter: the subject
        # filter above admits every session of the kept subjects
        if self.split_key_kind == 'session':
            n_before = len(self.entries)
            keep_set = set(self.split_keys)
            self.entries = [e for e in self.entries if e['session'] in keep_set]
            if not self.entries:
                raise RuntimeError(
                    f'split_by=session: none of the {n_before} clip(s) of the '
                    f'kept subjects belongs to the '
                    f'{"train" if self.is_train else "val"} session list '
                    f'{self.split_keys}.')
        self.split_sessions = sorted({e['session'] for e in self.entries})

    # ------------------------------------------------------------------ item
    def __getitem__(self, index: int):
        """``(tir [3, T, S, S] float32, waveform [L] float32)``."""
        # Stage-2 geometry; RAW volts unless physio_norm='session', where the
        # base already applied the whole-session z-score.
        item = super().__getitem__(index)
        tir = item['tir']
        if not torch.is_tensor(tir):
            tir = torch.from_numpy(np.ascontiguousarray(tir))
        w = item['resp']
        if torch.is_tensor(w):
            w = w.numpy()
        return tir, torch.from_numpy(
            self._normalize_target(np.asarray(w)[0]))

    def _normalize_target(self, w: np.ndarray) -> np.ndarray:
        """Stage-3 target normalisation.

        ``'zscore'`` / ``'ac'`` are per CLIP -- the Stage-3 analogue of the
        Stage-2 model's internal ``target_norm: clip`` (same population std and
        the same 1e-6 epsilon as ``data.paired_dataset``).

        ``'session'`` is a NO-OP here: the base dataset was constructed with
        ``physio_norm='session'`` and already z-scored the window with the WHOLE
        session's mean/std (the Stage-2 ``physio_norm: session`` target space),
        so re-normalising would silently collapse it back to a per-clip z-score.
        """
        w = np.asarray(w, dtype=np.float64)
        if self.physio_norm in ('none', 'session'):
            return w.astype(np.float32)
        w = w - float(w.mean())
        if self.physio_norm == 'zscore':
            w = w / (float(w.std()) + 1e-6)
        return w.astype(np.float32)

    def describe(self) -> str:
        return (super().describe()
                + f'\n  Stage-3 view: target={self.target} '
                  f'physio_norm={self.physio_norm} '
                  f'split={self.split_key_kind} '
                  f'({"train" if self.is_train else "val"}) = {self.split_keys}'
                  f'\n  clips {len(self)} over {len(self.split_sessions)} '
                  f'session(s): {self.split_sessions}')


def _log_corpus_cleaning(stats: dict, stage: str) -> None:
    """Print the cleaning knobs a run ACTUALLY used (never silent).

    Both knobs are opt-in (``0.0`` = off) and are threaded from the YAML/CLI, so
    a config or entry point that forgets ``rail_touch_v`` would train on
    rail-valued respiration targets without saying anything. This line makes the
    effective policy visible in the log of EVERY stage that goes through a
    builder (Stage 2, Stage 3, classification runs, ...).
    """
    touch = float(stats.get('rail_touch_v') or 0.0)
    spread = float(stats.get('min_signal_spread') or 0.0)
    if touch > 0.0:
        rule = (f'DROP a clip when abs(x) >= {touch:g} V -- it TOUCHED the '
                f'+/-{abs(RESP_RAIL_V):g} V recorder rail')
    else:
        rule = 'OFF -- rail-valued targets are NOT filtered'
    print(f'[data] {stage} corpus cleaning: rail_touch_v={touch:g} V '
          f'({rule}); min_signal_spread={spread:g} V')


def build_tir_roi_finetune_dataset(is_train: bool, test_mode: bool,
                                   args) -> TirRoiRespFinetuneDataset:
    """Build the Stage-3 TIR-ROI -> respiration dataset from runner ``args``.

    Selected by ``data_set: tir_roi`` / ``tir_roi_resp`` in a FINETUNE config
    (the same value in a PRETRAIN config selects the Stage-2 view; see
    ``data/datasets.build_pretraining_dataset``). Only ``getattr``-reads, so an
    args namespace from any runner/YAML works. ``test_mode`` is accepted for
    signature parity with the other Stage-3 builders and is unused (the val
    split is the evaluation set; there is no third split).
    """
    tubelet = str(getattr(args, 'tubelet', '2,16,16')).split(',')
    val_subject = getattr(args, 'val_subject', None)
    if isinstance(val_subject, (list, tuple)):
        val_subject = val_subject[0] if val_subject else None
    ds = TirRoiRespFinetuneDataset(
        raw_root=(getattr(args, 'raw_root', None)
                  or getattr(args, 'data_path', None) or None),
        target=str(getattr(args, 'target', 'resp')),
        is_train=is_train,
        train_ratio=float(getattr(args, 'train_ratio', 0.8)),
        split_by=str(getattr(args, 'split_by', 'subject')),
        val_subject=val_subject,
        fs=float(getattr(args, 'fs', 100.0)),
        clip_duration=float(getattr(args, 'clip_duration', 8.0)),
        clip_stride=getattr(args, 'clip_stride', None),
        temporal_stride=int(getattr(args, 'temporal_stride', 1) or 1),
        tubelet_t=int(tubelet[0]),
        input_size=int(getattr(args, 'input_size', DEFAULT_INPUT_SIZE)),
        roi_padding=float(getattr(args, 'roi_padding', DEFAULT_ROI_PADDING)),
        roi_quantile=float(getattr(args, 'roi_quantile', DEFAULT_ROI_QUANTILE)),
        target_landmarks=resolve_roi_landmarks(
            getattr(args, 'roi_landmarks', None)),
        resp_fs=float(getattr(args, 'resp_fs', DEFAULT_RESP_FS)),
        fps=float(getattr(args, 'fps', DEFAULT_FPS)),
        physio_norm=str(getattr(args, 'physio_norm', 'zscore')),
        min_signal_spread=float(getattr(args, 'min_signal_spread', 0.0) or 0.0),
        rail_touch_v=float(getattr(args, 'rail_touch_v', 0.0) or 0.0),
        subjects=getattr(args, 'subjects', None),
        tasks=getattr(args, 'tasks', None),
        task_groups=getattr(args, 'task_groups', None),
        task_set=getattr(args, 'task_set', None),
        max_clips_per_session=getattr(args, 'max_clips', None),
        max_entries=getattr(args, 'max_entries', None),
        verbose=bool(getattr(args, 'verbose', False)))
    _log_corpus_cleaning(ds.stats, 'stage3')
    return ds


def build_tir_roi_pretrain_dataset(args) -> TirRoiRespPretrainDataset:
    """Build the Stage-2 TIR-ROI + RESP dataset from a run_pretrain ``args``.

    Selected by ``data_set: tir_roi`` (see ``data/datasets.py``). Only
    ``getattr``-reads, so an args namespace from any runner/config works.

    ``--physio_norm`` is THE target-normalisation knob of the pipeline; the
    model side is pinned to ``target_norm='none'`` by ``runners/run_pretrain.py``
    for every ROI lineage, so there is no second knob to keep in sync.
    """
    tubelet = str(getattr(args, 'tubelet', '2,16,16')).split(',')
    ds = TirRoiRespPretrainDataset(
        raw_root=getattr(args, 'raw_root', None) or default_raw_root(),
        streams=tuple(s.strip() for s in
                      str(getattr(args, 'streams', 'tir,resp')).split(',')
                      if s.strip()),
        fs=getattr(args, 'fs', 100.0),
        fps=getattr(args, 'fps', DEFAULT_FPS),
        clip_duration=getattr(args, 'clip_duration', 4.0),
        clip_stride=getattr(args, 'clip_stride', None) or None,
        temporal_stride=int(getattr(args, 'temporal_stride', 1) or 1),
        tubelet_t=int(tubelet[0]),
        input_size=getattr(args, 'input_size', DEFAULT_INPUT_SIZE),
        roi_padding=float(getattr(args, 'roi_padding', DEFAULT_ROI_PADDING)),
        roi_quantile=float(getattr(args, 'roi_quantile', DEFAULT_ROI_QUANTILE)),
        target_landmarks=resolve_roi_landmarks(
            getattr(args, 'roi_landmarks', None)),
        min_signal_spread=float(getattr(args, 'min_signal_spread', 0.0) or 0.0),
        rail_touch_v=float(getattr(args, 'rail_touch_v', 0.0) or 0.0),
        physio_norm=str(getattr(args, 'physio_norm', 'none') or 'none'),
        subjects=getattr(args, 'subjects', None),
        tasks=getattr(args, 'tasks', None),
        task_groups=getattr(args, 'task_groups', None),
        task_set=getattr(args, 'task_set', None),
        max_clips_per_session=getattr(args, 'max_clips', None),
        max_entries=getattr(args, 'max_entries', None))
    _log_corpus_cleaning(ds.stats, 'stage2')
    return ds


# --------------------------------------------------------------------------- #
# data verification / shape tests
# --------------------------------------------------------------------------- #
def _check_clip(ds: BP4DPlusTIRRespDataset, index: int) -> List[str]:
    """Verify one sample; returns a list of failure strings (empty == pass)."""
    fails: List[str] = []
    item = ds[index]
    entry = ds.entries[index]
    sess = entry['session']

    keys = set(item.keys())
    if keys != {'tir_video', 'resp_signal', 'subject_task'}:
        fails.append(f'keys {sorted(keys)} != the documented three')

    tir, resp, tag = item['tir_video'], item['resp_signal'], item['subject_task']
    if not isinstance(tag, str) or tag != sess:
        fails.append(f'subject_task {tag!r} != entry session {sess!r}')
    if resp.shape != (ds.resp_len,) or resp.dtype != torch.float32:
        fails.append(f'resp_signal {tuple(resp.shape)}/{resp.dtype} != '
                     f'({ds.resp_len},)/float32')
    elif not torch.isfinite(resp).all():
        fails.append('resp_signal holds non-finite values')

    if tir.dtype != torch.float32:
        fails.append(f'tir_video dtype {tir.dtype} != float32')
    if tir.ndim != 4 or tir.shape != (3, ds.clip_frames, ds.input_size, ds.input_size):
        fails.append(f'tir_video {tuple(tir.shape)} != '
                     f'(3, {ds.clip_frames}, {ds.input_size}, {ds.input_size})')
    else:
        lo, hi = float(tir.min()), float(tir.max())
        if not torch.isfinite(tir).all():
            fails.append('tir_video holds non-finite values')
        if lo < 0.0 or hi > 1.0:
            fails.append(f'tir_video range [{lo:.4f}, {hi:.4f}] outside [0, 1]')
        if hi < 0.05:
            fails.append(f'tir_video looks empty (max {hi:.4f})')

    if ds.norm == 'clip' and resp.numel() > 1:
        mu, sd = float(resp.mean()), float(resp.std(unbiased=False))
        if abs(mu) > 1e-5 or abs(sd - 1.0) > 1e-3:
            fails.append(f'per-clip z-score off: mean {mu:.3e}, std {sd:.6f}')

    if ds.norm == 'session' and resp.numel() > 1:
        # Session-level z-score: the CLIP mean/std are deliberately NOT 0/1
        # (that is the point) -- what must hold is that the returned window is
        # exactly the raw window under the SESSION's affine map.
        ds._resp_full(sess)                         # fills _norm_cache
        mu_s, sd_s = ds._norm_cache[sess]
        raw = ds.respiration_clip(index, normalized=False)
        expect = (raw - mu_s) / (sd_s + 1e-8)
        if not np.allclose(resp.numpy(), expect, atol=1e-5):
            fails.append('session z-score != '
                         '(raw - session_mean)/(session_std + 1e-8)')

    # --- task-group bookkeeping (the "distortion level" the clip belongs to)
    want = list(ds.task_selection.levels_of(entry['task']))
    if list(entry.get('task_level') or []) != want:
        fails.append(f'task_level {entry.get("task_level")!r} != {want!r} '
                     f'for task {entry["task"]!r}')

    # --- ROI strategy: box inside the frame AND covering every target point
    ir = ds._landmarks(sess)
    pts = ir[entry['frame_start']:entry['frame_end']][:, ds.target_idx, :]
    frames = ds._frames(entry)
    h, w = frames.shape[1], frames.shape[2]
    box = roi_box_from_landmarks(pts, w, h, ds.roi_padding,
                                 quantile=ds.roi_quantile)
    x0, x1, y0, y1 = box
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        fails.append(f'ROI box {box} outside the {w}x{h} frame')
    fx, fy = pts[..., 0].ravel(), pts[..., 1].ravel()
    # The box is built from the min/max of the landmark cloud OR from a
    # PERCENTILE band of it (ds.roi_quantile), so the containment test and the
    # independent recomputation below must use the SAME bounds -- checking the
    # raw min/max against a percentile box fails spuriously.
    q = min(max(float(ds.roi_quantile), 0.0), 0.49)
    if q > 0.0:
        lo, hi = 100.0 * q, 100.0 * (1.0 - q)
        xlo, xhi = [float(v) for v in np.percentile(fx, [lo, hi])]
        ylo, yhi = [float(v) for v in np.percentile(fy, [lo, hi])]
    else:
        xlo, xhi = float(fx.min()), float(fx.max())
        ylo, yhi = float(fy.min()), float(fy.max())
    band = (fx >= xlo) & (fx <= xhi) & (fy >= ylo) & (fy <= yhi)
    if band.any() and not (x0 <= fx[band].min() and fx[band].max() < x1
                           and y0 <= fy[band].min() and fy[band].max() < y1):
        fails.append('ROI box does not contain every target landmark')
    # A mouth+nose box that swallowed most of the frame means the landmarks are
    # not what we think they are (the clip would carry no ROI localisation).
    if (x1 - x0) * (y1 - y0) > 0.5 * w * h:
        fails.append(f'ROI box {box} covers >50% of the {w}x{h} frame')

    # --- independent recomputation of the padded box (guards _clamp_box)
    sw, sh = xhi - xlo, yhi - ylo
    nx0 = math.floor(xlo - ds.roi_padding * sw)
    nx1 = math.ceil(xhi + ds.roi_padding * sw) + 1
    ny0 = math.floor(ylo - ds.roi_padding * sh)
    ny1 = math.ceil(yhi + ds.roi_padding * sh) + 1
    if 0 < nx0 and nx1 < w and 0 < ny0 and ny1 < h:
        if (x0, x1, y0, y1) != (nx0, nx1, ny0, ny1):
            fails.append(f'ROI box {box} != the padded landmark box '
                         f'{(nx0, nx1, ny0, ny1)}')

    # --- the returned tensor IS the crop of that box (not a centred re-crop)
    cv2 = _cv2()
    interp = (cv2.INTER_AREA if ds.input_size <= min(x1 - x0, y1 - y0)
              else cv2.INTER_LINEAR)
    expect = cv2.resize(frames[0][y0:y1, x0:x1], (ds.input_size, ds.input_size),
                        interpolation=interp)
    got = np.rint(tir[:, 0].permute(1, 2, 0).numpy() * 255.0)
    if expect.shape != got.shape or not np.allclose(expect, got, atol=1.0):
        fails.append('tir_video[:, 0] is not the resize of the ROI crop')

    # --- respiration alignment: frames x (resp_fs/fps) == the raw-file slice
    ratio = ds.resp_fs / ds.fps
    if abs(ratio - round(ratio)) < 1e-9:
        y_full = _load_1d(entry['resp_file'])
        s0 = entry['frame_start'] * int(round(ratio))
        brute = y_full[s0:s0 + ds.resp_len]
        raw = ds.respiration_clip(index, normalized=False)
        if brute.shape != raw.shape or not np.allclose(brute, raw, atol=1e-3):
            fails.append(f'resp window != raw[{s0}:{s0 + ds.resp_len}] of '
                         f'{os.path.basename(entry["resp_file"])}')

    # --- temporal alignment: the seek must land on the right frame
    fresh = ds._frames(entry)
    if fresh.shape != frames.shape or not np.array_equal(fresh[0], frames[0]):
        fails.append('re-decoded first frame differs (seek/alignment)')

    # --- T frames x 1/fps == L samples x 1/resp_fs
    if abs(ds.clip_frames / ds.fps - ds.resp_len / ds.resp_fs) > 1e-9:
        fails.append(f'clip {ds.clip_frames / ds.fps:.6f} s != resp '
                     f'{ds.resp_len / ds.resp_fs:.6f} s')
    return fails


def check_view_parity(raw_root=None, subjects=None, tasks=None,
                      task_groups=None, task_set=None,
                      clip_seconds=DEFAULT_CLIP_SECONDS, clip_stride=None,
                      fps=DEFAULT_FPS, resp_fs=DEFAULT_RESP_FS, fs=100.0,
                      temporal_stride=2, tubelet_t=2,
                      input_size=DEFAULT_INPUT_SIZE,
                      roi_padding=DEFAULT_ROI_PADDING,
                      roi_quantile=DEFAULT_ROI_QUANTILE,
                      target_landmarks=TARGET_LANDMARKS,
                      min_signal_spread=0.0, rail_touch_v=0.0,
                      max_entries=None,
                      n_check=3) -> List[str]:
    """Prove that the Stage-2 and Stage-3 views crop the ROI IDENTICALLY.

    Builds BOTH views on the SAME clip grid -- the Stage-3 view with
    ``split_by='session', train_ratio=1.0`` so it keeps every session, because
    the split is the one Stage-3 difference that changes WHICH clips exist and
    not what a clip looks like -- then compares, for the same
    ``(session, frame_start)`` window:

    * the ROI box -- must be equal (same landmark set, padding and quantile);
    * the ``[3, T, S, S]`` clip tensor -- must be BIT-identical: both views go
      through ``BP4DPlusTIRRespDataset._load``, i.e. ONE static box from the
      landmarks of THIS window and ONE ``cv2.resize`` per frame;
    * the target -- the Stage-3 waveform must equal the per-clip z-score of the
      Stage-2 RAW volts, which is exactly the label the Stage-2 model saw
      internally under ``target_norm: clip``.

    Returns a list of failure strings (empty == PASS).
    """
    common = dict(raw_root=raw_root, subjects=subjects, tasks=tasks,
                  task_groups=task_groups, task_set=task_set,
                  fs=fs, fps=fps, resp_fs=resp_fs,
                  clip_duration=clip_seconds, clip_stride=clip_stride,
                  temporal_stride=temporal_stride, tubelet_t=tubelet_t,
                  input_size=input_size, roi_padding=roi_padding,
                  roi_quantile=roi_quantile,
                  target_landmarks=target_landmarks,
                  min_signal_spread=min_signal_spread,
                  rail_touch_v=rail_touch_v,
                  max_entries=max_entries)
    s2 = TirRoiRespPretrainDataset(streams=('tir', 'resp'), **common)
    s3 = TirRoiRespFinetuneDataset(target='resp', is_train=True,
                                   split_by='session', train_ratio=1.0,
                                   **common)
    fails: List[str] = []
    idx3 = {(e['session'], e['frame_start']): i
            for i, e in enumerate(s3.entries)}
    n = 0
    for i, e in enumerate(s2.entries):
        key = (e['session'], e['frame_start'])
        if key not in idx3:
            continue
        j = idx3[key]
        item = s2[i]
        tir3, wave3 = s3[j]
        label = f'{key[0]} frames {key[1]}..{e["frame_end"]}'

        if not torch.equal(item['tir'], tir3):
            d = float((item['tir'] - tir3).abs().max())
            fails.append(f'{label}: Stage-2 and Stage-3 clip tensors differ '
                         f'(max |d| {d:g})')

        box2 = s2.base.clip_roi_box(i)
        box3 = s3.base.clip_roi_box(j)
        if tuple(box2) != tuple(box3):
            fails.append(f'{label}: ROI box {tuple(box2)} != {tuple(box3)}')

        raw = item['resp'].numpy().astype(np.float64).reshape(-1)
        want = (raw - raw.mean()) / (raw.std() + 1e-6)
        got = wave3.numpy().astype(np.float64)
        if want.shape != got.shape:
            fails.append(f'{label}: Stage-3 target shape {got.shape} != the '
                         f'Stage-2 window {want.shape}')
        elif not np.allclose(want, got, atol=1e-6):
            d = float(np.abs(want - got).max())
            fails.append(f'{label}: Stage-3 target != zscore(Stage-2 raw '
                         f'volts) (max |d| {d:g})')

        n += 1
        if n >= n_check:
            break

    if n == 0:
        fails.append('the Stage-2 and Stage-3 views share NO window '
                     '(check clip_stride / split)')
    return fails


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python data/tir_resp_dataset.py`` -> data verification + shape tests."""
    import argparse
    import time

    p = argparse.ArgumentParser(
        description='Verify BP4DPlusTIRRespDataset (data checks + shape tests).')
    p.add_argument('--raw_root', default=default_raw_root(),
                   help='raw BP4D root (default $RAW_DATA_PATH)')
    p.add_argument('--subject', default=None, help='comma list, e.g. F001')
    p.add_argument('--task', default=None, help='comma list, e.g. T1,T2')
    p.add_argument('--task_set', default=None,
                   help="named task group(s) to select, e.g. 'low' or "
                        "'low,high' (UNIONED with --task); '' = all tasks")
    p.add_argument('--task_groups', default=None,
                   help="group definitions, e.g. 'low=T1|T2;high=T7|T8|T9' "
                        '(the config\'s task_groups: mapping in string form)')
    p.add_argument('--clip_seconds', type=float, default=DEFAULT_CLIP_SECONDS)
    p.add_argument('--clip_stride', type=float, default=None)
    p.add_argument('--fps', type=float, default=DEFAULT_FPS)
    p.add_argument('--resp_fs', type=float, default=DEFAULT_RESP_FS)
    p.add_argument('--input_size', type=int, default=DEFAULT_INPUT_SIZE)
    p.add_argument('--roi_padding', type=float, default=DEFAULT_ROI_PADDING)
    p.add_argument('--roi_landmarks', default=None,
                   help="ROI landmark set: a preset (nose_mouth | nostrils | "
                        "nostril_mouth | nose_tip) or a CSV of 1-indexed "
                        "labels, e.g. '10,21'; '' = the 12-point default")
    p.add_argument('--roi_quantile', type=float, default=DEFAULT_ROI_QUANTILE,
                   help='0.0 = min/max box (historical); >0 = percentile box')
    p.add_argument('--min_signal_spread', type=float, default=0.0,
                   help='VOLTS: drop a clip whose respiration window is '
                        '(near-)constant (0.0 = off)')
    p.add_argument('--rail_touch_v', type=float, default=0.0,
                   help='VOLTS: drop a clip whose respiration window contains '
                        'ANY sample that touched the rail, i.e. '
                        'abs(x) >= this (0.0 = off). The recorder clamps at '
                        '+/-10 V, so 9.90 rejects every window with a single '
                        'rail-valued reading -- clipped troughs included')
    p.add_argument('--temporal_stride', type=int, default=2,
                   help='--check_views only: intra-window decimation')
    p.add_argument('--tubelet', type=int, default=2,
                   help='--check_views only: tubelet_t')
    p.add_argument('--check_views', action='store_true',
                   help='build the Stage-2 AND Stage-3 views on the same clip '
                        'grid and assert the ROI crop, the ROI box and the '
                        'target agree (the Stage-3 ROI contract)')
    p.add_argument('--norm', default='clip', choices=('clip', 'session', 'none'))
    p.add_argument('--max_entries', type=int, default=0, help='0 = no cap')
    p.add_argument('--n_check', type=int, default=3, help='clips to verify')
    args = p.parse_args(argv)

    split = lambda s: ([x.strip() for x in s.split(',') if x.strip()] if s else None)

    def build(**over):
        kw = dict(raw_root=args.raw_root, subjects=split(args.subject),
                  tasks=split(args.task), task_groups=args.task_groups,
                  task_set=args.task_set, clip_seconds=args.clip_seconds,
                  clip_stride=args.clip_stride, fps=args.fps,
                  resp_fs=args.resp_fs, input_size=args.input_size,
                  roi_padding=args.roi_padding,
                  roi_quantile=args.roi_quantile,
                  target_landmarks=resolve_roi_landmarks(args.roi_landmarks),
                  min_signal_spread=args.min_signal_spread,
                  rail_touch_v=args.rail_touch_v, norm=args.norm,
                  max_entries=args.max_entries or None, allow_empty=True)
        kw.update(over)
        return BP4DPlusTIRRespDataset(**kw)

    print('=' * 72)
    print('BP4DPlusTIRRespDataset -- data verification / shape tests')
    print('=' * 72)
    ds = build()
    print(ds.describe())
    print('-' * 72)
    if len(ds) == 0:
        print('[FAIL] no usable clips -- nothing to verify')
        return 1

    print(f'[ok] discovered {len(ds)} clip(s) from {len(ds.sessions)} session(s)')

    # Requirement 1: an untracked sequence (no IRFeatures) must NOT raise.
    # The corpus currently ships IRFeatures for every thermal video, so the
    # rule is verified against a SYNTHETIC tree (symlinks + an empty IRFeatures
    # dir) instead of relying on whichever sequences happen to be downloadless.
    tracked_subj = sorted({e['subject'] for e in ds.entries})
    untracked = sorted({s['subject'] for s in ds.skipped
                        if s['reason'] == 'missing_ir_features'})
    if untracked:
        probe = build(subjects=[untracked[0]], tasks=None,
                      task_set=None, task_groups=None)
        if len(probe) == 0 and any(s['reason'] == 'missing_ir_features'
                                   for s in probe.skipped):
            print(f'[ok] {untracked[0]} (no IRFeatures) -> skipped gracefully, '
                  f'0 clips')
        else:
            print(f'[FAIL] {untracked[0]} should have been skipped as untracked')
            return 1
    else:
        import tempfile
        src = next((s for s in discover_sessions(args.raw_root,
                                                subjects=split(args.subject),
                                                tasks=split(args.task))
                    if s['ir_file'] and s['resp_file']), None)
        if src is None:
            print('[--] no session to build the synthetic untracked probe from')
        else:
            with tempfile.TemporaryDirectory() as tmp:
                # NOTE: the relpath ALREADY starts with the tree name
                # ('Thermal/...', 'Physiology/...'), so it is not re-joined.
                for path in (src['video'], src['resp_file']):
                    dst = os.path.join(tmp, os.path.relpath(path, args.raw_root))
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    os.symlink(path, dst)
                    if not os.path.isfile(dst):
                        raise RuntimeError(f'probe symlink is dangling: {dst}')
                os.makedirs(os.path.join(tmp, IR_TREE), exist_ok=True)  # empty!
                probe = build(raw_root=tmp)
                if len(probe) == 0 and any(
                        s['reason'] == 'missing_ir_features' for s in probe.skipped):
                    print(f'[ok] synthetic {src["session"]} without '
                          f'{IR_TREE} -> skipped gracefully, 0 clips')
                else:
                    print('[FAIL] a session without IRFeatures was not skipped')
                    return 1

    # Requirement 2: no surviving clip may overlap a (0,0) sentinel line.
    sentinel_sess = None
    for spec in discover_sessions(args.raw_root, subjects=split(args.subject),
                                  tasks=split(args.task)):
        if spec['ir_file'] is None:
            continue
        try:
            mask = missing_frame_mask(parse_ir_features(spec['ir_file']),
                                      TARGET_LANDMARK_IDX)
        except Exception:
            continue
        if mask.any():
            sentinel_sess, sentinel_mask = spec['session'], mask
            break
    if sentinel_sess:
        subj, task = sentinel_sess.rsplit('_', 1)
        # the sentinel probes are about the DROP RULES, not the selection ->
        # build them with the group filter switched off
        probe = build(subjects=[subj], tasks=[task], task_set=None,
                      task_groups=None)
        dropped = int(probe.stats['clips_dropped_sentinel'])
        overlap = [e for e in probe.entries
                   if e['session'] == sentinel_sess
                   and sentinel_mask[e['frame_start']:e['frame_end']].any()]
        if overlap:
            print(f'[FAIL] {sentinel_sess}: {len(overlap)} kept clip(s) overlap '
                  f'a (0,0) sentinel line')
            return 1
        if dropped:
            print(f'[ok] {sentinel_sess} has (0,0) lines -> {dropped} window(s) '
                  f'dropped, {len(probe)} kept with no sentinel overlap')
        else:
            print(f'[--] {sentinel_sess} has (0,0) lines but no window '
                  f'overlaps them')
    else:
        print('[--] no sentinel-bearing session in this selection to probe')

    # Shape + normalisation + ROI + alignment checks on real clips.
    idxs = np.unique(np.linspace(0, len(ds) - 1,
                                 min(args.n_check, len(ds))).round().astype(int))
    n_fail = 0
    for i in idxs:
        t0 = time.time()
        fails = _check_clip(ds, int(i))
        e = ds.entries[int(i)]
        label = (f'{e["session"]} frames {e["frame_start"]}..{e["frame_end"]} '
                 f'({e["t_start"]:.2f}-{e["t_end"]:.2f} s)')
        if fails:
            n_fail += 1
            print(f'[FAIL] #{i} {label}')
            for f in fails:
                print(f'       - {f}')
        else:
            print(f'[ok] #{i} {label}: tir [3,{ds.clip_frames},'
                  f'{ds.input_size},{ds.input_size}], resp [{ds.resp_len}] '
                  f'({time.time() - t0:.1f} s)')

    # Requirement 3: the Stage-3 view must crop the ROI EXACTLY as Stage 2 did.
    # Same landmark set / padding / quantile / input_size + the same window ->
    # bit-identical pixels and the matching target (otherwise a Stage-3 run
    # fine-tunes an encoder on an input it was never pre-trained on, while every
    # tensor shape still matches, so nothing else can catch it).
    parity_fail = 0
    if args.check_views:
        print('-' * 72)
        t0 = time.time()
        pfails = check_view_parity(
            raw_root=args.raw_root, subjects=split(args.subject),
            tasks=split(args.task), task_groups=args.task_groups,
            task_set=args.task_set, clip_seconds=args.clip_seconds,
            clip_stride=args.clip_stride, fps=args.fps,
            resp_fs=args.resp_fs, temporal_stride=args.temporal_stride,
            tubelet_t=args.tubelet, input_size=args.input_size,
            roi_padding=args.roi_padding, roi_quantile=args.roi_quantile,
            target_landmarks=resolve_roi_landmarks(args.roi_landmarks),
            min_signal_spread=args.min_signal_spread,
            rail_touch_v=args.rail_touch_v,
            max_entries=args.max_entries or None, n_check=args.n_check)
        elapsed = time.time() - t0
        if pfails:
            parity_fail = len(pfails)
            print(f'[FAIL] Stage-2 vs Stage-3 ROI parity: {parity_fail} '
                  f'issue(s) ({elapsed:.1f} s)')
            for f in pfails:
                print(f'       - {f}')
        else:
            print(f'[ok] Stage-2 and Stage-3 views return the SAME ROI crop, '
                  f'the SAME box and the matching target ({elapsed:.1f} s)')
    else:
        print('[--] ROI parity not checked (pass --check_views)')

    print('-' * 72)
    print(f'{"FAIL" if (n_fail or parity_fail) else "PASS"}: '
          f'{len(idxs) - n_fail}/{len(idxs)} clip check(s) passed'
          + (f', {parity_fail} ROI-parity issue(s)' if parity_fail else '')
          + f', {len(ds)} clip(s) in the dataset')
    return 1 if (n_fail or parity_fail) else 0


if __name__ == '__main__':
    raise SystemExit(main())
