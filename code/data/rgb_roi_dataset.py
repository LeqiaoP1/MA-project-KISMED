"""BP4D+ RGB ROI -> 1-D physiology dataset provider (RAW tree, ADD-ON).

Why this exists
---------------
``simplifiedPlan.md`` locks ONE visual front-end for BOTH stages: the encoder
must see the identical input distribution at Stage-2 pre-training and at Stage-3
fine-tuning, so the RGB preprocessing is a *contract*, not a per-stage choice.
The stock path (:mod:`data.paired_dataset` + :mod:`data.video_io`) serves the
FULL frame through a centre crop, which on these portrait frames keeps a lot of
background: the face is only part of the crop, and at the plan's 0.90 tube mask
the encoder sees only ~20 patches per frame -- so a large share of the visible
(hence attendable) tokens can be background instead of skin.

This module provides the **RGB ROI** variant: the crop is derived from the
``2DFeatures`` landmark track, so the visible tokens are face pixels.

Trees read (RAW only, never modified)::

    <raw_root>/2D+3D/<S>/<T>/%04d.jpg        visible-light frames (25 fps)
    <raw_root>/2DFeatures/<S>_<T>.mat        49 (x, y) landmarks + head pose
    <raw_root>/Physiology/<S>/<T>/<signal>   1-D physiology (1000 Hz nominal)

The "unusable" rule (explicit requirement)
------------------------------------------
A ``(subject, task)`` pair whose ``2DFeatures`` frame count differs from the
number of JPEG files is **unusable**. The two streams cannot be aligned
frame-by-frame, and a mis-aligned crop would silently corrupt the label. Such a
pair is

* recorded in :attr:`RGBRoiDataset.unusable` with ``status='unusable'``, the two
  counts, and a ``reason``, and
* **completely excluded**, so its 1-D physiological data is never loaded or used
  either (the exclusion happens at discovery, before any signal file is opened).

On the local 40-session corpus this fires exactly once -- ``F003_T8``
(1518 jpg vs 1559 mat frames), which leaves 39 usable sessions.

Index alignment
---------------
The JPEG indices are contiguous and start at 0, and the ``frame`` field in the
``.mat`` runs 1..N monotonically, so once the counts agree, *rank == index* and
the clip's frames are a plain slice of the sorted file list. (Cross-TREE
alignment still keys on the ``frame`` field -- see :mod:`data.rgb_features`.)

This is an ADD-ON: nothing in :mod:`data.paired_dataset` /
:mod:`data.tir_resp_dataset` changes, and existing ``rgb``/``bp4d+`` configs are
untouched.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:                                          # package import
    from .rgb_features import (NUM_LANDMARKS_2D, ROI_LANDMARKS_2D,
                               find_2d_features, parse_2d_features)
    from .tir_resp_dataset import (_count_samples, _load_1d, _resp_start_limit,
                                   roi_box_from_landmarks)
except ImportError:                           # direct script execution
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.rgb_features import (NUM_LANDMARKS_2D, ROI_LANDMARKS_2D,
                                   find_2d_features, parse_2d_features)
    from data.tir_resp_dataset import (_count_samples, _load_1d,
                                       _resp_start_limit, roi_box_from_landmarks)

__all__ = [
    'RGB_TREE', 'FEAT_TREE', 'PHYS_TREE', 'SIGNAL_FILES', 'FACE_LANDMARKS',
    'resolve_face_landmarks', 'find_rgb_dir', 'find_signal_file',
    'discover_rgb_sessions', 'RGBRoiDataset', 'RGBRoiPretrainDataset',
    'RGBRoiFinetuneDataset', 'build_rgb_roi_pretrain_dataset',
    'build_rgb_roi_finetune_dataset', 'default_raw_root',
]

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
RGB_TREE = '2D+3D'
FEAT_TREE = '2DFeatures'
PHYS_TREE = 'Physiology'

#: signal name -> file inside ``Physiology/<S>/<T>/`` (RAW, one value per line)
SIGNAL_FILES = {
    'bp': 'BP_mmHg.txt',
    'resp': 'Resp_Volts.txt',
    'eda': 'EDA_microsiemens.txt',
}

DEFAULT_FPS = 25.0
DEFAULT_PHYS_FS = 1000.0
DEFAULT_CLIP_SECONDS = 8.0
DEFAULT_INPUT_SIZE = 224
DEFAULT_ROI_PADDING = 0.2
DEFAULT_ROI_QUANTILE = 0.0

#: the "RGB facial" ROI: the bounding box of ALL 49 landmarks.
FACE_LANDMARKS = tuple(range(1, NUM_LANDMARKS_2D + 1))

#: permitted ``decode_scale`` values (libjpeg DCT scales)
DECODE_FACTORS = (1, 2, 4, 8)

_IMAGE_EXTS = ('.jpg', '.jpeg', '.png')


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def default_raw_root() -> str:
    """``$RAW_DATA_PATH`` when set, else the in-repo raw BP4D root."""
    return os.environ.get('RAW_DATA_PATH') or os.path.join(
        _repo_root(), 'data', 'raw', 'BP4D')


def _cv2():
    import cv2
    return cv2


def _as_list(value) -> Optional[List[str]]:
    """``None``/``''`` -> ``None``; ``'a,b'`` or ``['a','b']`` -> ``['a','b']``."""
    if value is None:
        return None
    if isinstance(value, str):
        items = [v.strip() for v in value.split(',') if v.strip()]
        return items or None
    items = [str(v) for v in value]
    return items or None


# --------------------------------------------------------------------------- #
# landmark set resolution
# --------------------------------------------------------------------------- #
def resolve_face_landmarks(spec) -> Tuple[int, ...]:
    """Resolve a landmark spec -> a 1-indexed tuple in ``[1, 49]``.

    Accepts

    * ``None`` / ``''`` / ``'face'`` -> :data:`FACE_LANDMARKS` (all 49 points);
    * a preset name from :data:`data.rgb_features.ROI_LANDMARKS_2D`
      (``nose_mouth`` / ``nostrils`` / ``nostril_mouth`` / ``nose_tip``);
    * a comma-separated string of labels, e.g. ``'11,12,13,14'``;
    * any sequence of labels.
    """
    if spec is None:
        return FACE_LANDMARKS
    if isinstance(spec, str):
        key = spec.strip()
        if not key or key.lower() == 'face':
            return FACE_LANDMARKS
        if key in ROI_LANDMARKS_2D:
            return tuple(int(i) for i in ROI_LANDMARKS_2D[key])
        if not all(tok.strip().lstrip('+-').isdigit() for tok in key.split(',')):
            raise ValueError(
                f'unknown face-landmark spec {spec!r}; expected "face", one of '
                f'{sorted(ROI_LANDMARKS_2D)}, or a comma-separated list of '
                f'1-indexed labels in [1, {NUM_LANDMARKS_2D}]')
        labels = tuple(int(tok) for tok in key.split(','))
    else:
        labels = tuple(int(i) for i in spec)
    if not labels:
        raise ValueError('empty landmark set')
    bad = [i for i in labels if not 1 <= i <= NUM_LANDMARKS_2D]
    if bad:
        raise ValueError(f'landmark labels out of range 1..{NUM_LANDMARKS_2D}: {bad}')
    return labels


# --------------------------------------------------------------------------- #
# paths / discovery
# --------------------------------------------------------------------------- #
def find_rgb_dir(raw_root: str, subject: str, task: str) -> Optional[str]:
    """``<raw_root>/2D+3D/<S>/<T>/`` (also the ``<S>_<T>`` flat variant)."""
    for cand in (os.path.join(raw_root, RGB_TREE, subject, str(task)),
                 os.path.join(raw_root, RGB_TREE, f'{subject}_{task}')):
        if os.path.isdir(cand):
            return cand
    return None


def find_signal_file(raw_root: str, subject: str, task: str,
                     name: str) -> Optional[str]:
    """``<raw_root>/Physiology/<S>/<T>/<SIGNAL_FILES[name]>``."""
    if name not in SIGNAL_FILES:
        return None
    path = os.path.join(raw_root, PHYS_TREE, subject, str(task), SIGNAL_FILES[name])
    return path if os.path.isfile(path) else None


def _list_rgb_files(directory: str) -> List[str]:
    return sorted(f for f in os.listdir(directory)
                  if os.path.splitext(f)[1].lower() in _IMAGE_EXTS)


def discover_rgb_sessions(raw_root: str,
                          subjects: Optional[Sequence[str]] = None,
                          tasks: Optional[Sequence[str]] = None,
                          attach_landmarks: bool = True) -> List[dict]:
    """Every ``(subject, task)`` under ``2D+3D/``, with the usability verdict.

    Discovery keys on the JPEG directory (that is the modality the dataset
    predicts from). Each returned spec holds

    ``session`` / ``subject`` / ``task`` / ``rgb_dir`` / ``feat_file`` /
    ``signals`` (name -> path, only those that exist) / ``n_jpg`` / ``n_mat`` /
    ``status`` (``'usable'`` | ``'unusable'``) / ``reason``.

    The FRAME-COUNT check lives here so the exclusion is decided in exactly one
    place: a pair is ``'unusable'`` when the ``2DFeatures`` track is missing,
    cannot be parsed, or its frame count differs from the JPEG count. Its
    physiological files are still listed for diagnostics but are never loaded by
    the dataset.

    :param attach_landmarks: cache the parsed ``(N, 49, 2)`` track on the spec
        under ``'_pts'`` so the dataset does not parse the ``.mat`` twice.
    """
    if not os.path.isdir(raw_root):
        raise FileNotFoundError(f'raw_root does not exist: {raw_root}')
    rgb_root = os.path.join(raw_root, RGB_TREE)
    if not os.path.isdir(rgb_root):
        raise FileNotFoundError(f'no {RGB_TREE}/ tree under {raw_root}')
    want_subj = {str(s) for s in subjects} if subjects else None
    want_task = {str(t) for t in tasks} if tasks else None

    out: List[dict] = []
    for subj in sorted(os.listdir(rgb_root)):
        subj_dir = os.path.join(rgb_root, subj)
        if not os.path.isdir(subj_dir) or (want_subj and subj not in want_subj):
            continue
        dirs: Dict[str, str] = {}
        for name in sorted(os.listdir(subj_dir)):
            path = os.path.join(subj_dir, name)
            if os.path.isdir(path):
                dirs[name] = path
        for task in sorted(dirs):
            if want_task and task not in want_task:
                continue
            rgb_dir = dirs[task]
            spec = {
                'session': f'{subj}_{task}',
                'subject': subj,
                'task': task,
                'rgb_dir': rgb_dir,
                'feat_file': find_2d_features(raw_root, subj, task),
                'signals': {n: p for n in SIGNAL_FILES
                            if (p := find_signal_file(raw_root, subj, task, n))},
                'n_jpg': len(_list_rgb_files(rgb_dir)),
                'n_mat': -1,
                'status': 'usable',
                'reason': '',
            }
            pts: Optional[np.ndarray] = None
            # ---- usability: the landmark track must exist and match exactly --
            if spec['feat_file'] is None:
                spec['status'] = 'unusable'
                spec['reason'] = f'unusable: missing {FEAT_TREE} track'
            else:
                try:
                    pts, _pose, frame, _valid = parse_2d_features(spec['feat_file'])
                    spec['n_mat'] = int(len(frame))
                except Exception as exc:                      # malformed .mat
                    spec['status'] = 'unusable'
                    spec['reason'] = f'unusable: invalid {FEAT_TREE} track ({exc})'
            if spec['status'] == 'usable' and spec['n_mat'] != spec['n_jpg']:
                spec['status'] = 'unusable'
                spec['reason'] = (f'unusable: frame count mismatch '
                                  f'({spec["n_jpg"]} jpg vs {spec["n_mat"]} mat)')
            if pts is not None and attach_landmarks and spec['status'] == 'usable':
                spec['_pts'] = pts
            out.append(spec)
    if not out:
        raise FileNotFoundError(
            f'No 2D+3D sequences under {rgb_root}. Check --raw_root / $RAW_DATA_PATH.')
    return out


# --------------------------------------------------------------------------- #
# base dataset: RGB ROI clip -> raw 1-D physiology
# --------------------------------------------------------------------------- #
class RGBRoiDataset(Dataset):
    """RGB face-ROI clip -> raw 1-D physiological window(s), RAW BP4D tree.

    :param raw_root: raw BP4D root (default :func:`default_raw_root`).
    :param subjects, tasks: optional subset, e.g. ``['F001']`` / ``['T1','T2']``.
    :param signals: 1-D streams to serve; subset of :data:`SIGNAL_FILES`
        (``'bp'`` = ``BP_mmHg.txt``, ``'resp'`` = ``Resp_Volts.txt``,
        ``'eda'`` = ``EDA_microsiemens.txt``).
    :param clip_seconds: clip length in seconds (``clip_frames =
        round(clip_seconds * fps)``).
    :param fps: nominal frame rate used for the frame<->time mapping (25 Hz).
    :param phys_fs: sample rate of the RAW 1-D files (1000 Hz nominal).
    :param input_size: ROI patch side after ``cv2.resize``.
    :param clip_stride: clip hop in SECONDS (``None``/``0`` -> non-overlapping).
    :param roi_padding: per-side ROI margin, see :func:`roi_box_from_landmarks`
        (``0.2`` grows the landmark box by 40 % overall).
    :param roi_quantile: ``0.0`` = min/max box (historical); ``(0, 0.5)`` clips
        each side to that percentile of the clip's landmark cloud, which makes
        the box robust to head-motion outliers.
    :param landmarks: 1-indexed landmarks whose bounding box IS the ROI; defaults
        to all 49 (the whole face). See :func:`resolve_face_landmarks`.
    :param decode_scale: JPEG decode scale -- ``1`` = native full decode (most
        faithful), ``2/4/8`` = libjpeg DCT downscale (faster, and the ROI box is
        rescaled accordingly). The plane's materialised ``.npy`` frame store is
        the real answer to decode cost; this knob is for quick experiments.
    :param norm: ``'none'`` (default -- RAW values, because the model z-scores a
        clip internally under ``target_norm``), ``'clip'`` (per-clip z-score) or
        ``'session'``.
    :param max_clips_per_session, max_entries: dev caps (smoke runs).
    :param preload: materialise every clip at init (small: a 224 px clip of 200
        frames is ~30 MB as uint8).
    :param allow_empty: allow zero clips (diagnostics only).
    """

    def __init__(self, raw_root: Optional[str] = None,
                 subjects: Optional[Sequence[str]] = None,
                 tasks: Optional[Sequence[str]] = None,
                 signals: Sequence[str] = ('resp',),
                 clip_seconds: float = DEFAULT_CLIP_SECONDS,
                 fps: float = DEFAULT_FPS,
                 phys_fs: float = DEFAULT_PHYS_FS,
                 input_size: int = DEFAULT_INPUT_SIZE,
                 clip_stride: Optional[float] = None,
                 roi_padding: float = DEFAULT_ROI_PADDING,
                 roi_quantile: float = DEFAULT_ROI_QUANTILE,
                 landmarks: Sequence[int] = FACE_LANDMARKS,
                 decode_scale: int = 1,
                 norm: str = 'none',
                 max_clips_per_session: Optional[int] = None,
                 max_entries: Optional[int] = None,
                 preload: bool = False,
                 allow_empty: bool = False,
                 verbose: bool = False):
        if norm not in ('none', 'clip', 'session'):
            raise ValueError(f"norm must be 'none', 'clip' or 'session', got {norm!r}")
        if clip_seconds <= 0 or fps <= 0 or phys_fs <= 0:
            raise ValueError('clip_seconds, fps and phys_fs must be positive')
        if int(decode_scale) not in DECODE_FACTORS:
            raise ValueError(f'decode_scale must be one of {DECODE_FACTORS}, '
                             f'got {decode_scale!r}')
        self.signal_names = tuple(str(s).strip() for s in signals if str(s).strip())
        if not self.signal_names:
            raise ValueError('RGBRoiDataset: need at least one 1-D signal')
        unknown = [s for s in self.signal_names if s not in SIGNAL_FILES]
        if unknown:
            raise ValueError(
                f'RGBRoiDataset: unknown signal(s) {unknown}; expected a subset '
                f'of {sorted(SIGNAL_FILES)}.')

        self.raw_root = raw_root or default_raw_root()
        self.clip_seconds = float(clip_seconds)
        self.fps = float(fps)
        self.phys_fs = float(phys_fs)
        self.clip_frames = max(1, int(round(self.clip_seconds * self.fps)))
        self.phys_len = max(1, int(round(self.clip_seconds * self.phys_fs)))
        self.stride_frames = (self.clip_frames if not clip_stride
                              else max(1, int(round(float(clip_stride) * self.fps))))
        self.input_size = int(input_size)
        self.roi_padding = float(roi_padding)
        self.roi_quantile = float(roi_quantile)
        if not 0.0 <= self.roi_quantile < 0.5:
            raise ValueError(f'roi_quantile must be in [0, 0.5), got '
                             f'{roi_quantile!r}')
        self.landmarks = resolve_face_landmarks(landmarks)
        self.target_idx = np.asarray([l - 1 for l in self.landmarks], dtype=np.int64)
        self.decode_scale = int(decode_scale)
        self.norm = norm
        self.preload = bool(preload)
        self.verbose = bool(verbose)

        self.entries: List[dict] = []
        self.sessions: Dict[str, dict] = {}
        self.skipped: List[dict] = []
        self.unusable: List[dict] = []
        self.warnings: List[str] = []
        self.stats: Dict[str, object] = {}

        self._feat_cache: Dict[str, np.ndarray] = {}
        self._files: Dict[str, List[str]] = {}
        self._sig_cache: Dict[str, np.ndarray] = {}
        self._norm_cache: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self._src_size: Dict[str, Tuple[int, int]] = {}
        self._preloaded: List[dict] = []

        discovered = discover_rgb_sessions(self.raw_root, subjects=_as_list(subjects),
                                           tasks=_as_list(tasks))
        self._build_entries(discovered, max_clips_per_session, max_entries)
        if not self.entries and not allow_empty:
            reasons = '; '.join(f"{s['session']}: {s['reason']}"
                                for s in (self.unusable + self.skipped))
            raise RuntimeError(
                f'No usable RGB-ROI clips under {self.raw_root} for '
                f'clip_seconds={self.clip_seconds} ({self.clip_frames} frames). '
                f'Excluded: {reasons or "none"}')
        if self.preload:
            self._preload()

    # ------------------------------------------------------------------ build
    def _build_entries(self, discovered: List[dict],
                       max_clips_per_session: Optional[int],
                       max_entries: Optional[int]) -> None:
        n_dropped_nan = 0
        for spec in discovered:
            sess, subj, task = spec['session'], spec['subject'], spec['task']
            pts = spec.pop('_pts', None)

            # --- 1. the "unusable" gate: no landmarks / frame-count mismatch.
            #        The pair is annotated and dropped here, so its 1-D
            #        physiology is never opened.
            if spec['status'] != 'usable':
                self.unusable.append({k: v for k, v in spec.items()
                                      if k != 'signals'}
                                     | {'signals': dict(spec['signals'])})
                if self.verbose:
                    print(f'[rgb_roi] unusable {sess}: {spec["reason"]}')
                continue
            self._feat_cache[sess] = pts

            # --- 2. every requested 1-D signal must exist and be long enough
            missing = [n for n in self.signal_names if n not in spec['signals']]
            if missing:
                self._skip(spec, f'missing_signal: {",".join(missing)}')
                continue
            n_sig = {n: _count_samples(spec['signals'][n]) for n in self.signal_names}
            start_limit = min(_resp_start_limit(n_sig[n], self.phys_fs, self.fps,
                                                self.phys_len)
                              for n in self.signal_names)
            if start_limit < 0:
                self._skip(spec, f'signal_too_short: {n_sig} samples < '
                                 f'{self.phys_len} needed')
                continue

            # --- 3. clip slicing (frames == mat rows, both = n_jpg)
            n_frames = int(spec['n_jpg'])
            n_avail = min(n_frames, start_limit + self.clip_frames)
            if n_avail < self.clip_frames:
                self._skip(spec, f'too_short: {n_frames} frames < '
                                 f'{self.clip_frames} needed')
                continue

            bad = ~np.isfinite(pts[:, self.target_idx, :]).all(axis=(1, 2))

            kept, dropped, starts = 0, 0, []
            for start in range(0, n_avail - self.clip_frames + 1,
                               self.stride_frames):
                if bad[start:start + self.clip_frames].any():
                    dropped += 1
                    continue
                starts.append(start)
                kept += 1
                if max_clips_per_session and kept >= max_clips_per_session:
                    break
            n_dropped_nan += dropped
            if not starts:
                self._skip(spec, f'all_clips_dropped: {dropped} window(s) '
                                 f'contain a non-finite ROI landmark')
                continue

            self.sessions[sess] = dict(spec, n_frames=n_frames, n_sig=n_sig,
                                       signal_start_limit=start_limit,
                                       clips_dropped_nan=dropped)
            for start in starts:
                self.entries.append(self._make_entry(spec, start))
                if max_entries and len(self.entries) >= max_entries:
                    break
            if max_entries and len(self.entries) >= max_entries:
                break

        self.stats = {
            'raw_root': self.raw_root,
            'sessions_discovered': len(discovered),
            'sessions_used': len(self.sessions),
            'sessions_unusable': len(self.unusable),
            'sessions_skipped': len(self.skipped),
            'clips': len(self.entries),
            'clips_dropped_nan': n_dropped_nan,
            'clip_seconds': self.clip_seconds,
            'clip_frames': self.clip_frames,
            'clip_stride_frames': self.stride_frames,
            'fps': self.fps,
            'phys_fs': self.phys_fs,
            'phys_len': self.phys_len,
            'input_size': self.input_size,
            'roi_padding': self.roi_padding,
            'roi_quantile': self.roi_quantile,
            'landmarks': list(self.landmarks),
            'decode_scale': self.decode_scale,
            'norm': self.norm,
            'signals': list(self.signal_names),
            'warnings': list(self.warnings),
            'unusable': [{'session': u['session'], 'status': 'unusable',
                          'reason': u['reason'], 'n_jpg': u['n_jpg'],
                          'n_mat': u['n_mat']} for u in self.unusable],
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
            'rgb_dir': spec['rgb_dir'],
            'feat_file': spec['feat_file'],
            'signals': dict(spec['signals']),
        }

    def _skip(self, spec: dict, reason: str) -> None:
        self.skipped.append({'session': spec['session'], 'subject': spec['subject'],
                             'task': spec['task'], 'reason': reason,
                             'status': 'skipped',
                             'feat_file': spec['feat_file'],
                             'signals': dict(spec['signals'])})
        if self.verbose:
            print(f'[rgb_roi] skip {spec["session"]}: {reason}')

    # ------------------------------------------------------------------ caches
    def _files_of(self, session: str) -> List[str]:
        if session not in self._files:
            self._files[session] = _list_rgb_files(self.sessions[session]['rgb_dir'])
        return self._files[session]

    def _feat(self, session: str) -> np.ndarray:
        if session not in self._feat_cache:
            pts, _pose, _frame, _valid = parse_2d_features(
                self.sessions[session]['feat_file'])
            self._feat_cache[session] = pts
        return self._feat_cache[session]

    def _sig_full(self, session: str, name: str) -> np.ndarray:
        key = f'{session}||{name}'
        if key not in self._sig_cache:
            y = _load_1d(self.sessions[session]['signals'][name])
            if self.norm == 'session':
                self._norm_cache[(session, name)] = (float(np.nanmean(y)),
                                                     float(np.nanstd(y)))
            self._sig_cache[key] = y
        return self._sig_cache[key]

    def _sig_clip(self, entry: dict, name: str) -> np.ndarray:
        """Resample the clip's physiology window onto the ``[L]`` target grid."""
        y = self._sig_full(entry['session'], name)
        idx = (entry['t_start'] + np.arange(self.phys_len) / self.phys_fs) * self.phys_fs
        return np.interp(idx, np.arange(y.shape[0], dtype=np.float64),
                         y.astype(np.float64)).astype(np.float32)

    def _normalize(self, y: np.ndarray, session: str, name: str) -> np.ndarray:
        if self.norm == 'none':
            return y.astype(np.float32)
        if self.norm == 'session':
            mu, sd = self._norm_cache.get((session, name),
                                          (float(np.nanmean(y)), float(np.nanstd(y))))
        else:                                        # per-clip
            mu, sd = float(np.nanmean(y)), float(np.nanstd(y))
        return ((y - mu) / (sd + 1e-8)).astype(np.float32)

    # ------------------------------------------------------------------ frames
    def _read_rgb(self, path: str) -> np.ndarray:
        """One JPEG as uint8 **RGB** ``[h, w, 3]`` (BGR reversed), DCT-scaled."""
        cv2 = _cv2()
        flag = (cv2.IMREAD_COLOR if self.decode_scale == 1
                else getattr(cv2, f'IMREAD_REDUCED_COLOR_{self.decode_scale}'))
        img = cv2.imread(path, flag)
        if img is None:
            raise IOError(f'Failed to read image {path}')
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _source_size(self, session: str) -> Tuple[int, int]:
        """Native ``(h, w)`` of the session's frames (one full decode, cached)."""
        if session not in self._src_size:
            first = self.entries and next(
                (e for e in self.entries if e['session'] == session), None)
            path = (os.path.join(self.sessions[session]['rgb_dir'],
                                 self._files_of(session)[0])
                    if session in self.sessions else first['rgb_dir'])
            cv2 = _cv2()
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f'Failed to probe frame size for {session}: {path}')
            self._src_size[session] = (int(img.shape[0]), int(img.shape[1]))
        return self._src_size[session]

    # ------------------------------------------------------------------ roi
    def clip_roi_box(self, index: int) -> Tuple[int, int, int, int]:
        """The clip's STATIC ROI box ``(x0, x1, y0, y1)`` in SOURCE pixels."""
        entry = self.entries[index]
        pts = self._feat(entry['session'])[entry['frame_start']:entry['frame_end']]
        pts = pts[:, self.target_idx, :]
        h, w = self._source_size(entry['session'])
        return roi_box_from_landmarks(pts, w, h, self.roi_padding,
                                      quantile=self.roi_quantile)

    def clip_landmarks(self, index: int) -> np.ndarray:
        """All 49 landmarks of the clip's frames -> ``[T, 49, 2]`` (source px)."""
        entry = self.entries[index]
        return self._feat(entry['session'])[entry['frame_start']:entry['frame_end']]

    def signal_clip(self, index: int, name: str,
                    normalized: bool = False) -> np.ndarray:
        """The clip's ``[L]`` window of ``name`` (RAW by default)."""
        entry = self.entries[index]
        y = self._sig_clip(entry, name)
        return self._normalize(y, entry['session'], name) if normalized else y

    def _roi_patches(self, entry: dict) -> np.ndarray:
        """Per-frame decode -> ROI crop -> resize; ``[T, s, s, 3]`` uint8.

        The cue box is computed from the LANDMARKS (no decoding needed), then
        mapped into the decoded frame, so the big native frames are never all
        held in memory at once.
        """
        cv2 = _cv2()
        session = entry['session']
        files = self._files_of(session)
        pts = self._feat(session)[entry['frame_start']:entry['frame_end']]
        pts = pts[:, self.target_idx, :]
        src_h, src_w = self._source_size(session)
        x0, x1, y0, y1 = roi_box_from_landmarks(pts, src_w, src_h, self.roi_padding,
                                                quantile=self.roi_quantile)

        s = self.input_size
        out: Optional[np.ndarray] = None
        for i in range(entry['frame_start'], entry['frame_end']):
            img = self._read_rgb(os.path.join(entry['rgb_dir'], files[i]))
            h, w = img.shape[:2]
            sx, sy = w / src_w, h / src_h
            a0 = max(0, min(w - 1, int(math.floor(x0 * sx))))
            a1 = max(a0 + 1, min(w, int(math.ceil(x1 * sx))))
            b0 = max(0, min(h - 1, int(math.floor(y0 * sy))))
            b1 = max(b0 + 1, min(h, int(math.ceil(y1 * sy))))
            interp = (cv2.INTER_AREA if s <= min(a1 - a0, b1 - b0)
                      else cv2.INTER_LINEAR)
            patch = cv2.resize(img[b0:b1, a0:a1], (s, s), interpolation=interp)
            if out is None:
                out = np.empty((entry['frame_end'] - entry['frame_start'], s, s, 3),
                               np.uint8)
            out[i - entry['frame_start']] = patch
        if out is None:
            raise IOError(f'{session}: empty clip {entry["frame_start"]}')
        return out

    # ------------------------------------------------------------------ item
    def _load(self, index: int) -> dict:
        entry = self.entries[index]
        session = entry['session']
        patches = self._roi_patches(entry)                     # [T, s, s, 3] uint8

        # [T, H, W, C] uint8 RGB -> [C, T, H, W] float in [0, 1]
        rgb = torch.from_numpy(np.ascontiguousarray(patches.transpose(3, 0, 1, 2)))
        rgb = rgb.to(torch.float32).div_(255.0)

        item = {'rgb_video': rgb, 'subject_task': session,
                '_entry': entry, '_roi_box': self.clip_roi_box(index)}
        for name in self.signal_names:
            item[f'{name}_signal'] = torch.from_numpy(self.signal_clip(index, name))
        return item

    def _preload(self) -> None:
        self._preloaded = [self._load(i) for i in range(len(self.entries))]

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self.entries)
        if not 0 <= index < len(self.entries):
            raise IndexError(index)
        if self._preloaded:
            item = self._preloaded[index]
            out = {'rgb_video': item['rgb_video'].clone(),
                   'subject_task': item['subject_task']}
            for name in self.signal_names:
                out[f'{name}_signal'] = item[f'{name}_signal'].clone()
            return out
        item = self._load(index)
        return {k: v for k, v in item.items() if not k.startswith('_')}

    def __len__(self) -> int:
        return len(self.entries)

    # ------------------------------------------------------------------ report
    def describe(self) -> str:
        s = self.stats
        lines = [
            f'raw_root        : {s["raw_root"]}',
            f'sessions        : {s["sessions_used"]} used / '
            f'{s["sessions_discovered"]} found / {s["sessions_unusable"]} unusable / '
            f'{s["sessions_skipped"]} skipped',
            f'clips           : {s["clips"]} '
            f'({s["clips_dropped_nan"]} window(s) dropped for a non-finite landmark)',
            f'clip            : {s["clip_seconds"]:g} s = {s["clip_frames"]} '
            f'frames @ {s["fps"]:g} fps, stride {s["clip_stride_frames"]} frames; '
            f'phys {s["phys_len"]} samples @ {s["phys_fs"]:g} Hz',
            f'roi             : landmarks {len(self.landmarks)} of '
            f'{NUM_LANDMARKS_2D} padding {s["roi_padding"]:g} quantile '
            f'{s["roi_quantile"]:g} -> {self.input_size}x{self.input_size} '
            f'(decode_scale {s["decode_scale"]})',
            f'norm            : {s["norm"]}',
            f'signals         : {list(self.signal_names)}',
        ]
        for u in self.unusable:
            lines.append(f'unusable        : {u["session"]} ({u["reason"]})')
        for sk in self.skipped:
            lines.append(f'skipped         : {sk["session"]} ({sk["reason"]})')
        return '\n'.join(lines)


# --------------------------------------------------------------------------- #
# Stage-2 (masked pre-training) view
# --------------------------------------------------------------------------- #
class RGBRoiPretrainDataset(Dataset):
    """Stage-2 masked-pretraining dataset: RGB face ROI -> 1-D physiology.

    Wraps :class:`RGBRoiDataset` (clip enumeration, ROI crop, RAW physiology)
    and returns exactly the stream dict ``MultiModalMAE`` consumes::

        {'rgb':  float32 [3, T, input_size, input_size]   ROI crop, [0, 1]
         'resp': float32 [1, L]                           RAW values}

    The 1-D values are deliberately RAW: the model z-scores a clip internally
    under ``target_norm`` (the same contract ``PairedPretrainDataset`` follows).
    Do not normalise here as well.

    Geometry mirrors ``core.multimae.build_pretraining_model``::

        n = round(clip_duration * fps / temporal_stride)
        T = n rounded DOWN to a multiple of ``tubelet_t``   (>= tubelet_t)
        L = round((T * temporal_stride / fps) * fs)

    so ``grid_t = T / tubelet_t`` equals ``n_signal = L / sig_kernel`` and the
    model's hard space-time alignment check passes.
    """

    def __init__(self, raw_root: Optional[str] = None,
                 streams: Sequence[str] = ('rgb', 'resp'),
                 fs: float = 100.0, fps: float = DEFAULT_FPS,
                 clip_duration: float = DEFAULT_CLIP_SECONDS,
                 clip_stride: Optional[float] = None,
                 temporal_stride: int = 1, tubelet_t: int = 2,
                 input_size: int = DEFAULT_INPUT_SIZE,
                 roi_padding: float = DEFAULT_ROI_PADDING,
                 roi_quantile: float = DEFAULT_ROI_QUANTILE,
                 landmarks: Sequence[int] = FACE_LANDMARKS,
                 decode_scale: int = 1,
                 phys_fs: float = DEFAULT_PHYS_FS,
                 physio_norm: str = 'none',
                 subjects=None, tasks=None,
                 max_clips_per_session: Optional[int] = None,
                 max_entries: Optional[int] = None,
                 verbose: bool = False):
        self.streams = tuple(str(s).strip() for s in streams if str(s).strip())
        if 'rgb' not in self.streams:
            raise ValueError(
                f'RGBRoiPretrainDataset: streams must include "rgb" (the visual '
                f'stream); got {self.streams}.')
        sig = tuple(s for s in self.streams if s != 'rgb')
        if not sig:
            raise ValueError(
                'RGBRoiPretrainDataset: Stage 2 needs at least one 1-D stream '
                f'(bp/resp/eda) next to rgb; got {self.streams}.')
        unknown = [s for s in sig if s not in SIGNAL_FILES]
        if unknown:
            raise ValueError(f'RGBRoiPretrainDataset: unknown stream(s) {unknown}; '
                             f'expected a subset of {sorted(SIGNAL_FILES)}.')

        # WHERE the 1-D target is normalised -- the SAME single knob (and the
        # same values) as the tir_roi Stage-2 view and as Stage 3. 'zscore' is
        # accepted as an ALIAS of 'clip'. The model never normalises a 1-D
        # stream: run_pretrain.py pins target_norm='none' for every ROI path.
        self.physio_norm = str(physio_norm or 'none').strip().lower()
        if self.physio_norm == 'zscore':
            self.physio_norm = 'clip'
        if self.physio_norm not in ('none', 'clip', 'session'):
            raise ValueError(
                f"RGBRoiPretrainDataset: physio_norm must be 'none', 'clip' "
                f"('zscore') or 'session'; got {physio_norm!r}")

        self.fs = float(fs)
        self.fps = float(fps)
        self.clip_duration = float(clip_duration)
        self.temporal_stride = max(1, int(temporal_stride))
        self.tubelet_t = max(1, int(tubelet_t))
        self.phys_fs = float(phys_fs)
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

        self.base = RGBRoiDataset(
            raw_root=raw_root, subjects=_as_list(subjects), tasks=_as_list(tasks),
            signals=sig, clip_seconds=self.clip_duration, fps=self.fps,
            phys_fs=self.phys_fs, input_size=self.input_size,
            clip_stride=clip_stride, roi_padding=self.roi_padding,
            roi_quantile=self.roi_quantile, landmarks=landmarks,
            decode_scale=decode_scale, norm=self.physio_norm,
            max_clips_per_session=max_clips_per_session,
            max_entries=max_entries, verbose=verbose)
        self.entries = self.base.entries
        self.unusable = self.base.unusable
        self.skipped = self.base.skipped
        self.stats = self.base.stats

        self._phys_idx = (np.arange(self.seq_len) / self.fs) * self.phys_fs

    def __getitem__(self, index: int) -> dict:
        item = self.base[index]
        rgb = item['rgb_video']
        if self.temporal_stride > 1:
            rgb = rgb[:, ::self.temporal_stride]         # decimate INSIDE the window
        rgb = rgb[:, :self.num_frames]                   # drop the tubelet tail
        if rgb.shape[1] != self.num_frames:
            raise RuntimeError(
                f'{self.entries[index]["session"]}: ROI clip has '
                f'{rgb.shape[1]} frames, geometry needs {self.num_frames}')

        out = {'rgb': rgb.contiguous()}
        for name in self.streams:
            if name == 'rgb':
                continue
            raw = item[f'{name}_signal'].numpy()
            y = np.interp(self._phys_idx,
                          np.arange(raw.shape[0], dtype=np.float64),
                          raw.astype(np.float64)).astype(np.float32)
            out[name] = torch.from_numpy(y).unsqueeze(0)          # [1, L]
        return out

    def __len__(self) -> int:
        return len(self.entries)

    def describe(self) -> str:
        return (f'RGBRoiPretrainDataset streams={list(self.streams)}\n'
                f'  clips {len(self)} from {len(self.base.sessions)} session(s), '
                f'ROI {self.input_size}px padding {self.roi_padding:g}\n'
                f'  geometry T={self.num_frames} frames '
                f'({self.num_frames * self.temporal_stride / self.fps:.4f} s), '
                f'L={self.seq_len} @ {self.fs:g} Hz, '
                f'temporal_stride={self.temporal_stride}\n' + self.base.describe())


# --------------------------------------------------------------------------- #
# Stage-3 (waveform regression) view
# --------------------------------------------------------------------------- #
class RGBRoiFinetuneDataset(RGBRoiPretrainDataset):
    """Stage-3 view: RAW RGB-ROI clip (input) -> one 1-D waveform (target).

    Stage 3 simulates COMPLETE sensor failure: only the RGB ROI enters the model
    (no 1-D input, no masking), and the whole window is regressed at once. This
    is the Stage-3 counterpart of :class:`RGBRoiPretrainDataset`, so a Stage-2
    checkpoint sees exactly the input it was pre-trained on:

    * the same ROI crop (same landmark set, same per-clip STATIC box, same
      ``roi_padding``, same ``cv2.resize``, same ``decode_scale``) and the same
      exclusion rules -- an unusable pair stays unusable, and a clip whose frame
      span holds a non-finite ROI landmark is dropped;
    * the same geometry expressions (``n`` rounded down to ``tubelet_t``,
      ``L = round(kept_seconds * fs)``), so ``T``/``L`` match and
      ``_fit_time`` / ``samples_per_token == sig_kernel`` pass.

    Stage-3-specific: a SUBJECT-disjoint split by default (``split_by``), an
    explicit ``val_subject`` override for leave-one-subject-out, and a
    ``(rgb, waveform)`` tuple item; the target is normalised HERE
    (``physio_norm``) because Stage 3 scores the final waveform against the
    label, while the base class returns RAW values.
    """

    def __init__(self, raw_root: Optional[str] = None,
                 target: str = 'resp',
                 is_train: bool = True,
                 train_ratio: float = 0.8,
                 split_by: str = 'subject',
                 val_subject: Optional[str] = None,
                 fs: float = 100.0,
                 clip_duration: float = DEFAULT_CLIP_SECONDS,
                 clip_stride: Optional[float] = None,
                 temporal_stride: int = 1, tubelet_t: int = 2,
                 input_size: int = DEFAULT_INPUT_SIZE,
                 roi_padding: float = DEFAULT_ROI_PADDING,
                 roi_quantile: float = DEFAULT_ROI_QUANTILE,
                 landmarks: Sequence[int] = FACE_LANDMARKS,
                 decode_scale: int = 1,
                 phys_fs: float = DEFAULT_PHYS_FS,
                 fps: float = DEFAULT_FPS,
                 physio_norm: str = 'zscore',
                 subjects: Optional[Sequence[str]] = None,
                 tasks: Optional[Sequence[str]] = None,
                 max_clips_per_session: Optional[int] = None,
                 max_entries: Optional[int] = None,
                 verbose: bool = False):
        if target not in SIGNAL_FILES:
            raise ValueError(f'target must be one of {sorted(SIGNAL_FILES)}, '
                             f'got {target!r}')
        physio_norm = str(physio_norm or 'zscore').strip().lower()
        if physio_norm == 'clip':                       # Stage-2 name
            physio_norm = 'zscore'
        if physio_norm not in ('none', 'ac', 'zscore', 'session'):
            raise ValueError(
                f"physio_norm must be 'none', 'ac', 'zscore' ('clip') or "
                f"'session'; got {physio_norm!r}")
        if split_by not in ('session', 'subject'):
            raise ValueError(f"split_by must be 'session' or 'subject'; got "
                             f'{split_by!r}')
        if not 0.0 < float(train_ratio) <= 1.0:
            raise ValueError(f'train_ratio must be in (0, 1]; got {train_ratio!r}')

        self.target = target
        self.physio_norm = physio_norm
        self.split_by = split_by
        self.is_train = bool(is_train)
        self.train_ratio = float(train_ratio)
        self.val_subject = (str(val_subject).strip() or None
                            if val_subject is not None else None)

        # ---- the split, computed on USABLE sessions, before any clip work ---
        # "usable" = the (subject, task) pair passed the frame-count gate AND
        # carries this target signal, so a subject whose sessions are all
        # unusable cannot silently sit on one side of the split.
        root = raw_root or default_raw_root()
        usable = [s for s in discover_rgb_sessions(root,
                                                  subjects=_as_list(subjects),
                                                  tasks=_as_list(tasks),
                                                  attach_landmarks=False)
                  if s['status'] == 'usable' and target in s['signals']]
        if not usable:
            raise RuntimeError(
                f'RGBRoiFinetuneDataset: no usable session under {root} carries '
                f'{SIGNAL_FILES[target]}.')
        if self.val_subject:
            subs = sorted({s['subject'] for s in usable})
            if self.val_subject not in subs:
                raise ValueError(f'val_subject={self.val_subject!r} is not among '
                                 f'the usable subjects {subs}.')
            keep = ({self.val_subject} if not self.is_train
                    else set(subs) - {self.val_subject})
            if not keep:
                raise ValueError(f'val_subject={self.val_subject!r} leaves the '
                                 f'train split empty.')
            self.split_keys = sorted(keep)
            self.split_key_kind = 'subject'
        else:
            keys = sorted({s['subject'] if split_by == 'subject' else s['session']
                           for s in usable})
            n_keep = max(1, int(round(len(keys) * self.train_ratio)))
            keep = set(keys[:n_keep]) if self.is_train else set(keys[n_keep:])
            if not keep:
                raise ValueError(
                    f"split_by={split_by!r} with train_ratio {self.train_ratio} "
                    f"leaves the {'train' if self.is_train else 'val'} split "
                    f'empty: {len(keys)} {split_by}(s) -> {n_keep} train side(s). '
                    f'Lower train_ratio, or pass val_subject.')
            self.split_keys = sorted(keep)
            self.split_key_kind = split_by

        print(f'[data] rgb_roi split_by={self.split_key_kind}: '
              f'{"train" if self.is_train else "val"} = {self.split_keys} '
              f'({len(usable)} usable session(s), '
              f'{len({s["subject"] for s in usable})} subject(s))')

        keep_subjects = ([k.rsplit('_', 1)[0] for k in self.split_keys]
                         if self.split_key_kind == 'session' else self.split_keys)

        super().__init__(
            raw_root=root, streams=('rgb', target), fs=fs, fps=fps,
            clip_duration=clip_duration, clip_stride=clip_stride,
            temporal_stride=temporal_stride, tubelet_t=tubelet_t,
            input_size=input_size, roi_padding=roi_padding,
            roi_quantile=roi_quantile, landmarks=landmarks,
            decode_scale=decode_scale, phys_fs=phys_fs,
            # 'session' is implemented by the BASE dataset (whole-session
            # statistics); this class then leaves the window alone -- see
            # _normalize_target.
            physio_norm=('session' if self.physio_norm == 'session' else 'none'),
            subjects=_as_list(keep_subjects), tasks=tasks,
            max_clips_per_session=max_clips_per_session,
            max_entries=max_entries, verbose=verbose)

        # The parent stores ITS canonical dataset norm under the same attribute
        # name (see the tir_roi twin), so restore this class's STAGE-3 value.
        self.physio_norm = physio_norm

        if self.split_key_kind == 'session':
            keep_set = set(self.split_keys)
            self.entries = [e for e in self.entries if e['session'] in keep_set]
            if not self.entries:
                raise RuntimeError(
                    f'split_by=session: no clip of the kept subjects belongs to '
                    f'the {"train" if self.is_train else "val"} session list '
                    f'{self.split_keys}.')
        self.split_sessions = sorted({e['session'] for e in self.entries})

    def __getitem__(self, index: int):
        """``(rgb [3, T, S, S] float32, waveform [L] float32)``."""
        item = super().__getitem__(index)                 # {'rgb': ..., target: ...}
        rgb = item['rgb']
        if not torch.is_tensor(rgb):
            rgb = torch.from_numpy(np.ascontiguousarray(rgb))
        w = item[self.target]
        if torch.is_tensor(w):
            w = w.numpy()
        return rgb, torch.from_numpy(self._normalize_target(np.asarray(w)[0]))

    def _normalize_target(self, w: np.ndarray) -> np.ndarray:
        """Stage-3 target normalisation.

        ``'zscore'`` / ``'ac'`` are per CLIP -- the Stage-3 analogue of the
        Stage-2 model's internal ``target_norm: clip`` (same population std,
        same 1e-6 epsilon as ``data.paired_dataset``).

        ``'session'`` is a NO-OP here: the base dataset was constructed with
        ``physio_norm='session'`` and already z-scored the window with the
        WHOLE session's statistics -- re-normalising would collapse it back to a
        per-clip z-score.
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
                  f'physio_norm={self.physio_norm} split={self.split_key_kind} '
                  f'({"train" if self.is_train else "val"}) = {self.split_keys}'
                  f'\n  clips {len(self)} over {len(self.split_sessions)} '
                  f'session(s): {self.split_sessions}')


# --------------------------------------------------------------------------- #
# builders (runner glue)
# --------------------------------------------------------------------------- #
def build_rgb_roi_pretrain_dataset(args) -> RGBRoiPretrainDataset:
    """Build the Stage-2 RGB-ROI + 1-D dataset from a run_pretrain ``args``.

    Selected by ``data_set: rgb_roi`` (see ``data/datasets.py``). Only
    ``getattr``-reads, so an args namespace from any runner/config works.
    """
    tubelet = str(getattr(args, 'tubelet', '2,16,16')).split(',')
    return RGBRoiPretrainDataset(
        raw_root=getattr(args, 'raw_root', None) or default_raw_root(),
        streams=tuple(s.strip() for s in
                      str(getattr(args, 'streams', 'rgb,bp,resp')).split(',')
                      if s.strip()),
        fs=float(getattr(args, 'fs', 100.0)),
        fps=float(getattr(args, 'fps', DEFAULT_FPS)),
        clip_duration=float(getattr(args, 'clip_duration', DEFAULT_CLIP_SECONDS)),
        clip_stride=getattr(args, 'clip_stride', None) or None,
        temporal_stride=int(getattr(args, 'temporal_stride', 1) or 1),
        tubelet_t=int(tubelet[0]),
        input_size=int(getattr(args, 'input_size', DEFAULT_INPUT_SIZE)),
        roi_padding=float(getattr(args, 'roi_padding', DEFAULT_ROI_PADDING)),
        roi_quantile=float(getattr(args, 'roi_quantile', DEFAULT_ROI_QUANTILE)),
        landmarks=resolve_face_landmarks(
            getattr(args, 'roi_landmarks', None)
            or getattr(args, 'landmarks', None)),
        decode_scale=int(getattr(args, 'decode_scale', 1) or 1),
        phys_fs=float(getattr(args, 'phys_fs', DEFAULT_PHYS_FS)),
        physio_norm=str(getattr(args, 'physio_norm', 'none') or 'none'),
        subjects=getattr(args, 'subjects', None),
        tasks=getattr(args, 'tasks', None),
        max_clips_per_session=getattr(args, 'max_clips', None),
        max_entries=getattr(args, 'max_entries', None),
        verbose=bool(getattr(args, 'verbose', False)))


def build_rgb_roi_finetune_dataset(is_train: bool, test_mode: bool,
                                   args) -> RGBRoiFinetuneDataset:
    """Build the Stage-3 RGB-ROI -> waveform dataset from runner ``args``.

    Selected by ``data_set: rgb_roi`` / ``rgb_roi_resp`` in a FINETUNE config
    (the same value in a PRETRAIN config selects the Stage-2 view). ``test_mode``
    is accepted for signature parity and unused (the val split is the evaluation
    set; there is no third split).
    """
    tubelet = str(getattr(args, 'tubelet', '2,16,16')).split(',')
    val_subject = getattr(args, 'val_subject', None)
    if isinstance(val_subject, (list, tuple)):
        val_subject = val_subject[0] if val_subject else None
    return RGBRoiFinetuneDataset(
        raw_root=(getattr(args, 'raw_root', None)
                  or getattr(args, 'data_path', None) or None),
        target=str(getattr(args, 'target', 'resp')),
        is_train=is_train,
        train_ratio=float(getattr(args, 'train_ratio', 0.8)),
        split_by=str(getattr(args, 'split_by', 'subject')),
        val_subject=val_subject,
        fs=float(getattr(args, 'fs', 100.0)),
        clip_duration=float(getattr(args, 'clip_duration', DEFAULT_CLIP_SECONDS)),
        clip_stride=getattr(args, 'clip_stride', None),
        temporal_stride=int(getattr(args, 'temporal_stride', 1) or 1),
        tubelet_t=int(tubelet[0]),
        input_size=int(getattr(args, 'input_size', DEFAULT_INPUT_SIZE)),
        roi_padding=float(getattr(args, 'roi_padding', DEFAULT_ROI_PADDING)),
        roi_quantile=float(getattr(args, 'roi_quantile', DEFAULT_ROI_QUANTILE)),
        landmarks=resolve_face_landmarks(
            getattr(args, 'roi_landmarks', None)
            or getattr(args, 'landmarks', None)),
        decode_scale=int(getattr(args, 'decode_scale', 1) or 1),
        phys_fs=float(getattr(args, 'phys_fs', DEFAULT_PHYS_FS)),
        fps=float(getattr(args, 'fps', DEFAULT_FPS)),
        physio_norm=str(getattr(args, 'physio_norm', 'zscore')),
        subjects=getattr(args, 'subjects', None),
        tasks=getattr(args, 'tasks', None),
        max_clips_per_session=getattr(args, 'max_clips', None),
        max_entries=getattr(args, 'max_entries', None),
        verbose=bool(getattr(args, 'verbose', False)))


# --------------------------------------------------------------------------- #
# data verification / shape tests
# --------------------------------------------------------------------------- #
def _check_clip(ds: RGBRoiDataset, index: int) -> List[str]:
    """Verify one sample; returns a list of failure strings (empty == pass)."""
    fails: List[str] = []
    item = ds[index]
    entry = ds.entries[index]
    sess = entry['session']

    want = {'rgb_video', 'subject_task'} | {f'{n}_signal' for n in ds.signal_names}
    if set(item.keys()) != want:
        fails.append(f'keys {sorted(item.keys())} != {sorted(want)}')
    if item.get('subject_task') != sess:
        fails.append(f'subject_task {item.get("subject_task")!r} != {sess!r}')

    rgb = item['rgb_video']
    if rgb.dtype != torch.float32:
        fails.append(f'rgb_video dtype {rgb.dtype} != float32')
    if rgb.ndim != 4 or rgb.shape != (3, ds.clip_frames, ds.input_size,
                                      ds.input_size):
        fails.append(f'rgb_video {tuple(rgb.shape)} != (3, {ds.clip_frames}, '
                     f'{ds.input_size}, {ds.input_size})')
    else:
        lo, hi = float(rgb.min()), float(rgb.max())
        if not torch.isfinite(rgb).all():
            fails.append('rgb_video holds non-finite values')
        if lo < 0.0 or hi > 1.0:
            fails.append(f'rgb_video range [{lo:.4f}, {hi:.4f}] outside [0, 1]')
        if hi < 0.05:
            fails.append(f'rgb_video looks empty (max {hi:.4f})')

    for name in ds.signal_names:
        y = item[f'{name}_signal']
        if y.shape != (ds.phys_len,) or y.dtype != torch.float32:
            fails.append(f'{name}_signal {tuple(y.shape)}/{y.dtype} != '
                         f'({ds.phys_len},)/float32')
        elif not torch.isfinite(y).all():
            fails.append(f'{name}_signal holds non-finite values')

    # --- the unusable gate: a USED session must have jpg count == mat rows
    n_jpg = len(ds._files_of(sess))
    n_mat = int(ds._feat(sess).shape[0])
    if n_jpg != n_mat:
        fails.append(f'used session {sess} has {n_jpg} jpg vs {n_mat} mat '
                     f'frames (should have been marked unusable)')

    # --- ROI: box inside the frame and covering every target landmark
    src_h, src_w = ds._source_size(sess)
    pts = ds._feat(sess)[entry['frame_start']:entry['frame_end']][:, ds.target_idx, :]
    box = ds.clip_roi_box(index)
    x0, x1, y0, y1 = box
    if not (0 <= x0 < x1 <= src_w and 0 <= y0 < y1 <= src_h):
        fails.append(f'ROI box {box} outside the {src_w}x{src_h} frame')
    fx, fy = pts[..., 0].ravel(), pts[..., 1].ravel()
    if not (x0 <= fx.min() and fx.max() < x1 and y0 <= fy.min() and fy.max() < y1):
        fails.append('ROI box does not contain every target landmark')

    # --- the returned tensor IS the resize of that crop (frame 0)
    cv2 = _cv2()
    first = ds._read_rgb(os.path.join(entry['rgb_dir'],
                                      ds._files_of(sess)[entry['frame_start']]))
    h, w = first.shape[:2]
    sx, sy = w / src_w, h / src_h
    a0, a1 = int(math.floor(x0 * sx)), int(math.ceil(x1 * sx))
    b0, b1 = int(math.floor(y0 * sy)), int(math.ceil(y1 * sy))
    interp = (cv2.INTER_AREA if ds.input_size <= min(a1 - a0, b1 - b0)
              else cv2.INTER_LINEAR)
    expect = cv2.resize(first[b0:b1, a0:a1], (ds.input_size, ds.input_size),
                        interpolation=interp)
    got = np.rint(rgb[:, 0].permute(1, 2, 0).numpy() * 255.0)
    if expect.shape != got.shape or not np.allclose(expect, got, atol=1.0):
        fails.append('rgb_video[:, 0] is not the resize of the ROI crop')

    # --- 1-D alignment: frames x (phys_fs/fps) == the RAW-file slice
    ratio = ds.phys_fs / ds.fps
    if abs(ratio - round(ratio)) < 1e-9:
        for name in ds.signal_names:
            y_full = _load_1d(entry['signals'][name])
            s0 = entry['frame_start'] * int(round(ratio))
            brute = y_full[s0:s0 + ds.phys_len]
            raw = ds.signal_clip(index, name, normalized=False)
            if brute.shape != raw.shape or not np.allclose(brute, raw, atol=1e-3):
                fails.append(f'{name} window != raw[{s0}:{s0 + ds.phys_len}] of '
                             f'{os.path.basename(entry["signals"][name])}')

    # --- T frames x 1/fps == L samples x 1/phys_fs
    if abs(ds.clip_frames / ds.fps - ds.phys_len / ds.phys_fs) > 1e-9:
        fails.append(f'clip {ds.clip_frames / ds.fps:.6f} s != phys '
                     f'{ds.phys_len / ds.phys_fs:.6f} s')
    return fails


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python data/rgb_roi_dataset.py`` -> discovery + data verification."""
    import argparse

    p = argparse.ArgumentParser(
        description='Verify the RGB ROI dataset provider (discovery + checks).')
    p.add_argument('--raw_root', default=default_raw_root())
    p.add_argument('--subject', default=None, help='comma list, e.g. F001')
    p.add_argument('--task', default=None, help='comma list, e.g. T1,T2')
    p.add_argument('--signals', default='resp', help='comma list bp,resp,eda')
    p.add_argument('--clip_seconds', type=float, default=DEFAULT_CLIP_SECONDS)
    p.add_argument('--clip_stride', type=float, default=None)
    p.add_argument('--input_size', type=int, default=DEFAULT_INPUT_SIZE)
    p.add_argument('--roi_padding', type=float, default=DEFAULT_ROI_PADDING)
    p.add_argument('--roi_quantile', type=float, default=DEFAULT_ROI_QUANTILE)
    p.add_argument('--landmarks', default='face',
                   help='"face", a ROI_LANDMARKS_2D preset, or 1-indexed CSV')
    p.add_argument('--decode_scale', type=int, default=1, choices=DECODE_FACTORS)
    p.add_argument('--norm', default='none', choices=('none', 'clip', 'session'))
    p.add_argument('--max_entries', type=int, default=0, help='0 = no cap')
    p.add_argument('--n_check', type=int, default=3, help='clips to verify')
    p.add_argument('--list', action='store_true', help='print every session verdict')
    args = p.parse_args(argv)

    print('=' * 72)
    print('RGB ROI dataset provider -- discovery / data verification')
    print('=' * 72)

    if args.list:
        specs = discover_rgb_sessions(args.raw_root, subjects=_as_list(args.subject),
                                      tasks=_as_list(args.task))
        n_un = sum(1 for s in specs if s['status'] != 'usable')
        print(f'{len(specs)} sequence(s), {n_un} unusable:')
        for s in specs:
            tag = 'ok     ' if s['status'] == 'usable' else 'UNUSABLE'
            sig = ','.join(sorted(s['signals'])) or '-'
            print(f'  {tag} {s["session"]:10s} jpg={s["n_jpg"]:5d} '
                  f'mat={s["n_mat"]:5d} signals={sig:10s} {s["reason"]}')
        return 0

    ds = RGBRoiDataset(
        raw_root=args.raw_root, subjects=_as_list(args.subject),
        tasks=_as_list(args.task), signals=_as_list(args.signals) or ('resp',),
        clip_seconds=args.clip_seconds, clip_stride=args.clip_stride,
        input_size=args.input_size, roi_padding=args.roi_padding,
        roi_quantile=args.roi_quantile,
        landmarks=resolve_face_landmarks(args.landmarks),
        decode_scale=args.decode_scale, norm=args.norm,
        max_entries=args.max_entries or None, allow_empty=True)
    print(ds.describe())
    print('-' * 72)

    n = min(args.n_check, len(ds))
    for i in range(n):
        fails = _check_clip(ds, i)
        tag = ds.entries[i]['session']
        if fails:
            print(f'[FAIL] clip {i} ({tag})')
            for f in fails:
                print(f'       - {f}')
        else:
            print(f'[ok]   clip {i} ({tag}) rgb {tuple(ds[i]["rgb_video"].shape)}')
    if n == 0:
        print('[warn] no clips to verify (nothing usable passed the gates)')
    ok = all(not _check_clip(ds, i) for i in range(n))
    print('-' * 72)
    print('RESULT:', 'PASS' if ok and n else 'FAIL')
    return 0 if (ok and n) else 1


if __name__ == '__main__':                                    # pragma: no cover
    import sys as _sys
    _sys.exit(main())
