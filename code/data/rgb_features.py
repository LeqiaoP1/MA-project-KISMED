"""BP4D+ ``2DFeatures`` reader (RGB 49-point landmarks + head pose).

Companion to :mod:`data.tir_resp_dataset`, which owns the *thermal* raw tree.
This module owns the *visible-light* one::

    <raw_root>/2DFeatures/<S>_<T>.mat      # MATLAB v5, struct array 'fit'

Verified on the shipped files (F001_T1.mat, F001_T2.mat, 2026-09-26)
-------------------------------------------------------------------
``fit`` is a 1 x N struct array with fields ``frame`` / ``pts_2d`` / ``headPose``::

    fit[0, i]['frame']      scalar, 1-based frame index   (dtype varies per
                            element! uint8 for small values, uint16 later --
                            scipy picks the smallest type per cell, so ALWAYS
                            cast with int(np.ravel(f)[0]))
    fit[0, i]['pts_2d']     (49, 2) float64, (x, y) in RGB TEXTURE pixels
                            (the frame is 1392 rows x 1040 cols -- x < 1040,
                            y < 1392)
    fit[0, i]['headPose']   (3, 1) float32, Euler (pitch, yaw, roll) RADIANS

A frame the tracker lost is stored as an EMPTY field (``[]``), NOT as the
all-``(0, 0)`` sentinel the thermal ``IRFeatures`` track uses. Do not reuse
:func:`data.tir_resp_dataset.missing_frame_mask` here.

Index layout -- THE TRAP
------------------------
``pts_2d`` is indexed by the user-guide **Figure 1** (49 points); the thermal
track is indexed by **Figure 3** (28 points). The two are unrelated::

    thermal 28-pt (Figure 3)          2D 49-pt (Figure 1)
    ----------------------------------------------------------------
    9, 20   nose bridge R/L           11-14  nose bridge (VERTICAL chain)
    10, 21  nostril wings             15-19  nose base row (horizontal)
    11, 22  mouth corners             20-25  LEFT EYE
    12, 23  upper lip R/L             26-31  RIGHT EYE
    13, 24  lower lip R/L             32-49  mouth (32-43 outer, 44-49 inner)
    25, 26  lip centres               1-10   eyebrows

So passing the thermal ``TARGET_LANDMARKS = (9, 10, 11, 12, 13, 20, ..., 26)``
to this stream selects *brow + bridge + BOTH EYES*. Use :data:`ROI_LANDMARKS_2D`.

Verified by overlaying frame 1 of F001_T1 on ``2D+3D/F001/T1/0000.jpg``: the
groups above land on the anatomy named, e.g. ``15,19`` are the alar outer bases
(x 489.6 / 597.9), ``16,18`` the nares wings (516.9 / 572.2), ``17`` the tip on
the nose axis (x 544.8), ``32``/``38`` the mouth corners (x 447.2 / 639.2).
"""
import math
import os
from typing import Optional, Sequence, Tuple

import numpy as np

try:                                          # package import
    from .tir_resp_dataset import roi_box_from_landmarks
except ImportError:                           # direct script execution
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.tir_resp_dataset import roi_box_from_landmarks

__all__ = ['NUM_LANDMARKS_2D', 'GROUP_RANGES', 'ROI_LANDMARKS_2D',
           'find_2d_features', 'parse_2d_features', 'jpg_index_of',
           'roi_box_2d', 'resolve_roi_landmarks_2d']

#: 49 (x, y) points per frame in the 2DFeatures track (user-guide Figure 1).
NUM_LANDMARKS_2D = 49

#: Pose vector order, matching the .mat field.
POSE_NAMES = ('pitch', 'yaw', 'roll')

#: Anatomy of the 49-point layout, 1-indexed INCLUSIVE ranges. Single source of
#: truth for the overlay tool and any ROI reasoning.
GROUP_RANGES = (
    ('brow_left', 1, 5),
    ('brow_right', 6, 10),
    ('nose_bridge', 11, 14),
    ('nose_base', 15, 19),
    ('eye_left', 20, 25),
    ('eye_right', 26, 31),
    ('mouth_outer', 32, 43),
    ('mouth_inner', 44, 49),
)

#: Named 1-indexed landmark sets for the 2D ROI box, mirroring the thermal
#: presets in :mod:`data.tir_resp_dataset` so the same clip can be cropped
#: from either stream. ``nose_mouth`` is the anatomically matched default.
#:
#: NB ``nostrils`` / ``nose_tip`` are nearly COLLINEAR rows (the five nose-base
#: points span ~9 px vertically but ~108 px horizontally), so their min/max box
#: is a thin strip -- pass ``min_extent`` to :func:`roi_box_2d` (or the runner's
#: equivalent) or the 64x64 resize stretches a 13-px band into pure interpolation.
ROI_LANDMARKS_2D = {
    # thermal 9,10,11,12,13 / 20..26  ->  lower bridge + nose base + mouth
    'nose_mouth': (13, 14) + tuple(range(15, 20)) + tuple(range(32, 50)),
    # thermal 10,21 -> the paired nares wings (16,18) or the alar bases (15,19)
    'nostrils': (16, 18),
    # thermal 10,11,21,22 -> nares wings + mouth corners
    'nostril_mouth': (16, 18, 32, 38),
    # thermal 9,10,20,21 -> the whole nose-base row
    'nose_tip': tuple(range(15, 20)),
}


def resolve_roi_landmarks_2d(spec) -> Tuple[int, ...]:
    """Resolve a preset name / CSV of 1-indexed labels / sequence -> tuple.

    Same contract as :func:`data.tir_resp_dataset.resolve_roi_landmarks`, so a
    CLI flag can accept either stream's naming.
    """
    if isinstance(spec, str):
        key = spec.strip()
        if key in ROI_LANDMARKS_2D:
            return tuple(int(i) for i in ROI_LANDMARKS_2D[key])
        if not key:
            return tuple(int(i) for i in ROI_LANDMARKS_2D['nose_mouth'])
        if not all(tok.strip().lstrip('+-').isdigit() for tok in key.split(',')):
            raise ValueError(
                f'unknown 2D ROI landmarks {spec!r}; expected one of '
                f'{sorted(ROI_LANDMARKS_2D)} or a comma-separated list of '
                f'1-indexed labels in [1, {NUM_LANDMARKS_2D}]')
        labels = tuple(int(tok) for tok in key.split(','))
    else:
        labels = tuple(int(i) for i in spec)
    if not labels:
        raise ValueError('empty landmark set')
    bad = [i for i in labels if not 1 <= i <= NUM_LANDMARKS_2D]
    if bad:
        raise ValueError(
            f'2D landmark labels out of range 1..{NUM_LANDMARKS_2D}: {bad}')
    return labels


def find_2d_features(raw_root: str, subject: str, task: str) -> Optional[str]:
    """``<raw_root>/2DFeatures/<S>_<T>.mat`` (flat, like ``IRFeatures/``)."""
    for cand in (os.path.join(raw_root, '2DFeatures', f'{subject}_{task}.mat'),
                 os.path.join(raw_root, '2DFeatures', subject, f'{task}.mat')):
        if os.path.exists(cand):
            return cand
    return None


def parse_2d_features(path: str,
                      num_landmarks: int = NUM_LANDMARKS_2D
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a ``2DFeatures/<S>_<T>.mat`` track.

    Returns ``(pts, pose, frame, valid)``:

    * ``pts``   ``(N, 49, 2)`` float64, ``(x, y)`` in RGB texture pixels; NaN
      on frames the tracker lost;
    * ``pose``  ``(N, 3)`` float64, Euler (pitch, yaw, roll) in RADIANS; NaN
      where lost;
    * ``frame`` ``(N,)`` int64, the 1-based frame index from the file (verified
      strictly monotonic 1..N on the two shipped files);
    * ``valid`` ``(N,)`` bool, False where either field was empty ``[]`` or
      non-finite.

    The field dtype is not stable across cells (scipy stores the smallest type
    that fits each value), so every access goes through ``int(...)``/``float``.
    """
    try:
        from scipy.io import loadmat
    except ImportError as exc:                # pragma: no cover
        raise ImportError('scipy is required to read 2DFeatures .mat files') from exc

    if not os.path.exists(path):
        raise FileNotFoundError(f'2DFeatures file not found: {path}')
    try:
        mat = loadmat(path)
    except NotImplementedError as exc:
        raise NotImplementedError(
            f'{path} is MATLAB v7.3; read it with h5py instead of scipy.io') from exc
    if 'fit' not in mat:
        raise ValueError(f'{path}: no "fit" variable (keys: '
                         f'{[k for k in mat if not k.startswith("__")]})')
    fit = mat['fit']
    if fit.dtype.names is None or 'pts_2d' not in fit.dtype.names:
        raise ValueError(f'{path}: "fit" is not a struct with pts_2d '
                         f'(fields: {fit.dtype.names})')
    n = int(fit.shape[1])

    pts = np.full((n, num_landmarks, 2), np.nan, dtype=np.float64)
    pose = np.full((n, 3), np.nan, dtype=np.float64)
    frame = np.zeros(n, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        cell = fit[0, i]
        f = np.ravel(np.asarray(cell['frame']))
        frame[i] = int(f[0]) if f.size else i + 1
        p = np.asarray(cell['pts_2d'], dtype=np.float64)
        h = np.asarray(cell['headPose'], dtype=np.float64).ravel()
        if p.shape == (2, num_landmarks):     # defensive: transposed storage
            p = p.T
        if p.shape == (num_landmarks, 2) and h.size == 3:
            pts[i] = p
            pose[i] = h
            valid[i] = True
    valid &= np.isfinite(pts).all(axis=(1, 2)) & np.isfinite(pose).all(axis=1)
    pts[~valid] = np.nan
    pose[~valid] = np.nan
    return pts, pose, frame, valid


def jpg_index_of(frame_value: int) -> int:
    """Map the 1-based ``frame`` field to the 0-based ``%04d.jpg`` index.

    Verified: F001_T1 has ``frame`` 1..1612 and ``2D+3D/F001/T1/0000.jpg`` ..
    ``1611.jpg`` (all contiguous, so rank also equals index). The guide warns
    that occluded frames are simply NOT written while indices keep increasing,
    so index and rank can diverge in other sessions -- prefer this mapping.
    """
    return int(frame_value) - 1


def roi_box_2d(pts: np.ndarray, width: int, height: int,
               padding: float = 0.2, min_extent: int = 0,
               quantile: float = 0.0) -> Tuple[int, int, int, int]:
    """Clip-level static ROI box ``(x0, x1, y0, y1)`` for a 2D landmark clip.

    Thin wrapper over :func:`data.tir_resp_dataset.roi_box_from_landmarks` that
    additionally grows the SHORTER side to ``min_extent`` pixels about the box
    centre. Without it a collinear landmark set (``nostrils`` / ``nose_tip``,
    ~108 x 13 px) yields a strip, and resizing that to 64x64 fabricates ~5x
    vertical upsampling.

    :param pts: ``(T, 49, 2)`` (or any ``(..., 2)``) landmark coordinates.
    :param min_extent: 0 disables the guard (thermal behaviour).
    """
    x0, x1, y0, y1 = roi_box_from_landmarks(pts, width, height,
                                            padding=padding, quantile=quantile)
    if min_extent and min_extent > 0:
        me = int(min_extent)
        for short, lo, hi, lim in (('x', x0, x1, width), ('y', y0, y1, height)):
            if hi - lo < me:
                grow = me - (hi - lo)
                a = lo - grow // 2
                a = max(0, min(a, lim - me))
                b = min(lim, a + me)
                if short == 'x':
                    x0, x1 = a, b
                else:
                    y0, y1 = a, b
    return x0, x1, y0, y1


def _self_test(raw_root: Optional[str] = None) -> int:
    """Exercise the reader on whatever 2DFeatures files are present."""
    raw_root = raw_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        '..', 'data', 'raw', 'BP4D')
    raw_root = os.path.normpath(raw_root)
    d = os.path.join(raw_root, '2DFeatures')
    if not os.path.isdir(d):
        print(f'[skip] no {d}')
        return 0
    files = sorted(f for f in os.listdir(d) if f.endswith('.mat'))
    if not files:
        print(f'[skip] no .mat in {d}')
        return 0

    print('ROI_LANDMARKS_2D presets:')
    for k, v in ROI_LANDMARKS_2D.items():
        print(f'  {k:15s} n={len(v):2d}  {v if len(v) <= 8 else str(v[:6]) + "..."}')

    fails = 0
    for fname in files:
        subj, task = os.path.splitext(fname)[0].split('_')
        pts, pose, frame, valid = parse_2d_features(os.path.join(d, fname))
        n = len(frame)
        print(f'\n{fname}: N={n}  valid={valid.sum()}  '
              f'pose rad range={np.nanmin(pose):+.3f}..{np.nanmax(pose):+.3f}')
        mono = bool(np.all(np.diff(frame) == 1)) and frame[0] == 1
        print(f'  frame 1..{frame[-1]} strictly monotonic: {mono}')
        if not mono:
            print('  [warn] frame index is not 1..N contiguous -- map with '
                  'jpg_index_of(frame), do not assume rank')
        jd = os.path.join(raw_root, '2D+3D', subj, task)
        if os.path.isdir(jd):
            jpgs = sorted(f for f in os.listdir(jd) if f.endswith('.jpg'))
            ok = len(jpgs) == n
            print(f'  vs 2D+3D/{subj}/{task}: {len(jpgs)} jpgs vs {n} mat frames '
                  f'-> {"aligned" if ok else "MISMATCH"}')
            fails += 0 if ok else 1
        irf = os.path.join(raw_root, 'IRFeatures', f'{subj}_{task}.txt')
        if os.path.exists(irf):
            nir = sum(1 for _ in open(irf))
            print(f'  vs IRFeatures/{subj}_{task}.txt: {nir} lines '
                  f'({"match" if nir == n else f"DIFFERS by {nir - n}"})')

    # the anatomy claim must hold on real data: nose base is a flat row and the
    # mouth is wider than the bridge chain
    pts, pose, frame, valid = parse_2d_features(os.path.join(d, files[0]))
    p = pts[valid][0]
    nb = p[[14, 15, 16, 17, 18]]              # nose base row, 0-indexed
    # NB np.ptp(arr), not arr.ptp(): the ndarray method was removed in numpy 2.
    nb_dx, nb_dy = float(np.ptp(nb[:, 0])), float(np.ptp(nb[:, 1]))
    assert nb_dy < 0.25 * nb_dx, \
        'nose_base (15..19) should be a near-horizontal row'
    eye_l = p[19:25].mean(0)
    eye_r = p[25:31].mean(0)
    print(f'\n[ok] anatomy checks pass on {files[0]} frame 1: '
          f'nose_base is a flat row (dy={nb_dy:.1f} < 0.25*dx={0.25 * nb_dx:.1f}); '
          f'inter-ocular distance = {np.linalg.norm(eye_l - eye_r):.1f} px')
    return fails


if __name__ == '__main__':                    # pragma: no cover
    import sys as _s
    _s.exit(1 if _self_test(_s.argv[1] if len(_s.argv) > 1 else None) else 0)
