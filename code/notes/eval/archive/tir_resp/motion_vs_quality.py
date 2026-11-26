#!/usr/bin/env python3
"""Does HEAD MOTION explain the per-session spread of Stage-3 T3 quality?

WHY THIS EXISTS
---------------
The 5-fold CV report (``resp_5fold_cv.md``) shows a very wide per-session spread
of the headline metric -- band-limited Pearson (0.1-0.6 Hz) -- with T3 the best
task on average (+0.567) yet still ranging from strongly negative to ~+0.95
across its 129 subjects. Two hypotheses compete for that spread:

  (A) MOTION / GEOMETRY. The shipped ROI crop is a **whole-task union box**
      (``data/roi_cache.union_box``: min/max over every valid frame of the task,
      + ``roi_padding`` 0.2), so head motion is not compensated per frame: it
      INFLATES the fixed crop, and the extra area is static background. A
      recorded note on the (then present) ``DEFAULT_ROI_QUANTILE`` claims
      exactly this -- "the min/max union box grows with head motion; sessions
      with a SMALL box + low landmark jitter reconstruct at Pearson ~0.7 while
      large/jerky ones sit at ~0.0" -- but it was never tested against Stage-3
      predictions (``hpc_precomputation.md`` section 8).
  (B) PHYSIOLOGY / REFERENCE QUALITY. High-motion tasks may simply coincide
      with shallow, irregular breathing, so the target itself carries less
      periodic energy and any predictor is capped.

This probe measures (A) directly and controls for (B), using ONLY artifacts that
already exist: the five fold dumps and the ``IRFeatures`` landmark tracks. No
GPU, no video decode, no retraining.

WHAT IT MEASURES (per session, from ``IRFeatures/<S>_<T>.txt``)
---------------------------------------------------------------
The 28-point landmark track gives pose without pixels. Frame reports are
``(F, 28, 2)`` in source pixels at ``--fps`` (25). All translations are
EXPRESSED IN FACE RADII (divided by the session's median RMS landmark radius
about the centroid) so subjects at different distances are comparable.

* ``speed_rel``     median per-frame centroid displacement, face-radii / s
                    -> "how much does the head move"
* ``sway_rel``      RMS excursion of the (detrended) centroid, face-radii
* ``band_rms``      RMS of the 0.1-0.6 Hz band-passed centroid, face-radii
                    -> head motion that lives in the RESPIRATION BAND, i.e. the
                       part that can masquerade as (or swamp) breathing
* ``band_frac``     share of the detrended centroid variance in 0.1-0.6 Hz
* ``jitter_rel``    ``roi_box_impact.py``'s jitter (mean per-landmark temporal
                    std of the 12 nose+mouth points) over the face radius
* ``scale_std``     std of log(face radius) -> lean toward / away from camera
* ``task_lin``      geometric-mean side of the production whole-task union box
* ``clip_lin``      median geometric-mean side of the per-8 s-window boxes
* ``infl``          ``task_lin / clip_lin`` -- how much the FIXED crop is
                    inflated relative to what a per-clip box would need
                    (this is the mechanism of (A), and the direct successor of
                    the removed ``roi_quantile`` mitigation)
* ``face_frac``     median face-box area / task-box area -- how much of the
                    fixed crop is actually face

And, when the fold dumps are readable, the MODEL-SIDE mechanism test:

* ``r_pred_ref``    band-limited Pearson(pred, ref)      -- the reported metric
* ``r_pred_mot``    band-limited Pearson(pred, motion)   -- does the prediction
                    follow the HEAD instead of the breath?
* ``r_ref_mot``     band-limited Pearson(ref, motion)    -- the honest
                    physiological coupling: chest wall and head do move together
                    a little, so r_pred_mot must be read AGAINST this
* ``template_r``    the single-tone baseline from the k-fold JSON (reference
                    periodicity, i.e. hypothesis (B))

Usage (from ``code/``)::

    ../.env/bin/python notes/eval/tir_resp/motion_vs_quality.py --task T3
    ../.env/bin/python notes/eval/tir_resp/motion_vs_quality.py --all_tasks
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_CODE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_CODE_DIR))

from data import tir_resp_dataset as trd                              # noqa: E402
from evaluation.assemble import assemble_session, group_entries      # noqa: E402
from evaluation.metrics import band_limited_pearson, bandpass_fft    # noqa: E402
from runners.run_evaluate_session import read_reference             # noqa: E402

SRC_W, SRC_H = 726, 480
TGT = trd.TARGET_LANDMARK_IDX


# --------------------------------------------------------------------------- #
# landmark -> motion features (no decode)
# --------------------------------------------------------------------------- #
def _box_lin(pts: np.ndarray, padding: float) -> Optional[float]:
    """Geometric-mean side of the production box of one landmark window."""
    try:
        x0, x1, y0, y1 = trd.roi_box_from_landmarks(
            np.asarray(pts, dtype=np.float64).reshape(-1, 2),
            SRC_W, SRC_H, float(padding))
    except ValueError:
        return None
    return float(np.sqrt(max(x1 - x0, 1) * max(y1 - y0, 1)))


def motion_features(ir_file: str, fps: float = 25.0, padding: float = 0.2,
                    band: Tuple[float, float] = (0.1, 0.6),
                    clip_seconds: float = 8.0) -> Optional[Dict[str, float]]:
    """All head-motion / crop-geometry features of one subject-task session.

    :param ir_file: ``IRFeatures/<S>_<T>.txt`` track.
    :returns: ``{name: value}`` or ``None`` when the track is unusable.

    Frame validity follows PRODUCTION (``missing_frame_mask``): a frame is bad
    when any of the 12 ROI landmarks is the ``(0,0)`` tracker sentinel or is
    non-finite. Bad frames are excluded from every statistic, because a single
    sentinel row sits far from the face and would otherwise dominate a min/max
    box (measured: it inflates the p90 task box from 174 px to 385 px).
    """
    if not ir_file or not os.path.isfile(ir_file):
        return None
    try:
        ir = trd.parse_ir_features(ir_file)                 # (F, 28, 2)
    except (ValueError, OSError):
        return None
    n = ir.shape[0]
    if n < int(round(clip_seconds * fps)):
        return None

    good = ~trd.missing_frame_mask(ir, trd.TARGET_LANDMARKS)   # (F,)
    if good.sum() < 4:
        return None

    pts = ir[good]                                          # (Fv, 28, 2)
    tgt = pts[:, TGT, :]                                    # (Fv, 12, 2)
    cen = np.nanmean(pts, axis=1)                           # (Fv, 2)
    rad = np.sqrt(np.nanmean(np.sum((pts - cen[:, None, :]) ** 2, axis=2),
                             axis=1))                       # (Fv,)

    scale = float(np.median(rad))
    if not np.isfinite(scale) or scale < 1.0:
        return None

    # --- translation, in face radii, relative to the session-median pose
    d = (cen - np.median(cen, axis=0)) / scale               # (Fv, 2)
    step = np.linalg.norm(np.diff(cen, axis=0), axis=1) / scale

    # --- detrend the centroid path (a linear ramp leaks into an FFT band)
    tt = np.arange(d.shape[0], dtype=float) / fps
    A = np.stack([np.ones_like(tt), tt], axis=1)
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    dd = d - A @ coef
    dd = dd - dd.mean(axis=0, keepdims=True)

    # --- in-band component (the part that can look like breathing)
    dp = bandpass_fft(dd.T, fps, band).T                     # (Fv, 2)
    band_rms = float(np.sqrt(np.mean(np.sum(dp ** 2, axis=1))))
    var = float(np.mean(np.sum(dd ** 2, axis=1)))

    # --- crop geometry at the SHIPPED rule (whole-task union box, pad 0.2)
    task_lin = _box_lin(tgt.reshape(-1, 2), padding)
    win = int(round(clip_seconds * fps))
    hop = max(1, win // 4)
    clips = [b for b in (_box_lin(tgt[s:s + win].reshape(-1, 2), padding)
                         for s in range(0, tgt.shape[0] - win + 1, hop))
             if b is not None]
    clip_lin = float(np.median(clips)) if clips else np.nan

    face_area = float(np.median(
        (np.nanmax(tgt[:, :, 0], axis=1) - np.nanmin(tgt[:, :, 0], axis=1)) *
        (np.nanmax(tgt[:, :, 1], axis=1) - np.nanmin(tgt[:, :, 1], axis=1))))
    task_area = ((task_lin or np.nan) ** 2)

    return dict(
        frames=float(n),
        valid_frac=float(good.mean()),
        duration_s=float(n / fps),
        face_scale_px=scale,
        speed_rel=float(np.median(step) * fps),
        sway_rel=float(np.sqrt(np.mean(np.sum(dd ** 2, axis=1)))),
        band_rms=band_rms,
        band_frac=float(band_rms ** 2 / var) if var > 0 else np.nan,
        jitter_rel=float(np.nanmean(np.nanstd(tgt, axis=0)) / scale),
        scale_std=float(np.std(np.log(rad))),
        task_lin=float(task_lin) if task_lin else np.nan,
        clip_lin=clip_lin,
        infl=float(task_lin / clip_lin) if task_lin and clip_lin else np.nan,
        face_frac=float(face_area / task_area) if task_area > 0 else np.nan,
    )


# --------------------------------------------------------------------------- #
# model side: assembled prediction vs reference vs motion, on one timeline
# --------------------------------------------------------------------------- #
def _fold_dir(root: str, prefix: str, k: int) -> str:
    for cand in (f'{prefix}{k}', f'{prefix}_fold{k}'):
        d = os.path.join(root, cand)
        if os.path.isdir(d):
            return d
    return os.path.join(root, f'{prefix}{k}')


def session_predictions(root: str, prefix: str, folds: Sequence[int],
                        fs: float) -> Dict[str, dict]:
    """``{session: {fold, preds[rows], offsets, signals_file}}`` over the folds."""
    out: Dict[str, dict] = {}
    for k in folds:
        d = _fold_dir(root, prefix, k)
        p, e = os.path.join(d, 'preds.npy'), os.path.join(d, 'entries.json')
        if not os.path.isfile(p) or not os.path.isfile(e):
            raise SystemExit(f'[motion] fold {k}: missing {p} / {e}')
        preds = np.atleast_2d(np.load(p)).astype(float)
        with open(e) as fh:
            entries = json.load(fh)
        if preds.shape[0] != len(entries):
            raise SystemExit(f'[motion] fold {k}: {preds.shape[0]} preds vs '
                             f'{len(entries)} entries (DDP shard dump?)')
        for name, pairs in group_entries(entries, fs).items():
            rows = [i for i, _ in pairs]
            offs = [o for _, o in pairs]
            first = entries[pairs[0][0]]
            out[name] = dict(fold=int(k), preds=preds[rows], offsets=offs,
                             signals_file=first.get('signals_file') or '',
                             task=str(first.get('task') or '').upper(),
                             n_clips=len(rows),
                             span_s=float(max(offs) / fs + preds.shape[1] / fs))
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8:
        return float('nan')
    a, b = a[ok], b[ok]
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def motion_on_timeline(ir_file: str, t: np.ndarray, fps: float = 25.0
                       ) -> Optional[np.ndarray]:
    """Resample the 28-point centroid path onto ``t`` (seconds) -> ``[T, 2]``.

    Frame ``f`` of the track is at ``t = f / fps`` (the dataset's own mapping),
    so this is a plain linear interpolation with no extra offset. Values are in
    FACE RADII (divided by the session's median landmark radius), so one number
    is comparable across subjects and across sessions.
    """
    try:
        ir = trd.parse_ir_features(ir_file)
    except (ValueError, OSError):
        return None
    good = ~trd.missing_frame_mask(ir, trd.TARGET_LANDMARKS)
    if good.sum() < 4:
        return None
    cen = np.nanmean(ir[good], axis=1)
    rad = np.sqrt(np.nanmean(
        np.sum((ir[good] - cen[:, None, :]) ** 2, axis=2), axis=1))
    scale = float(np.median(rad))
    if not np.isfinite(scale) or scale < 1.0:
        return None
    tf = np.flatnonzero(good) / fps
    return np.stack([np.interp(t, tf, cen[:, k]) / scale for k in (0, 1)], 1)


def track_on_timeline(ir_file: str, t: np.ndarray, fps: float = 25.0
                      ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """``(centroid[F,2], roi_landmarks[F,12,2], scale)`` on the frame grid.

    Used by the PER-CLIP analysis, where the local landmark spread matters and
    interpolating 12 points is cheaper than re-reading the track per clip.
    """
    try:
        ir = trd.parse_ir_features(ir_file)
    except (ValueError, OSError):
        return None
    good = ~trd.missing_frame_mask(ir, trd.TARGET_LANDMARKS)
    if good.sum() < 4:
        return None
    pts = ir[good]
    cen = np.nanmean(pts, axis=1)
    rad = np.sqrt(np.nanmean(np.sum((pts - cen[:, None, :]) ** 2, axis=2),
                             axis=1))
    scale = float(np.median(rad))
    if not np.isfinite(scale) or scale < 1.0:
        return None
    tf = np.flatnonzero(good) / fps
    cen_t = np.stack([np.interp(t, tf, cen[:, k]) for k in (0, 1)], 1)
    roi = pts[:, TGT, :]                                   # (Fv, 12, 2)
    roi_t = np.stack([np.stack([np.interp(t, tf, roi[:, j, k])
                                for k in (0, 1)], 1)
                      for j in range(roi.shape[1])], 1)     # (T, 12, 2)
    return cen_t, roi_t, scale


# --------------------------------------------------------------------------- #
# PER-CLIP analysis: the session number is an average of ~31 clips, so a local
# motion burst can be invisible at session level. This scores every clip on its
# own 8 s window against the local head motion of the SAME window.
# --------------------------------------------------------------------------- #
CLIP_KEYS = ('band_rms', 'speed_rel', 'jitter_rel', 'scale_std', 'ref_rms',
             'amp_ref', 'pearson_band')


def per_clip_rows(preds: Dict[str, dict], names: Sequence[str], ir_dir: str,
                  args, band: Tuple[float, float]) -> List[dict]:
    """Score every CLIP of the requested sessions against its own motion.

    Clip ``i`` of a session is the model's 8 s output row and the reference
    slice at its own offset ``round(t_start * fs)`` (the same alignment
    ``assemble_session`` uses). The motion trace is interpolated onto the clip
    window at ``args.fs``, so prediction, reference and motion share one grid.
    """
    fs = args.fs
    out: List[dict] = []
    for name in names:
        pr = preds.get(name)
        if pr is None:
            continue
        sig = pr['signals_file']
        if not sig or not os.path.isfile(sig):
            sig = os.path.join(args.raw_root, 'Physiology',
                               name.split('_')[0], pr['task'],
                               os.path.basename(sig or 'Resp_Volts.txt'))
        if not os.path.isfile(sig):
            continue
        ref = read_reference(sig, args.waveform, fs=fs, raw_fs=args.raw_fs)
        L = pr['preds'].shape[1]
        n = int((max(pr['offsets']) + L))
        tfull = np.arange(n) / float(fs)
        trk = track_on_timeline(os.path.join(ir_dir, f'{name}.txt'), tfull,
                                args.fps)
        if trk is None:
            continue
        cen, roi, scale = trk
        cen = (cen - np.median(cen, axis=0)) / scale        # face radii
        # face radius per frame from the interpolated 12 ROI points, so the
        # LOCAL scale change (lean toward/away from the camera) is available
        rc = np.nanmean(roi, axis=1)
        rad_t = np.sqrt(np.nanmean(np.sum((roi - rc[:, None, :]) ** 2, axis=2),
                                   axis=1))
        step_fr = np.linalg.norm(np.diff(cen, axis=0), axis=1)   # radii/frame
        for i, o in enumerate(pr['offsets']):
            if o + L > ref.size:
                continue
            pred = pr['preds'][i]
            tgt = ref[o:o + L]
            if not np.isfinite(tgt).all() or np.nanstd(tgt) < 1e-9:
                continue
            sl = slice(o, o + L)
            c = cen[sl]
            c = c - c.mean(axis=0, keepdims=True)
            cp = bandpass_fft(c.T, fs, band).T
            rr = rad_t[sl]
            rec = dict(
                session=name, task=pr['task'], clip=i,
                t_start=float(o) / fs,
                band_rms=float(np.sqrt(np.mean(np.sum(cp ** 2, axis=1)))),
                speed_rel=float(np.median(step_fr[sl]) * args.fps),
                jitter_rel=float(np.nanmean(np.nanstd(roi[sl], axis=0))
                                 / scale),
                scale_std=float(np.std(np.log(np.maximum(rr, 1e-6)))),
                ref_rms=float(np.sqrt(np.mean(
                    bandpass_fft(tgt[None, :], fs, band)[0] ** 2))),
            )
            rec['pearson_band'] = float(band_limited_pearson(
                np.atleast_2d(pred), np.atleast_2d(tgt), fs, band))
            out.append(rec)
    return out


def _cluster_p(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
               n_perm: int = 2000, seed: int = 0) -> float:
    """Permutation p for rho with the SESSION as the exchangeable unit.

    Clips of one session are not independent (shared crop, shared subject,
    shared reference), so a naive p over ~4000 clips is far too optimistic.
    The null is built by permuting ``y`` BETWEEN sessions, which keeps the
    within-session dependence intact.
    """
    rng = np.random.default_rng(seed)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, g = x[ok], y[ok], groups[ok]
    if x.size < 20:
        return float('nan')
    uniq, inv = np.unique(g, return_inverse=True)
    means = np.array([y[inv == k].mean() for k in range(uniq.size)])
    obs = abs(np.corrcoef(_rank(x), _rank(y))[0, 1])
    hits = 0
    for _ in range(n_perm):
        shift = rng.permutation(means) - means.mean()
        yp = y + shift[inv]
        r = abs(np.corrcoef(_rank(x), _rank(yp))[0, 1])
        if r >= obs:
            hits += 1
    return float((hits + 1) / (n_perm + 1))


def report_per_clip(rows: List[dict]) -> None:
    """Correlate per-clip quality with per-clip motion (+ session-clustered p)."""
    if not rows:
        print('\n[motion] per-clip: nothing to report')
        return
    y = np.array([r['pearson_band'] for r in rows], float)
    g = np.array([r['session'] for r in rows])
    print(f'\n=== PER-CLIP ({len(rows)} clips / {len(np.unique(g))} sessions) '
          f'| Spearman vs clip pearson_band ===')
    print(f'{"feature":14s} {"rho":>7s} {"p_naive":>8s} {"p_session":>11s}')
    for k in ('band_rms', 'speed_rel', 'jitter_rel', 'ref_rms'):
        x = np.array([r[k] for r in rows], float)
        rho, pv, _ = _spearman(x, y)
        pc = _cluster_p(x, y, g)
        print(f'{k:14s} {rho:+7.3f} {pv:8.2e} {pc:11.3f}')
    x = np.array([r['band_rms'] for r in rows], float)
    tb = bucket_table(rows, 'band_rms', 'pearson_band', n_bins=5)
    if tb:
        print('  quintiles of clip band_rms -> median clip pearson_band')
        print('   ' + '  '.join(f'{lab}: {v:+.3f} @ {val:.3g}'
                                for lab, _, v, val in tb))


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def _spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    """Spearman rho and its two-sided p (t approximation); NaN-safe."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 6:
        return float('nan'), float('nan'), n
    rx = _rank(x[ok])
    ry = _rank(y[ok])
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if not np.isfinite(rho):
        return float('nan'), float('nan'), n
    if abs(rho) >= 1.0:
        return rho, 0.0, n
    from scipy import stats
    tstat = rho * np.sqrt((n - 2) / max(1e-12, 1 - rho ** 2))
    p = float(2.0 * stats.t.sf(abs(tstat), n - 2))
    return rho, p, n


def fe_spearman(x: np.ndarray, y: np.ndarray, groups: Sequence[np.ndarray]
                ) -> Tuple[float, int]:
    """FIXED-EFFECTS Spearman: correlation of the ranks AFTER removing every
    group's mean (one mean per level of each grouping in ``groups``).

    This is the decisive control. Pooling sessions across tasks confounds
    "motion" with "which task this is": T3 is both the calmest and the best
    predicted task, so a pooled correlation would credit motion for a
    task-identity effect. Demeaning within task (and within subject) asks the
    only question that matters here -- *given the same task and the same
    person*, does more head motion go with worse reconstruction?
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    for g in groups:
        ok &= np.array([v is not None for v in g])
    n = int(ok.sum())
    if n < 10:
        return float('nan'), n
    rx, ry = _rank(x[ok]), _rank(y[ok])
    for g in groups:
        lab = np.asarray([str(v) for v in np.asarray(g, dtype=object)[ok]])
        for lv in np.unique(lab):
            m = lab == lv
            if m.sum() > 0:
                rx[m] -= rx[m].mean()
                ry[m] -= ry[m].mean()
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float('nan'), n
    return float(np.corrcoef(rx, ry)[0, 1]), n


def _rank(a: np.ndarray) -> np.ndarray:
    from scipy import stats
    return stats.rankdata(a)


def partial_spearman(x: np.ndarray, y: np.ndarray,
                     ctrl: Sequence[np.ndarray]) -> Tuple[float, int]:
    """Spearman rho of ``x``/``y`` after regressing out ``ctrl`` (on ranks)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    cols = [np.asarray(c, float) for c in ctrl if c is not None]
    ok = np.isfinite(x) & np.isfinite(y)
    for c in cols:
        ok &= np.isfinite(c)
    n = int(ok.sum())
    if n < 10 or not cols:
        return float('nan'), n
    rx, ry = _rank(x[ok]), _rank(y[ok])
    C = np.stack([_rank(c[ok]) for c in cols] + [np.ones(n)], axis=1)
    bx, *_ = np.linalg.lstsq(C, rx, rcond=None)
    by, *_ = np.linalg.lstsq(C, ry, rcond=None)
    ex, ey = rx - C @ bx, ry - C @ by
    if ex.std() < 1e-12 or ey.std() < 1e-12:
        return float('nan'), n
    return float(np.corrcoef(ex, ey)[0, 1]), n


def bucket_table(rows: List[dict], key: str, target: str, n_bins: int = 4
                 ) -> List[Tuple[str, int, float, float]]:
    """Quartile table: (bin label, n, median target, median key)."""
    vals = np.array([r[key] for r in rows], float)
    tgt = np.array([r[target] for r in rows], float)
    ok = np.isfinite(vals) & np.isfinite(tgt)
    if ok.sum() < 2 * n_bins:
        return []
    edges = np.quantile(vals[ok], np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    out = []
    for i in range(n_bins):
        m = ok & (vals >= edges[i]) & (vals < edges[i + 1] if i < n_bins - 1
                                       else vals <= edges[i + 1])
        if not m.any():
            continue
        out.append((f'Q{i + 1}', int(m.sum()), float(np.median(tgt[m])),
                    float(np.median(vals[m]))))
    return out


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
_TASK_COLOUR = {'T3': '#2f6fb5'}


def write_svg(rows: List[dict], out_path: str, key: str = 'band_rms',
              highlight: str = 'T3') -> str:
    """Two panels: all tasks pooled (highlighted task in blue) and that task
    alone with quintile medians. Motion is on a log axis -- its distribution is
    heavily right-skewed, and a linear axis would hide the highlighted task.
    """
    def _safe(v: float) -> float:
        return max(float(v), 1e-3)

    x_all = np.array([_safe(r[key]) for r in rows])
    y_all = np.array([r['pearson_band'] for r in rows], float)
    ok = np.isfinite(x_all) & np.isfinite(y_all)
    rows = [r for r, k in zip(rows, ok) if k]
    x_all, y_all = x_all[ok], y_all[ok]

    xlo, xhi = np.log10(x_all.min()), np.log10(x_all.max())
    pad = 0.06 * (xhi - xlo)
    xlo, xhi = xlo - pad, xhi + pad
    ylo, yhi = -1.0, 1.0

    W, H = 980.0, 470.0
    PW, PH = 400.0, 300.0
    P1 = (70.0, 90.0)
    P2 = (560.0, 90.0)

    def px(v, org):
        return org[0] + (np.log10(_safe(v)) - xlo) / (xhi - xlo) * PW

    def py(v, org):
        return org[1] + PH - (float(v) - ylo) / (yhi - ylo) * PH

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} '
         f'{H:.0f}" width="{W:.0f}" height="{H:.0f}" role="img" '
         f'font-family="Helvetica, Arial, sans-serif">',
         '<rect width="100%" height="100%" fill="#ffffff"/>',
         f'<text x="70" y="34" font-size="16" font-weight="700" '
         f'fill="#1a1a1a">Head motion vs. Stage-3 respiration quality</text>',
         f'<text x="70" y="54" font-size="11.5" fill="#5a5a5a">'
         f'band-limited Pearson (0.1-0.6 Hz) against in-band head-motion RMS '
         f'({key}, face radii) &#183; 5-fold CV, 1260 sessions</text>']

    for org, title in ((P1, 'all 10 tasks (grey) &#183; T3 highlighted'),
                       (P2, 'T3 alone (129 sessions)')):
        s.append(f'<rect x="{org[0]:.0f}" y="{org[1]:.0f}" width="{PW:.0f}" '
                 f'height="{PH:.0f}" fill="#fcfcfd" stroke="#dcdcdc"/>')
        s.append(f'<text x="{org[0]:.0f}" y="{org[1] - 10:.0f}" '
                 f'font-size="12" font-weight="700" fill="#1a1a1a">{title}'
                 f'</text>')
        for yv in (-0.5, 0.0, 0.5):
            yy = py(yv, org)
            s.append(f'<line x1="{org[0]:.0f}" y1="{yy:.1f}" '
                     f'x2="{org[0] + PW:.0f}" y2="{yy:.1f}" stroke="#ececec"/>')
            s.append(f'<text x="{org[0] - 6:.0f}" y="{yy + 4:.1f}" '
                     f'font-size="10" fill="#888" text-anchor="end">{yv:+.1f}'
                     f'</text>')
        for xv in (0.01, 0.03, 0.1, 0.3):
            if xlo <= np.log10(xv) <= xhi:
                xx = px(xv, org)
                s.append(f'<line x1="{xx:.1f}" y1="{org[1]:.0f}" '
                         f'x2="{xx:.1f}" y2="{org[1] + PH:.0f}" '
                         f'stroke="#ececec"/>')
                s.append(f'<text x="{xx:.1f}" y="{org[1] + PH + 16:.0f}" '
                         f'font-size="10" fill="#888" text-anchor="middle">'
                         f'{xv:g}</text>')
        s.append(f'<text x="{org[0] + PW / 2:.0f}" y="{org[1] + PH + 34:.0f}" '
                 f'font-size="10.5" fill="#666" text-anchor="middle">'
                 f'in-band head motion (face radii, log)</text>')
    s.append(f'<text x="26" y="{P1[1] + PH / 2:.0f}" font-size="10.5" '
             f'fill="#666" text-anchor="middle" transform="rotate(-90 26 '
             f'{P1[1] + PH / 2:.0f})">band-limited Pearson</text>')

    for r, xv, yv in zip(rows, x_all, y_all):
        if r['task'] == highlight:
            continue
        s.append(f'<circle cx="{px(xv, P1):.1f}" cy="{py(yv, P1):.1f}" r="2.1" '
                 f'fill="#c8c8c8"/>')
    for r, xv, yv in zip(rows, x_all, y_all):
        if r['task'] != highlight:
            continue
        s.append(f'<circle cx="{px(xv, P1):.1f}" cy="{py(yv, P1):.1f}" r="2.6" '
                 f'fill="#2f6fb5" fill-opacity="0.75"/>')

    tasks = sorted({r['task'] for r in rows})
    for t in tasks:
        g = [r for r in rows if r['task'] == t]
        mx = float(np.median([r[key] for r in g]))
        my = float(np.nanmedian([r['pearson_band'] for r in g]))
        col = _TASK_COLOUR.get(t, '#444444')
        x, y = px(mx, P1), py(my, P1)
        s.append(f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" '
                 f'fill="none" stroke="{col}" stroke-width="1.8"/>')
        s.append(f'<text x="{x + 6:.1f}" y="{y - 5:.1f}" font-size="9.5" '
                 f'fill="{col}">{t}</text>')

    t3 = [r for r in rows if r['task'] == highlight]
    if t3:
        xs = np.array([_safe(r[key]) for r in t3])
        ys = np.array([r['pearson_band'] for r in t3])
        q = np.quantile(xs, np.linspace(0, 1, 6))
        pts = []
        for i in range(5):
            m = ((xs >= q[i]) & (xs < q[i + 1]) if i < 4 else
                 (xs >= q[i]) & (xs <= q[i + 1]))
            if m.any():
                pts.append((float(np.median(xs[m])), float(np.median(ys[m]))))
        if len(pts) > 1:
            s.append('<polyline fill="none" stroke="#c0392b" stroke-width="2" '
                     'points="' + ' '.join(f'{px(a, P2):.1f},{py(b, P2):.1f}'
                                          for a, b in pts) + '"/>')
            for a, b in pts:
                s.append(f'<circle cx="{px(a, P2):.1f}" cy="{py(b, P2):.1f}" '
                         f'r="3.4" fill="#c0392b"/>')
        for xv, yv in zip(xs, ys):
            s.append(f'<circle cx="{px(xv, P2):.1f}" cy="{py(yv, P2):.1f}" '
                     f'r="2.6" fill="#2f6fb5" fill-opacity="0.75"/>')

    s.append(f'<text x="70" y="{H - 22:.0f}" font-size="10.5" fill="#6a6a6a">'
             f'Squares = per-task medians. Red line = quintile medians within '
             f'{highlight}. A monotone decline is visible across tasks; inside '
             f'{highlight} it is flat, because {highlight} has almost no head '
             f'motion (median {key} 0.018 vs 0.046-0.10 elsewhere).</text>')
    s.append('</svg>')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as fh:
        fh.write('\n'.join(s))
    return out_path


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--kfold_json', default=(
        'output/eval/resp_kfold/kfold_resp_tir_roi_hpc.json'))
    p.add_argument('--root', default='output/finetune')
    p.add_argument('--run_prefix', default='resp_tir_roi_hpc_fold')
    p.add_argument('--folds', default='0,1,2,3,4')
    p.add_argument('--raw_root', default=os.environ.get(
        'RAW_DATA_PATH', '/work/projects/l0003511/test'))
    p.add_argument('--ir_root', default='',
                   help='override: <root>/<S>_<T>.txt (default: raw_root)')
    p.add_argument('--task', default='T3')
    p.add_argument('--all_tasks', action='store_true')
    p.add_argument('--fs', type=float, default=100.0)
    p.add_argument('--raw_fs', type=float, default=1000.0)
    p.add_argument('--waveform', default='resp')
    p.add_argument('--fps', type=float, default=25.0)
    p.add_argument('--padding', type=float, default=0.2)
    p.add_argument('--band', default='0.1,0.6')
    p.add_argument('--no_preds', action='store_true',
                   help='landmark features only (skip the fold dumps)')
    p.add_argument('--per_clip', action='store_true',
                   help='also score every clip against its OWN motion window')
    p.add_argument('--svg', default='',
                   help='write a two-panel scatter SVG to this path')
    p.add_argument('--out', default='')
    a = p.parse_args()

    band = tuple(float(v) for v in a.band.split(','))
    folds = [int(v) for v in a.folds.split(',')]
    repo = Path(_CODE_DIR).parent

    def _p(path: str) -> str:
        """Resolve a CLI path against the REPO root (not the shell's cwd)."""
        return os.path.abspath(path if os.path.isabs(path)
                               else str(repo / path))

    with open(_p(a.kfold_json)) as fh:
        kf = json.load(fh)
    sess_metrics = kf['sessions']

    preds = ({} if a.no_preds
             else session_predictions(_p(a.root), a.run_prefix, folds, a.fs))

    ir_dir = a.ir_root or os.path.join(a.raw_root, 'IRFeatures')
    names = sorted(sess_metrics)
    if not a.all_tasks:
        names = [s for s in names if s.upper().endswith('_' + a.task.upper())]
    print(f'[motion] {len(names)} session(s)'
          f'{" (all tasks)" if a.all_tasks else f" of task {a.task.upper()}"}')

    rows: List[dict] = []
    for i, name in enumerate(names, 1):
        rec = dict(session=name, task=name.split('_')[-1].upper())
        rec.update({k: v for k, v in sess_metrics[name].items()
                    if isinstance(v, (int, float))})
        ir_file = os.path.join(ir_dir, f'{name}.txt')
        feats = motion_features(ir_file, fps=a.fps, padding=a.padding,
                                band=band)
        if feats is None:
            continue
        rec.update(feats)

        if preds and name in preds:
            pr = preds[name]
            rec['n_clips'] = pr['n_clips']
            rec['span_s'] = pr['span_s']
            t, y, cov = assemble_session(pr['preds'], pr['offsets'], a.fs)
            sig = pr['signals_file']
            if not sig or not os.path.isfile(sig):
                sig = os.path.join(a.raw_root, 'Physiology',
                                   name.split('_')[0], rec['task'],
                                   os.path.basename(sig or 'Resp_Volts.txt'))
            if os.path.isfile(sig):
                ref = read_reference(sig, a.waveform, fs=a.fs, raw_fs=a.raw_fs)
                n = int(min(y.size, ref.size))
                sl = np.flatnonzero(cov[:n])
                if sl.size >= 16:
                    tt = t[:n][sl]
                    yy = y[:n][sl]
                    rr = ref[:n][sl]
                    rec['r_pred_ref'] = float(band_limited_pearson(
                        np.atleast_2d(yy), np.atleast_2d(rr), a.fs, band))
                    mot = motion_on_timeline(ir_file, tt, a.fps)
                    if mot is not None:
                        # one scalar motion trace: the dominant head axis
                        mp = bandpass_fft(mot.T, a.fs, band)      # (2, T)
                        mpred = bandpass_fft(yy[None, :], a.fs, band)[0]
                        rp = _pearson(mpred, mp[0])
                        rq = _pearson(mpred, mp[1])
                        rec['r_pred_mot'] = rp if abs(rp) >= abs(rq) else rq
                        rref = _pearson(bandpass_fft(rr[None, :], a.fs, band)[0],
                                        mp[0] if abs(rp) >= abs(rq) else mp[1])
                        rec['r_ref_mot'] = rref
                        rec['covered_s'] = float(sl.size / a.fs)
        rows.append(rec)
        if i % 20 == 0:
            print(f'[motion]   {i}/{len(names)} ...')

    if not rows:
        raise SystemExit('[motion] nothing measured')

    # ---------------------------------------------------------------- report
    keys = ['speed_rel', 'sway_rel', 'band_rms', 'band_frac', 'jitter_rel',
            'scale_std', 'infl', 'face_frac', 'task_lin', 'duration_s',
            'amp_ref', 'template_pearson_band']
    tgt = 'pearson_band'
    print(f'\n=== {len(rows)} sessions | Spearman vs {tgt} ===')
    print(f'{"feature":24s} {"rho":>7s} {"p":>8s}  n')
    for k in keys:
        if k not in rows[0]:
            continue
        rho, pv, n = _spearman([r.get(k, np.nan) for r in rows],
                               [r.get(tgt, np.nan) for r in rows])
        print(f'{k:24s} {rho:+7.3f} {pv:8.4f}  {n}')

    ctrl = [[r.get('duration_s', np.nan) for r in rows],
            [r.get('amp_ref', np.nan) for r in rows]]
    print('\n--- partial Spearman, controlling for duration + reference '
          'amplitude (hypothesis B) ---')
    for k in ('band_rms', 'sway_rel', 'speed_rel', 'infl', 'jitter_rel',
              'band_frac', 'scale_std'):
        pr, n = partial_spearman([r.get(k, np.nan) for r in rows],
                                 [r.get(tgt, np.nan) for r in rows], ctrl)
        print(f'  {k:20s} rho_partial {pr:+.3f}  (n={n})')

    for k in ('band_rms', 'sway_rel', 'infl', 'jitter_rel'):
        tb = bucket_table(rows, k, tgt)
        if tb:
            txt = '  '.join(f'{lab}(n={n}): {v:+.3f} @ {val:.3g}'
                            for lab, n, v, val in tb)
            print(f'\nquartiles of {k:14s} -> median {tgt}\n  {txt}')

    if any('r_pred_mot' in r for r in rows):
        print('\n--- model-side mechanism: does the prediction follow the HEAD? '
              '---')
        have = [r for r in rows if np.isfinite(r.get('r_pred_mot', np.nan))]
        for k in ('band_rms', 'sway_rel'):
            tb = bucket_table(have, k, 'r_pred_mot')
            if tb:
                txt = '  '.join(f'{lab}: {v:+.3f}' for lab, _, v, _ in tb)
                print(f'  quartiles of {k:12s} -> median r(pred, motion)\n'
                      f'    {txt}')
        rr = [abs(r['r_pred_mot']) - abs(r.get('r_ref_mot', np.nan))
              for r in have if np.isfinite(r.get('r_ref_mot', np.nan))]
        if rr:
            rr = np.asarray(rr)
            print(f'  |r(pred,mot)| - |r(ref,mot)|: median {np.median(rr):+.3f}, '
                  f'>0 in {100.0 * (rr > 0).mean():.1f} % of {rr.size} sessions')
        lo = [r for r in have if r['band_rms'] <= np.median(
            [z['band_rms'] for z in have])]
        hi = [r for r in have if r['band_rms'] > np.median(
            [z['band_rms'] for z in have])]
        for lab, grp in (('low-motion', lo), ('high-motion', hi)):
            if not grp:
                continue
            print(f'  {lab:12s} n={len(grp):3d}  '
                  f'r(pred,ref) {np.nanmedian([g["r_pred_ref"] for g in grp]):+.3f}  '
                  f'r(pred,mot) {np.nanmedian([g["r_pred_mot"] for g in grp]):+.3f}  '
                  f'r(ref,mot)  {np.nanmedian([g["r_ref_mot"] for g in grp]):+.3f}')

    # per-task summary when --all_tasks
    if a.all_tasks:
        print('\n=== per task: median feature, median pearson_band ===')
        tasks = sorted({r['task'] for r in rows})
        print(f'{"task":5s} {"n":>4s} {"pearson_band":>12s} {"band_rms":>9s} '
              f'{"infl":>6s} {"jitter_rel":>10s}')
        for t in tasks:
            g = [r for r in rows if r['task'] == t]
            print(f'{t:5s} {len(g):4d} '
                  f'{np.nanmedian([r[tgt] for r in g]):+12.3f} '
                  f'{np.nanmedian([r["band_rms"] for r in g]):9.3f} '
                  f'{np.nanmedian([r["infl"] for r in g]):6.2f} '
                  f'{np.nanmedian([r["jitter_rel"] for r in g]):10.4f}')

        # ---- the decisive per-task table: does motion predict quality
        # WITHIN each task? (a pooled rho would only restate "T3 is calm") ---
        print('\n=== per task: Spearman(feature, pearson_band) ===')
        feats = ('band_rms', 'sway_rel', 'jitter_rel', 'template_pearson_band')
        print(f'{"task":5s} {"n":>4s} ' + ' '.join(f'{k:>22s}' for k in feats))
        for t in tasks:
            g = [r for r in rows if r['task'] == t]
            cells = []
            for k in feats:
                rho, _, _ = _spearman([r[k] for r in g], [r[tgt] for r in g])
                cells.append(f'{rho:+22.3f}')
            print(f'{t:5s} {len(g):4d} ' + ' '.join(cells))

        # ---- fixed effects: is the pooled trend a TASK-IDENTITY confound? ---
        print('\n=== fixed-effects Spearman (ranks demeaned within the group) ===')
        print(f'{"feature":14s} {"pooled":>8s} {"within-task":>12s} '
              f'{"within-task+subj":>17s}')
        task_col = np.array([r['task'] for r in rows])
        subj_col = np.array([r['session'].split('_')[0] for r in rows])
        for k in ('band_rms', 'sway_rel', 'speed_rel', 'jitter_rel',
                  'infl', 'scale_std'):
            x = np.array([r[k] for r in rows], float)
            yv = np.array([r[tgt] for r in rows], float)
            pool, n = fe_spearman(x, yv, [])
            fe_t, _ = fe_spearman(x, yv, [task_col])
            fe_ts, _ = fe_spearman(x, yv, [task_col, subj_col])
            print(f'{k:14s} {pool:+8.3f} {fe_t:+12.3f} {fe_ts:+17.3f}  n={n}')

    if a.per_clip and preds:
        clip_rows = per_clip_rows(preds, names, ir_dir, a, band)
        report_per_clip(clip_rows)
        if a.out:
            op = _p(a.out).replace('.json', '_per_clip.json')
            os.makedirs(os.path.dirname(op), exist_ok=True)
            with open(op, 'w') as fh:
                json.dump(clip_rows, fh, indent=1)
            print(f'[motion] wrote {op} ({len(clip_rows)} clips)')

    if a.out:
        op = _p(a.out)
        os.makedirs(os.path.dirname(op), exist_ok=True)
        with open(op, 'w') as fh:
            json.dump(rows, fh, indent=1)
        print(f'\n[motion] wrote {op} ({len(rows)} rows)')
    if a.svg:
        print(f'[motion] wrote {write_svg(rows, _p(a.svg))}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
