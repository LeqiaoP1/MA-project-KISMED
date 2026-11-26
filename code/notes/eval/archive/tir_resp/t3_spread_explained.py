#!/usr/bin/env python3
"""What ELSE explains the T3 spread, once head motion is ruled out?

``resp_motion_confound.md`` established that head motion does not move T3
quality (rho -0.046). But the only significant T3 correlate found there was
``template_pearson_band`` (+0.349, R^2 = 12 %), i.e. ~85 % of the spread was
still unexplained. This probe tests the remaining candidate axes in one place
and ranks them by how much HONEST (leave-one-out) variance each one adds.

The candidates, and why each is plausible
-----------------------------------------
REFERENCE SIDE -- the target itself may be unlearnable:
* ``template_pearson_band``  how periodic the reference is (known: +0.349)
* ``rate_bpm``               breathing rate; the loss only polices rate down to
                             ~0.2 Hz (a 5.12 s STFT window), and the band is
                             0.1-0.6 Hz, so a slow or fast breather can sit at
                             the edge of what the model can express
* ``band_snr``               in-band power / total detrended power -- a clean
                             breath against drift and noise
* ``drift_ratio``            <0.1 Hz drift RMS / in-band RMS
* ``period_jitter``          std/mean of the instantaneous breathing frequency
                             (how irregular the rhythm is)
* ``amp_cv``                 breath-to-breath depth variability
* ``cycles``                 duration x rate = how many breaths the metric sees

SUBJECT SIDE -- a stable trait, not a session accident:
* ``other_tasks_quality``    mean ``pearson_band`` of the SAME subject on the
                             OTHER tasks. If this is large, the T3 spread is a
                             person effect (anatomy / thermal contrast /
                             breathing habit), not a T3 effect.
* ``gender``                 F/M from the subject id
* ``face_scale_px``          landmark RMS radius -- distance from the camera
* ``face_px_in_input``       face side x input_size / crop side = how many of
                             the 112 input pixels the face actually occupies
                             (a small face in a motion-inflated crop is heavily
                             downsampled -- the "resolution" axis, distinct
                             from motion)

MODEL SIDE -- 5 folds are 5 different models:
* ``fold``                   each fold has its own Stage-2 encoder and Stage-3
                             head, so part of the spread is model variance

Ranking method: univariate Spearman, then a nested OLS on standardised ranks
with LEAVE-ONE-OUT R^2, and a per-feature increment. LOOCV is the honest
number here -- 129 sessions and a dozen correlated features will otherwise
produce a flattering in-sample R^2.

Usage (from ``code/``)::

    ../.env/bin/python notes/eval/tir_resp/t3_spread_explained.py
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

from evaluation.assemble import assemble_session, group_entries      # noqa: E402
from evaluation.metrics import bandpass_fft                          # noqa: E402
from runners.run_evaluate_session import read_reference             # noqa: E402


# --------------------------------------------------------------------------- #
# reference-side features
# --------------------------------------------------------------------------- #
def reference_features(ref: np.ndarray, fs: float, band: Tuple[float, float]
                       ) -> Dict[str, float]:
    """Describe the TARGET of one session: rate, SNR, drift, irregularity.

    :param ref: the reference samples on the assembled covered span.
    :param fs: sample rate of ``ref``.
    """
    x = np.asarray(ref, float)
    n = x.size
    if n < 32:
        return {}
    t = np.arange(n) / fs
    A = np.stack([np.ones_like(t), t], axis=1)
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    d = x - A @ coef                                   # detrended
    d = d - d.mean()
    lo, hi = band
    bp = bandpass_fft(d[None, :], fs, band)[0]
    lp = d - bp                                        # < lo and > hi
    var_d = float(np.var(d))
    var_b = float(np.var(bp))

    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    m = (freqs >= lo) & (freqs <= hi)
    spec = np.abs(np.fft.rfft(bp))
    if not m.any():
        return {}
    k = int(np.argmax(spec[m]))
    rate_hz = float(freqs[m][k])

    # instantaneous frequency / envelope from the analytic signal
    from scipy import signal as sg
    z = sg.hilbert(bp)
    env = np.abs(z)
    ph = np.unwrap(np.angle(z))
    ifreq = np.gradient(ph) * fs / (2.0 * np.pi)
    ok = np.isfinite(ifreq) & (ifreq > 0)
    pj = (float(np.std(ifreq[ok]) / np.mean(ifreq[ok]))
          if ok.sum() > 8 and np.mean(ifreq[ok]) > 0 else np.nan)

    return dict(
        rate_bpm=rate_hz * 60.0,
        band_snr=var_b / var_d if var_d > 0 else np.nan,
        drift_ratio=(float(np.sqrt(np.var(lp))) /
                     float(np.sqrt(var_b))) if var_b > 0 else np.nan,
        period_jitter=pj,
        amp_cv=(float(np.std(env) / np.mean(env))
                if np.mean(env) > 0 else np.nan),
        cycles=float(n / fs) * rate_hz,
        ref_rms=float(np.sqrt(var_b)),
        ref_std=float(np.std(x)),
    )


# --------------------------------------------------------------------------- #
# thermal contrast of the model's INPUT crop (needs the ROI cache)
# --------------------------------------------------------------------------- #
def roi_contrast(shard_path: str, fps: float, band: Tuple[float, float],
                 stride: int = 5) -> Dict[str, float]:
    """How much thermal detail the model's fixed crop actually contains.

    The shipped input is the whole-task union crop, resized to 112x112, so a
    subject whose palette rendering is flat (saturated in the red channel, little
    spatial spread in green) has little for the encoder to read no matter how
    still the head is. This is the one candidate axis the landmark track cannot
    see, and it is a SUBJECT property (skin, ambient, distance), which matches
    the cross-task consistency measured below.

    :param shard_path: one ``tasks/<S>_<T>.npz`` from the ROI cache.
    :param stride: keep every ``stride``-th frame (temporal statistics do not
        need 25 fps; the 0.1-0.6 Hz band is far below the resulting rate).
    """
    from data import roi_cache as rc
    from data import tir_resp_dataset as trd
    sh = rc.read_shard(shard_path)
    frames = sh['frames'][::stride]
    lm = sh['landmarks'][::stride] if sh['landmarks'] is not None else None
    if frames.shape[0] < 8:
        return {}
    x0, y0 = sh['box'][0], sh['box'][2]
    H, W = frames.shape[1], frames.shape[2]
    f = frames.astype(np.float32)
    g = f[..., 1]                                        # red is saturated
    glob = g.reshape(g.shape[0], -1).mean(axis=1)        # AGC / global drift
    gq = g - glob[:, None, None]

    out: Dict[str, float] = {
        'sat_frac': float((f[..., 0] >= 250).mean()),
        'crop_g_std': float(np.median(g.reshape(g.shape[0], -1).std(axis=1))),
    }
    if lm is not None:
        tgt = lm[:, trd.TARGET_LANDMARK_IDX, :].copy()
        tgt[..., 0] -= x0
        tgt[..., 1] -= y0
        q = np.full(g.shape[0], np.nan)
        spread = np.full(g.shape[0], np.nan)
        sat = np.full(g.shape[0], np.nan)
        for i in range(g.shape[0]):
            p = tgt[i]
            if not np.isfinite(p).all():
                continue
            a = int(np.clip(np.floor(p[:, 0].min()), 0, W - 2))
            b = int(np.clip(np.ceil(p[:, 0].max()), a + 1, W - 1))
            c = int(np.clip(np.floor(p[:, 1].min()), 0, H - 2))
            d = int(np.clip(np.ceil(p[:, 1].max()), c + 1, H - 1))
            patch = gq[i, c:d, a:b]
            if patch.size < 4:
                continue
            q[i] = float(patch.mean())
            spread[i] = float(patch.std())
            sat[i] = float((f[i, c:d, a:b, 0] >= 250).mean())
        if np.isfinite(q).sum() > 8:
            rate = fps / float(stride)
            idx = np.flatnonzero(np.isfinite(q))
            qq = q[idx]
            qq = qq - qq.mean()
            bp = bandpass_fft(qq[None, :], rate, band)[0]
            out.update(
                roi_thermal_tot=float(np.std(qq)),
                roi_thermal_band=float(np.sqrt(np.mean(bp ** 2))),
                roi_thermal_frac=(float(np.mean(bp ** 2) /
                                        max(np.mean(qq ** 2), 1e-12))),
                roi_g_std=float(np.nanmedian(spread)),
                roi_sat_frac=float(np.nanmedian(sat)),
            )
    return out


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def _rank(a: np.ndarray) -> np.ndarray:
    from scipy import stats
    return stats.rankdata(np.asarray(a, float))


def _z(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    s = a.std()
    return (a - a.mean()) / s if s > 1e-12 else np.zeros_like(a)


def spearman(x: Sequence[float], y: Sequence[float]
             ) -> Tuple[float, float, int]:
    from scipy import stats
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 6:
        return float('nan'), float('nan'), n
    rho, p = stats.spearmanr(x[ok], y[ok])
    return float(rho), float(p), n


def loo_r2(X: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out R^2 of an OLS with intercept (0.0 = no better than mean)."""
    n = y.size
    if X.shape[0] != n or n < 8:
        return float('nan')
    Xd = np.column_stack([np.ones(n), X])
    pred = np.zeros(n)
    for i in range(n):
        keep = np.arange(n) != i
        b, *_ = np.linalg.lstsq(Xd[keep], y[keep], rcond=None)
        pred[i] = Xd[i] @ b
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default='output/finetune')
    p.add_argument('--run_prefix', default='resp_tir_roi_hpc_fold')
    p.add_argument('--folds', default='0,1,2,3,4')
    p.add_argument('--rows', default=(
        'output/eval/resp_kfold/all_tasks_motion_vs_quality.json'),
        help='per-session rows from motion_vs_quality.py --all_tasks')
    p.add_argument('--kfold_json', default=(
        'output/eval/resp_kfold/kfold_resp_tir_roi_hpc.json'))
    p.add_argument('--raw_root', default=os.environ.get(
        'RAW_DATA_PATH', '/work/projects/l0003511/test'))
    p.add_argument('--task', default='T3')
    p.add_argument('--fs', type=float, default=100.0)
    p.add_argument('--raw_fs', type=float, default=1000.0)
    p.add_argument('--waveform', default='resp')
    p.add_argument('--band', default='0.1,0.6')
    p.add_argument('--input_size', type=int, default=112,
                   help='model input side (Stage-3 ships 112)')
    p.add_argument('--roi_cache', default=os.environ.get(
        'TIR_ROI_CACHE',
        '/work/scratch/ne95ocyg/tir_roi_cache/'
        'tirroi_nose_mouth_pad0.20_minmax_src726x480_native_v2_9d0214aefa'),
        help='ROI-cache root; empty string disables the thermal-contrast axis')
    p.add_argument('--stride', type=int, default=5,
                   help='frame stride for the thermal-contrast statistics')
    p.add_argument('--out', default='')
    a = p.parse_args()

    repo = Path(_CODE_DIR).parent

    def _p(path: str) -> str:
        return os.path.abspath(path if os.path.isabs(path)
                               else str(repo / path))

    band = tuple(float(v) for v in a.band.split(','))
    folds = [int(v) for v in a.folds.split(',')]
    task = a.task.upper()

    allrows = json.load(open(_p(a.rows)))
    rows = [dict(r) for r in allrows if r['task'] == task]
    print(f'[spread] {len(rows)} {task} sessions; '
          f'{len(allrows)} sessions in the corpus for cross-task features')

    # --- fold identity (which MODEL produced this session's prediction) ------
    fold_of: Dict[str, int] = {}
    for k in folds:
        for cand in (f'{a.run_prefix}{k}', f'{a.run_prefix}_fold{k}'):
            d = os.path.join(_p(a.root), cand)
            if os.path.isdir(d):
                with open(os.path.join(d, 'entries.json')) as fh:
                    for e in json.load(fh):
                        fold_of[e['session']] = k
                break

    # --- cross-task subject trait -------------------------------------------
    by_subject: Dict[str, List[dict]] = {}
    for r in allrows:
        by_subject.setdefault(r['session'].split('_')[0], []).append(r)

    # --- reference features (needs only the reference file per session) -----
    sig_of: Dict[str, str] = {}
    for k in folds:
        for cand in (f'{a.run_prefix}{k}', f'{a.run_prefix}_fold{k}'):
            d = os.path.join(_p(a.root), cand)
            if not os.path.isdir(d):
                break
            with open(os.path.join(d, 'entries.json')) as fh:
                for e in json.load(fh):
                    sig_of[e['session']] = e.get('signals_file') or ''
            break

    for r in rows:
        name = r['session']
        subj = name.split('_')[0]
        r['fold'] = fold_of.get(name, -1)
        r['gender'] = 1.0 if subj.upper().startswith('F') else 0.0
        others = [z['pearson_band'] for z in by_subject.get(subj, [])
                  if z['task'] != task]
        r['n_other_tasks'] = float(len(others))
        r['other_tasks_quality'] = float(np.median(others)) if others else np.nan
        # effective face resolution inside the model's fixed input
        face_frac = float(r.get('face_frac', np.nan))
        r['face_px_in_input'] = (np.sqrt(face_frac) * a.input_size
                                 if np.isfinite(face_frac) else np.nan)

        sig = sig_of.get(name, '')
        if not sig or not os.path.isfile(sig):
            sig = os.path.join(a.raw_root, 'Physiology', subj, task,
                               os.path.basename(sig or 'Resp_Volts.txt'))
        if not os.path.isfile(sig):
            continue
        # the assembled covered span, reconstructed from duration_s
        ref_full = read_reference(sig, a.waveform, fs=a.fs, raw_fs=a.raw_fs)
        L = int(round(float(r['duration_s']) * a.fs))
        r.update(reference_features(ref_full[:L], a.fs, band))

        if a.roi_cache:
            sh = os.path.join(a.roi_cache, 'tasks', f'{name}.npz')
            if os.path.isfile(sh):
                try:
                    r.update(roi_contrast(sh, 25.0, band, a.stride))
                except (ValueError, OSError, KeyError) as exc:
                    print(f'[spread] {name}: roi_contrast failed ({exc})')

    # ---------------------------------------------------------------- report
    feats = ['template_pearson_band', 'rate_bpm', 'band_snr', 'drift_ratio',
             'period_jitter', 'amp_cv', 'cycles', 'ref_rms', 'ref_std',
             'amp_ref', 'duration_s', 'n_clips', 'other_tasks_quality',
             'gender', 'face_scale_px', 'face_px_in_input', 'task_lin',
             'clip_lin', 'infl', 'jitter_rel', 'band_rms', 'valid_frac',
             'sat_frac', 'crop_g_std', 'roi_thermal_tot', 'roi_thermal_band',
             'roi_thermal_frac', 'roi_g_std', 'roi_sat_frac']
    tgt = 'pearson_band'

    print(f'\n=== univariate Spearman vs {tgt} (n = {len(rows)}) ===')
    print(f'{"feature":22s} {"rho":>7s} {"p":>9s} {"n":>4s}')
    uni = []
    for f in feats:
        if f not in rows[0]:
            continue
        rho, pv, n = spearman([r.get(f, np.nan) for r in rows],
                              [r[tgt] for r in rows])
        uni.append((f, rho, pv, n))
    for f, rho, pv, n in sorted(uni, key=lambda z: -abs(z[1])):
        print(f'{f:22s} {rho:+7.3f} {pv:9.2e} {n:4d}')

    print('\n=== fold (model) effect ===')
    for k in folds:
        g = [r[tgt] for r in rows if r.get('fold') == k]
        if g:
            print(f'  fold {k}: n={len(g):3d} median {np.median(g):+.3f} '
                  f'mean {np.mean(g):+.3f}')
    from scipy import stats as st
    groups = [[r[tgt] for r in rows if r.get('fold') == k] for k in folds]
    groups = [g for g in groups if len(g) > 3]
    if len(groups) > 1:
        H, pv = st.kruskal(*groups)
        print(f'  Kruskal-Wallis H={H:.2f} p={pv:.4f} across {len(groups)} folds')
    yv = np.array([r[tgt] for r in rows], float)
    fold_ok = np.array([r.get('fold', -1) >= 0 for r in rows])
    F = np.zeros((len(rows), len(folds)))
    for i, r in enumerate(rows):
        if r.get('fold') in folds:
            F[i, folds.index(r['fold'])] = 1.0
    # drop the first dummy so the intercept is absorbed
    print(f'  LOOCV R2 of fold dummies alone: '
          f'{loo_r2(F[:, 1:][fold_ok], yv[fold_ok]):+.4f}')

    # ---- nested models, LOOCV ---------------------------------------------
    print('\n=== nested OLS on standardised ranks, LEAVE-ONE-OUT R2 ===')

    def _build(cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        use = [r for r in rows
               if all(np.isfinite(float(r.get(c, np.nan))) for c in cols)]
        X = np.column_stack([_z(_rank([r[c] for r in use])) for c in cols])
        y = _z(_rank([r[tgt] for r in use]))
        return X, y

    ref_cols = ['template_pearson_band', 'rate_bpm', 'band_snr', 'drift_ratio',
                'period_jitter', 'amp_cv', 'cycles', 'ref_rms']
    subj_cols = ['other_tasks_quality', 'gender', 'face_px_in_input']
    for label, cols in (
            ('reference only', ref_cols),
            ('subject only', subj_cols),
            ('subject + reference', subj_cols + ref_cols),
            ('+ fold dummies', subj_cols + ref_cols),
    ):
        if label.startswith('+ fold'):
            use = [r for r in rows
                   if r.get('fold') in folds
                   and all(np.isfinite(float(r.get(c, np.nan)))
                           for c in cols)]
            X = np.column_stack(
                [_z(_rank([r[c] for r in use])) for c in cols]
                + [np.array([1.0 if r['fold'] == k else 0.0 for r in use])
                   for k in folds[1:]])
            y = _z(_rank([r[tgt] for r in use]))
        else:
            X, y = _build(cols)
        print(f'  {label:22s} k={len(cols):2d} n={y.size:3d} '
              f'LOOCV R2 = {loo_r2(X, y):+.4f}')

    # ---- per-feature increment over the best single predictor --------------
    print('\n=== increment of each feature over template_pearson_band ===')
    base = ['template_pearson_band']
    Xb, yb = _build(base)
    r2b = loo_r2(Xb, yb)
    print(f'  {"(base)":22s} LOOCV R2 = {r2b:+.4f}')
    inc = []
    for f in feats:
        if f in base or f not in rows[0]:
            continue
        try:
            X, y = _build(base + [f])
        except ValueError:
            continue
        if y.size != yb.size:
            continue
        inc.append((f, loo_r2(X, y) - r2b))
    for f, d in sorted(inc, key=lambda z: -z[1])[:8]:
        print(f'  + {f:20s} dR2 = {d:+.4f}')

    if a.out:
        op = _p(a.out)
        os.makedirs(os.path.dirname(op), exist_ok=True)
        with open(op, 'w') as fh:
            json.dump(rows, fh, indent=1)
        print(f'\n[spread] wrote {op} ({len(rows)} rows)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
