"""Stage-3 MATCHED PAIR readout -- Solution C vs Solution A encoder.

The two arms differ ONLY in which pre-training produced the frozen encoder
(configs/finetune/resp_tir_roi_local_matched.yaml vs resp_tir_roi_crossmae.yaml),
so any difference here is attributable to the pre-training objective.

It reads only the saved arrays of a finished run -- no model, no GPU, no repo
imports beyond the sibling ``waveform_diagnostics`` helpers:

    python analysis/tir_resp/matched_pair_stage3.py
    python analysis/tir_resp/matched_pair_stage3.py --arms C=<dir> A=<dir>

WHY THIS SCRIPT EXISTS. ``mse_resp`` / ``psd_mae`` / raw Pearson are all
MISLEADING on this task: a fully collapsed (near-constant) prediction can score
a better ``psd_mae`` than a jittery one, and any shared slow trend inflates every
correlation. The discriminative family is the PHASE-LOCKING CONTRAST

    contrast = mean_clip |r(0)|  -  mean_clip mean_|shift|>=1.2s |r(shift)|

i.e. "does the prediction track THIS clip's phase, or is it just a waveform shape
that correlates with anything slowly varying?". A pure shared trend scores
contrast ~= 0 even when |r(0)| is visibly non-zero, because the same waveform also
correlates with the time-shifted target. Calibration: contrast > 0.15 is real
phase locking; |contrast| < 0.03 with a large cross-clip prediction correlation
means the model emits ONE waveform for every clip.

The two controls that make the verdict airtight:
  * BROADCAST -- correlate a single FIXED waveform (the mean prediction over all
    clips) with every target. If this scores about the same as the model's own
    |r(0)|, the model's correlation is entirely that shared trend.
  * BAND-LIMITED -- band-pass the prediction to the respiration band before
    scoring. A model that "wins" only on out-of-band jitter loses its edge here.
"""
import argparse
import json
import os

import numpy as np
from scipy import signal as sg

from waveform_diagnostics import BAND, FS, load, bandpass, seam_report

SHIFT_S = 1.2                 # s; the minimum lag for the artefact floor
# Above the respiration band: the belt's own 0.6-1.5 Hz content is larger than
# the model's, so "everything above the band" misleads (see the diagnostics doc).
HF_LO_HZ = 2.0

DEFAULT_ARMS = {
    'C (shared-encoder MAE)': 'resp_tir_roi_local_matched',
    'A (asymmetric cross-MAE)': 'resp_tir_roi_crossmae',
}


def default_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, '..', '..', '..', 'output',
                                         'finetune'))


def key_of(e):
    return (e['session'], e['task'], round(float(e['t_start']), 3))


def mae_of(pred, targ):
    return float(np.abs(pred - targ).mean())


def phase_locking(pred, targ, fs=FS, shift_s=SHIFT_S):
    """|r(0)| vs the |r| floor over large lags, per clip, then averaged.

    Returns (mean|r(0)|, mean floor, median r(0), n_used).
    """
    lag = int(round(shift_s * fs))
    r0, floor, med = [], [], []
    max_lag = targ.shape[-1] - 1
    lags = list(range(lag, max_lag + 1, max(1, lag // 4)))
    for a, b in zip(pred, targ):
        if b.std() < 1e-8 or a.std() < 1e-8:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if not np.isfinite(c):
            continue
        row = []
        for s_ in lags:
            for rolled in (np.roll(a, s_), np.roll(a, -s_)):
                cc = float(np.corrcoef(rolled, b)[0, 1])
                if np.isfinite(cc):
                    row.append(abs(cc))
        r0.append(abs(c))
        med.append(c)
        if row:
            floor.append(float(np.mean(row)))
    if not r0:
        return float('nan'), float('nan'), float('nan'), 0
    return (float(np.mean(r0)), float(np.mean(floor)) if floor else float('nan'),
            float(np.median(med)), len(r0))


def rel_hf(x, lo=HF_LO_HZ, nperseg=600):
    f, P = sg.welch(x, FS, nperseg=nperseg, axis=-1)
    hi = f >= lo
    return float((P[:, hi].sum(-1) / np.maximum(P.sum(-1), 1e-30)).mean())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', default=default_root(),
                    help='directory holding the run folders')
    ap.add_argument('--arms', nargs='*', default=None,
                    help='override the arms as NAME=<run folder>')
    args = ap.parse_args(argv)

    arms = dict(DEFAULT_ARMS)
    if args.arms:
        arms = {}
        for spec in args.arms:
            name, _, run = spec.partition('=')
            arms[name] = run

    raw = {}
    for name, run in arms.items():
        path = os.path.join(args.root, run)
        if not os.path.isdir(path):
            print(f'[skip] {name}: {path} does not exist')
            continue
        try:
            raw[name] = load(args.root, run)
        except FileNotFoundError as exc:
            print(f'[skip] {name}: {exc}')
    if not raw:
        raise SystemExit(f'no usable run folders under {args.root}')

    # Score every arm on the SAME clips, with the SAME key, and drop
    # (near-)constant targets: np.corrcoef on a constant target is NaN and would
    # poison every mean.
    shared = set.intersection(*[{key_of(x) for x in e} for _, _, e in raw.values()])
    data = {}
    for name, (p, t, e) in raw.items():
        idx = [i for i, x in enumerate(e) if key_of(x) in shared]
        p, t = p[idx], t[idx]
        dev = np.abs(t - t.mean(-1, keepdims=True)).max(-1)
        keep = dev > 1e-6 * max(dev.max(), 1e-30)
        data[name] = (p[keep], t[keep])
    n = next(iter(data.values()))[0].shape[0]

    print('=' * 78)
    print(f'STAGE-3 MATCHED PAIR -- {n} shared clip(s), respiration band '
          f'{BAND[0]}-{BAND[1]} Hz')
    print('=' * 78)

    hdr = f'{"metric":<42}' + ''.join(f'{k:>17}' for k in data)
    print(hdr)
    print('-' * len(hdr))

    targ = next(iter(data.values()))[1]
    base0 = float(np.abs(targ).mean())          # constant-0 predictor
    print(f'{"constant-0 predictor MAE (floor)":<42}' + f'{base0:>17.4f}')

    summary = {}
    for name, (p, t) in data.items():
        r0, floor, med, used = phase_locking(p, t)
        r0b, floorb, medb, _ = phase_locking(bandpass(p), t)
        mean_pred = p.mean(0, keepdims=True)
        broadcast = float(np.mean([abs(np.corrcoef(mean_pred[0], b)[0, 1])
                                   for b in t if b.std() > 1e-8]))
        pb = bandpass(p)
        seam_v, within_v, seam_s, within_s = seam_report(p)
        summary[name] = dict(
            mae=mae_of(p, t), rmse=float(np.sqrt(((p - t) ** 2).mean())),
            pearson=float(np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(p, t)])),
            base0=base0, pred_std=float(p.std()), targ_std=float(t.std()),
            r0=r0, floor=floor, contrast=r0 - floor, med_r0=med, used=used,
            r0_band=r0b, floor_band=floorb, contrast_band=r0b - floorb,
            broadcast=broadcast,
            mae_band=mae_of(pb, t),
            hf=rel_hf(p), hf_target=rel_hf(t),
            d2=float(np.abs(np.diff(p, n=2, axis=-1)).mean()),
            d2_target=float(np.abs(np.diff(t, n=2, axis=-1)).mean()),
            seam_v=seam_v, within_v=within_v, seam_s=seam_s, within_s=within_s,
            cross_clip=float(np.mean(np.corrcoef(p[:min(40, len(p))])[
                np.triu_indices(min(40, len(p)), 1)])) if len(p) > 1 else float('nan'),
        )

    print(f'{"MAE(model)":<42}' + ''.join(f'{s["mae"]:>17.4f}' for s in summary.values()))
    print(f'{"MAE vs constant-0 (fraction)":<42}' + ''.join(
        f'{s["mae"] / s["base0"]:>17.2f}' for s in summary.values()))
    print(f'{"MAE after band-limiting the pred":<42}' + ''.join(
        f'{s["mae_band"]:>17.4f}' for s in summary.values()))
    print(f'{"pearson (mean per clip)":<42}' + ''.join(
        f'{s["pearson"]:>17.4f}' for s in summary.values()))
    print(f'{"pred std / target std":<42}' + ''.join(
        f'{s["pred_std"] / s["targ_std"]:>17.4f}' for s in summary.values()))
    print(f'{"|r(0)|":<42}' + ''.join(f'{s["r0"]:>17.4f}' for s in summary.values()))
    print(f'{"|r(|shift| >= 1.2 s)|  (floor)":<42}' + ''.join(
        f'{s["floor"]:>17.4f}' for s in summary.values()))
    print(f'{"PHASE-LOCKING CONTRAST":<42}' + ''.join(
        f'{s["contrast"]:>17.4f}' for s in summary.values()))
    print(f'{"  ... on the band-limited pred":<42}' + ''.join(
        f'{s["contrast_band"]:>17.4f}' for s in summary.values()))
    print(f'{"|r| of ONE FIXED waveform":<42}' + ''.join(
        f'{s["broadcast"]:>17.4f}' for s in summary.values()))
    print(f'{"median per-clip r(0)":<42}' + ''.join(
        f'{s["med_r0"]:>17.4f}' for s in summary.values()))
    print(f'{"cross-clip corr of preds":<42}' + ''.join(
        f'{s["cross_clip"]:>17.4f}' for s in summary.values()))
    print(f'{"hf_power_rel (>{:g} Hz)":<32}'.format(HF_LO_HZ) + ''.join(
        f'{s["hf"]:>17.3f}' for s in summary.values()))
    print(f'{"  target hf_power_rel":<42}' + ''.join(
        f'{s["hf_target"]:>17.5f}' for s in summary.values()))
    print(f'{"pred |2nd diff|":<42}' + ''.join(
        f'{s["d2"]:>17.5f}' for s in summary.values()))
    print(f'{"  target |2nd diff|":<42}' + ''.join(
        f'{s["d2_target"]:>17.5f}' for s in summary.values()))
    print(f'{"seam/within SLOPE ratio":<42}' + ''.join(
        f'{s["seam_s"] / max(s["within_s"], 1e-30):>17.2f}' for s in summary.values()))

    print()
    print('VERDICT')
    print('-' * 78)
    for name, s in summary.items():
        if abs(s['contrast']) >= 0.15:
            verdict = 'REAL phase locking'
        elif abs(s['contrast']) < 0.03 and s['cross_clip'] > 0.3:
            verdict = ('NO phase info -- one shared waveform for every clip '
                       f'(|r(0)| {s["r0"]:.3f} vs broadcast {s["broadcast"]:.3f})')
        else:
            verdict = 'inconclusive'
        print(f'  {name:<26} contrast {s["contrast"]:+.4f}  -> {verdict}')
    print()
    print('Read this as: MAE(model) must beat the constant-0 floor AND the '
          'contrast must exceed ~0.15 before "the encoder carries respiration '
          'phase" is a defensible claim. A high |r(0)| with contrast ~= 0 and a '
          'broadcast score equal to |r(0)| is a shared slow trend, not phase.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
