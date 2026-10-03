"""Diagnostics for the Stage-3 TIR-ROI -> RESP predictions (the CLOSED line).

Persisted copy (2026-10-01) of the /tmp diagnostics that produced the numbers in
``code/plan/TirROI_Resp_plan.md`` §12. It reads only the saved arrays of a finished
run, so it needs no model, no GPU and no repo imports:

    python analysis/tir_resp/waveform_diagnostics.py                 # all sections
    python analysis/tir_resp/waveform_diagnostics.py --root <output/finetune>

Recorded values (g000 = gamma 0, 75 val clips; g001 = gamma 0.01, 75 clips):

  §2  MAE 0.9706 / 1.1932 vs the constant-0 baseline **0.7545** -> both lose
  §2  spectral excess (g001): x143 (4-8 Hz), x553 (8-20 Hz), x991 at 6.25 Hz
  §3  per-band |r|: 0.1-0.3 -0.20, 0.3-0.6 -0.03, 0.6-1.0 +0.05, 1.0-2.0 -0.03
  §4  |2nd diff| 0.4178 vs the target's 0.0014; seam-SLOPE ratio 2.64 (g001)
      vs the target's own 1.43 and **1.44 at gamma 0** (artifact gone);
      band-limiting the prediction: MAE 1.1932 -> 0.8870, pearson -0.0272 -> -0.0023
  §5  cross-clip correlation of the predictions: 0.49 (g000) / 0.756 (g001)
      vs the targets' -0.004
  §7  why ``hf_lo`` defaults to 2 Hz: the belt's OWN 0.6-1.5 Hz content is
      larger than the model's, so an "everything above the band" window misleads

The clinching fixed-prior control is in ``prior_control.py`` next to this file.
"""
import argparse
import json
import os

import numpy as np
from scipy import signal as sg

FS = 100.0
TOK = 16                      # sig_kernel: samples the head emits per visual token
ARMS = {'g000 (gamma 0)': 'resp_tir_roi_low_mid_g000',
        'g001 (gamma 0.01)': 'resp_tir_roi_low_mid_g001',
        'g10 (gamma 10)': 'resp_tir_roi_low_mid_g10'}
BAND = (0.1, 0.6)


def default_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, '..', '..', '..', 'output',
                                         'finetune'))


def load(root, run):
    p = np.load(os.path.join(root, run, 'preds.npy')).astype(np.float64)
    t = np.load(os.path.join(root, run, 'targets.npy')).astype(np.float64)
    with open(os.path.join(root, run, 'entries.json')) as fh:
        e = json.load(fh)
    return p, t, e


def stats(pred, targ):
    mae = float(np.abs(pred - targ).mean())
    rmse = float(np.sqrt(((pred - targ) ** 2).mean()))
    r = float(np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(pred, targ)]))
    return mae, rmse, r


def bandpass(x, lo=BAND[0], hi=BAND[1]):
    b, a = sg.butter(4, [lo / (FS / 2), hi / (FS / 2)], btype='band')
    return sg.filtfilt(b, a, x, axis=-1)


def psd(x, nperseg):
    return sg.welch(x, FS, nperseg=nperseg, axis=-1)


def seam_report(x):
    """Value-continuity and SLOPE-continuity at the token boundaries."""
    q = x[:, :x.shape[-1] // TOK * TOK].reshape(x.shape[0], -1, TOK)
    seam_v = float(np.abs(q[:, :-1, -1] - q[:, 1:, 0]).mean())
    within_v = float(np.abs(np.diff(q, axis=2)).mean())
    d = np.diff(q, axis=2)
    seam_s = float(np.abs(d[:, :-1, -1] - d[:, 1:, 0]).mean())
    within_s = float(np.abs(np.diff(q, n=2, axis=2)[:, :-1, -1]).mean())
    return seam_v, within_v, seam_s, within_s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', default=default_root(),
                    help='directory holding the run folders (default: '
                         '<repo>/output/finetune)')
    ap.add_argument('--nperseg', type=int, default=600,
                    help='Welch window for the spectrum sections (600 = the '
                         'value evaluation.metrics resolves for 0.1-0.6 Hz)')
    args = ap.parse_args(argv)

    raw = {}
    for name, run in ARMS.items():
        path = os.path.join(args.root, run)
        if os.path.isdir(path):
            raw[name] = load(args.root, run)
    if not raw:
        raise SystemExit(f'no run folders under {args.root}')

    # Score every arm on the SAME clips, and drop (near-)constant targets:
    # np.corrcoef on a constant target is NaN and would poison every mean. The
    # g10 arm still carries the 4 railed windows (its run predates
    # min_signal_spread), while g000/g001 do not -- without this, g10 prints NaN.
    key = lambda x: (x['session'], x['task'], round(float(x['t_start']), 3))
    shared = set.intersection(*[{key(x) for x in e} for _, _, e in raw.values()])
    data = {}
    for name, (p, t, e) in raw.items():
        idx = [i for i, x in enumerate(e) if key(x) in shared]
        p, t = p[idx], t[idx]
        dev = np.abs(t - t.mean(-1, keepdims=True)).max(-1)
        keep = dev > 1e-6 * dev.max()
        data[name] = (p[keep], t[keep], [e[i] for i in idx])
    ref_key = next(iter(data))
    n_clips = data[ref_key][0].shape[0]

    print('=' * 78)
    print(f'2. ERROR AND OUT-OF-BAND (JITTER) POWER   [nperseg {args.nperseg}, '
          f'{n_clips} shared clip(s)]')
    print('=' * 78)
    for name, (p, t, _) in data.items():
        mae, rmse, r = stats(p, t)
        base = float(np.abs(t).mean())
        f, Pu = psd(p, args.nperseg)
        Pp = Pu.mean(0)
        Pt = psd(t, args.nperseg)[1].mean(0)
        print(f'  {name:<18} MAE {mae:.4f}  RMSE {rmse:.4f}  pearson {r:+.4f}  '
              f'| constant-0 MAE {base:.4f} ({"WORSE" if mae > base else "better"})')
        for lo, hi in ((0.1, 0.6), (0.6, 1.5), (1.5, 4.0), (4.0, 8.0),
                       (8.0, 20.0)):
            m = (f >= lo) & (f <= hi)
            ex = Pp[m].sum() / Pt[m].sum()
            print(f'       {lo:4.1f}-{hi:4.1f} Hz  excess x{ex:8.1f}'
                  + ('   <== jitter' if ex > 20 else ''))
        tr = (f >= 6.0) & (f <= 6.5)
        print(f'       token rate {FS / TOK:.2f} Hz  excess '
              f'x{Pp[tr].mean() / Pt[tr].mean():.0f}')

    print()
    print('=' * 78)
    print('3. IS ANY BAND PHASE-LOCKED?  (band-pass both sides, then correlate)')
    print('=' * 78)
    for name, (p, t, _) in data.items():
        row = []
        for lo, hi in ((0.1, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 2.0)):
            rr = [np.corrcoef(a, b)[0, 1]
                  for a, b in zip(bandpass(p, lo, hi), bandpass(t, lo, hi))]
            row.append(f'{lo:4.2f}-{hi:4.2f}: {np.mean(rr):+.3f}')
        print(f'  {name:<18} ' + '  '.join(row))

    print()
    print('=' * 78)
    print('4. SMOOTHNESS, TOKEN SEAMS, AND WHAT THE JITTER COSTS')
    print('=' * 78)
    for name, (p, t, _) in data.items():
        sv, wv, ss, ws = seam_report(p)
        print(f'  {name:<18} |2nd diff| {np.abs(np.diff(p, n=2, axis=-1)).mean():.4f}'
              f'  seam value x{sv / wv:.2f}  seam SLOPE x{ss / ws:.2f}')
    tgt = data[ref_key][1]
    print(f'  {"target":<18} |2nd diff| '
          f'{np.abs(np.diff(tgt, n=2, axis=-1)).mean():.4f}  '
          f'seam value x{seam_report(tgt)[0] / seam_report(tgt)[1]:.2f}  '
          f'seam SLOPE x{seam_report(tgt)[2] / seam_report(tgt)[3]:.2f}')
    for name, (p, t, _) in data.items():
        low = bandpass(p)
        m_low, _, r_low = stats(low, t)
        print(f'  {name:<18} band-limited pred vs raw target: MAE {m_low:.4f}  '
              f'pearson {r_low:+.4f}   (raw: {stats(p, t)[0]:.4f} / '
              f'{stats(p, t)[2]:+.4f})')

    print()
    print('=' * 78)
    print('5. CLIP INDEPENDENCE  (one fixed waveform broadcast to every clip?)')
    print('=' * 78)
    for name, (p, t, _) in data.items():
        out = []
        for nm, x in (('pred', p), ('target', t)):
            c = x - x.mean(-1, keepdims=True)
            C = np.corrcoef(c)
            out.append(f'{nm} {(C.sum() - np.trace(C)) / (C.size - c.shape[0]):+.3f}')
        print(f'  {name:<18} mean pairwise corr between DIFFERENT clips: '
              + '   '.join(out))

    print()
    print('=' * 78)
    print('7. HF WINDOW: why hf_lo defaults to 2 Hz')
    print('=' * 78)
    print(f'  {"hf_lo":>6} | {"pred HF / target total":>22} | {"pred frac":>14}')
    fine = args.nperseg
    f, Pu = sg.welch(data[ref_key][0], FS, nperseg=fine, axis=-1)
    Pt = sg.welch(data[ref_key][1], FS, nperseg=fine, axis=-1)[1].mean(0)
    df = float(f[1] - f[0])
    tot = float(Pt.sum() * df)
    for hf_lo in (1.0, 2.0, 3.0, 4.0):
        m = f >= hf_lo
        rel = float(Pu.mean(0)[m].sum() * df) / tot
        frac = float(Pu.mean(0)[m].sum() / Pu.mean(0).sum())
        print(f'  {hf_lo:6.1f} | {rel:22.5f} | {frac:14.3f}')
    print('  the reference prediction scores the target\'s own value; 2 Hz is '
          'above any respiration harmonic')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
