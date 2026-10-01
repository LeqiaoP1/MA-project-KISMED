"""The clinching control for the closed TIR-ROI -> RESP line (plan §12.1(iv)).

Persisted copy (2026-10-01) of the /tmp control. Two questions, both answered
from the SAVED arrays of a finished run (no model, no GPU):

* is the per-clip output better than ONE FIXED WAVEFORM broadcast to every clip?
  -> g000: |r| model 0.217 vs |r| of the broadcast prior **0.229**. The model is
     no better than ignoring the input. (Cross-clip corr of the predictions:
     0.49 at gamma 0, 0.756 at gamma 0.01; the targets' is -0.004.)
* could a rate-only model have done better? The best single sinusoid fitted to
  the TARGET reaches **0.638**, so the belt is strongly sinusoidal and the
  failure is PHASE, not expressivity.

Also prints the phase-locking contrast, the established control: |r(0)| against
the anti-phase floor (shifts 1.2-2.5 s). g000 measures |r(0)| 0.217 vs a floor of
**0.365** -> contrast **-0.148**, i.e. the prediction is BELOW its own artifact
floor, the signature of zero phase information.

    python analysis/tir_resp/prior_control.py
    python analysis/tir_resp/prior_control.py --root <output/finetune>
"""
import argparse
import json
import os

import numpy as np

FS = 100.0
ARMS = [('g000', 'resp_tir_roi_low_mid_g000'),
        ('g001', 'resp_tir_roi_low_mid_g001')]


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


def z(x):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-12)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def best_sine_r(x, lo=0.15, hi=1.2, step=0.005):
    """Multiple correlation of x with the best sin/cos pair in [lo, hi] Hz.

    NOTE the per-ROW norm: a plain ``np.linalg.norm(x)`` is the Frobenius norm of
    the whole [N, L] batch and makes every value ~9x too small (measured bug).
    """
    n = x.shape[-1]
    idx = np.arange(n)
    xn = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
    best = np.zeros(x.shape[0])
    for f in np.arange(lo, hi + 1e-9, step):
        c = np.cos(2 * np.pi * f * idx / FS)
        s = np.sin(2 * np.pi * f * idx / FS)
        c = c / np.linalg.norm(c)
        s = s / np.linalg.norm(s)
        best = np.maximum(best, np.sqrt((xn @ c) ** 2 + (xn @ s) ** 2))
    return best


def contrast(pred, targ, lo=1.2, hi=2.5):
    """|r(0)| and the anti-phase floor -- report the CONTRAST, never bare |r|."""
    r0 = np.array([abs(corr(a, b)) for a, b in zip(pred, targ)])
    anti = np.array([max(abs(corr(a, np.roll(b, s)))
                         for s in range(int(lo * FS), int(hi * FS) + 1, 5))
                     for a, b in zip(pred, targ)])
    return r0.mean(), anti.mean()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', default=default_root())
    args = ap.parse_args(argv)

    for name, run in ARMS:
        path = os.path.join(args.root, run)
        if not os.path.isdir(path):
            print(f'[skip] {path} not found')
            continue
        p, t, _ = load(args.root, run)
        prior = z(p).mean(0)
        prior = prior / np.linalg.norm(prior)
        r_model = np.array([abs(corr(a, b)) for a, b in zip(p, t)])
        r_prior = (np.abs(z(t) @ prior)
                   / (np.linalg.norm(z(t), axis=-1) + 1e-12))
        rs_t = best_sine_r(t)
        rs_p = best_sine_r(p)
        c_m, a_m = contrast(p, t)
        c_p, a_p = contrast(np.tile(prior, (p.shape[0], 1)), t)

        print(f'=== {name} ({p.shape[0]} clips) ===')
        print(f'  |r| model vs its own target        {r_model.mean():+.4f}'
              f'   (median {np.median(r_model):+.4f})')
        print(f'  |r| FIXED prior vs target          {r_prior.mean():+.4f}'
              f'   <-- one waveform broadcast to every clip')
        print(f'  |r| best single sinusoid, TARGET   {rs_t.mean():.4f}'
              f'   <-- rate-only ceiling')
        print(f'  |r| best single sinusoid, PRED     {rs_p.mean():.4f}')
        print(f'  phase-lock contrast: model {c_m - a_m:+.3f} '
              f'(|r0| {c_m:.3f} vs anti {a_m:.3f}) | prior {c_p - a_p:+.3f}')
        print(f'  constant-0 MAE {np.abs(t).mean():.4f}   model MAE '
              f'{np.abs(p - t).mean():.4f}')
        print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
