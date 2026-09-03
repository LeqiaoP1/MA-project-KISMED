"""Offline post-processing evaluation of saved waveform predictions.

Usage (from ``code/``)::

    python runners/run_evaluate.py --pred_path pred_bvp.npy \
        --target_path gt_bvp.npy --fs 100 --tier 1,2,3 --waveform bvp

Loads ``[T]`` or ``[N, T]`` numpy predictions + ground truth and prints the
tiered metrics from ``evaluation/``:
  Tier 1 -- MAE / RMSE / Pearson
  Tier 2 -- Welch PSD consistency (needs scipy)
  Tier 3 -- NeuroKit2 HRV: RMSSD / pNN50 / MedianNN / ShanEn (needs neurokit2)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from evaluation import clinical as _clinical
from evaluation import metrics as _metrics


def _load(path):
    """Load a .npy/.npz array, preferring keys pred/target when present."""
    if path.endswith('.npz'):
        with np.load(path, allow_pickle=True) as d:
            for key in ('pred', 'target', 'arr_0', 'data'):
                if key in d:
                    return np.asarray(d[key], dtype=np.float32)
            raise KeyError(f'No recognised array key in {path}')
    return np.asarray(np.load(path), dtype=np.float32)


def get_args():
    parser = argparse.ArgumentParser('Offline waveform evaluation', add_help=False)
    parser.add_argument('--pred_path', required=True, type=str)
    parser.add_argument('--target_path', required=True, type=str)
    parser.add_argument('--fs', default=100.0, type=float)
    parser.add_argument('--tier', default='1,2,3', type=str,
                        help='comma-separated tiers to run')
    parser.add_argument('--waveform', default='bvp', choices=['bvp', 'resp'])
    parser.add_argument('--band', default=None, type=str,
                        help='e.g. "1.0,2.5" spectral band override')
    parser.add_argument('--out', default='', type=str,
                        help='optional .json path to write results')
    return parser.parse_args()


def main(args):
    pred = _load(args.pred_path)
    target = _load(args.target_path)
    tiers = {int(t) for t in args.tier.split(',')}
    results = {'fs': args.fs, 'waveform': args.waveform}

    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    if target.ndim == 1:
        target = target.reshape(1, -1)

    if 1 in tiers:
        r = _metrics.time_domain_metrics(pred, target)
        print('Tier 1 (temporal):', r)
        results.update(r)

    if 2 in tiers:
        band = args.band
        if band:
            band = tuple(float(x) for x in band.split(','))
        else:
            # plan defaults: BVP 1.0-2.5 Hz, RESP 0.16-0.4 Hz
            band = (1.0, 2.5) if args.waveform == 'bvp' else (0.16, 0.4)
        r = _metrics.spectral_metrics(pred, target, fs=args.fs, band=band)
        print('Tier 2 (spectral):', r)
        results.update(r)

    if 3 in tiers:
        if args.waveform != 'bvp':
            print('Tier 3: clinical HRV metrics are defined for BVP only; skipping.')
        else:
            recs = [_clinical.extract_hrv_metrics(p, fs=args.fs) for p in pred]
            agg = {}
            for key in recs[0]:
                vals = [r[key] for r in recs if not np.isnan(r[key])]
                agg[key] = float(np.mean(vals)) if vals else float('nan')
            print('Tier 3 (clinical, NeuroKit2):', agg)
            results['tier3'] = agg

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f'Wrote results to {args.out}')


if __name__ == '__main__':
    main(get_args())
