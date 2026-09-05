"""Data-pipeline smoke test / inspection runner (no training).

Loads aligned RGB + TIR + physiological tensors from the canonical layout
(produced by ``data/prepare_bp4d.py``) for a SMALL number of sessions/clips and
reports tensor shapes, ranges, finite-ness, per-signal statistics and an
optional sanity figure. Intended for quick local checks before launching real
training on an HPC cluster.

Usage (from ``code/``)::

    python runners/run_inspect_data.py --data_path data/processed/bp4d_canonical
    python runners/run_inspect_data.py --data_path ... --max_sessions 2 \
        --max_entries 2 --input_size 64 --plot

The same command runs on a SLURM cluster (see ``scripts/hpc/submit_inspect.sbatch``);
paths are typically injected through environment variables (see scripts/env_*.sh).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data import alignment as align
from data import video_io as vio
from data.paired_dataset import PairedSessionDataset, scan_sessions


def _env(key: str, default: str = '') -> str:
    return os.environ.get(key, default)


def _read_signals_csv(path: str):
    """Read canonical signals.csv -> ({col: array}, fs)."""
    with open(path) as f:
        header = [h.strip() for h in next(f).strip().split(',')]
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    cols = {name: data[:, i] for i, name in enumerate(header)}
    fs = None
    if 'time' in cols and cols['time'].size > 1:
        t = cols['time'][~np.isnan(cols['time'])]
        dt = np.diff(t)
        dt = dt[dt > 0]
        if dt.size:
            fs = 1.0 / float(np.median(dt))
    return cols, fs


def _tensor_stats(name: str, arr) -> dict:
    a = np.asarray(arr)
    return {
        'name': name, 'shape': list(a.shape), 'dtype': str(a.dtype),
        'min': float(np.nanmin(a)) if a.size else None,
        'max': float(np.nanmax(a)) if a.size else None,
        'mean': float(np.nanmean(a)) if a.size else None,
        'finite_frac': float(np.isfinite(a).mean()) if a.size else None,
    }


def get_args():
    p = argparse.ArgumentParser('BP4D data-pipeline smoke test', add_help=False)
    p.add_argument('--data_path', default=_env('DATA_PATH', ''),
                   help='canonical sessions root (see data/prepare_bp4d.py)')
    p.add_argument('--output_dir', default=_env('OUTPUT_DIR',
                                                './output/inspect_data'), type=str)
    p.add_argument('--target', default='bvp', choices=['bvp', 'resp', 'eda'])
    # small-data controls (dev/smoke)
    p.add_argument('--max_sessions', default=2, type=int,
                   help='cap the number of sessions loaded')
    p.add_argument('--max_entries', default=2, type=int,
                   help='cap the number of clips inspected per split')
    p.add_argument('--train_ratio', default=0.5, type=float)
    # dataset geometry (must match what the model will see later)
    p.add_argument('--fs', default=100.0, type=float)
    p.add_argument('--fps', default=25.0, type=float)
    p.add_argument('--clip_duration', default=10.0, type=float)
    p.add_argument('--seq_len', default=0, type=int,
                   help='0 => clip_duration * fs')
    p.add_argument('--input_size', default=64, type=int,
                   help='frame short-side resize/crop (small = faster smoke)')
    p.add_argument('--plot', action='store_true',
                   help='render one aligned sanity figure (RGB+TIR+signals)')
    return p.parse_args()


def build(args, is_train: bool):
    return PairedSessionDataset(
        data_path=args.data_path, target=args.target,
        is_train=is_train, test_mode=True,
        fs=args.fs, fps=args.fps, clip_duration=args.clip_duration,
        seq_len=args.seq_len or None, input_size=args.input_size,
        train_ratio=args.train_ratio,
        max_sessions=args.max_sessions, max_entries=args.max_entries)


def main():
    args = get_args()
    if not os.path.isdir(args.data_path):
        raise SystemExit(f'data_path not found: {args.data_path} '
                         f'(run data/prepare_bp4d.py first)')
    os.makedirs(args.output_dir, exist_ok=True)
    report = {'data_path': args.data_path, 'args': vars(args)}

    sessions = scan_sessions(args.data_path)
    report['sessions_found'] = [s['session'] for s in sessions]
    print(f'Sessions found: {report["sessions_found"]}')

    # --- lightweight overview of the first session (no full decode) ------- #
    first = sessions[0]
    rgb_n = len(vio.list_image_files(first['rgb_dir']))
    sig_cols, sig_fs = _read_signals_csv(first['signals_file'])
    report['first_session'] = {
        'session': first['session'], 'rgb_frames': rgb_n,
        'signals_fs': sig_fs,
        'signals': {k: {'len': int(v.size),
                        'min': float(np.nanmin(v)),
                        'max': float(np.nanmax(v)),
                        'finite_frac': float(np.isfinite(v).mean())}
                    for k, v in sig_cols.items()},
    }
    print('First-session signal overview:')
    for k, v in report['first_session']['signals'].items():
        print(f"  {k:6s} len={v['len']:6d}  [{v['min']:.4g}, {v['max']:.4g}]  "
              f"finite={v['finite_frac']:.3f}")
    try:
        with vio.open_video(first['tir_file']) as reader:
            print(f"  TIR     {first['tir_file']}: {reader.fps:.2f} fps, "
                  f"{reader.num_frames} frames")
            report['first_session']['tir'] = {'fps': reader.fps,
                                              'num_frames': reader.num_frames}
    except Exception as exc:
        print(f'  TIR probe failed: {exc}')
        report['first_session']['tir_error'] = str(exc)

    # --- build datasets (decodes TIR for the capped sessions) -------------- #
    for split, is_train in (('train', True), ('val', False)):
        ds = build(args, is_train)
        print(f'[{split}] entries: {len(ds)}')
        if len(ds) == 0:
            report[split] = {'entries': 0}
            continue
        samples = []
        n = min(args.max_entries, len(ds))
        for idx in range(n):
            sample, target = ds[idx]
            sample = np.asarray(sample)
            target = np.asarray(target)
            s = {
                'clip_index': idx,
                'session': ds.entries[idx][0]['session'],
                't_start': ds.entries[idx][1],
                'samples': _tensor_stats('samples[4,T,H,W]', sample),
                'target': _tensor_stats(f'target[{args.target}]', target),
            }
            samples.append(s)
            print(f'  [{split}] clip {idx} '
                  f'(session {s["session"]}, t0={s["t_start"]:.2f}s): '
                  f'samples {s["samples"]["shape"]} range '
                  f'[{s["samples"]["min"]:.3f}, {s["samples"]["max"]:.3f}] | '
                  f'target {s["target"]["shape"]}')
        report[split] = {'entries': len(ds), 'clips': samples}
        if args.plot and split == 'train':
            _make_figure(args, ds, samples[0], sig_cols, sig_fs)

    out_json = os.path.join(args.output_dir, 'inspect_summary.json')
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'Wrote summary to {out_json}')


def _make_figure(args, ds, clip, sig_cols, sig_fs):
    """One sanity figure: RGB + TIR middle frame and aligned 1D signals."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f'[plot] matplotlib unavailable, skipping ({exc})')
        return

    session = clip['session']
    idx = clip['clip_index']
    sample, target = ds[idx]
    sample = np.asarray(sample)
    target = np.asarray(target)
    t_mid = sample.shape[1] // 2

    rgb = (np.clip(sample[0:3, t_mid].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
    tir = (np.clip(sample[3, t_mid], 0, 1) * 255).astype(np.uint8)

    fs = sig_fs or args.fs
    t0, dur = clip['t_start'], args.clip_duration
    start = int(round(t0 * fs))
    n_win = int(round(dur * fs))
    fig, axes = plt.subplots(3, 2, figsize=(11, 8),
                             gridspec_kw={'width_ratios': [1, 2]})
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f'RGB (middle frame) {rgb.shape[0]}x{rgb.shape[1]}')
    axes[0, 0].axis('off')
    axes[1, 0].imshow(tir, cmap='gray')
    axes[1, 0].set_title(f'TIR gray (middle frame) {tir.shape[0]}x{tir.shape[1]}')
    axes[1, 0].axis('off')
    axes[2, 0].axis('off')

    t_tgt = np.arange(target.size) / args.fs
    names = ['bvp', 'resp', 'eda']
    # vertical reference lines (every second) + shared x-range so the panels
    # line up: lets you eyeball that blue (raw window) and orange (resampled
    # dataset target) features occur at the same times in every row
    step = 1.0
    refs = np.arange(0.0, args.clip_duration, step)
    for row, col in enumerate(names):
        ax = axes[row, 1]
        if col in sig_cols:
            seg = align.slice_1d(sig_cols[col], fs, start=start, n=n_win)
            t_ax = np.arange(seg.size) / fs
            ax.plot(t_ax, seg, lw=0.8, color='tab:blue')
        if col == args.target:
            ax.plot(t_tgt, target, lw=0.8, color='tab:orange')
        for t in refs:
            ax.axvline(t, color='0.4', lw=0.6, ls=':', alpha=0.7)
        ax.set_xlim(0, args.clip_duration)
        ax.set_title(f'{col} (blue=window, orange=dataset target)'
                     if col == args.target else f'{col} window')
        if row == 2:
            ax.set_xlabel('time (s)')

    # --- metadata block in the empty bottom-left panel ---------------------- #
    meta = ds.entries[idx][0]                      # full per-session meta
    tir_fps = meta.get('tir_fps') or args.fps      # actual TIR frame rate
    sig_fs = fs or args.fs                         # signal sample rate in use
    t_end = t0 + args.clip_duration
    info = [
        f'session: {session}   clip #{idx}',
        f'RGB {args.fps:.0f} fps | TIR {tir_fps:.0f} fps | signal fs {sig_fs:.1f} Hz',
        f't-start {t0:.2f} s -> t-end {t_end:.2f} s  (dur {args.clip_duration:.0f} s)',
        f'seq_len (target): {target.size} | input_size: {args.input_size}',
    ]
    axes[2, 0].axis('off')
    axes[2, 0].text(0.0, 0.5, '\n'.join(info),
                    transform=axes[2, 0].transAxes,
                    ha='left', va='center', fontsize=8, family='monospace')

    fig.tight_layout()
    path = os.path.join(args.output_dir, f'fig_{session}_clip{idx}.png')
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f'[plot] saved {path}')


if __name__ == '__main__':
    main()
