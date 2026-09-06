"""Data-pipeline smoke test / inspection runner (no training).

Loads aligned RGB + TIR + physiological tensors from the canonical layout
(produced by ``data/prepare_bp4d.py``) for a SMALL number of sessions/clips and
reports tensor shapes, ranges, finite-ness, per-signal statistics and an
optional sanity figure. Intended for quick local checks before launching real
training on an HPC cluster.

Usage (from ``code/``)::

    python runners/run_inspect_data.py --data_path data/processed/bp4d_canonical
    python runners/run_inspect_data.py --data_path ... --max_sessions 6 \
        --max_clips 2 --clip_duration 2 --input_size 64 --plot

``--plot`` saves one PNG for EVERY clip of both splits (train + val) into
``<output_dir>/figures/<split>/``; tensor stats always go to
``<output_dir>/inspect_summary.json``.

Window geometry: ``--clip_duration`` (default 10 s; ``seq_len = clip_duration *
fs`` when ``--seq_len 0``) and per-session ``--max_clips``. The dataset is
bounded by ``--max_sessions`` (decode budget) and ``--max_clips`` (per-session
windows), so every extracted clip is processed and windows never collapse onto
the leading sessions. ``--max_entries`` is only an optional per-split safety
cap (``0`` = no cap; default), see :func:`build`.

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


def _session_counts(ds) -> dict:
    """Map session name -> number of clips in ``ds``, in session order."""
    counts, order = {}, []
    for meta, _ in ds.entries:
        name = meta['session']
        if name not in counts:
            counts[name] = 0
            order.append(name)
        counts[name] += 1
    return counts


def get_args():
    p = argparse.ArgumentParser('BP4D data-pipeline smoke test', add_help=False)
    p.add_argument('--data_path', default=_env('DATA_PATH', ''),
                   help='canonical sessions root (see data/prepare_bp4d.py)')
    p.add_argument('--output_dir', default=_env('OUTPUT_DIR',
                                                './output/inspect_data'), type=str)
    p.add_argument('--target', default='bvp', choices=['bvp', 'resp', 'eda'])
    # small-data controls (dev/smoke)
    p.add_argument('--max_sessions', default=3, type=int,
                   help='cap the number of sessions loaded')
    p.add_argument('--max_clips', default=None, type=int,
                   help='cap clips per session (None = every clip of a session)')
    p.add_argument('--max_entries', default=0, type=int,
                   help='optional cap on clips processed per split; '
                        '0 = no cap (all clips)')
    p.add_argument('--train_ratio', default=0.5, type=float)
    # dataset geometry (must match what the model will see later)
    p.add_argument('--fs', default=100.0, type=float)
    p.add_argument('--fps', default=25.0, type=float)
    p.add_argument('--clip_duration', default=10.0, type=float,
                   help='window length in seconds (seq_len = clip_duration*fs)')
    p.add_argument('--seq_len', default=0, type=int,
                   help='0 => clip_duration * fs')
    p.add_argument('--input_size', default=64, type=int,
                   help='frame short-side resize/crop (small = faster smoke)')
    p.add_argument('--plot', action='store_true',
                   help='save a PNG for every clip of train+val into '
                        '<output_dir>/figures/<split>/')
    return p.parse_args()


def build(args, is_train: bool):
    # NOTE: the global ``max_entries`` cap is deliberately NOT forwarded to the
    # dataset. Forwarding it truncates ``self.entries`` to the leading clips of
    # the leading sessions, which defeats the ``max_clips`` per-session spread.
    # Instead the dataset is bounded by max_sessions (decode budget) and
    # max_clips (per-session windows); the inspect loop below processes every
    # entry unless the user set an explicit ``max_entries`` per-split cap.
    return PairedSessionDataset(
        data_path=args.data_path, target=args.target,
        is_train=is_train, test_mode=True,
        fs=args.fs, fps=args.fps, clip_duration=args.clip_duration,
        seq_len=args.seq_len or None, input_size=args.input_size,
        train_ratio=args.train_ratio,
        max_sessions=args.max_sessions, max_clips=args.max_clips)


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
        counts = _session_counts(ds)
        samples = []
        # process every clip of the split; max_entries (0 = no cap) may limit it
        n = (len(ds) if (args.max_entries or 0) <= 0
             else min(args.max_entries, len(ds)))
        for idx in range(n):
            sample, target = ds[idx]
            sample = np.asarray(sample)
            target = np.asarray(target)
            meta = ds.entries[idx][0]            # full per-session meta dict
            t_start = ds.entries[idx][1]
            clip_k = int(round(t_start / args.clip_duration))
            s = {
                'clip_index': idx,
                'session': meta['session'],
                't_start': t_start,
                'clip_k': clip_k,                # 0-based window within session
                'n_clips': meta.get('n_clips'),
                'n_clips_raw': meta.get('n_clips_raw'),
                'tir_fps': meta.get('tir_fps') or args.fps,
                'samples': _tensor_stats('samples[4,T,H,W]', sample),
                'target': _tensor_stats(f'target[{args.target}]', target),
            }
            samples.append(s)
            print(f'  [{split}] clip {idx} '
                  f'(session {s["session"]}, k={clip_k + 1}/{s["n_clips"]}, '
                  f't0={t_start:.2f}s): '
                  f'samples {s["samples"]["shape"]} range '
                  f'[{s["samples"]["min"]:.3f}, {s["samples"]["max"]:.3f}] | '
                  f'target {s["target"]["shape"]}')
            if args.plot:
                # preview uses that session's own signals for correct overlay
                cols_plot, fs_plot = _read_signals_csv(meta['signals_file'])
                _make_figure(args, s, sample, target, cols_plot, fs_plot,
                             split=split)
        report[split] = {
            'entries': len(ds),                  # after session + max_clips caps
            'inspected': len(samples),           # visited (<= max_entries)
            'entries_per_session': counts,       # full dataset, in session order
            'window': {'clip_duration': args.clip_duration,
                       'seq_len': int(ds.seq_len), 'fs': args.fs},
            'clips': samples,
        }

    out_json = os.path.join(args.output_dir, 'inspect_summary.json')
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'Wrote summary to {out_json}')


def _make_figure(args, clip, sample, target, sig_cols, sig_fs, split='train'):
    """Save one preview PNG per inspected clip: RGB + TIR middle frame and the
    aligned 1-D signal windows (blue) with the dataset target (orange).

    ``clip`` is the per-clip report dict from :func:`main`; ``sample``/``target``
    are the already-decoded arrays. The figure is written to
    ``<output_dir>/figures/<split>/<session>_t<t0>s.png``.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f'[plot] matplotlib unavailable, skipping ({exc})')
        return

    session = clip['session']
    idx = clip['clip_index']
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
    tir_fps = clip.get('tir_fps') or args.fps    # actual TIR frame rate
    sig_fs = fs or args.fs                       # signal sample rate in use
    t_end = t0 + args.clip_duration
    n_cap = clip.get('n_clips')
    n_raw = clip.get('n_clips_raw') or n_cap
    clip_k1 = int(clip.get('clip_k') or 0) + 1   # 1-based window in the session
    capped = (n_cap is not None and n_raw is not None and n_cap < n_raw)
    info = [
        f'session: {session}   clip #{idx}  k={clip_k1}/{n_cap or "?"}   '
        f'split: {split}',
        f'clip dur {args.clip_duration:.1f} s | seq_len {target.size} | '
        f'input {args.input_size}',
        f'RGB {args.fps:.0f} fps | TIR {tir_fps:.0f} fps | '
        f'signal fs {sig_fs:.1f} Hz',
        f't-start {t0:.2f} s -> t-end {t_end:.2f} s',
    ]
    if capped:
        info.append(f'max_clips applied: {n_cap} of {n_raw} session windows shown')
    axes[2, 0].axis('off')
    axes[2, 0].text(0.0, 0.5, '\n'.join(info),
                    transform=axes[2, 0].transAxes,
                    ha='left', va='center', fontsize=8, family='monospace')

    fig.tight_layout()
    out_dir = os.path.join(args.output_dir, 'figures', split)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{session}_t{t0:06.2f}s.png')
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f'[plot] saved {path}')


if __name__ == '__main__':
    main()
