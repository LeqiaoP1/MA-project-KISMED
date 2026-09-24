"""Inspect ``BP4DPlusTIRRespDataset`` -- the thermal -> respiration clip dataset.

Draws, for one clip of one session, WHAT THE MODEL WILL ACTUALLY RECEIVE and
checks that it is what the ROI strategy promises:

* **TIR + clip-static ROI** -- the clip's first thermal frame with the single
  crop box that is applied to ALL 200 frames, the 12 target landmarks of the
  mouth + nose (1-indexed user-guide labels 9, 10, 11, 12, 13, 20, 21, 22, 23,
  24, 25, 26) drawn colour-coded by region, and the same target points of every
  frame scattered faintly behind them -- the points that min/max produced the
  box, so the "one box per clip, no per-frame jitter" claim is visible.
* **ROI patch** -- exactly the ``[3, T, H, W]`` tensor slice the dataset
  returns (first frame), NOT a re-crop: if the tensor and the box disagreed,
  this panel would show it.
* **respiration (raw)** -- the ``Resp_Volts.txt`` window over the clip's own
  time span, in volts (the dataset's ``norm='none'`` output).
* **respiration (target)** -- the normalised ``resp_signal`` the dataset
  returns, i.e. the per-clip z-score.

The LAYOUT follows the same priority: the annotated frame owns the full-width top
row (it is the panel that has to be read closely -- box edges and the 28-point
track), while the ROI patch is a small thumbnail next to the wide respiration
strip in the bottom row.

The same clip is run through the dataset's own verification
(``data.tir_resp_dataset._check_clip``) and both the checks and every shape /
range / ROI number land in the JSON report, so a run is auditable without the
figure. Figures are deliberately NUMBER-FREE (repo convention): the numbers
live in the JSON and on stdout.

Artifacts::

    <output_dir>/<subject>_<task>/clip_0000.png     # 4-panel figure (or .json only)
    <output_dir>/<subject>_<task>/clip_0000.json    # clip + dataset report
    <output_dir>/dataset_summary.json               # whole-dataset summary
    <output_dir>/tir_resp_index.json                # one row per clip inspected

Usage (from ``code/``)::

    python runners/run_inspect_tir_resp.py --list
    python runners/run_inspect_tir_resp.py --subject F001 --task T1 --clip_index 0
    python runners/run_inspect_tir_resp.py --subject F001 --task T1 --clip_index 0,1,7
    python runners/run_inspect_tir_resp.py --input_size 224 --no-plot

Paths default to ``$RAW_DATA_PATH`` (raw BP4D root) and
``$OUTPUT_DIR/inspect_data/tir_resp`` (see ``scripts/env_local.sh``); without the
env profile they fall back to the in-repo ``data/raw/BP4D`` and
``<repo>/output``.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data import tir_resp_dataset as trd

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_DIR = os.path.dirname(_CODE_DIR)

FIG_TAG = 'clip'

#: the 12 ROI landmarks split by region, for the colours in panel 1
_NOSE = (9, 10, 20, 21)
_MOUTH = (11, 12, 13, 22, 23, 24, 25, 26)


# --------------------------------------------------------------------------- #
# path defaults
# --------------------------------------------------------------------------- #
def _env(key: str, default: str = '') -> str:
    return os.environ.get(key, default)


def default_output_dir() -> str:
    """``$OUTPUT_DIR/inspect_data/tir_resp``, else ``<repo>/output/...``.

    Absolute on purpose: a bare ``./output`` would land inside ``code/`` when
    the env profile was not sourced.
    """
    root = _env('OUTPUT_DIR', '') or os.path.join(_REPO_DIR, 'output')
    return os.path.join(root, 'inspect_data', 'tir_resp')


def _split(value):
    """``'a,b' -> ['a', 'b']``; ``'all'``/empty -> ``None`` (no filter)."""
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() == 'all':
        return None
    return [v.strip() for v in value.split(',') if v.strip()]


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def _plot_clip(ds, index: int, path: str, box, lm, frame0, tir, raw, tgt):
    """4-panel figure for one clip; see the module docstring for the content."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    entry = ds.entries[index]
    # The annotated frame IS the point of this figure, so it gets the whole top
    # row and most of the height; the ROI patch is a thumbnail and the
    # respiration window is wide, so they share the smaller bottom row.
    fig = plt.figure(figsize=(12.5, 10.0), layout='constrained')
    ax = fig.subplot_mosaic([['roi', 'roi'], ['patch', 'resp']],
                            height_ratios=[1.75, 1.0], width_ratios=[1.0, 2.4])

    # --- panel 1: the frame, the static box, the landmarks that made it
    x0, x1, y0, y1 = box
    ax['roi'].imshow(frame0, interpolation='nearest')
    ax['roi'].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                  ec='#ff2020', lw=2.0))
    p = np.asarray(lm)[:, np.asarray(_NOSE + _MOUTH) - 1, :]
    ax['roi'].scatter(p[..., 0].ravel(), p[..., 1].ravel(), s=3.0, c='#00e5ff',
                      alpha=0.45, lw=0)
    ax['roi'].scatter(p[0, :4, 0], p[0, :4, 1], s=44, c='#ffe600',
                      edgecolors='k', lw=0.6, label='nose')
    ax['roi'].scatter(p[0, 4:, 0], p[0, 4:, 1], s=44, c='#ff3df2',
                      edgecolors='k', lw=0.6, label='mouth')
    ax['roi'].legend(loc='upper right', fontsize=9, framealpha=0.7)
    ax['roi'].set_title(f'{entry["session"]} clip {index}: '
                        f'frames {entry["frame_start"]}-{entry["frame_end"] - 1}',
                        fontsize=12)

    # --- panel 2: the tensor slice the dataset returns (not a re-crop)
    patch = tir[:, 0].permute(1, 2, 0).numpy()
    ax['patch'].imshow(np.clip(patch, 0.0, 1.0), interpolation='nearest',
                       aspect='equal')
    ax['patch'].set_title(f'ROI patch {ds.input_size}x{ds.input_size} (tensor)')

    # --- panel 3: raw respiration volts + the normalised target
    t = entry['t_start'] + np.arange(ds.resp_len) / ds.resp_fs
    ax['resp'].plot(t, raw, color='0.35', lw=0.8, label='Resp_Volts.txt')
    ax['resp'].set_ylabel('respiration [V]')
    ax['resp'].set_xlabel('time [s]')
    ax['resp'].set_xlim(float(t[0]), float(t[-1]))
    ax['resp'].set_title(f'clip {index} respiration window '
                         f'({ds.resp_len} samples @ {ds.resp_fs:g} Hz)')
    tw = ax['resp'].twinx()
    tw.plot(t, tgt, color='#d62728', lw=0.9, label='resp_signal (z-scored)')
    tw.set_ylabel('resp_signal [z]')
    h1, l1 = ax['resp'].get_legend_handles_labels()
    h2, l2 = tw.get_legend_handles_labels()
    ax['resp'].legend(h1 + h2, l1 + l2, loc='lower right', fontsize=7,
                      framealpha=0.7)

    for key in ('roi', 'patch'):
        ax[key].set_xticks([])
        ax[key].set_yticks([])
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _clip_report(ds, index: int, box, fails) -> dict:
    entry = ds.entries[index]
    item = ds[index]
    tir, resp = item['tir_video'], item['resp_signal']
    raw = ds.respiration_clip(index, normalized=False)
    lm = ds.clip_landmarks(index)
    x0, x1, y0, y1 = box
    return {
        'index': int(index),
        'session': entry['session'],
        'subject': entry['subject'],
        'task': entry['task'],
        'frame_start': int(entry['frame_start']),
        'frame_end': int(entry['frame_end']),
        't_start_s': float(entry['t_start']),
        't_end_s': float(entry['t_end']),
        'video': entry['video'],
        'ir_file': entry['ir_file'],
        'resp_file': entry['resp_file'],
        'roi': {
            'box_px': [int(v) for v in box],
            'width_px': int(x1 - x0),
            'height_px': int(y1 - y0),
            'padding': float(ds.roi_padding),
            'landmarks_1based': list(ds.target_landmarks),
            'landmarks_0based': [int(i) for i in ds.target_idx],
            'static_over_frames': int(ds.clip_frames),
            'landmark_span_px': [float(np.nanmin(lm[..., 0])),
                                 float(np.nanmax(lm[..., 0])),
                                 float(np.nanmin(lm[..., 1])),
                                 float(np.nanmax(lm[..., 1]))],
        },
        'tir_video': {
            'shape': list(tir.shape),
            'dtype': str(tir.dtype),
            'min': float(tir.min()),
            'max': float(tir.max()),
            'mean': float(tir.mean()),
        },
        'resp_signal': {
            'shape': list(resp.shape),
            'dtype': str(resp.dtype),
            'min': float(resp.min()),
            'max': float(resp.max()),
            'mean': float(resp.mean()),
            'std': float(resp.std(unbiased=False)),
            'raw_min_v': float(np.nanmin(raw)),
            'raw_max_v': float(np.nanmax(raw)),
            'norm': ds.norm,
        },
        'subject_task': item['subject_task'],
        'checks_failed': list(fails),
    }


def _append_index(output_dir: str, rows) -> str:
    path = os.path.join(output_dir, 'tir_resp_index.json')
    existing = []
    if os.path.isfile(path):
        try:
            with open(path) as fh:
                existing = json.load(fh)
        except (OSError, ValueError):
            existing = []
    if not isinstance(existing, list):
        existing = []
    existing.extend(rows)
    os.makedirs(output_dir, exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(existing, fh, indent=2)
    return path


# --------------------------------------------------------------------------- #
# list mode
# --------------------------------------------------------------------------- #
def list_sessions(raw_root: str) -> int:
    """Print the discovered sessions and what each one is missing."""
    try:
        specs = trd.discover_sessions(raw_root)
    except FileNotFoundError as exc:
        print(f'[error] {exc}')
        return 1
    print(f'raw root: {raw_root}')
    print(f'{"session":<10} {"video":<6} {"IR":<4} {"resp":<6} note')
    n_ok = 0
    for spec in specs:
        has = [spec['video'] is not None,
               spec['ir_file'] is not None,
               spec['resp_file'] is not None]
        note = ''
        if not has[1]:
            note = 'no IRFeatures -> skipped (untracked / glasses sequence)'
        elif not has[2]:
            note = f'no {trd.RESP_FILE} -> skipped'
        else:
            n_ok += 1
            try:
                mask = trd.missing_frame_mask(
                    trd.parse_ir_features(spec['ir_file']))
                if mask.any():
                    note = (f'{int(mask.sum())} (0,0) sentinel line(s) -> '
                            f'clips touching them are dropped')
            except Exception as exc:
                note = f'IRFeatures unusable: {exc}'
        flag = '*' if (not has[1] or not has[2]) else ' '
        print(f'{spec["session"]:<10} {str(has[0]):<6} {str(has[1]):<4} '
              f'{str(has[2]):<6}{flag} {note}')
    print(f'{len(specs)} thermal session(s) on disk, {n_ok} usable '
          f'("*" = would be skipped)')
    return 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description='Inspect BP4DPlusTIRRespDataset clips (ROI + respiration).')
    p.add_argument('--raw_root', default=trd.default_raw_root(),
                   help='raw BP4D root (default $RAW_DATA_PATH)')
    p.add_argument('--subject', default='F001', help='comma list, or all')
    p.add_argument('--task', default='T1', help='comma list, or all')
    p.add_argument('--clip_seconds', type=float, default=trd.DEFAULT_CLIP_SECONDS)
    p.add_argument('--clip_stride', type=float, default=None,
                   help='clip hop in seconds (default: non-overlapping)')
    p.add_argument('--fps', type=float, default=trd.DEFAULT_FPS)
    p.add_argument('--resp_fs', type=float, default=trd.DEFAULT_RESP_FS)
    p.add_argument('--input_size', type=int, default=trd.DEFAULT_INPUT_SIZE)
    p.add_argument('--roi_padding', type=float, default=trd.DEFAULT_ROI_PADDING)
    p.add_argument('--clip_index', default='0',
                   help='comma list of clip indices in the selection')
    p.add_argument('--max_entries', type=int, default=0, help='0 = no cap')
    p.add_argument('--output_dir', default=default_output_dir())
    p.add_argument('--no-plot', dest='plot', action='store_false',
                   help='write the JSON reports only')
    p.add_argument('--list', action='store_true',
                   help='list the thermal sessions on disk and exit')
    args = p.parse_args(argv)

    if args.list:
        return list_sessions(args.raw_root)

    ds = trd.BP4DPlusTIRRespDataset(
        raw_root=args.raw_root, subjects=_split(args.subject),
        tasks=_split(args.task), clip_seconds=args.clip_seconds,
        clip_stride=args.clip_stride, fps=args.fps, resp_fs=args.resp_fs,
        input_size=args.input_size, roi_padding=args.roi_padding,
        max_entries=args.max_entries or None)

    print('=' * 76)
    print('BP4DPlusTIRRespDataset -- clip inspection')
    print('=' * 76)
    print(ds.describe())
    print('-' * 76)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'dataset_summary.json'), 'w') as fh:
        json.dump(ds.stats, fh, indent=2)

    wanted = [int(v) for v in str(args.clip_index).split(',') if v.strip()]
    rows, n_fail = [], 0
    for index in wanted:
        if not 0 <= index < len(ds):
            print(f'[skip] clip {index}: out of range (0..{len(ds) - 1})')
            continue
        entry = ds.entries[index]
        lm = ds.clip_landmarks(index)
        box = ds.clip_roi_box(index)
        frames = ds._frames(entry)
        fails = trd._check_clip(ds, index)
        rep = _clip_report(ds, index, box, fails)
        if fails:
            n_fail += 1

        out_dir = os.path.join(args.output_dir, entry['session'])
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.join(out_dir, f'{FIG_TAG}_{index:04d}')
        if args.plot:
            try:
                _plot_clip(ds, index, stem + '.png', box, lm, frames[0],
                           ds[index]['tir_video'],
                           ds.respiration_clip(index, normalized=False),
                           ds[index]['resp_signal'])
                rep['figure'] = stem + '.png'
            except Exception as exc:                     # never lose the JSON
                rep['figure_error'] = str(exc)
                print(f'[warn] figure failed: {exc}')
        with open(stem + '.json', 'w') as fh:
            json.dump(rep, fh, indent=2)

        roi = rep['roi']
        tag = 'FAIL' if fails else 'ok'
        print(f'[{tag}] clip {index} {entry["session"]} '
              f'frames {entry["frame_start"]}-{entry["frame_end"] - 1} '
              f'({entry["t_start"]:.2f}-{entry["t_end"]:.2f} s) '
              f'roi {roi["width_px"]}x{roi["height_px"]} px -> '
              f'{tuple(rep["tir_video"]["shape"])} + resp '
              f'{tuple(rep["resp_signal"]["shape"])}')
        for f in fails:
            print(f'       - {f}')
        rows.append({'session': entry['session'], 'clip_index': index,
                     'figure': rep.get('figure'), 'json': stem + '.json',
                     'roi_box_px': roi['box_px'],
                     'tir_shape': rep['tir_video']['shape'],
                     'resp_shape': rep['resp_signal']['shape'],
                     'checks_failed': fails})

    if rows:
        idx_path = _append_index(args.output_dir, rows)
        print('-' * 76)
        print(f'wrote {len(rows)} clip report(s) -> {args.output_dir}')
        print(f'index: {idx_path}')
    print('-' * 76)
    print(f'{"FAIL" if n_fail else "PASS"}: {len(rows) - n_fail}/{len(rows)} '
          f'clip(s) checked, {len(ds)} clip(s) in the selection')
    return 1 if n_fail else 0


if __name__ == '__main__':
    raise SystemExit(main())
