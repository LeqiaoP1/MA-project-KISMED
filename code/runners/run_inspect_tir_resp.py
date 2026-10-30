"""Inspect ``BP4DPlusTIRRespDataset`` -- the thermal -> respiration clip dataset.

Draws, for one clip of one session, WHAT THE MODEL WILL ACTUALLY RECEIVE and
checks that it is what the ROI strategy promises:

* **TIR + clip-static ROI** -- the clip's first thermal frame with the single
  crop box that is applied to ALL 200 frames, the target landmarks of the
  mouth + nose drawn colour-coded by region, and the same target points of every
  frame scattered faintly behind them -- the points the box was built from, so
  the "one box per clip, no per-frame jitter" claim is visible.
* **ROI patch** -- exactly the ``[3, T, H, W]`` tensor slice the dataset
  returns (first frame), NOT a re-crop: if the tensor and the box disagreed,
  this panel would show it. The SAME 12 landmarks are drawn again, mapped into
  PATCH pixels (``(x - x0) * S/(x1 - x0)``), so the crop's own framing of the
  nose and mouth is visible -- not just the box on the source frame.
* **respiration (raw VOLTS)** -- the ``Resp_Volts.txt`` window over the clip's
  own time span, in volts. The dataset's ``resp_signal`` is the SAME array: the
  default is ``norm='none'``, i.e. **no per-clip z-score**. ``--norm clip`` /
  ``--norm session`` exist only to inspect an explicitly normalised run, and the
  panel and the JSON then say so.

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
    <output_dir>/<subject>_<task>/clip_0000.gif     # the WHOLE 8 s clip as an
                                                    # animation (--video; .mp4
                                                    # instead with --video mp4)
    <output_dir>/<subject>_<task>/clip_0000.json    # clip + dataset report
    <output_dir>/dataset_summary.json               # whole-dataset summary
    <output_dir>/tir_resp_index.json                # one row per clip inspected

Usage (from ``code/``)::

    python runners/run_inspect_tir_resp.py --list
    python runners/run_inspect_tir_resp.py --subject F001 --task T1 --clip_index 0
    python runners/run_inspect_tir_resp.py --subject F001 --task T1 --clip_index 0,1,7
    python runners/run_inspect_tir_resp.py --subject F001 --task T1 --t_start 12
    python runners/run_inspect_tir_resp.py --input_size 64 --no-plot

``--clip_index`` addresses the ENTRY list, which the motion gate renumbers: an
index means a different window before and after a geometry change. ``--t_start``
(seconds) / ``--frame_start`` (frames) address the WINDOW itself and override
``--clip_index``; if the window is not in the entry list the tool re-applies the
gates and reports which one dropped it, exiting non-zero.

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

#: Inspection default ROI-patch side. ``trd.DEFAULT_INPUT_SIZE`` (64) is the
#: LOCAL/smoke training geometry -- a DOWN-sample of the ~73x89 px native box,
#: which is too small to read in the figure. 112 is both the HPC Stage-2/3
#: lineage's ``input_size`` and an up-sample that starts to look like the face
#: it is; it does NOT change what the model sees (pass ``--input_size 64`` to
#: inspect the local training geometry exactly).
DEFAULT_INSPECT_INPUT_SIZE = 112

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
def _plot_clip(ds, index: int, path: str, box, lm, frame0, tir, resp):
    """4-panel figure for one clip; see the module docstring for the content.

    ``resp`` is the raw respiration window in VOLTS (the dataset's
    ``resp_signal`` under the default ``norm='none'``).
    """
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
    # draw the landmarks the ROI was ACTUALLY built from (--roi_landmarks)
    # instead of a hardcoded set: with a preset (or a custom CSV) the box would
    # otherwise be annotated with points that did NOT produce it.
    want = [int(l) for l in ds.target_landmarks]
    p = np.asarray(lm)[:, np.asarray(want) - 1, :]
    ax['roi'].scatter(p[..., 0].ravel(), p[..., 1].ravel(), s=3.0, c='#00e5ff',
                      alpha=0.45, lw=0)
    for labels, colour, name in ((_NOSE, '#ffe600', 'nose'),
                                 (_MOUTH, '#ff3df2', 'mouth')):
        sel = [want.index(lab) for lab in labels if lab in want]
        if not sel:
            continue
        ax['roi'].scatter(p[0, sel, 0], p[0, sel, 1], s=44, c=colour,
                          edgecolors='k', lw=0.6, label=name)
    ax['roi'].legend(loc='upper right', fontsize=9, framealpha=0.7)
    ax['roi'].set_title(f'{entry["session"]} clip {index}: '
                        f'frames {entry["frame_start"]}-{entry["frame_end"] - 1}'
                        f'  [ROI: {len(want)} landmark(s)]',
                        fontsize=12)

    # --- panel 2: the tensor slice the dataset returns (not a re-crop)
    patch = tir[:, 0].permute(1, 2, 0).numpy()
    ax['patch'].imshow(np.clip(patch, 0.0, 1.0), interpolation='nearest',
                       aspect='equal')
    # The SAME landmarks, re-expressed in PATCH pixels. The crop is rescaled with
    # ONE factor sigma = S / side: with crop_rule='square' the box is square, so
    # sx == sy and the anatomy is not deformed. A per-axis factor here would
    # draw a mapping the pipeline does not perform.
    s_px = float(ds.input_size)
    if getattr(ds, 'crop_rule', 'box') == 'square' and (x1 - x0) != (y1 - y0):
        raise ValueError(f'crop_rule=square but the box is '
                         f'{x1 - x0}x{y1 - y0} (invariant I6 violated)')
    sigma = s_px / float(max(x1 - x0, 1))
    px = (p[..., 0] - x0) * sigma
    py = (p[..., 1] - y0) * sigma
    ax['patch'].scatter(px.ravel(), py.ravel(), s=3.0, c='#00e5ff',
                        alpha=0.45, lw=0)
    for labels, colour in ((_NOSE, '#ffe600'), (_MOUTH, '#ff3df2')):
        sel = [want.index(lab) for lab in labels if lab in want]
        if not sel:
            continue
        ax['patch'].scatter(px[0, sel], py[0, sel], s=34, c=colour,
                            edgecolors='k', lw=0.5)
    ax['patch'].set_title(f'ROI patch {ds.input_size}x{ds.input_size} '
                          f'(tensor + {len(want)} landmarks)')

    # --- panel 3: the respiration window, RAW VOLTS (no z-score) ----------
    # The legend never mentions the scale story: 'resp_signal [V]' when the
    # dataset is raw (the default), 'resp_signal [z]' only for an explicit
    # --norm clip/session run.
    raw_units = ds.norm == 'none'
    unit = 'V' if raw_units else 'z'
    t = entry['t_start'] + np.arange(ds.resp_len) / ds.resp_fs
    ax['resp'].plot(t, resp, color='#1f4e9c', lw=0.9,
                    label=f'resp_signal [{unit}]')
    ax['resp'].set_ylabel(f'respiration [{unit}]')
    ax['resp'].set_xlabel('time [s]')
    ax['resp'].set_xlim(float(t[0]), float(t[-1]))
    ax['resp'].set_title(
        f'clip {index} respiration window, '
        f'{"raw volts" if raw_units else f"norm={ds.norm}"} '
        f'({ds.resp_len} samples @ {ds.resp_fs:g} Hz)')
    ax['resp'].legend(loc='lower right', fontsize=7, framealpha=0.7)

    for key in ('roi', 'patch'):
        ax[key].set_xticks([])
        ax[key].set_yticks([])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _write_video(ds, index: int, stem: str, box, lm, frames, tir, resp,
                 fps: float = 25.0, dpi: int = 70,
                 fmt: str = 'gif'):
    """Write the WHOLE clip (8 s) as an animation, not one frame.

    Layout mirrors the static figure: source frame + box + landmark track,
    the ``[3, T, S, S]`` ROI patch as the dataset returns it, and the raw
    respiration window with a playhead that follows the frame's time. Frames
    are rendered with the Agg canvas (no ffmpeg needed); ``mp4`` uses OpenCV's
    ``mp4v`` (MPEG-4 Part 2), ``gif`` uses Pillow with ONE shared palette so
    the false-colour thermal image does not flicker.

    ``fps`` is the OUTPUT rate: source frames are decimated to the nearest
    integral ratio (``25`` = every frame, ``12.5`` = every 2nd). Returns the
    written paths.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    entry = ds.entries[index]
    step = max(1, int(round(ds.fps / float(fps))))
    out_fps = ds.fps / step
    resp = np.asarray(resp, dtype=np.float32)

    want = [int(l) for l in ds.target_landmarks]
    p = np.asarray(lm)[:, np.asarray(want) - 1, :]
    x0, x1, y0, y1 = box
    s_px = float(ds.input_size)
    sigma = s_px / float(max(x1 - x0, 1))      # ONE scale: the box is square
    px = (p[..., 0] - x0) * sigma
    py = (p[..., 1] - y0) * sigma
    nose_sel = [want.index(lab) for lab in _NOSE if lab in want]
    mouth_sel = [want.index(lab) for lab in _MOUTH if lab in want]

    fig = plt.figure(figsize=(9.0, 6.0), dpi=int(dpi))
    ax = fig.subplot_mosaic([['roi', 'patch'], ['resp', 'resp']],
                            height_ratios=[1.8, 1.0], width_ratios=[2.0, 1.0])
    fig.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.09,
                        hspace=0.16, wspace=0.05)

    # source frame + the clip-static box + the WHOLE track (static)
    im_src = ax['roi'].imshow(frames[0], interpolation='nearest')
    ax['roi'].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                  ec='#ff2020', lw=2.0))
    ax['roi'].scatter(p[..., 0].ravel(), p[..., 1].ravel(), s=2.5,
                      c='#00e5ff', alpha=0.35, lw=0)
    src_nose = ax['roi'].scatter([], [], s=34, c='#ffe600', edgecolors='k',
                                 lw=0.5, label='nose')
    src_mouth = ax['roi'].scatter([], [], s=34, c='#ff3df2', edgecolors='k',
                                  lw=0.5, label='mouth')
    ax['roi'].legend(loc='upper right', fontsize=8, framealpha=0.7)

    # the ROI patch exactly as the dataset returns it (this frame)
    im_patch = ax['patch'].imshow(
        np.clip(tir[:, 0].permute(1, 2, 0).numpy(), 0.0, 1.0),
        interpolation='nearest', aspect='equal')
    ax['patch'].scatter(px.ravel(), py.ravel(), s=2.0, c='#00e5ff',
                        alpha=0.3, lw=0)
    patch_nose = ax['patch'].scatter([], [], s=28, c='#ffe600',
                                     edgecolors='k', lw=0.4)
    patch_mouth = ax['patch'].scatter([], [], s=28, c='#ff3df2',
                                      edgecolors='k', lw=0.4)

    # raw respiration + playhead
    tt = entry['t_start'] + np.arange(ds.resp_len) / ds.resp_fs
    ax['resp'].plot(tt, resp, color='#1f4e9c', lw=0.9,
                    label='resp_signal [V]' if ds.norm == 'none'
                    else f'resp_signal [{ds.norm}]')
    play = ax['resp'].axvline(float(tt[0]), color='#d62728', lw=1.2)
    dot, = ax['resp'].plot([], [], 'o', color='#d62728', ms=5)
    ax['resp'].set_xlim(float(tt[0]), float(tt[-1]))
    ax['resp'].set_xlabel('time [s]')
    ax['resp'].set_ylabel('respiration [V]' if ds.norm == 'none'
                          else 'respiration [z]')
    ax['resp'].legend(loc='lower right', fontsize=7, framealpha=0.7)
    for key in ('roi', 'patch'):
        ax[key].set_xticks([])
        ax[key].set_yticks([])

    writer = None
    gif_frames, gif_base = [], None
    if fmt in ('mp4', 'both'):
        import cv2
    from PIL import Image
    n = int(tir.shape[1])
    for i in range(0, n, step):
        im_src.set_data(frames[i])
        im_patch.set_data(np.clip(tir[:, i].permute(1, 2, 0).numpy(), 0.0, 1.0))
        src_nose.set_offsets(np.column_stack([p[i, nose_sel, 0],
                                              p[i, nose_sel, 1]])
                             if nose_sel else np.empty((0, 2)))
        src_mouth.set_offsets(np.column_stack([p[i, mouth_sel, 0],
                                               p[i, mouth_sel, 1]])
                              if mouth_sel else np.empty((0, 2)))
        patch_nose.set_offsets(np.column_stack([px[i, nose_sel],
                                                py[i, nose_sel]])
                               if nose_sel else np.empty((0, 2)))
        patch_mouth.set_offsets(np.column_stack([px[i, mouth_sel],
                                                 py[i, mouth_sel]])
                                if mouth_sel else np.empty((0, 2)))
        t_i = entry['t_start'] + i / ds.fps
        k = int(min(max(round((t_i - entry['t_start']) * ds.resp_fs), 0),
                    ds.resp_len - 1))
        play.set_xdata([t_i, t_i])
        dot.set_data([t_i], [resp[k]])
        ax['roi'].set_title(
            f'{entry["session"]} clip {index}   t = {t_i:5.2f} s   '
            f'frame {i}/{n - 1}', fontsize=11)
        fig.canvas.draw()
        a = np.asarray(fig.canvas.buffer_rgba())[..., :3]

        if fmt in ('mp4', 'both'):
            if writer is None:
                h, w = a.shape[:2]
                writer = cv2.VideoWriter(
                    stem + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                    out_fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError('cv2.VideoWriter(mp4v) refused to open')
            writer.write(np.ascontiguousarray(a[..., ::-1]))
        if fmt in ('gif', 'both'):
            im = Image.fromarray(np.ascontiguousarray(a))
            if gif_base is None:
                gif_base = im.quantize(colors=255,
                                       method=Image.Quantize.MEDIANCUT)
                gif_frames.append(gif_base)
            else:
                gif_frames.append(im.quantize(palette=gif_base,
                                              dither=Image.Dither.NONE))
    if writer is not None:
        writer.release()
    plt.close(fig)

    paths = []
    if fmt in ('mp4', 'both'):
        paths.append(stem + '.mp4')
    if fmt in ('gif', 'both'):
        if not gif_frames:
            raise RuntimeError('no frames rendered for the GIF')
        gif_frames[0].save(stem + '.gif', save_all=True,
                           append_images=gif_frames[1:], loop=0,
                           duration=int(round(1000.0 / out_fps)),
                           optimize=True, disposal=1)
        paths.append(stem + '.gif')
    return paths


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _coverage(box, lm, target_idx, src_h: int, src_w: int) -> dict:
    """Square/coverage metrics of one clip's box, over the TARGET landmarks.

    Pure arithmetic on (box, landmark track, frame size) so it can be checked
    without a decode. The box is built from the TARGET points only, so the
    coverage metrics must use exactly those: taking the span of all 28 landmarks
    once reported ``envelope_frac_of_patch = 1.36`` (eyes and ears are far wider
    than the nose+mouth box) and silently disabled the coverage flag.
    """
    x0, x1, y0, y1 = box
    side = max(x1 - x0, y1 - y0)
    tgt = np.asarray(lm)[:, np.asarray(target_idx), :]
    tx0, tx1 = float(np.nanmin(tgt[..., 0])), float(np.nanmax(tgt[..., 0]))
    ty0, ty1 = float(np.nanmin(tgt[..., 1])), float(np.nanmax(tgt[..., 1]))
    return {
        'side_px': int(side),
        'is_square': bool((x1 - x0) == (y1 - y0)),
        'envelope_frac_of_patch': float((tx1 - tx0) * (ty1 - ty0)
                                       / float(max(side * side, 1))),
        'area_frac_of_frame': float((side * side)
                                    / float(max(src_w * src_h, 1))),
        'target_span_px': [tx0, tx1, ty0, ty1],
        'targets_inside': bool(x0 <= tx0 and tx1 < x1
                               and y0 <= ty0 and ty1 < y1),
    }


def _clip_report(ds, index: int, box, fails) -> dict:
    entry = ds.entries[index]
    item = ds[index]
    tir, resp = item['tir_video'], item['resp_signal']
    raw = ds.respiration_clip(index, normalized=False)
    lm = ds.clip_landmarks(index)
    x0, x1, y0, y1 = box
    src_h, src_w = ds._source_hw(entry)
    crop_rule = str(getattr(ds, 'crop_rule', 'box'))
    max_sway = float(getattr(ds, 'max_sway_px', 0.0))
    # The gate statistic of THIS window, whether or not the gate is enabled: on
    # a kept clip it is the only observable of how close the window came to
    # being dropped, and how much of the patch is actually face.
    sway = trd.landmark_sway(lm, ds.target_idx)
    cov = _coverage(box, lm, ds.target_idx, src_h, src_w)
    flags = []
    if not cov['is_square']:
        flags.append('box_not_square')
    if not cov['targets_inside']:
        flags.append('targets_outside_box')
    if max_sway > 0.0 and (not np.isfinite(sway) or sway > 0.8 * max_sway):
        flags.append('near_gate')
    if cov['area_frac_of_frame'] > 0.4:
        flags.append('box_gt_40pct_frame')
    if cov['envelope_frac_of_patch'] < 0.25:
        flags.append('envelope_lt_25pct_patch')
    dropped_motion = int(ds.sessions.get(entry['session'], {})
                         .get('clips_dropped_motion', 0) or 0)
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
            'side_px': cov['side_px'],
            'is_square': cov['is_square'],
            'crop_rule': crop_rule,
            'padding': float(ds.roi_padding),
            'max_sway_px': max_sway,
            'sway_px': float(sway),
            'gate_margin_px': float(max_sway - sway) if max_sway > 0 else None,
            'clips_dropped_motion': dropped_motion,
            'source_hw': [int(src_h), int(src_w)],
            'area_frac_of_frame': cov['area_frac_of_frame'],
            'envelope_frac_of_patch': cov['envelope_frac_of_patch'],
            'target_span_px': cov['target_span_px'],
            'landmarks_1based': list(ds.target_landmarks),
            'landmarks_0based': [int(i) for i in ds.target_idx],
            'static_over_frames': int(ds.clip_frames),
            'landmark_span_px': [float(np.nanmin(lm[..., 0])),
                                 float(np.nanmax(lm[..., 0])),
                                 float(np.nanmin(lm[..., 1])),
                                 float(np.nanmax(lm[..., 1]))],
        },
        'flags': flags,
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
            'units': 'V' if ds.norm == 'none' else 'z',
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
# why is a window absent?
# --------------------------------------------------------------------------- #
def _diagnose_window(ds, frame_start: int, max_sessions: int = 5
                     ) -> tuple:
    """Why is ``[frame_start, frame_start + clip_frames)`` not an entry?

    Re-applies the gates in the SAME order the ENTRY BUILDER does -- session
    gates, sentinel, motion, rail, flat -- and, separately, the box geometry the
    LOADER applies (``RoiBoxOutOfFrame``). This is the only way to review a clip
    the gate removed: a dropped window is not in ``ds.entries``, so no index can
    address it.

    :returns: ``(lines, reasons)`` -- human-readable detail plus a short tag list
        (``sentinel`` / ``motion`` / ``rail`` / ``flat`` / ``envelope`` /
        ``beyond_usable`` / session-skip reasons).
    """
    cf = ds.clip_frames
    lines: list = []
    reasons: list = []
    if frame_start < 0:
        return [f'frame_start {frame_start} is negative'], ['negative']
    if any(int(e['frame_start']) == frame_start for e in ds.entries):
        return ([f'frame_start {frame_start} IS in the entry list -- nothing to '
                 f'diagnose'], ['present'])
    if ds.stride_frames > 1 and frame_start % ds.stride_frames:
        lines.append(f'NOT on the clip hop grid: stride is '
                     f'{ds.stride_frames} frames, so the emitted windows start '
                     f'at multiples of it')

    used = sorted(ds.sessions)
    skipped = {s['session']: s['reason'] for s in ds.skipped}
    names = used + [k for k in sorted(skipped) if k not in ds.sessions]
    if not names:
        return ['no session in the selection'], ['no_session']

    n_shown = 0
    for sess in names:
        if n_shown >= max_sessions:
            lines.append(f'... and {len(names) - n_shown} more session(s) in the '
                         f'selection; narrow with --subject/--task')
            break
        if sess in skipped:
            lines.append(f'{sess}: session dropped whole ({skipped[sess]})')
            reasons.append('session_skip')
            n_shown += 1
            continue
        spec = ds.sessions[sess]
        n_avail = min(int(spec['n_frames']),
                      int(spec['resp_start_limit']) + cf)
        if frame_start + cf > n_avail:
            lines.append(f'{sess}: window ends at frame {frame_start + cf}, '
                         f'beyond the {n_avail} usable frame(s)')
            reasons.append('beyond_usable')
            n_shown += 1
            continue
        ir = ds._landmarks(sess)
        if trd.missing_frame_mask(ir[frame_start:frame_start + cf],
                                  ds.target_idx).any():
            lines.append(f'{sess}: window overlaps a (0,0)/unparseable '
                         f'target landmark -> dropped as SENTINEL')
            reasons.append('sentinel')
            n_shown += 1
            continue
        if ds.max_sway_px > 0.0:
            sway = trd.landmark_sway(ir[frame_start:frame_start + cf],
                                     ds.target_idx)
            if sway > ds.max_sway_px:
                lines.append(f'{sess}: head sway {sway:.1f} px > '
                             f'max_sway_px={ds.max_sway_px:g} -> dropped by the '
                             f'MOTION gate')
                reasons.append('motion')
                n_shown += 1
                continue
        try:
            frames = ds._frames({'session': sess, 'video': spec['video'],
                                 'frame_start': frame_start,
                                 'frame_end': frame_start + cf})
            h, w = int(frames.shape[1]), int(frames.shape[2])
            trd.roi_box_from_landmarks(
                ir[frame_start:frame_start + cf][:, ds.target_idx, :], w, h,
                ds.roi_padding, crop_rule=ds.crop_rule)
        except trd.RoiBoxOutOfFrame as exc:
            lines.append(f'{sess}: GEOMETRY -- {exc}')
            reasons.append('envelope')
            n_shown += 1
            continue
        except Exception as exc:                     # undecodable, short read
            lines.append(f'{sess}: could not decode the window ({exc})')
            reasons.append('decode')
            n_shown += 1
            continue
        window = ds._resp_clip({'session': sess,
                                't_start': frame_start / ds.fps})
        if ds.rail_touch_v > 0.0 and window.size and \
                bool(np.any(np.abs(window) >= ds.rail_touch_v)):
            lines.append(f'{sess}: window touches the recorder rail '
                         f'(|x| >= {ds.rail_touch_v:g} V) -> dropped as RAIL')
            reasons.append('rail')
        elif ds.min_signal_spread > 0.0 and \
                float(window.max() - window.min()) < ds.min_signal_spread:
            lines.append(f'{sess}: window spread '
                         f'{float(window.max() - window.min()):.3f} V < '
                         f'min_signal_spread={ds.min_signal_spread:g} V '
                         f'-> dropped as FLAT')
            reasons.append('flat')
        else:
            lines.append(f'{sess}: PASSES every filter it can be checked '
                         f'against -- the window exists but its frame_start is '
                         f'{frame_start}, which is not on this run\'s hop grid '
                         f'({ds.stride_frames} frames); try '
                         f'--frame_start {frame_start - frame_start % ds.stride_frames}'
                         f'.')
            reasons.append('off_grid')
        n_shown += 1
    return lines, sorted(set(reasons))


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
    p.add_argument('--task_set', default=None,
                   help="named task group(s) to select, e.g. 'low' or "
                        "'low,high' (UNIONED with --task); default: no group "
                        'filter. See code/data/task_groups.py')
    p.add_argument('--task_groups', default=None,
                   help="group definitions, e.g. 'low=T1|T2;high=T7|T8|T9' "
                        "(the config's task_groups: mapping in string form)")
    p.add_argument('--clip_seconds', type=float, default=trd.DEFAULT_CLIP_SECONDS)
    p.add_argument('--clip_stride', type=float, default=None,
                   help='clip hop in seconds (default: non-overlapping)')
    p.add_argument('--fps', type=float, default=trd.DEFAULT_FPS)
    p.add_argument('--resp_fs', type=float, default=trd.DEFAULT_RESP_FS)
    p.add_argument('--input_size', type=int, default=DEFAULT_INSPECT_INPUT_SIZE,
                   help='ROI patch side in the figure/report (default 112; the '
                        'TRAINING geometry is 64 -- pass 64 to inspect exactly '
                        'what the model receives)')
    p.add_argument('--roi_padding', type=float, default=trd.DEFAULT_ROI_PADDING)
    p.add_argument('--crop_rule', default=trd.DEFAULT_CROP_RULE,
                   choices=list(trd.CROP_RULES),
                   help="how the padded landmark box becomes a crop: 'square' "
                        "(default) squares it before the rescale, so ONE "
                        "scale factor is applied and the anatomy is not "
                        "deformed; 'box' is the legacy per-axis geometry. "
                        'Must match the run whose input you are inspecting.')
    p.add_argument('--max_sway_px', type=float,
                   default=trd.DEFAULT_MAX_SWAY_PX,
                   help='motion gate of the run being inspected: a clip whose '
                        'target-landmark centroid travels further than this '
                        'many source pixels from its own median is NOT in the '
                        'entry list (0 = gate disabled). Must match that run; '
                        'the reported sway_px is the window statistic for the '
                        'clip you asked for.')
    p.add_argument('--roi_landmarks', default='',
                   help='the landmarks whose bounding box IS the ROI: a preset '
                        'name (nose_mouth | nostrils | nostril_mouth | '
                        'nose_tip) or a comma list of 1-indexed labels, e.g. '
                        '"10,21"; empty = the 12-point nose+mouth default. '
                        'Must match the run whose input you are inspecting '
                        '(measured: "nostrils" gives an unstable thin slab, '
                        'prefer "nose_tip")')
    p.add_argument('--min_signal_spread', type=float, default=0.0,
                   help='DROP a clip whose respiration window spread '
                        '(max-min) is below this many VOLTS (0.0 = off; the '
                        'study value is 0.1). Must match the run being '
                        'inspected.')
    p.add_argument('--rail_touch_v', type=float, default=0.0,
                   help='DROP a clip whose respiration window contains any '
                        'sample that touched the rail, i.e. abs(x) >= this '
                        'many VOLTS (0.0 = off; the study value is 9.90). Must '
                        'match the run being inspected.')
    p.add_argument('--norm', default='none', type=str,
                   choices=['none', 'clip', 'session'],
                   help="respiration scale of resp_signal: 'none' (default) = "
                        "RAW VOLTS, NO per-clip z-score; 'clip' = per-clip "
                        "z-score; 'session' = session z-score. Only use a "
                        'normalised value to inspect a run that was trained '
                        'with --physio_norm clip/session.')
    p.add_argument('--clip_index', default='0',
                   help='comma list of clip indices in the selection')
    p.add_argument('--frame_start', default='',
                   help='comma list of WINDOW start frames; OVERRIDES '
                        '--clip_index. A window address is stable across a '
                        'geometry change (the motion gate renumbers the entry '
                        'list), and a window that is NOT in the entry list is '
                        'reported together with the gate that dropped it.')
    p.add_argument('--t_start', default='',
                   help='comma list of WINDOW start times in SECONDS, converted '
                        'with --fps; an alternative to --frame_start')
    p.add_argument('--max_entries', type=int, default=0, help='0 = no cap')
    p.add_argument('--output_dir', default=default_output_dir())
    p.add_argument('--no-plot', dest='plot', action='store_false',
                   help='write the JSON reports only')
    p.add_argument('--video', default='gif',
                   choices=['none', 'gif', 'mp4', 'both'],
                   help="write the WHOLE clip as an animation next to the "
                        "figure: 'gif' (default, Pillow, plays in VS Code), "
                        "'mp4' (OpenCV mp4v -- MPEG-4 Part 2, NOT H.264, so "
                        "some players refuse it), 'both' or 'none'")
    p.add_argument('--video_fps', type=float, default=25.0,
                   help='output frame rate of the animation: source frames are '
                        'decimated to the nearest integral ratio (25 = every '
                        'frame, 12.5 = every 2nd; default 25)')
    p.add_argument('--video_dpi', type=int, default=70,
                   help='render dpi of the animation (default 70 -> about '
                        '630x420 px; raise for a sharper/larger file)')
    p.add_argument('--list', action='store_true',
                   help='list the thermal sessions on disk and exit')
    args = p.parse_args(argv)

    if args.list:
        return list_sessions(args.raw_root)

    ds = trd.BP4DPlusTIRRespDataset(
        raw_root=args.raw_root, subjects=_split(args.subject),
        tasks=_split(args.task), task_groups=args.task_groups,
        task_set=args.task_set, clip_seconds=args.clip_seconds,
        clip_stride=args.clip_stride, fps=args.fps, resp_fs=args.resp_fs,
        input_size=args.input_size, roi_padding=args.roi_padding,
        crop_rule=args.crop_rule, max_sway_px=args.max_sway_px,
        target_landmarks=trd.resolve_roi_landmarks(args.roi_landmarks),
        min_signal_spread=args.min_signal_spread,
        rail_touch_v=args.rail_touch_v,
        norm=args.norm,
        max_entries=args.max_entries or None)

    print('=' * 76)
    print('BP4DPlusTIRRespDataset -- clip inspection')
    print('=' * 76)
    print(ds.describe())
    print('-' * 76)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'dataset_summary.json'), 'w') as fh:
        json.dump(ds.stats, fh, indent=2)

    # A request is either an ENTRY index or an absolute WINDOW start. The window
    # form survives a geometry change; the index form does not (the motion gate
    # renumbers entries), so the window form is the one to cite in a report.
    requests = []
    if str(args.frame_start).strip():
        requests += [('frame_start', int(v)) for v in
                     str(args.frame_start).split(',') if v.strip()]
    if str(args.t_start).strip():
        requests += [('frame_start', int(round(float(v) * args.fps))) for v in
                     str(args.t_start).split(',') if v.strip()]
    if not requests:
        requests = [('index', int(v)) for v in
                    str(args.clip_index).split(',') if v.strip()]

    rows, n_fail, n_skipped, n_absent = [], 0, 0, 0
    by_frame_start = {int(e['frame_start']): i
                      for i, e in enumerate(ds.entries)}
    for kind, value in requests:
        if kind == 'index':
            if not 0 <= value < len(ds):
                print(f'[skip] clip {value}: out of range (0..{len(ds) - 1}). '
                      f'NOTE the motion gate renumbers the entry list, so an '
                      f'index is only meaningful for the SAME geometry that '
                      f'produced it -- prefer --t_start/--frame_start.')
                n_skipped += 1
                continue
            index = value
        else:
            index = by_frame_start.get(int(value))
            if index is None:
                n_absent += 1
                print(f'[absent] frame_start {value}: no such window in the '
                      f'entry list ({len(ds)} clip(s) in the selection)')
                lines, reasons = _diagnose_window(ds, int(value))
                for line in lines:
                    print(f'       - {line}')
                rows.append({'frame_start': int(value), 'absent': True,
                             'reasons': reasons, 'diagnosis': lines,
                             'checks_failed': [f'window absent: {reasons}']})
                continue
            print(f'[resolve] frame_start {value} -> clip index {index}')
        entry = ds.entries[index]
        # Everything below can raise a CONTRACT error (a clip box escaping the
        # cache hull, a square box that cannot be placed, a decode that gives
        # up). That must land in the report as a FAIL, not as a traceback that
        # loses the JSON for the clips that were fine.
        try:
            lm = ds.clip_landmarks(index)
            box = ds.clip_roi_box(index)
            frames = ds._frames(entry)
            item = ds[index]
            fails = trd._check_clip(ds, index)
            rep = _clip_report(ds, index, box, fails)
        except Exception as exc:
            fails = [f'{type(exc).__name__}: {exc}']
            rep = {'index': int(index), 'session': entry['session'],
                   'checks_failed': list(fails)}
            lm = box = frames = item = None
        if fails:
            n_fail += 1

        out_dir = os.path.join(args.output_dir, entry['session'])
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.join(out_dir, f'{FIG_TAG}_{index:04d}')
        if args.plot and item is not None:
            try:
                _plot_clip(ds, index, stem + '.png', box, lm, frames[0],
                           item['tir_video'], item['resp_signal'])
                rep['figure'] = stem + '.png'
            except Exception as exc:                     # never lose the JSON
                rep['figure_error'] = str(exc)
                print(f'[warn] figure failed: {exc}')
            if args.video != 'none':
                try:
                    rep['videos'] = _write_video(
                        ds, index, stem, box, lm, frames,
                        item['tir_video'], item['resp_signal'],
                        fps=args.video_fps, dpi=args.video_dpi,
                        fmt=args.video)
                except Exception as exc:                 # never lose the JSON
                    rep['video_error'] = str(exc)
                    print(f'[warn] video failed: {exc}')
        with open(stem + '.json', 'w') as fh:
            json.dump(rep, fh, indent=2)

        roi = rep.get('roi') or {}
        tag = 'FAIL' if fails else 'ok'
        print(f'[{tag}] clip {index} {entry["session"]} '
              f'frames {entry["frame_start"]}-{entry["frame_end"] - 1} '
              f'({entry["t_start"]:.2f}-{entry["t_end"]:.2f} s) '
              f'roi {roi.get("width_px", "?")}x{roi.get("height_px", "?")} px '
              f'sway {roi.get("sway_px", "?")} px '
              f'-> {(rep.get("tir_video") or {}).get("shape", "?")} + resp '
              f'{(rep.get("resp_signal") or {}).get("shape", "?")}')
        for f in fails:
            print(f'       - {f}')
        rows.append({'session': entry['session'], 'clip_index': index,
                     'figure': rep.get('figure'), 'videos': rep.get('videos'),
                     'json': stem + '.json',
                     'roi_box_px': roi.get('box_px'),
                     'sway_px': roi.get('sway_px'),
                     'flags': rep.get('flags'),
                     'tir_shape': (rep.get('tir_video') or {}).get('shape'),
                     'resp_shape': (rep.get('resp_signal') or {}).get('shape'),
                     'checks_failed': fails})

    if rows:
        idx_path = _append_index(args.output_dir, rows)
        print('-' * 76)
        print(f'wrote {len(rows)} clip report(s) -> {args.output_dir}')
        print(f'index: {idx_path}')
    print('-' * 76)
    if not rows:
        # Nothing was inspected: reporting PASS here would be a lie (a stale or
        # gate-renumbered --clip_index used to do exactly that).
        print(f'FAIL: 0/{len(requests)} window(s) inspected '
              f'({n_skipped} index/indices out of range), '
              f'{len(ds)} clip(s) in the selection')
        return 1
    print(f'{"FAIL" if (n_fail or n_absent) else "PASS"}: '
          f'{len(rows) - n_fail - n_absent}/{len(rows)} window(s) checked, '
          f'{len(ds)} clip(s) in the selection'
          + (f', {n_absent} absent from the entry list' if n_absent else '')
          + (f', {n_skipped} index/indices skipped' if n_skipped else ''))
    return 1 if (n_fail or n_absent) else 0


if __name__ == '__main__':
    raise SystemExit(main())
