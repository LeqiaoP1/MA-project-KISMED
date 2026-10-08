"""Inspect ``RGBRoiDataset`` -- the visible-light face-ROI -> blood-pressure clip dataset.

Draws, for one clip of one session, WHAT THE MODEL WILL ACTUALLY RECEIVE and
checks that it is what the ROI strategy promises:

* **RGB + clip-static ROI** -- the clip's first visible-light frame with the
  single crop box that is applied to ALL ``clip_frames`` frames, the target
  landmarks (``--landmarks``; ``'face'`` = all 49 by default) colour-coded by
  anatomy group, and the same points of every frame scattered faintly behind
  them -- the points whose min/max produced the box, so the "one box per clip,
  no per-frame jitter" claim is visible. The rectangle is drawn in the DECODED
  frame's pixels with the same floor/ceil/clamp arithmetic the dataset uses
  (:func:`_crop_bounds`, a verbatim copy of ``RGBRoiDataset._roi_patches``), so
  the box on screen IS the slice that gets resized -- it is not a re-derivation.
* **ROI patch** -- exactly the ``[3, T, H, W]`` tensor slice the dataset returns
  (first frame), NOT a re-crop: if the tensor and the box disagreed, this panel
  would show it.
* **skin mask (optional, ``--skin_mask``)** -- the two-stage mask that produced
  the validated rPPG gate: the motion-bearing landmark groups (brows / eyes /
  mouth) are excluded PER FRAME and the surviving pixels must pass the YCrCb
  skin gate. Excluded pixels are tinted red on both the frame and the patch, and
  the per-frame retained fraction + the retained/dropped mean RGB go into the
  JSON. The mask is REPORTED ONLY here -- it is not applied to the tensor (wiring
  it into the provider is a separate, deliberate step). Reference implementation:
  ``code/analysis/rgb_bp/rppg_skin_chrom.py``; measured there: 34.4 % of the box
  retained on average (13.4-51.2 %), retained pixels brighter (R ~145 vs ~119),
  and CHROM detection rising 83 % -> 90 % against the calibrated control.
* **1-D target (raw)** -- the ``BP_mmHg.txt`` window over the clip's own time
  span, in mmHg (``signal_clip(..., normalized=False)``).
* **1-D overlay** -- the transform selected by ``--norm``. NB the RGB provider's
  item is ALWAYS RAW (``RGBRoiDataset._load`` calls ``signal_clip(...,
  normalized=False)``; the model applies ``target_norm: clip`` internally), so
  this is NOT "what the dataset returns" -- ``--norm clip`` (the default) draws
  the per-clip z-score on a twin ``[z]`` axis, i.e. exactly the target the model
  computes, while ``--norm none`` draws no overlay at all, because then the grey
  raw curve IS the item.

The LAYOUT follows the same priority as the thermal inspector: the annotated
frame owns the full-width top row (it is the panel that has to be read closely
-- box edges and the landmark track), while the ROI patch is a small thumbnail
next to the wide 1-D strip in the bottom row.

The same clip is run through the dataset's own verification
(``data.rgb_roi_dataset._check_clip``: keys / dtypes / [0, 1] range, the
unusable gate, the ROI box inside the frame AND containing every target
landmark, "the tensor IS the resize of that crop", each 1-D window against the
brute-force raw slice, and ``T/fps == L/phys_fs``) and both the checks and every
shape / range / ROI number land in the JSON report, so a run is auditable
without the figure. Figures are deliberately NUMBER-FREE (repo convention): the
numbers live in the JSON and on stdout.

Padding default
---------------
``--roi_padding`` defaults to **0.1** (+20 % overall) here, NOT to the RGB
provider's ``rgb_roi_dataset.DEFAULT_ROI_PADDING = 0.2`` (+40 % overall). The
inspector's job is to show the box, and the tighter default keeps the crop tied
visibly to the landmarks; both numbers are recorded in the JSON (``padding``
and ``provider_default_padding``), and ``--roi_padding 0.2`` (or whatever a
config pins) reproduces a training run exactly.

Artifacts::

    <output_dir>/<subject>_<task>/clip_0000.png     # 4-panel figure (or .json only)
    <output_dir>/<subject>_<task>/clip_0000.json    # clip + dataset report
    <output_dir>/dataset_summary.json               # whole-dataset summary
    <output_dir>/rgb_bp_index.json                  # one row per clip inspected

Usage (from ``code/``)::

    python runners/run_inspect_rgb_bp.py --list
    python runners/run_inspect_rgb_bp.py --subject F001 --task T1 --clip_index 0
    python runners/run_inspect_rgb_bp.py --subject F001 --task T1 --clip_index 0,7
    python runners/run_inspect_rgb_bp.py --landmarks nose_mouth --input_size 64 --no-plot
    python runners/run_inspect_rgb_bp.py --subject F001 --task T1 --clip_index 0 --skin_mask

Paths default to ``$RAW_DATA_PATH`` (raw BP4D root) and
``$OUTPUT_DIR/inspect_data/rgb_bp`` (see ``scripts/env_local.sh``); without the
env profile they fall back to the in-repo ``data/raw/BP4D`` and ``<repo>/output``.

NOTE: the RGB frames are ~1392x1040 JPEGs read from the RAW tree, so a native
(``--decode_scale 1``) clip costs a full clip decode per load -- the dataset's
check decodes it once more. ``--decode_scale 4`` (with the landmarks rescaled by
the same factor) is the fast path for a quick look; the report records both the
source and the decoded frame size, so the two are never confused.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data import rgb_features as rfeat
from data import rgb_roi_dataset as rrd

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_DIR = os.path.dirname(_CODE_DIR)

FIG_TAG = 'clip'

#: The inspector's default ROI margin, deliberately TIGHTER than the provider's
#: ``rrd.DEFAULT_ROI_PADDING`` (0.2): see the module docstring. The CLI flag and
#: the JSON report both carry it, so a run against a training config is
#: auditable via ``--roi_padding <config value>``.
DEFAULT_ROI_PADDING = 0.1

#: the 1-D stream this inspector defaults to (the filename says "bp")
DEFAULT_SIGNAL = 'bp'

#: display units of the raw physiology files (see ``rrd.SIGNAL_FILES``)
SIGNAL_UNITS = {'bp': 'mmHg', 'resp': 'V', 'eda': 'uS'}

#: colours for the anatomy groups of the ROI landmark set, in the order
#: ``rgb_features.GROUP_RANGES`` lists them
_GROUP_COLOURS = ('#ffe600', '#ff3df2', '#00e5ff', '#7cff4d',
                  '#ff7a1a', '#b9a2ff', '#ff4d4d', '#ffffff')

#: landmark groups EXCLUDED by the skin mask's first stage, by PREFIX against
#: ``rgb_features.GROUP_RANGES`` -> the fraction of the group's extent added
#: around its per-frame bbox before exclusion. The mouth is tightened (0.15) so
#: the lips are dropped without eating the cheeks; brows/eyes get 0.25. Exactly
#: the analysis script's ``EXCL`` + pads (which produced the validated gate).
SKIN_EXCLUDE_PAD = {'brow': 0.25, 'eye': 0.25, 'mouth': 0.15}

#: report/legend names of the excluded groups (the keys above are PREFIXES, so
#: ``mouth`` must not be pluralised mechanically)
SKIN_GROUP_NAME = {'brow': 'brows', 'eye': 'eyes', 'mouth': 'mouth'}

#: YCrCb skin gate + luminance floor -- verbatim the analysis script's constants
SKIN_CR = (133.0, 173.0)
SKIN_CB = (77.0, 127.0)
SKIN_Y_MIN = 40.0


# --------------------------------------------------------------------------- #
# path defaults
# --------------------------------------------------------------------------- #
def _env(key: str, default: str = '') -> str:
    return os.environ.get(key, default)


def default_output_dir() -> str:
    """``$OUTPUT_DIR/inspect_data/rgb_bp``, else ``<repo>/output/...``.

    Absolute on purpose: a bare ``./output`` would land inside ``code/`` when
    the env profile was not sourced.
    """
    root = _env('OUTPUT_DIR', '') or os.path.join(_REPO_DIR, 'output')
    return os.path.join(root, 'inspect_data', 'rgb_bp')


def _split(value):
    """``'a,b' -> ['a', 'b']``; ``'all'``/empty -> ``None`` (no filter)."""
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() == 'all':
        return None
    return [v.strip() for v in value.split(',') if v.strip()]


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def _crop_bounds(box, src_hw, dec_hw):
    """ROI slice ``(a0, a1, b0, b1)`` in DECODED pixels.

    A verbatim copy of the arithmetic in ``RGBRoiDataset._roi_patches`` (scale
    by ``dec/src``, floor the near edge, ceil the far one, clamp into the
    decoded frame), so the rectangle the figure draws is the crop the dataset
    takes -- not an approximation of it. ``box`` is the SOURCE-pixel
    ``(x0, x1, y0, y1)`` from :meth:`RGBRoiDataset.clip_roi_box`.
    """
    x0, x1, y0, y1 = box
    src_h, src_w = src_hw
    h, w = dec_hw
    sx, sy = w / src_w, h / src_h
    a0 = max(0, min(w - 1, int(math.floor(x0 * sx))))
    a1 = max(a0 + 1, min(w, int(math.ceil(x1 * sx))))
    b0 = max(0, min(h - 1, int(math.floor(y0 * sy))))
    b1 = max(b0 + 1, min(h, int(math.ceil(y1 * sy))))
    return a0, a1, b0, b1


def _first_frame(ds, entry) -> np.ndarray:
    """The clip's first frame as the dataset decodes it -> uint8 RGB ``[h, w, 3]``.

    ``decode_scale`` is applied, so ``h/w`` are the DECODED size the ROI slice
    is taken in (``_read_rgb`` returns RGB, i.e. the BGR->RGB fix is included).
    """
    session = entry['session']
    path = os.path.join(entry['rgb_dir'],
                        ds._files_of(session)[entry['frame_start']])
    return ds._read_rgb(path)


def _group_members(labels):
    """1-indexed labels -> ``[(anatomy group name, [labels]), ...]``.

    Splits the ROI set with ``rgb_features.GROUP_RANGES`` (brow / nose bridge /
    nose base / eyes / mouth outer+inner) so ANY ``--landmarks`` choice gets a
    readable legend. Labels outside every range land in a trailing ``'other'``
    group, so nothing is silently dropped from the figure.
    """
    out, used = [], set()
    for name, lo, hi in rfeat.GROUP_RANGES:
        sel = [int(l) for l in labels if lo <= l <= hi]
        if sel:
            out.append((name, sel))
            used.update(sel)
    rest = [int(l) for l in labels if int(l) not in used]
    if rest:
        out.append(('other', rest))
    return out


# --------------------------------------------------------------------------- #
# skin mask (mirrors analysis/rgb_bp/rppg_skin_chrom.py)
# --------------------------------------------------------------------------- #
def _skin_exclude_groups():
    """``[(name, labels, pad), ...]`` -- the mask's landmark-exclusion stage.

    The label sets are DERIVED from ``rgb_features.GROUP_RANGES`` by prefix, so
    they cannot drift from the 49-point layout: brows = 1-10, eyes = 20-31,
    mouth = 32-49 (1-indexed) -- exactly the exclusions of the analysis script
    that measured the +7-point CHROM gain.
    """
    out = []
    for prefix, pad in SKIN_EXCLUDE_PAD.items():
        sel = []
        for name, lo, hi in rfeat.GROUP_RANGES:
            if name.startswith(prefix):
                sel.extend(range(lo, hi + 1))
        if sel:
            out.append((SKIN_GROUP_NAME.get(prefix, prefix), sel, float(pad)))
    return out


def _skin_mask(crop_rgb: np.ndarray, pts_frame: np.ndarray, box, src_hw,
               dec_hw):
    """Two-stage skin mask of ONE decoded crop -> bool ``[ch, cw]`` (True = skin).

    Stage 1 blacks out the bbox of this frame's brow / eye / mouth landmarks
    (padded by :data:`SKIN_EXCLUDE_PAD`); stage 2 keeps only pixels passing the
    YCrCb gate. A frame whose group landmarks are non-finite skips its exclusion
    (defensive -- the provider already drops such clips).

    :param crop_rgb: the decoded ROI crop, **RGB** (``_read_rgb`` reverses BGR),
        and exactly ``_crop_bounds(box, src_hw, dec_hw)`` -- guarded below, since
        a mismatch silently misplaces every exclusion box.
    :param pts_frame: ``[49, 2]`` landmarks of THIS frame, SOURCE pixels.
    :param box: the clip's static ROI box (source px), for the source->crop map.
    :param src_hw: ``(h, w)`` of the SOURCE frame, same map.
    :param dec_hw: ``(h, w)`` of the DECODED frame -- the scale denominator.
        NOTE the mask must be mapped with the *frame* size; using the crop size
        here shrinks every exclusion rectangle towards the crop origin.
    """
    cv2 = rrd._cv2()
    src_h, src_w = src_hw
    dec_h, dec_w = dec_hw
    ch, cw = crop_rgb.shape[:2]
    a0, a1, b0, b1 = _crop_bounds(box, src_hw, dec_hw)
    if (ch, cw) != (b1 - b0, a1 - a0):
        raise ValueError(
            f'skin mask: crop {ch}x{cw} != the ROI slice {b1 - b0}x{a1 - a0} of '
            f'box {box} in the {dec_h}x{dec_w} decoded frame -- the exclusion '
            f'boxes would be misaligned')
    sx, sy = dec_w / src_w, dec_h / src_h      # the SAME map _roi_patches uses
    bx0, _bx1, by0, _by1 = box

    mask = np.ones((ch, cw), bool)
    for _name, labels, pad in _skin_exclude_groups():
        p = np.asarray(pts_frame, np.float64)[np.asarray(labels, np.int64) - 1]
        if not np.isfinite(p).all():
            continue
        x0, x1 = float(p[:, 0].min()), float(p[:, 0].max())
        y0, y1 = float(p[:, 1].min()), float(p[:, 1].max())
        px, py = pad * (x1 - x0), pad * (y1 - y0)
        i0 = max(0, min(cw, int((x0 - px - bx0) * sx)))
        i1 = max(0, min(cw, int((x1 + px - bx0) * sx)))
        j0 = max(0, min(ch, int((y0 - py - by0) * sy)))
        j1 = max(0, min(ch, int((y1 + py - by0) * sy)))
        if i1 > i0 and j1 > j0:
            mask[j0:j1, i0:i1] = False

    # RGB -> YCrCb (the crop is RGB because _read_rgb already fixed the channel
    # order; the analysis script used BGR2YCrCb on a raw imread crop).
    ycc = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2YCrCb)
    cr, cb, y = ycc[..., 1], ycc[..., 2], ycc[..., 0]
    skin = ((cr >= SKIN_CR[0]) & (cr <= SKIN_CR[1])
            & (cb >= SKIN_CB[0]) & (cb <= SKIN_CB[1]) & (y >= SKIN_Y_MIN))
    return mask & skin


def _skin_pass(ds, index: int) -> dict:
    """Decode the clip once and build the two-stage skin mask for EVERY frame.

    Returns ``{'first', 'mask0', 'stats'}``: the decoded first frame, its
    crop-space mask (for the figure) and the per-frame statistics -- mean / min /
    max retained fraction INSIDE the box (comparable with the 0.344 the analysis
    script measured) plus the mean RGB of the retained vs the dropped pixels.
    Costs one extra clip decode (``--decode_scale 4`` makes it cheap).
    """
    entry = ds.entries[index]
    session = entry['session']
    lm = ds.clip_landmarks(index)
    box = ds.clip_roi_box(index)
    src_hw = ds._source_size(session)
    files = ds._files_of(session)

    fracs, kept, dropped = [], [], []
    first = mask0 = None
    for k, i in enumerate(range(entry['frame_start'], entry['frame_end'])):
        img = ds._read_rgb(os.path.join(entry['rgb_dir'], files[i]))
        dec_hw = img.shape[:2]
        a0, a1, b0, b1 = _crop_bounds(box, src_hw, dec_hw)
        crop = img[b0:b1, a0:a1]
        m = _skin_mask(crop, lm[k], box, src_hw, dec_hw)
        if k == 0:
            first, mask0 = img, m
        fracs.append(float(m.mean()))
        pix, flat = crop.reshape(-1, 3).astype(np.float64), m.reshape(-1)
        if flat.any():
            kept.append(pix[flat].mean(0))
        if (~flat).any():
            dropped.append(pix[~flat].mean(0))

    k_rgb = np.mean(kept, axis=0) if kept else np.full(3, np.nan)
    d_rgb = np.mean(dropped, axis=0) if dropped else np.full(3, np.nan)
    stats = {
        'enabled': True,
        'method': 'per-frame landmark exclusion (brows/eyes/mouth) + YCrCb gate',
        'reference': 'code/analysis/rgb_bp/rppg_skin_chrom.py',
        'exclude_groups': {n: list(l) for n, l, _p in _skin_exclude_groups()},
        'exclude_pad': {n: float(p) for n, _l, p in _skin_exclude_groups()},
        'gate': {'colorspace': 'RGB2YCrCb', 'cr': list(SKIN_CR),
                 'cb': list(SKIN_CB), 'y_min': SKIN_Y_MIN},
        'frames': len(fracs),
        'retained_fraction_mean': float(np.mean(fracs)),
        'retained_fraction_min': float(np.min(fracs)),
        'retained_fraction_max': float(np.max(fracs)),
        'frame0_retained_fraction': float(fracs[0]),
        'retained_mean_rgb': [float(v) for v in k_rgb],
        'dropped_mean_rgb': [float(v) for v in d_rgb],
        # sanity check kept from the analysis script: skin must read R > G > B
        'retained_rgb_order_r_gt_g_gt_b': bool(
            np.isfinite(k_rgb).all() and k_rgb[0] > k_rgb[1] > k_rgb[2]),
        'applied_to_tensor': False,
    }
    return {'first': first, 'mask0': mask0, 'stats': stats}


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def _plot_clip(ds, index, path, box, lm, frame0, tensor, raw, over, sig, unit,
               skin=None):
    """4-panel figure for one clip; see the module docstring for the content."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    entry = ds.entries[index]
    session = entry['session']
    src_h, src_w = ds._source_size(session)
    h, w = frame0.shape[:2]
    a0, a1, b0, b1 = _crop_bounds(box, (src_h, src_w), (h, w))
    sx, sy = w / src_w, h / src_h

    # The annotated frame IS the point of this figure, so it gets the whole top
    # row and most of the height; the ROI patch is a thumbnail and the 1-D
    # target is wide, so they share the smaller bottom row.
    fig = plt.figure(figsize=(12.5, 10.0), layout='constrained')
    ax = fig.subplot_mosaic([['roi', 'roi'], ['patch', 'sig']],
                            height_ratios=[1.75, 1.0], width_ratios=[1.0, 2.4])

    # --- panel 1: the frame, the static box, the landmarks that made it
    ax['roi'].imshow(frame0, interpolation='nearest')
    ax['roi'].add_patch(Rectangle((a0, b0), a1 - a0, b1 - b0, fill=False,
                                  ec='#ff2020', lw=2.0))
    # --skin_mask: tint the pixels the two-stage mask REJECTS -- the per-frame
    # brow/eye/mouth boxes AND everything that fails the YCrCb gate -- so the
    # loss is visible instead of implied.
    if skin is not None and skin.get('mask0') is not None:
        ov = np.zeros((h, w, 4), np.float32)
        ov[b0:b1, a0:a1][~skin['mask0']] = (1.0, 0.15, 0.15, 0.35)
        ax['roi'].imshow(ov, interpolation='nearest')
    pts = np.asarray(lm, dtype=np.float64)[:, ds.target_idx, :]
    pts[..., 0] *= sx
    pts[..., 1] *= sy
    ax['roi'].scatter(pts[..., 0].ravel(), pts[..., 1].ravel(), s=3.0,
                      c='#00e5ff', alpha=0.35, lw=0)
    handles, legend_labels = [], []
    for (name, labels), colour in zip(_group_members(ds.landmarks),
                                      _GROUP_COLOURS):
        cols = [ds.landmarks.index(l) for l in labels]
        handles.append(ax['roi'].scatter(pts[0, cols, 0], pts[0, cols, 1], s=34,
                                         c=colour, edgecolors='k', lw=0.6))
        legend_labels.append(name)
    if skin is not None and skin.get('mask0') is not None:
        handles.append(Patch(facecolor=(1.0, 0.15, 0.15), alpha=0.35,
                             edgecolor='none'))
        legend_labels.append('excluded (non-skin)')
    ax['roi'].legend(handles, legend_labels, loc='upper right', fontsize=8,
                     ncol=2, framealpha=0.7)
    ax['roi'].set_title(f'{session} clip {index}: '
                        f'frames {entry["frame_start"]}-{entry["frame_end"] - 1} '
                        f'({w}x{h}, decode_scale {ds.decode_scale})',
                        fontsize=12)

    # --- panel 2: the tensor slice the dataset returns (not a re-crop)
    patch = tensor[:, 0].permute(1, 2, 0).numpy()
    ax['patch'].imshow(np.clip(patch, 0.0, 1.0), interpolation='nearest',
                       aspect='equal')
    if skin is not None and skin.get('mask0') is not None:
        # the same mask at TENSOR resolution (nearest, so no fractional edges):
        # this is exactly the fraction of the tensor a provider-side mask drops
        cv2 = rrd._cv2()
        small = cv2.resize(skin['mask0'].astype(np.uint8),
                           (ds.input_size, ds.input_size),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        ov = np.zeros((ds.input_size, ds.input_size, 4), np.float32)
        ov[~small] = (1.0, 0.15, 0.15, 0.35)
        ax['patch'].imshow(ov, interpolation='nearest', aspect='equal')
    ax['patch'].set_title(f'ROI patch {ds.input_size}x{ds.input_size} (tensor)')

    # --- panel 3: the RAW window the dataset returns + the --norm overlay
    t = entry['t_start'] + np.arange(ds.phys_len) / ds.phys_fs
    # The raw curve is drawn THICKER than the overlay: a z-score is an affine map
    # of the same waveform and each curve fills its own axis, so the two trace
    # the SAME path in pixels -- a thin red curve would hide the grey one
    # completely and the panel would look like a single trace.
    ax['sig'].plot(t, raw, color='0.55', lw=2.0, alpha=0.9,
                   label=os.path.basename(entry['signals'][sig]))
    ax['sig'].set_ylabel(f'{sig} [{unit}]')
    ax['sig'].set_xlabel('time [s]')
    ax['sig'].set_xlim(float(t[0]), float(t[-1]))
    ax['sig'].set_title(f'clip {index} {sig} window '
                        f'({ds.phys_len} samples @ {ds.phys_fs:g} Hz)'
                        + ('' if over is not None else ' -- RAW, norm=none'))
    if over is None:
        ax['sig'].legend(loc='lower right', fontsize=7, framealpha=0.7)
    else:
        tw = ax['sig'].twinx()
        tw.plot(t, over, color='#d62728', lw=0.9,
                label=f'{sig}_signal (norm={ds.norm})')
        tw.set_ylabel(f'{sig}_signal [z]')
        h1, l1 = ax['sig'].get_legend_handles_labels()
        h2, l2 = tw.get_legend_handles_labels()
        ax['sig'].legend(h1 + h2, l1 + l2, loc='lower right', fontsize=7,
                         framealpha=0.7)

    for key in ('roi', 'patch'):
        ax[key].set_xticks([])
        ax[key].set_yticks([])
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _overlay_curve(ds, index, sig):
    """The extra curve the figure overlays, or ``None``.

    ``RGBRoiDataset._load`` always returns RAW values (``normalized=False``), so
    there is nothing to overlay under ``norm='none'`` -- the grey raw trace IS
    the item. Otherwise this is the ``norm`` z-score (``clip`` = the transform
    the model applies internally under ``target_norm: clip``; ``session`` = the
    whole-session statistics).
    """
    if ds.norm == 'none':
        return None
    return ds.signal_clip(index, sig, normalized=True)


def _clip_report(ds, index, box, fails, skin=None) -> dict:
    entry = ds.entries[index]
    session = entry['session']
    sig = ds.signal_names[0]
    item = ds[index]
    rgb, item_sig = item['rgb_video'], item[f'{sig}_signal']
    raw = ds.signal_clip(index, sig, normalized=False)
    over = _overlay_curve(ds, index, sig)
    lm = ds.clip_landmarks(index)
    src_h, src_w = ds._source_size(session)
    dec_h, dec_w = ((skin['first'].shape[:2] if skin is not None
                     else _first_frame(ds, entry).shape[:2]))
    x0, x1, y0, y1 = box
    crop = _crop_bounds(box, (src_h, src_w), (dec_h, dec_w))
    return {
        'index': int(index),
        'session': session,
        'subject': entry['subject'],
        'task': entry['task'],
        'frame_start': int(entry['frame_start']),
        'frame_end': int(entry['frame_end']),
        't_start_s': float(entry['t_start']),
        't_end_s': float(entry['t_end']),
        'rgb_dir': entry['rgb_dir'],
        'feat_file': entry['feat_file'],
        'signal_file': entry['signals'][sig],
        'roi': {
            'box_px': [int(v) for v in box],
            'width_px': int(x1 - x0),
            'height_px': int(y1 - y0),
            'padding': float(ds.roi_padding),
            'provider_default_padding': float(rrd.DEFAULT_ROI_PADDING),
            'landmarks_1based': list(ds.landmarks),
            'landmarks_0based': [int(i) for i in ds.target_idx],
            'landmark_groups': {n: list(l) for n, l in _group_members(ds.landmarks)},
            'static_over_frames': int(ds.clip_frames),
            'landmark_span_px': [float(np.nanmin(lm[..., 0])),
                                 float(np.nanmax(lm[..., 0])),
                                 float(np.nanmin(lm[..., 1])),
                                 float(np.nanmax(lm[..., 1]))],
            'source_frame_hw': [int(src_h), int(src_w)],
            'decoded_frame_hw': [int(dec_h), int(dec_w)],
            'decode_scale': int(ds.decode_scale),
            # same half-open (x0, x1, y0, y1) convention as box_px, but in
            # DECODED pixels -- the slice that is resized to input_size
            'crop_px_decoded': [int(v) for v in crop],
            'frame_fraction': float(((x1 - x0) * (y1 - y0)) / (src_w * src_h)),
        },
        'rgb_video': {
            'shape': list(rgb.shape),
            'dtype': str(rgb.dtype),
            'min': float(rgb.min()),
            'max': float(rgb.max()),
            'mean': float(rgb.mean()),
        },
        'signal': {
            'name': sig,
            'unit': SIGNAL_UNITS.get(sig, ''),
            'norm': ds.norm,
            # the provider's item: ALWAYS raw, whatever ds.norm says
            'item_shape': list(item_sig.shape),
            'item_dtype': str(item_sig.dtype),
            'item_min': float(item_sig.min()),
            'item_max': float(item_sig.max()),
            'item_mean': float(item_sig.mean()),
            'item_std': float(item_sig.std(unbiased=False)),
            'raw_min': float(np.nanmin(raw)),
            'raw_max': float(np.nanmax(raw)),
            # the --norm transform the figure overlays (None under norm='none')
            'overlay_min': None if over is None else float(over.min()),
            'overlay_max': None if over is None else float(over.max()),
            'overlay_mean': None if over is None else float(over.mean()),
            'overlay_std': None if over is None else float(np.std(over)),
        },
        'subject_task': item['subject_task'],
        # the skin mask is REPORTED here; nothing in this path applies it to the
        # tensor (``applied_to_tensor: false``), which is the provider's job.
        'skin': (skin['stats'] if skin is not None
                 else {'enabled': False, 'applied_to_tensor': False}),
        'checks_failed': list(fails),
    }


def _append_index(output_dir: str, rows) -> str:
    path = os.path.join(output_dir, 'rgb_bp_index.json')
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
    """Print the discovered sessions and each one's usability verdict."""
    try:
        specs = rrd.discover_rgb_sessions(raw_root)
    except FileNotFoundError as exc:
        print(f'[error] {exc}')
        return 1
    print(f'raw root: {raw_root}')
    print(f'{"session":<10} {"jpg":>5} {"mat":>5} {"status":<9} {"signals":<9} note')
    n_bad = 0
    for spec in specs:
        ok = spec['status'] == 'usable'
        n_bad += 0 if ok else 1
        sig = ','.join(sorted(spec['signals'])) or '-'
        note = '' if ok else spec['reason']
        print(f'{spec["session"]:<10} {spec["n_jpg"]:>5} {spec["n_mat"]:>5} '
              f'{"ok" if ok else "UNUSABLE":<9} {sig:<9} {note}')
    print(f'{len(specs)} RGB sequence(s) on disk, {len(specs) - n_bad} usable, '
          f'{n_bad} unusable (excluded whole -- their 1-D signals are never read)')
    return 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description='Inspect RGBRoiDataset clips (RGB face ROI + 1-D physiology).')
    p.add_argument('--raw_root', default=rrd.default_raw_root(),
                   help='raw BP4D root (default $RAW_DATA_PATH)')
    p.add_argument('--subject', default='F001', help='comma list, or all')
    p.add_argument('--task', default='T1', help='comma list, or all')
    p.add_argument('--signal', default=DEFAULT_SIGNAL,
                   choices=tuple(rrd.SIGNAL_FILES),
                   help='1-D stream to inspect (default bp)')
    p.add_argument('--clip_seconds', type=float, default=rrd.DEFAULT_CLIP_SECONDS)
    p.add_argument('--clip_stride', type=float, default=None,
                   help='clip hop in seconds (default: non-overlapping)')
    p.add_argument('--fps', type=float, default=rrd.DEFAULT_FPS)
    p.add_argument('--phys_fs', type=float, default=rrd.DEFAULT_PHYS_FS)
    p.add_argument('--input_size', type=int, default=rrd.DEFAULT_INPUT_SIZE)
    p.add_argument('--roi_padding', type=float, default=DEFAULT_ROI_PADDING,
                   help=f'per-side ROI margin added to the landmark box '
                        f'(default {DEFAULT_ROI_PADDING} = +20%% overall, NOT the '
                        f'provider default {rrd.DEFAULT_ROI_PADDING} = +40%%; pass '
                        f'{rrd.DEFAULT_ROI_PADDING} to mirror a training config)')
    p.add_argument('--landmarks', default='face',
                   help='"face" (all 49), a ROI_LANDMARKS_2D preset, or a '
                        '1-indexed CSV')
    p.add_argument('--skin_mask', action='store_true',
                   help='also build the two-stage SKIN mask (per-frame '
                        'brows/eyes/mouth landmark exclusion + YCrCb gate), tint '
                        'the rejected pixels on the frame AND the patch, and '
                        'report the per-frame retained fraction and the '
                        'retained vs dropped mean RGB. Costs one extra clip '
                        'decode. The mask is REPORTED, not applied to the tensor')
    p.add_argument('--decode_scale', type=int, default=1,
                   choices=rrd.DECODE_FACTORS,
                   help='1 = native full decode; 2/4/8 = libjpeg DCT downscale')
    p.add_argument('--norm', default='clip', choices=('none', 'clip', 'session'),
                   help="overlay drawn on the 1-D panel and recorded in the "
                        "report: 'clip' = the per-clip z-score the model "
                        "computes internally, 'session' = whole-session "
                        "z-score, 'none' = no overlay (the provider's item is "
                        'always RAW)')
    p.add_argument('--clip_index', default='0',
                   help='comma list of clip indices in the selection')
    p.add_argument('--max_entries', type=int, default=0, help='0 = no cap')
    p.add_argument('--output_dir', default=default_output_dir())
    p.add_argument('--no-plot', dest='plot', action='store_false',
                   help='write the JSON reports only')
    p.add_argument('--list', action='store_true',
                   help='list the RGB sequences on disk and exit')
    args = p.parse_args(argv)

    if args.list:
        return list_sessions(args.raw_root)

    unit = SIGNAL_UNITS.get(args.signal, '')
    try:
        ds = rrd.RGBRoiDataset(
            raw_root=args.raw_root, subjects=_split(args.subject),
            tasks=_split(args.task), signals=(args.signal,),
            clip_seconds=args.clip_seconds, clip_stride=args.clip_stride,
            fps=args.fps, phys_fs=args.phys_fs, input_size=args.input_size,
            roi_padding=args.roi_padding,
            landmarks=rrd.resolve_face_landmarks(args.landmarks),
            decode_scale=args.decode_scale, norm=args.norm,
            max_entries=args.max_entries or None)
    except (FileNotFoundError, RuntimeError) as exc:
        # A selection that yields nothing (bad raw_root, or a pair excluded by
        # the jpg==mat gate such as F003_T8) is a RESULT here, not a crash:
        # report it and exit non-zero so a script can branch on it.
        print(f'[error] {exc}')
        return 1

    print('=' * 76)
    print(f'RGBRoiDataset -- clip inspection ({args.signal})')
    print('=' * 76)
    print(ds.describe())
    if args.skin_mask:
        print('skin mask       : ON -- per-frame landmark exclusion '
              '(brows/eyes/mouth) + YCrCb gate; REPORTED, not applied to the '
              'tensor')
    if abs(args.roi_padding - rrd.DEFAULT_ROI_PADDING) > 1e-9:
        print(f'note            : roi_padding {args.roi_padding:g} (inspector '
              f'default {DEFAULT_ROI_PADDING:g}) != provider default '
              f'{rrd.DEFAULT_ROI_PADDING:g} -- the figure and the JSON show '
              f'THIS box; pass --roi_padding {rrd.DEFAULT_ROI_PADDING:g} to '
              f'mirror the training configs')
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
        session = entry['session']
        lm = ds.clip_landmarks(index)
        box = ds.clip_roi_box(index)
        item = ds[index]
        raw = ds.signal_clip(index, args.signal, normalized=False)
        over = _overlay_curve(ds, index, args.signal)
        skin = None
        if args.skin_mask:
            try:
                skin = _skin_pass(ds, index)
            except Exception as exc:                   # never lose the JSON
                print(f'[warn] skin mask failed: {exc}')
        fails = rrd._check_clip(ds, index)
        rep = _clip_report(ds, index, box, fails, skin)
        if fails:
            n_fail += 1

        out_dir = os.path.join(args.output_dir, session)
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.join(out_dir, f'{FIG_TAG}_{index:04d}')
        if args.plot:
            try:
                frame0 = (skin['first'] if skin is not None
                          else _first_frame(ds, entry))
                _plot_clip(ds, index, stem + '.png', box, lm, frame0,
                           item['rgb_video'], raw, over, args.signal, unit, skin)
                rep['figure'] = stem + '.png'
            except Exception as exc:                     # never lose the JSON
                rep['figure_error'] = str(exc)
                print(f'[warn] figure failed: {exc}')
        with open(stem + '.json', 'w') as fh:
            json.dump(rep, fh, indent=2)

        roi = rep['roi']
        tag = 'FAIL' if fails else 'ok'
        print(f'[{tag}] clip {index} {session} '
              f'frames {entry["frame_start"]}-{entry["frame_end"] - 1} '
              f'({entry["t_start"]:.2f}-{entry["t_end"]:.2f} s) '
              f'roi {roi["width_px"]}x{roi["height_px"]} px '
              f'({100.0 * roi["frame_fraction"]:.1f}% of the frame) -> '
              f'{tuple(rep["rgb_video"]["shape"])} + {args.signal} '
              f'{tuple(rep["signal"]["item_shape"])}')
        for f in fails:
            print(f'       - {f}')
        if skin is not None:
            st = skin['stats']
            print(f'       skin: {100.0 * st["retained_fraction_mean"]:.1f}% of '
                  f'the box retained '
                  f'({100.0 * st["retained_fraction_min"]:.1f}-'
                  f'{100.0 * st["retained_fraction_max"]:.1f}%), retained RGB '
                  f'{[round(v, 1) for v in st["retained_mean_rgb"]]} vs dropped '
                  f'{[round(v, 1) for v in st["dropped_mean_rgb"]]}')
        rows.append({'session': session, 'clip_index': index,
                     'figure': rep.get('figure'), 'json': stem + '.json',
                     'roi_box_px': roi['box_px'],
                     'rgb_shape': rep['rgb_video']['shape'],
                     f'{args.signal}_shape': rep['signal']['item_shape'],
                     'skin_retained_fraction': (
                         None if skin is None
                         else skin['stats']['retained_fraction_mean']),
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
