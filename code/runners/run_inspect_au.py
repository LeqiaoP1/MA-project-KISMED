"""Inspect the RAW BP4D+ FACS AU OCCURRENCE coding (``AUCoding/AU_OCC``).

Reads the per-session AU occurrence csvs straight out of the raw tree::

    <au_root>/<subject>_<task>.csv      # default data/raw/BP4D/AUCoding/AU_OCC

and reports, over the whole corpus (= 140 subjects x {T1,T6,T7,T8} = 560 files),
how often every AU actually occurs, how the AUs co-occur, how long their
activation segments are and how the coding blocks sit in the videos. The figures
are written to::

    <output_dir>/                      # default <repo>/output/inspect_data/AUCoding/
        AU_presence.png                # per-AU occurrence + missing rate
        AU_cooccurrence.png            # pairwise Jaccard (all AUs + the subset)
        AU_active_count.png            # how many AUs are active in a frame
        AU_segments.png                # activation segments per AU + durations
        AU_session_spread.png          # per-session spread (and per task)
        AU_task_presence.png           # subset AUs, grouped by task
        AU_session_timeline.png        # one session's coding as a raster
        AU_session_coverage.png        # where the coded blocks sit in the videos
        AU_summary.json                # every number behind the figures
        AU_index.json                  # what was inspected + what was written

FILE FORMAT (verified on all 560 local files)
--------------------------------------------
A csv with a header row and one row per coded video frame::

    1,1,2,4,5,6,7,...,39,99      <- header: frame-col label, then 35 AU ids
    1204,0,0,1,0,0,0,...,0,0     <- col0 = 1-BASED GLOBAL VIDEO FRAME index
    1205,0,0,1,0,0,0,...,0,0

Values are ``0`` (absent), ``1`` (present) or ``9`` (missing/unknown). The frame
column is 1-based and maps onto the canonical rgb frames as ``jpg index = f - 1``
(see ``data/au_dataset.py``).

**AU99 IS NOT AN AU, it is a per-frame "coding unreliable" flag.** Verified: the
944 frames of the corpus that contain a ``9`` are exactly the frames with
``AU99 == 1`` (923 of them have *all* 34 real AU columns set to ``9``, 21 have a
partial ``9``). AU99 is therefore excluded from every statistic and reported
separately as ``unknown_frames``; all occurrence rates are computed on the frames
that are NOT flagged, and ``n_missing`` still counts the per-AU ``9`` codes --
over ALL frames, because that is where those codes live. The upshot is that the
corpus-level missingness is essentially a *whole-frame* property (one uniform
~0.5% on every AU), not a per-AU one.

Only tasks T1/T6/T7/T8 are FACS-coded (4 x 140 = 560 files, 82 F + 58 M
subjects; ~198k coded frames). The coded rows of a session form ONE CONTIGUOUS
block -- but it usually starts deep inside the video (first coded frames range
from 1 to 2248), which is what ``AU_session_coverage.png`` makes visible and
what the clip sampler in ``data/au_dataset.py`` has to respect at the video
start/end.

WHY THIS MATTERS FOR THE THESIS
-------------------------------
The AU-occurrence probe (``core/au_probe.py``) is a *semantic representation
quality* diagnostic, so its label distribution is a first-class result, not an
implementation detail:

* Presence is extremely unbalanced -- AU6/7/10/12/14 fire on 50-66% of frames
  while AU13/27/29/33/34/35/36/37/39 are essentially absent. A probe trained on
  all 34 AUs is dominated by dead classes, which is why the standard BP4D 12-AU
  subset (``data.au_dataset.DEFAULT_AU_LIST``) is the default ``--au_list``.
* AU coding is done on the MOST EXPRESSIVE segments of a task, so several AUs
  are near-constant ON there. A degenerate all-positive head already scores a
  high macro-F1 on that val split -- always report the trivial baseline next to
  the probe's F1 (this inspector quantifies exactly how degenerate it is via
  ``mean_active_per_frame`` of the subset and the per-AU presence rates).

Usage (from ``code/``)::

    # whole corpus, every figure
    python runners/run_inspect_au.py

    # what is on disk?
    python runners/run_inspect_au.py --list

    # one task only, no figures, different AU subset
    python runners/run_inspect_au.py --task T7 --no-plot --au_list 6,7,10,12,14

    # smoke run on a handful of files
    python runners/run_inspect_au.py --max_files 20

``--au_root`` defaults to ``$AU_ROOT``, else
``data.au_dataset.default_au_root()`` (the in-repo raw ``AUCoding/AU_OCC``);
``--output_dir`` defaults to ``$OUTPUT_DIR/inspect_data/AUCoding``, else
``<repo>/output/inspect_data/AUCoding``.
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data.au_dataset import DEFAULT_AU_LIST, default_au_root

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
#: per-AU code for "not coded / unknown" (an AU NEVER has this as an AU id)
MISSING = 9
#: the corpus' per-frame reliability flag column (see the module docstring)
UNKNOWN_AU = 99
#: AU codes that are valid in the data (besides :data:`MISSING`)
PRESENT, ABSENT = 1, 0

SCHEMA = 'bp4d-au-inspect/1'
DEFAULT_FPS = 25.0
TASK_ORDER = ['T1', 'T6', 'T7', 'T8']

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_DIR = os.path.dirname(_CODE_DIR)

# colours: the default 12-AU subset is highlighted wherever AUs are laid out
_C_SUBSET = 'tab:orange'
_C_OTHER = '#7f7f7f'
# raster codes (absent / present / missing) for the timeline figure
_CMAP_RASTER = ['#f4f4f4', '#1f4e79', '#d62728']


# --------------------------------------------------------------------------- #
# path defaults
# --------------------------------------------------------------------------- #
def _env(key: str, default: str = '') -> str:
    return os.environ.get(key, default)


def default_output_dir() -> str:
    """``$OUTPUT_DIR/inspect_data/AUCoding``, else ``<repo>/output/...``.

    Absolute on purpose (same reasoning as the other inspectors): a bare
    ``./output/...`` would land inside ``code/`` when the env profile was not
    sourced.
    """
    root = _env('OUTPUT_DIR', '') or os.path.join(_REPO_DIR, 'output')
    return os.path.join(root, 'inspect_data', 'AUCoding')


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def load_au_table(path: str):
    """Read one AU_OCC csv -> ``(au_ids, frames, values)``.

    ``au_ids`` is the header's AU-id list (int64, AU99 *included* -- dropping it
    is the caller's decision), ``frames`` the 1-based global video frame indices
    and ``values`` the ``[n_frames, n_au]`` int8 code matrix (0/1/9).
    """
    with open(path, 'r', newline='') as fh:
        rows = [r for r in csv.reader(fh) if r and any(c.strip() for c in r)]
    if len(rows) < 2:
        raise ValueError(f'AU file has no data rows: {path}')
    header = [c.strip() for c in rows[0]]
    au_ids = np.array([int(c) for c in header[1:]], dtype=np.int64)
    n = len(rows) - 1
    frames = np.empty(n, dtype=np.int64)
    values = np.empty((n, au_ids.size), dtype=np.int8)
    for i, row in enumerate(rows[1:]):
        if len(row) != au_ids.size + 1:
            raise ValueError(f'{path}: row {i + 2} has {len(row)} fields, '
                             f'expected {au_ids.size + 1}')
        frames[i] = int(row[0])
        values[i] = [int(c) for c in row[1:]]
    bad = np.setdiff1d(np.unique(values), np.array([ABSENT, PRESENT, MISSING]))
    if bad.size:
        raise ValueError(f'{path}: unexpected AU codes {bad.tolist()} '
                         f'(expected 0/1/9)')
    return au_ids, frames, values


def discover_au_files(au_root: str, subjects=None, tasks=None):
    """All ``<subject>_<task>.csv`` files under ``au_root`` (sorted)."""
    if not os.path.isdir(au_root):
        raise FileNotFoundError(f'AU root does not exist: {au_root}')
    subs = {s.upper() for s in (subjects or [])}
    tsks = {t.upper() for t in (tasks or [])}
    out = []
    for path in sorted(glob.glob(os.path.join(au_root, '*.csv'))):
        session = os.path.splitext(os.path.basename(path))[0]
        subject, _, task = session.rpartition('_')
        if not subject:                       # a file without an '_' suffix
            subject, task = session, ''
        if subs and subject.upper() not in subs:
            continue
        if tsks and task.upper() not in tsks:
            continue
        out.append({'session': session, 'subject': subject, 'task': task,
                    'path': path})
    return out


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def _segments(present: np.ndarray):
    """Runs of ``True`` -> ``(n_segments, lengths)`` in frames.

    A ``9`` (missing) frame is NOT present, so it terminates a segment. The
    corpus has only 21 such frames inside otherwise-coded windows, so the effect
    on segment statistics is immaterial -- but the convention is documented here
    because it is the one place where a ``9`` is silently treated as a ``0``.
    """
    if present.size == 0:
        return 0, np.zeros(0, dtype=np.int64)
    padded = np.concatenate([[False], present.astype(bool), [False]])
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return int(starts.size), (ends - starts).astype(np.int64)


def au_statistics(values: np.ndarray, au_ids: np.ndarray, valid: np.ndarray,
                  fps: float) -> list:
    """Per-AU occurrence / missingness / segment statistics.

    Every rate is computed on the frames that are NOT flagged by AU99
    (``valid``); ``presence_rate`` additionally drops the per-AU ``9`` codes, so
    it is the number to compare across AUs. ``n_missing``/``missing_rate`` are
    the exception: they count the ``9`` codes over ALL frames, because the
    missing codes live inside the AU99-flagged rows -- measuring them on the
    unflagged frames only would report ~0 for every AU and hide that fact.
    """
    n_frames = int(values.shape[0])
    rows = []
    for j, au in enumerate(au_ids):
        col = values[:, j]
        n_m = int((col == MISSING).sum())
        vcol = col[valid]
        present = vcol == PRESENT
        n_p = int(present.sum())
        n_a = int((vcol == ABSENT).sum())
        known = n_p + n_a
        seg_count, seg_len = _segments(present)
        mean_len = float(seg_len.mean()) if seg_count else 0.0
        med_len = float(np.median(seg_len)) if seg_count else 0.0
        max_len = int(seg_len.max()) if seg_count else 0
        rows.append({
            'au': int(au),
            'n_frames': n_frames,
            'n_present': n_p,
            'n_absent': n_a,
            'n_missing': n_m,
            'presence_rate': (n_p / known) if known else 0.0,
            'presence_rate_all': (n_p / n_frames) if n_frames else 0.0,
            'missing_rate': (n_m / n_frames) if n_frames else 0.0,
            'n_segments': seg_count,
            'mean_segment_frames': mean_len,
            'median_segment_frames': med_len,
            'max_segment_frames': max_len,
            'mean_segment_s': mean_len / fps,
            'median_segment_s': med_len / fps,
            'max_segment_s': max_len / fps,
        })
    return rows


def cooccurrence(values: np.ndarray, valid: np.ndarray):
    """Pairwise Jaccard + conditional presence between all AUs.

    Both are computed only over frames where BOTH AUs of the pair are KNOWN, so
    a ``9`` in one column can never be counted as an absence for the other::

        jaccard[i, j]  = both / (present_i + present_j - both)
        p_cond[i, j]   = P(AU j present | AU i present, AU j known)
                       = both / present_i

    (the second condition on ``p_cond`` matters only for the 21 frames that are
    AU99-flagged without every column being ``9``).

    ``present_i`` (= AU i present AND known) and the pairwise counts are built as
    three matrix products, which keeps this O(n_frames * n_au^2) but in BLAS.
    ``present`` must be masked by ``valid`` as well: a flagged frame can carry a
    non-``9`` code, so ``present`` alone is not a subset of ``known``.
    """
    present = (values == PRESENT) & valid[:, None]
    known = valid[:, None] & (values != MISSING)
    P = present.astype(np.float64)
    K = known.astype(np.float64)
    both = P.T @ P                     # both present AND known
    pres_i_pair = P.T @ K              # AU i present AND AU j known
    pair_known = K.T @ K               # both AUs known
    # |A| and |B| are counted on the frames where BOTH are known, so the union is
    # |A| + |B| - |A and B| (the intersection is subtracted ONCE).
    union = pres_i_pair + pres_i_pair.T - both
    with np.errstate(divide='ignore', invalid='ignore'):
        jac = np.where(union > 0, both / union, 0.0)
        cond = np.where(pres_i_pair > 0, both / pres_i_pair, 0.0)
    # a Jaccard coefficient above 1 is impossible -> a formula regression here
    # would silently corrupt the figure, so check instead of clipping blindly
    if jac.max(initial=0.0) > 1.0 + 1e-9:                        # pragma: no cover
        raise AssertionError(f'Jaccard > 1 (max {jac.max():.4f}): the union is '
                             f'computed wrongly')
    jac = np.clip(jac, 0.0, 1.0)
    np.fill_diagonal(jac, 1.0)
    np.fill_diagonal(cond, 1.0)
    return {'jaccard': jac, 'p_cond': cond, 'both': both,
            'pair_known': pair_known}


def active_counts(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Number of active AUs in every frame (all frames; flagged ones are 0)."""
    n_au = values.shape[1]
    out = np.zeros(values.shape[0], dtype=np.int64)
    out[valid] = (values[valid] == PRESENT).sum(axis=1)
    return np.bincount(out, minlength=n_au + 1)[:n_au + 1]


def session_summary(frames: np.ndarray, sub: np.ndarray, valid: np.ndarray,
                    fps: float) -> dict:
    """Block coverage + activity of one session.

    ``sub`` is that session's ``[n_frames, len(--au_list)]`` code slice, so the
    activity numbers are the SUBSET's, not all 34 AUs'.
    """
    gaps = np.diff(frames) if frames.size > 1 else np.zeros(0, dtype=np.int64)
    active = (sub[valid] == PRESENT).sum(axis=1)
    return {
        'n_frames': int(frames.size),
        'frame_first': int(frames[0]) if frames.size else 0,
        'frame_last': int(frames[-1]) if frames.size else 0,
        'contiguous': bool(gaps.size == 0 or np.all(gaps == 1)),
        'n_gaps': int((gaps != 1).sum()),
        'largest_gap_frames': int(gaps.max()) if gaps.size else 0,
        'n_unknown_frames': int((~valid).sum()),
        'duration_s': frames.size / fps,
        'start_s': (int(frames[0]) - 1) / fps if frames.size else 0.0,
        'mean_active_per_frame': float(active.mean()) if active.size else 0.0,
        'max_active_per_frame': int(active.max()) if active.size else 0,
        'any_active_fraction': float((active > 0).mean()) if active.size else 0.0,
    }


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def _import_pyplot():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:                                     # pragma: no cover
        print(f'[plot] matplotlib unavailable, skipping figures ({exc})')
        return None


def _sort_key(rows, key):
    return sorted(rows, key=lambda r: r[key])


def plot_presence(stats: list, subset: set, out_dir: str) -> str:
    """Per-AU occurrence rate (the headline figure) + how much is missing."""
    plt = _import_pyplot()
    if plt is None:
        return ''
    rows = _sort_key(stats, 'presence_rate')
    ids = [r['au'] for r in rows]
    y = np.arange(len(rows))
    colors = [_C_SUBSET if a in subset else _C_OTHER for a in ids]

    fig, axes = plt.subplot_mosaic([['rate', 'miss']], figsize=(10.5, 8.0),
                                   layout='constrained')
    ax = axes['rate']
    ax.barh(y, [r['presence_rate'] for r in rows], color=colors, height=0.72)
    ax.set_yticks(y, [str(a) for a in ids], fontsize=6.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('frames with the AU active')
    ax.set_ylabel('AU')
    ax.set_title('AU occurrence rate (9 excluded)', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    ax = axes['miss']
    ax.barh(y, [r['missing_rate'] for r in rows], color=colors, height=0.72)
    ax.set_yticks(y, [str(a) for a in ids], fontsize=6.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel('frames coded 9 (unknown)')
    ax.set_title('per-AU missing rate', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    handles = [plt.Rectangle((0, 0), 1, 1, color=_C_SUBSET),
               plt.Rectangle((0, 0), 1, 1, color=_C_OTHER)]
    axes['rate'].legend(handles, [f'default subset ({len(subset)} AUs)',
                                  'other AUs'], fontsize=8, loc='lower right')
    path = os.path.join(out_dir, 'AU_presence.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_cooccurrence(co, au_ids, subset_ids, out_dir: str) -> str:
    """Pairwise Jaccard -- how redundant the AU set is."""
    plt = _import_pyplot()
    if plt is None:
        return ''
    ids = [int(a) for a in au_ids]
    jac = co['jaccard']
    sel = [i for i, a in enumerate(ids) if a in subset_ids]
    sub = jac[np.ix_(sel, sel)]
    tick = np.arange(len(ids))

    fig, axes = plt.subplot_mosaic([['all', 'sub']], figsize=(13.0, 6.4),
                                   layout='constrained')
    ax = axes['all']
    im = ax.imshow(jac, cmap='viridis', vmin=0.0, vmax=1.0)
    ax.set_xticks(tick, ids, fontsize=5.5, rotation=90)
    ax.set_yticks(tick, ids, fontsize=5.5)
    ax.set_title('Jaccard overlap, all AUs', fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes['sub']
    im = ax.imshow(sub, cmap='viridis', vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(sel)), [ids[i] for i in sel], fontsize=8)
    ax.set_yticks(np.arange(len(sel)), [ids[i] for i in sel], fontsize=8)
    ax.set_title('Jaccard overlap, default subset', fontsize=10)
    for i in range(len(sel)):
        for j in range(len(sel)):
            v = sub[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6.5,
                    color='white' if v < 0.6 else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046)

    path = os.path.join(out_dir, 'AU_cooccurrence.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_active_count(counts: np.ndarray, out_dir: str) -> str:
    """How many AUs are active in a typical frame (= how degenerate it is)."""
    plt = _import_pyplot()
    if plt is None:
        return ''
    k = np.arange(counts.size)
    fig, axes = plt.subplot_mosaic([['hist', 'cdf']], figsize=(11.0, 4.4),
                                   layout='constrained')
    ax = axes['hist']
    ax.bar(k, counts, color='tab:blue', width=0.8)
    ax.set_xlabel('AUs active in the frame (of 34)')
    ax.set_ylabel('frames')
    ax.set_title('active AUs per coded frame', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    ax = axes['cdf']
    frac = np.cumsum(counts) / max(1, counts.sum())
    ax.plot(k, frac, marker='o', ms=3, color='tab:red')
    ax.set_xlabel('AUs active in the frame (of 34)')
    ax.set_ylabel('fraction of frames <= k')
    ax.set_ylim(0, 1.02)
    ax.set_title('cumulative distribution', fontsize=10)
    ax.grid(alpha=0.3)

    path = os.path.join(out_dir, 'AU_active_count.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_segments(stats: list, subset: set, out_dir: str) -> str:
    """Segmentation of the coding: activations per AU and their duration."""
    plt = _import_pyplot()
    if plt is None:
        return ''
    rows = _sort_key(stats, 'n_segments')
    ids = [r['au'] for r in rows]
    colors = [_C_SUBSET if a in subset else _C_OTHER for a in ids]
    y = np.arange(len(rows))

    fig, axes = plt.subplot_mosaic([['cnt', 'dur']], figsize=(11.0, 8.0),
                                   layout='constrained')
    ax = axes['cnt']
    ax.barh(y, [r['n_segments'] for r in rows], color=colors, height=0.72)
    ax.set_yticks(y, [str(a) for a in ids], fontsize=6.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel('activation segments (whole corpus)')
    ax.set_ylabel('AU')
    ax.set_title('number of activation segments', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    ax = axes['dur']
    mean = np.array([r['mean_segment_s'] for r in rows])
    med = np.array([r['median_segment_s'] for r in rows])
    ax.barh(y, mean, color=colors, height=0.72, label='mean')
    ax.plot(med, y, 'k.', ms=3.5, label='median')
    ax.set_yticks(y, [str(a) for a in ids], fontsize=6.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel('segment duration (s)')
    ax.set_title('activation duration', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')

    path = os.path.join(out_dir, 'AU_segments.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_session_spread(per_session: dict, subset: set, per_task: dict,
                        out_dir: str) -> str:
    """Between-session variability: is an AU frequent everywhere or on average?"""
    plt = _import_pyplot()
    if plt is None:
        return ''
    order = sorted(subset, key=lambda a: -float(np.median(per_session[a])))
    data = [per_session[a] for a in order]
    colors = [_C_SUBSET] * len(order)

    fig, axes = plt.subplot_mosaic([['aus', 'tasks']], figsize=(12.5, 5.4),
                                   width_ratios=[2.4, 1.0], layout='constrained')
    ax = axes['aus']
    bp = ax.boxplot(data, positions=np.arange(len(order)), widths=0.6,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker='.', ms=3, alpha=0.5))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_xticks(np.arange(len(order)), [str(a) for a in order])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('AU')
    ax.set_ylabel('presence rate of the session')
    ax.set_title('per-session occurrence, default subset', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    ax = axes['tasks']
    tasks = [t for t in TASK_ORDER if t in per_task] or sorted(per_task)
    ax.boxplot([per_task[t] for t in tasks], positions=np.arange(len(tasks)),
               widths=0.55, patch_artist=True,
               boxprops=dict(facecolor='tab:cyan', alpha=0.7),
               flierprops=dict(marker='.', ms=3, alpha=0.5))
    ax.set_xticks(np.arange(len(tasks)), tasks)
    ax.set_xlabel('task')
    ax.set_ylabel('mean active AUs per frame')
    # a COUNT out of len(subset), not a rate -- a 0-1 axis would clip every box
    ax.set_ylim(-0.05, len(order) + 0.05)
    ax.set_title('activity by task', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    path = os.path.join(out_dir, 'AU_session_spread.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_task_presence(stats: list, per_task_au: dict, subset: set,
                       out_dir: str) -> str:
    """Per-AU occurrence split by task (T1 is much shorter than T6/T7/T8)."""
    plt = _import_pyplot()
    if plt is None:
        return ''
    rate = {r['au']: r['presence_rate'] for r in stats}
    aus = sorted(subset, key=lambda a: -rate.get(a, 0.0))
    tasks = [t for t in TASK_ORDER if t in per_task_au] or sorted(per_task_au)
    x = np.arange(len(aus))
    width = 0.8 / max(1, len(tasks))
    fig, ax = plt.subplots(figsize=(12.0, 4.6), layout='constrained')
    for i, task in enumerate(tasks):
        vals = [per_task_au[task].get(a, float('nan')) for a in aus]
        ax.bar(x + (i - (len(tasks) - 1) / 2) * width, vals, width,
               label=task, color=f'C{i}')
    ax.set_xticks(x, [str(a) for a in aus])
    ax.set_xlabel('AU')
    ax.set_ylabel('frames with the AU active')
    ax.set_ylim(0, 1.0)
    ax.set_title('occurrence rate by task (default subset)', fontsize=10)
    ax.legend(fontsize=8, ncols=len(tasks))
    ax.grid(axis='y', alpha=0.3)
    path = os.path.join(out_dir, 'AU_task_presence.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_session_timeline(frames, values, au_ids, subset: set, fps: float,
                          session: str, out_dir: str) -> str:
    """One session's coding as a raster: what the clip sampler has to deal with."""
    plt = _import_pyplot()
    if plt is None:
        return ''
    ids = [int(a) for a in au_ids]
    active = (values == PRESENT).sum(axis=1)
    x = frames

    fig, axes = plt.subplot_mosaic([['cnt'], ['rast']], figsize=(12.5, 7.0),
                                   height_ratios=[1.0, 2.6], sharex=True,
                                   layout='constrained')
    ax = axes['cnt']
    ax.fill_between(x, 0, active, step='mid', color='tab:blue', alpha=0.85)
    unknown_rows = np.flatnonzero((values == MISSING).any(axis=1))
    if unknown_rows.size:
        ax.plot(x[unknown_rows], np.full(unknown_rows.size, active.max()),
                '|', color='tab:red', ms=6, label='coded 9')
        ax.legend(fontsize=8, loc='upper right')
    ax.set_ylabel('active AUs\n(of 34)')
    ax.set_title(f'{session}: AU occurrence raster', fontsize=11)
    ax.grid(alpha=0.3)

    ax = axes['rast']
    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap = ListedColormap(_CMAP_RASTER)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 9.5], cmap.N)
    ax.imshow(values.T, aspect='auto', origin='upper', cmap=cmap, norm=norm,
              extent=[frames[0] - 0.5, frames[-1] + 0.5, len(ids) - 0.5,
                      -0.5], interpolation='nearest')
    ax.set_yticks(np.arange(len(ids)))
    ax.set_yticklabels([str(a) for a in ids], fontsize=6)
    for tick, a in zip(ax.get_yticklabels(), ids):
        if a in subset:
            tick.set_color(_C_SUBSET)
            tick.set_fontweight('bold')
    ax.set_xlabel(f'global video frame (1-based); coding starts at frame '
                  f'{int(frames[0])}')
    ax.set_ylabel('AU')

    # absolute video time on top (AU frame f <-> jpg f-1 -> (f-1)/fps seconds)
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    n_ticks = 6
    tpos = np.linspace(int(frames[0]), int(frames[-1]), n_ticks)
    top.set_xticks(tpos, [f'{(p - 1) / fps:.0f}' for p in tpos], fontsize=8)
    top.set_xlabel('video time (s)', fontsize=8)

    path = os.path.join(out_dir, 'AU_session_timeline.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_session_coverage(sessions: list, out_dir: str) -> str:
    """Where the coded blocks sit in the videos + how long they are."""
    plt = _import_pyplot()
    if plt is None:
        return ''
    tasks = [t for t in TASK_ORDER
             if any(s['task'] == t for s in sessions)] or \
        sorted({s['task'] for s in sessions})
    cmap = {t: f'C{i}' for i, t in enumerate(tasks)}
    ordered = sorted(sessions, key=lambda s: (s['frame_first'], s['session']))

    fig, axes = plt.subplot_mosaic([['blocks', 'start'], ['blocks', 'len']],
                                   figsize=(13.0, 8.0), width_ratios=[2.0, 1.0],
                                   layout='constrained')
    ax = axes['blocks']
    y = np.arange(len(ordered))
    ax.hlines(y, [s['frame_first'] for s in ordered],
              [s['frame_last'] for s in ordered],
              color=[cmap[s['task']] for s in ordered], lw=0.9)
    ax.axvline(1.0, color='0.4', lw=0.8, ls=':', label='video start')
    ax.set_yticks([])
    ax.set_ylim(-1, len(ordered))
    ax.set_xlabel('global video frame (1-based)')
    ax.set_ylabel(f'coded session (sorted, n={len(ordered)})')
    ax.set_title('coded frame blocks per session', fontsize=10)
    handles = [plt.Line2D([0], [0], color=cmap[t], lw=2.5) for t in tasks]
    ax.legend(handles, tasks, fontsize=8, loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    ax = axes['start']
    starts = [s['frame_first'] for s in sessions]
    bins = np.linspace(0, max(starts) if starts else 1, 40)
    for t in tasks:
        vals = [s['frame_first'] for s in sessions if s['task'] == t]
        ax.hist(vals, bins=bins, alpha=0.65, label=t, color=cmap[t], stacked=True)
    ax.set_xlabel('first coded frame')
    ax.set_ylabel('sessions')
    ax.set_title('where coding starts', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    ax = axes['len']
    hi = max([s['n_frames'] for s in sessions] or [1])
    bins = np.linspace(0, hi, 40)
    for t in tasks:
        vals = [s['n_frames'] for s in sessions if s['task'] == t]
        ax.hist(vals, bins=bins, alpha=0.65, label=t, color=cmap[t], stacked=True)
    ax.set_xlabel('coded frames per session')
    ax.set_ylabel('sessions')
    ax.set_title('coded length', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    path = os.path.join(out_dir, 'AU_session_coverage.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# console report
# --------------------------------------------------------------------------- #
def print_report(summary: dict) -> None:
    """Compact corpus + per-AU table (the same numbers as the JSON)."""
    corp = summary['corpus']
    print(f"corpus      : {corp['n_files']} files, {corp['n_subjects']} subjects "
          f"({corp['n_subjects_female']} F / {corp['n_subjects_male']} M), "
          f"tasks {', '.join(corp['tasks'])}")
    print(f"frames      : {corp['n_frames']} coded rows "
          f"({corp['total_duration_s'] / 60.0:.1f} min), "
          f"{corp['n_unknown_frames']} flagged unknown by AU99 "
          f"({corp['unknown_frame_fraction']:.3%}; "
          f"{corp['n_all_missing_frames']} of them all-AU-9)")
    print(f"per session : {corp['frames_per_session_min']}-"
          f"{corp['frames_per_session_max']} frames "
          f"(median {corp['frames_per_session_median']:.0f}), "
          f"first coded frame {corp['first_coded_frame_min']}-"
          f"{corp['first_coded_frame_max']}, "
          f"{corp['n_noncontiguous_sessions']} non-contiguous")
    print(f"active/frame: mean {corp['mean_active_per_frame']:.2f} of "
          f"{corp['n_au']} AUs, "
          f"{corp['any_active_fraction']:.1%} of frames have any AU at all")
    print(f"subset      : {summary['subset']} -> mean "
          f"{corp['mean_active_per_frame_subset']:.2f} active AUs/frame "
          f"({corp['mean_active_per_frame_subset'] / len(summary['subset']):.1%} "
          f"of the subset active)")
    print()
    print(f"  {'AU':>4} {'present%':>9} {'9%':>7} {'sess%':>7} {'segs':>7} "
          f"{'med dur s':>10} {'max dur s':>10}")
    for r in sorted(summary['aus'], key=lambda v: -v['presence_rate']):
        print(f"  {r['au']:>4} {r['presence_rate']:>8.1%} "
              f"{r['missing_rate']:>7.2%} "
              f"{r['frac_sessions_present']:>7.1%} {r['n_segments']:>7} "
              f"{r['median_segment_s']:>10.2f} {r['max_segment_s']:>10.1f}")
    print('  present%: AU active among the frames that are neither AU99-flagged '
          'nor coded 9 for that AU;  9%: frames coded 9 over ALL frames;')
    print('  sess%: sessions in which the AU fires at least once;  segs: '
          'activation segments over the corpus.')
    print()
    print('task means (default subset AUs, mean over sessions):')
    for task, block in summary['per_task'].items():
        vals = block['mean_active_per_session']
        if vals:
            print(f"  {task}: mean active {np.mean(vals):.2f} / "
                  f"{len(summary['subset'])} AUs per frame "
                  f"(min {np.min(vals):.2f}, max {np.max(vals):.2f})")
    if summary['figures']:
        print()
        print('figures:')
        for path in summary['figures']:
            print(f'  {path}')


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def inspect_au(au_root: str, output_dir: str, au_list=DEFAULT_AU_LIST,
               subjects=None, tasks=None, fps: float = DEFAULT_FPS,
               max_files=None, plot: bool = True,
               timeline_session: str = '') -> dict:
    """Inspect every selected AU file and write the figures + JSON."""
    files = discover_au_files(au_root, subjects, tasks)
    if max_files:
        files = files[:int(max_files)]
    if not files:
        raise FileNotFoundError(
            f'no AU_OCC csv matched under {au_root} '
            f'(subjects={subjects}, tasks={tasks})')

    os.makedirs(output_dir, exist_ok=True)
    print(f'au root     : {au_root}')
    print(f'output dir  : {output_dir}')
    print(f'files       : {len(files)}')

    # ---- ingest ---------------------------------------------------------
    # the header is identical in every file, but read it each time and fail
    # loudly if it drifts: the AU columns must line up across sessions
    per_file = []
    au_ids_ref = None
    for meta in files:
        ids, frames, values = load_au_table(meta['path'])
        if au_ids_ref is None:
            au_ids_ref = ids
        elif not np.array_equal(ids, au_ids_ref):
            raise ValueError(
                f'{meta["path"]}: AU header differs from {files[0]["path"]} '
                f'({ids.tolist()} vs {au_ids_ref.tolist()})')
        per_file.append({'meta': meta, 'frames': frames, 'values': values})

    # AU99 is a per-frame reliability FLAG, not an AU -> drop that column here,
    # once, so every statistic below only ever sees real AUs. The flag mask is
    # kept per frame (concatenated in file order) and excludes those frames from
    # all rates.
    flag_col = None
    if UNKNOWN_AU in au_ids_ref.tolist():
        flag_col = int(np.flatnonzero(au_ids_ref == UNKNOWN_AU)[0])
    keep = [j for j in range(au_ids_ref.size) if j != flag_col]
    au_ids = au_ids_ref[keep]
    flag = np.concatenate([
        (p['values'][:, flag_col] == PRESENT) if flag_col is not None
        else np.zeros(p['frames'].size, dtype=bool) for p in per_file])
    valid = ~flag
    frames_all = np.concatenate([p['frames'] for p in per_file])
    values_all = np.concatenate([p['values'][:, keep] for p in per_file], axis=0)
    offsets = np.cumsum([0] + [p['frames'].size for p in per_file])

    subset = sorted({int(a) for a in au_list} & set(au_ids.tolist()))
    unknown_subset = sorted({int(a) for a in au_list} - set(au_ids.tolist()))
    if not subset:
        raise ValueError(f'none of --au_list {list(au_list)} is in the corpus '
                         f'(available: {au_ids.tolist()})')
    sub_cols = np.array([int(np.flatnonzero(au_ids == a)[0]) for a in subset])

    # ---- statistics -----------------------------------------------------
    stats = au_statistics(values_all, au_ids, valid, fps)
    co = cooccurrence(values_all, valid)
    counts = active_counts(values_all, valid)
    sub_active = (values_all[np.ix_(valid, sub_cols)] == PRESENT).sum(axis=1)

    # per-session: AU rates + block coverage. Each session slices the SAME
    # concatenated arrays, so the flag mask and the values can never drift apart.
    # The rates are kept for EVERY AU (not just the subset) -- that is what makes
    # the 'sess%' column meaningful for rare AUs too, while the per-session table
    # itself stores only the subset keys to stay readable.
    sessions = []
    per_session = {int(a): [] for a in au_ids}
    for k, p in enumerate(per_file):
        meta = p['meta']
        sl = slice(int(offsets[k]), int(offsets[k + 1]))
        v_valid = valid[sl]
        vals = values_all[sl]
        n_known = (v_valid[:, None] & (vals != MISSING)).sum(axis=0)
        n_present = (vals == PRESENT).sum(axis=0)
        rate = np.where(n_known > 0, n_present / np.maximum(1, n_known), 0.0)
        for a, r in zip(au_ids.tolist(), rate):
            per_session[a].append(float(r))
        row = {'session': meta['session'], 'subject': meta['subject'],
               'task': meta['task'], 'path': meta['path']}
        row.update(session_summary(p['frames'], vals[:, sub_cols], v_valid, fps))
        row['presence_rate'] = {str(a): float(r)
                                for a, r in zip(subset, rate[sub_cols])}
        sessions.append(row)

    # how many sessions contain each AU at all (a 0 here is a real, reportable
    # state: the AU never fires in that session)
    n_sessions = len(per_file)
    for r in stats:
        hits = per_session.get(r['au'], [])
        r['n_sessions_present'] = int(sum(1 for v in hits if v > 0))
        r['frac_sessions_present'] = (r['n_sessions_present'] / n_sessions
                                      if n_sessions else 0.0)

    # per task: session-mean activity + per-AU occurrence
    per_task, per_task_au = {}, {}
    for task in sorted({p['meta']['task'] for p in per_file}):
        rows_t = [s for s in sessions if s['task'] == task]
        per_task[task] = [s['mean_active_per_frame'] for s in rows_t]
        per_task_au[task] = {a: float(np.mean([s['presence_rate'][str(a)]
                                               for s in rows_t]))
                             for a in subset}

    # ---- corpus headline numbers
    starts = np.array([s['frame_first'] for s in sessions], dtype=np.int64)
    lens = np.array([s['n_frames'] for s in sessions], dtype=np.int64)
    subjects_seen = sorted({p['meta']['subject'] for p in per_file})
    n_f = sum(1 for s in subjects_seen if s.upper().startswith('F'))
    tasks_seen = sorted({p['meta']['task'] for p in per_file})

    corpus = {
        'n_files': len(per_file),
        'n_subjects': len(subjects_seen),
        'n_subjects_female': n_f,
        'n_subjects_male': len(subjects_seen) - n_f,
        'tasks': tasks_seen,
        'n_au': int(au_ids.size),
        'au_ids': au_ids.tolist(),
        'n_frames': int(frames_all.size),
        'total_duration_s': float(frames_all.size / fps),
        'fps': float(fps),
        'n_unknown_frames': int(flag.sum()),
        'unknown_frame_fraction': float(flag.mean()),
        'n_all_missing_frames': int(((values_all == MISSING).all(axis=1)
                                    & flag).sum()),
        'frames_per_session_min': int(lens.min()),
        'frames_per_session_max': int(lens.max()),
        'frames_per_session_median': float(np.median(lens)),
        'first_coded_frame_min': int(starts.min()),
        'first_coded_frame_max': int(starts.max()),
        'last_coded_frame_max': int(max(s['frame_last'] for s in sessions)),
        'n_noncontiguous_sessions': int(sum(1 for s in sessions
                                            if not s['contiguous'])),
        'n_frames_per_task': {t: int(sum(s['n_frames'] for s in sessions
                                         if s['task'] == t))
                              for t in tasks_seen},
        'mean_active_per_frame': float(((values_all == PRESENT).sum(axis=1)
                                        * valid).sum() / max(1, int(valid.sum()))),
        'max_active_per_frame': int(((values_all == PRESENT).sum(axis=1)
                                     * valid).max()),
        'any_active_fraction': float((sub_active > 0).mean()),
        'mean_active_per_frame_subset': float(sub_active.mean()),
        'active_count_histogram': counts.tolist(),
    }

    summary = {
        'schema': SCHEMA,
        'au_root': au_root,
        'output_dir': output_dir,
        'subset': subset,
        'subset_missing_from_corpus': unknown_subset,
        'corpus': corpus,
        'aus': stats,
        'cooccurrence': {
            'au_ids': au_ids.tolist(),
            'jaccard': np.round(co['jaccard'], 4).tolist(),
            'p_cond_i_present_j_present': np.round(co['p_cond'], 4).tolist(),
            'both_present_counts': co['both'].astype(np.int64).tolist(),
        },
        'per_task': {t: {'mean_active_per_session': v, 'presence_rate': per_task_au[t]}
                     for t, v in per_task.items()},
        'sessions': sessions,
        'figures': [],
    }

    # ---- figures
    if plot:
        written = []
        written.append(plot_presence(stats, set(subset), output_dir))
        written.append(plot_cooccurrence(co, au_ids, set(subset), output_dir))
        written.append(plot_active_count(counts, output_dir))
        written.append(plot_segments(stats, set(subset), output_dir))
        written.append(plot_session_spread(per_session, set(subset), per_task,
                                           output_dir))
        written.append(plot_task_presence(stats, per_task_au, set(subset),
                                          output_dir))
        chosen = per_file[0]
        if timeline_session:
            for p in per_file:
                if p['meta']['session'].lower() == timeline_session.lower():
                    chosen = p
                    break
        k = per_file.index(chosen)
        written.append(plot_session_timeline(
            chosen['frames'], values_all[int(offsets[k]):int(offsets[k + 1])],
            au_ids, set(subset), fps, chosen['meta']['session'], output_dir))
        written.append(plot_session_coverage(sessions, output_dir))
        summary['figures'] = [w for w in written if w]
        summary['timeline_session'] = chosen['meta']['session']

    # ---- JSON
    with open(os.path.join(output_dir, 'AU_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)
    index = {
        'schema': SCHEMA,
        'au_root': au_root,
        'output_dir': output_dir,
        'n_files': len(per_file),
        'sessions': [p['meta']['session'] for p in per_file],
        'subjects': subjects_seen,
        'tasks': corpus['tasks'],
        'au_ids': au_ids.tolist(),
        'subset': subset,
        'fps': float(fps),
        'figures': summary['figures'],
        'summary': os.path.join(output_dir, 'AU_summary.json'),
    }
    with open(os.path.join(output_dir, 'AU_index.json'), 'w') as fh:
        json.dump(index, fh, indent=2, default=str)
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_list(spec: str):
    return [s.strip() for s in (spec or '').replace(';', ',').split(',')
            if s.strip()]


def _parse_au_list(spec: str):
    """``--au_list`` OR ``--au_list 'all'`` / ``''`` (default subset)."""
    toks = _parse_list(spec)
    if not toks:
        return list(DEFAULT_AU_LIST)
    if len(toks) == 1 and toks[0].lower() in ('all', '*'):
        return None                       # resolved after discovery
    try:
        return [int(t) for t in toks]
    except ValueError:
        raise SystemExit(f'--au_list expects AU numbers, got {spec!r}')


def get_args(argv=None):
    p = argparse.ArgumentParser(
        'BP4D+ raw AU occurrence coding inspection', add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('--au_root', default=_env('AU_ROOT') or default_au_root(),
                   help='raw AU_OCC directory (default: $AU_ROOT, else '
                        'data/raw/BP4D/AUCoding/AU_OCC)')
    p.add_argument('--output_dir', default=default_output_dir(),
                   help='figure + JSON directory (default: '
                        '$OUTPUT_DIR/inspect_data/AUCoding, else '
                        '<repo>/output/inspect_data/AUCoding)')
    p.add_argument('--subject', default='',
                   help='subject id(s), comma-separated (default: all 140)')
    p.add_argument('--task', default='all',
                   help='task id(s), comma-separated (T1,T6,T7,T8) or "all"')
    p.add_argument('--au_list', default='',
                   help='AUs to treat as the target subset (highlighted in the '
                        'figures, summarised per task/session); "all" = every '
                        f'AU. default: {list(DEFAULT_AU_LIST)}')
    p.add_argument('--fps', default=DEFAULT_FPS, type=float,
                   help='video frame rate used for frame -> second conversions '
                        '(default 25; BP4D nominal)')
    p.add_argument('--timeline_session', default='',
                   help='session shown in AU_session_timeline.png '
                        '(default: the first one discovered)')
    p.add_argument('--max_files', default=None, type=int,
                   help='cap the number of AU files (smoke runs)')
    p.add_argument('--no-plot', dest='plot', action='store_false', default=True,
                   help='write the JSON summaries only, skip the figures')
    p.add_argument('--list', action='store_true',
                   help='list the AU files found and exit')
    return p.parse_args(argv)


def main() -> int:
    args = get_args()
    try:
        available = discover_au_files(args.au_root)
    except FileNotFoundError as exc:
        raise SystemExit(f'{exc}\nset $AU_ROOT or pass --au_root')
    if args.list:
        print(f'au root: {args.au_root}')
        print(f'{len(available)} AU_OCC file(s):')
        for subject in sorted({m['subject'] for m in available}):
            tasks = [m['task'] for m in available if m['subject'] == subject]
            print(f'  {subject}: {" ".join(tasks)}')
        if not available:
            print('  (none - is this the right --au_root?)')
        return 0

    tasks = _parse_list(args.task)
    if any(t.lower() == 'all' for t in tasks) or not tasks:
        tasks = None
    subjects = _parse_list(args.subject) or None
    au_list = _parse_au_list(args.au_list)

    # '--au_list all' needs the corpus header, so read it from one file first
    if au_list is None:
        files = discover_au_files(args.au_root, subjects, tasks)
        if not files:
            raise SystemExit('no AU files matched --subject/--task')
        ids, _, _ = load_au_table(files[0]['path'])
        au_list = [int(a) for a in ids if int(a) != UNKNOWN_AU]

    try:
        summary = inspect_au(
            au_root=args.au_root, output_dir=args.output_dir, au_list=au_list,
            subjects=subjects, tasks=tasks, fps=args.fps,
            max_files=args.max_files, plot=args.plot,
            timeline_session=args.timeline_session)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f'{exc}')
    print()
    print_report(summary)
    return 0


if __name__ == '__main__':
    sys.exit(main())
