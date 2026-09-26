"""Evaluation artefacts for a Stage-3 run: metric logs and figures.

Everything here WRITES TO DISK (JSON / JSONL / PNG). matplotlib is imported
lazily and forced to the headless ``Agg`` backend, so these helpers are safe on
a machine without a display and they never affect the training path when
matplotlib is unavailable (callers guard the import).

Figures produced by the Stage-3 path:

* ``training_curves.png``      -- train loss, val Pearson, val MAE/RMSE, LR
* ``predictions_final.png``    -- per-clip predicted vs target waveform (+ PSD)
* ``session_<name>.png``       -- session-level assembled waveform (see
  ``runners/run_evaluate_session.py``)
"""
import json
import os

import numpy as np

__all__ = ['save_json', 'append_jsonl', 'plot_training_curves',
           'plot_waveform_panel', 'plot_session_waveform', 'jsonable']


# --------------------------------------------------------------------------- #
# metric documents
# --------------------------------------------------------------------------- #
def jsonable(obj):
    """Make metric dicts JSON-safe (numpy scalars / arrays -> python types)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float):
        return None if np.isnan(obj) else float(obj)
    return obj


def save_json(path, obj):
    """Write ``obj`` as indented JSON (creating parent dirs). Returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(jsonable(obj), f, indent=2, default=str)
    return path


def append_jsonl(path, record):
    """Append one record (usually one epoch) to a JSONL log. Returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(jsonable(record), default=str) + '\n')
    return path


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use('Agg')          # headless: never needs a display
    import matplotlib.pyplot as plt
    return plt


def _finish(fig, out_png):
    """Save + close a figure, creating the parent directory."""
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out_png


def _psd(x, fs):
    """Welch PSD if scipy is available, else the raw periodogram."""
    try:
        from scipy import signal as sps
        f, p = sps.welch(x, fs=fs, nperseg=int(min(256, len(x))))
        return f, p
    except Exception:
        x = x - np.mean(x)
        p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        f = np.fft.rfftfreq(len(x), d=1.0 / fs)
        return f, p / max(p.sum(), 1e-12)


def plot_training_curves(history, out_png, title=''):
    """2x2 summary of a run: loss, Pearson, MAE/RMSE, LR (all vs epoch)."""
    if not history:
        return None
    plt = _plt()
    ep = [h.get('epoch', i) for i, h in enumerate(history)]

    def col(key):
        vals = [h.get(key, np.nan) for h in history]
        return np.array([np.nan if v is None else v for v in vals], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), constrained_layout=True)
    ax = axes[0][0]
    loss = col('train_loss')
    if np.isfinite(loss).any():
        ax.plot(ep, loss, marker='o', ms=3, color='tab:red')
    ax.set_title('train loss'); ax.set_xlabel('epoch'); ax.grid(alpha=.3)

    ax = axes[0][1]
    for key, c in (('pearson', 'tab:blue'), ('mae', 'tab:orange'),
                   ('rmse', 'tab:green')):
        v = col(key)
        if np.isfinite(v).any():
            ax.plot(ep, v, marker='o', ms=3, color=c, label=key)
    ax.set_title('val Tier-1'); ax.set_xlabel('epoch'); ax.grid(alpha=.3)
    if ax.lines:
        ax.legend(fontsize=8)

    ax = axes[1][0]
    v = col('psd_mae')
    if np.isfinite(v).any():
        ax.plot(ep, v, marker='o', ms=3, color='tab:purple')
    ax.set_title('val Tier-2 (psd_mae)'); ax.set_xlabel('epoch'); ax.grid(alpha=.3)

    ax = axes[1][1]
    v = col('lr')
    if np.isfinite(v).any():
        ax.plot(ep, v, marker='o', ms=3, color='tab:gray')
        ax.set_yscale('log')
    ax.set_title('learning rate'); ax.set_xlabel('epoch'); ax.grid(alpha=.3)

    if title:
        fig.suptitle(title, fontsize=11)
    return _finish(fig, out_png)


def _spread_clip_indices(pred, target, sessions, max_clips):
    """Pick up to ``max_clips`` clips SPREAD ACROSS different sessions.

    One clip per chosen session, and within a session the clip whose per-clip
    Pearson is the MEDIAN of that session -- i.e. a representative clip, not the
    best one. Fully deterministic (evenly spaced sessions over the sorted unique
    labels), so a re-run reproduces the same figure.

    :returns: ``(indices, labels)`` where a label is ``'<session> clip <i>'``.
    """
    sess = [str(s) for s in sessions]
    order = sorted(set(sess))
    if len(order) <= max_clips:
        chosen = order
    else:
        pos = np.linspace(0, len(order) - 1, int(max_clips)).round().astype(int)
        chosen = [order[p] for p in dict.fromkeys(pos)]
    idx, labels = [], []
    for s in chosen:
        cand = [i for i, ss in enumerate(sess) if ss == s]
        scored = []
        for i in cand:
            r = (float(np.corrcoef(pred[i], target[i])[0, 1])
                 if target[i].std() > 1e-9 and pred[i].std() > 1e-9 else np.nan)
            scored.append((r, i))
        finite = [t for t in scored if np.isfinite(t[0])]
        scored = sorted(finite or scored, key=lambda t: np.nan_to_num(t[0]))
        _, pick = scored[len(scored) // 2]           # median-quality clip
        idx.append(pick)
        labels.append(f'{s} clip {pick}')
    return idx, labels


def plot_waveform_panel(pred, target, fs, out_png, title='', band=None,
                        metrics=None, max_clips=4, sessions=None,
                        clip_indices=None):
    """Per-clip predicted vs target: time overlay (left) + PSD (right).

    ``pred``/``target`` are ``[N, L]`` arrays in the same (z-scored) units.

    :param sessions: optional per-row session label (same length as ``pred``).
        When given, the panel shows ``max_clips`` clips SPREAD ACROSS DIFFERENT
        SESSIONS (one median-quality clip each) instead of the FIRST
        ``max_clips`` rows. Without it the panel silently plotted a single
        session -- which made a run look solved when only that session worked
        (measured 2026-09-26: val mean Pearson +0.139 overall but +0.961 for the
        session that happens to own rows 0-3). Passing ``sessions`` is strongly
        recommended for any reported figure.
    :param clip_indices: explicit row indices to plot; overrides ``sessions``.
    """
    plt = _plt()
    pred = np.atleast_2d(np.asarray(pred, float))
    target = np.atleast_2d(np.asarray(target, float))
    ntotal = pred.shape[0]
    note = None
    if clip_indices is not None:
        idx = [int(i) for i in clip_indices][:max_clips]
        labels = [f'clip {i}' for i in idx]
    elif sessions is not None and ntotal and len(sessions) == ntotal:
        idx, labels = _spread_clip_indices(pred, target, sessions, max_clips)
        nsess = len(set(str(s) for s in sessions))
        note = (f'{len(idx)} of {nsess} session(s) shown -- one median-quality '
                f'clip per session')
    else:
        if sessions is not None and len(sessions) != ntotal:
            note = (f'sessions length {len(sessions)} != {ntotal} predictions '
                    f'-- showing the FIRST clips (may be ONE session)')
        idx = list(range(min(int(max_clips), ntotal)))
        labels = [f'clip {i}' for i in idx]
    n = len(idx)
    if n == 0:
        return None
    L = pred.shape[1]
    t = np.arange(L) / float(fs)

    fig, axes = plt.subplots(n, 2, figsize=(12, 1.9 * n + 0.6),
                             constrained_layout=True, squeeze=False)
    for row, i in enumerate(idx):
        ax = axes[row][0]
        ax.plot(t, target[i], color='0.55', lw=1.2, label='target')
        ax.plot(t, pred[i], color='tab:blue', lw=1.0, label='pred')
        if band is not None:
            ax.axvspan(1.0 / band[1] if band[1] else 0, 1.0 / band[0],
                       color='orange', alpha=.08)
        r = np.corrcoef(pred[i], target[i])[0, 1] if L > 1 else np.nan
        ax.set_title(f'{labels[row]}   Pearson {r:+.3f}', fontsize=9)
        ax.grid(alpha=.3)
        if row == 0:
            ax.legend(fontsize=8, loc='upper right')
        if row == n - 1:
            ax.set_xlabel('time (s)')

        ax = axes[row][1]
        f0, p0 = _psd(pred[i], fs)
        f1, p1 = _psd(target[i], fs)
        ax.semilogy(f1, p1, color='0.55', lw=1.0, label='target')
        ax.semilogy(f0, p0, color='tab:blue', lw=1.0, label='pred')
        fmax = 8.0 if fs >= 16 else fs / 2
        ax.set_xlim(0, fmax)
        if band is not None:
            ax.axvspan(band[0], band[1], color='orange', alpha=.10)
        ax.set_title('PSD', fontsize=9); ax.grid(alpha=.3, which='both')
        if row == n - 1:
            ax.set_xlabel('Hz')

    sub = title or 'predicted vs target'
    if metrics:
        bits = ', '.join(f'{k}={v:.3g}' for k, v in sorted(metrics.items())
                         if isinstance(v, (int, float)) and np.isfinite(v))
        if bits:
            sub += f'\n{bits}'
    if note:
        sub += f'\n[{note}]'
    fig.suptitle(sub, fontsize=11)
    return _finish(fig, out_png)


def plot_session_waveform(t, target, pred, out_png, title='', fs=100.0,
                          zoom_s=20.0, metrics=None):
    """Session-level assembled prediction vs the reference (both z-scored).

    Top panel = whole session with the per-clip windows marked; bottom = a
    ``zoom_s`` second zoom so the morphology is readable.
    """
    plt = _plt()
    t = np.asarray(t, float)
    y = np.asarray(target, float)
    yh = np.asarray(pred, float)
    # be defensive: only the COMMON span is plottable (the assembled prediction
    # covers the clip range, the reference is the whole session)
    n = int(min(t.size, y.size, yh.size))
    if n == 0:
        return None
    t, y, yh = t[:n], y[:n], yh[:n]

    def z(a):
        s = np.nanstd(a)
        return (a - np.nanmean(a)) / (s if s > 1e-12 else 1.0)

    y, yh = z(y), z(yh)
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.2), constrained_layout=True)
    axes[0].plot(t, y, color='0.55', lw=0.9, label='reference (z)')
    axes[0].plot(t, yh, color='tab:blue', lw=0.9, label='assembled pred (z)')
    axes[0].set_title('session assembly'); axes[0].grid(alpha=.3)
    axes[0].legend(fontsize=8)
    axes[0].set_xlabel('time (s)')

    dur = float(t[-1] - t[0]) if t.size > 1 else 0.0
    t0 = max(0.0, (dur - min(zoom_s, dur)) / 2.0)
    m = (t >= t0) & (t <= t0 + min(zoom_s, dur))
    axes[1].plot(t[m], y[m], color='0.55', lw=1.2, label='reference (z)')
    axes[1].plot(t[m], yh[m], color='tab:blue', lw=1.2,
                 label='assembled pred (z)')
    axes[1].set_xlim(t0, t0 + min(zoom_s, dur))
    axes[1].set_title(f'zoom ({min(zoom_s, dur):.0f} s)')
    axes[1].grid(alpha=.3); axes[1].set_xlabel('time (s)'); axes[1].legend(fontsize=8)

    sub = title or 'session waveform'
    if metrics:
        bits = ', '.join(f'{k}={v:.3g}' for k, v in sorted(metrics.items())
                         if isinstance(v, (int, float)) and np.isfinite(v))
        if bits:
            sub += f'\n{bits}'
    fig.suptitle(sub, fontsize=11)
    return _finish(fig, out_png)
