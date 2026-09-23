"""Session-level assembly of per-clip Stage-3 predictions.

Stage 3 predicts one fixed-length window per clip (``output_len`` samples,
covering exactly the clip's own time span). To report clinical metrics the
whole session has to be reconstructed, which is a training-free post-processing
step: overlap-add the per-clip windows on the session timeline.

Two details that matter:

* **Offsets come from the dataset entries** (``t_start`` in seconds), and the
  1-D target is sampled at ``fs`` on the *original* window regardless of
  ``temporal_stride`` (only the VIDEO is decimated), so the offset is
  ``round(t_start * fs)`` -- NOT ``t_start * fs / temporal_stride``.
* **Normalisation.** With ``signal_norm: zscore`` each clip prediction is only
  defined up to a per-clip affine map. The assembled waveform is therefore
  meaningful for *affine-invariant* metrics (Pearson, PSD shape, RR intervals);
  absolute-amplitude metrics need the per-session affine calibration in
  ``affine_calibrate``.

Hann windows with weight-sum normalisation are used instead of cropping
``D - stride`` per clip, so overlapping windows cross-fade and no sample is
dropped at the session edges.
"""
import numpy as np

__all__ = ['clip_offsets', 'group_entries', 'assemble_session',
           'affine_calibrate']


def clip_offsets(t_starts, fs):
    """Clip start times (seconds) -> sample offsets on the session timeline."""
    return np.round(np.asarray(t_starts, dtype=float) * float(fs)).astype(int)


def group_entries(entries, fs):
    """``[{session, t_start}, ...]`` -> {session: [(row_index, offset), ...]}.

    Preserves the row order of ``entries`` (row *i* is the prediction matrix's
    row *i*), and sorts each session's clips by time.
    """
    out = {}
    for i, e in enumerate(entries):
        s = e['session'] if isinstance(e, dict) else e[0]
        t = e['t_start'] if isinstance(e, dict) else e[1]
        out.setdefault(s, []).append((i, int(clip_offsets([t], fs)[0])))
    for s in out:
        out[s].sort(key=lambda p: p[1])
    return out


def assemble_session(preds, offsets, fs, window=('hann', 'ones'), eps=1e-8):
    """Overlap-add per-clip windows onto one session timeline.

    :param preds: ``[N, L]`` waveforms (one row per clip, same L)
    :param offsets: ``[N]`` sample offsets (see :func:`clip_offsets`)
    :param fs: sampling rate of the waveforms (Hz)
    :param window: ``'hann'`` (default) or ``'ones'``
    :returns: ``(t, y)`` with ``t`` in seconds and ``y`` the assembled waveform
    """
    preds = np.atleast_2d(np.asarray(preds, dtype=float))
    offsets = np.asarray(offsets, dtype=int)
    if preds.shape[0] != offsets.size:
        raise ValueError(f'assemble_session: {preds.shape[0]} predictions but '
                         f'{offsets.size} offsets')
    if preds.shape[0] == 0:
        return np.zeros(0), np.zeros(0)
    L = preds.shape[1]
    n = int(offsets.max()) + L
    if isinstance(window, (tuple, list)):
        window = window[0]
    w = np.hanning(L) if str(window).lower().startswith('hann') else np.ones(L)
    acc = np.zeros(n, dtype=float)
    wsum = np.zeros(n, dtype=float)
    for p, o in zip(preds, offsets):
        acc[o:o + L] += w * p
        wsum[o:o + L] += w
    y = acc / np.maximum(wsum, eps)
    return np.arange(n) / float(fs), y


def affine_calibrate(pred, ref):
    """Least-squares ``a, b`` minimising ``||a*pred + b - ref||`` (+ the fit).

    Returns ``(a, b, fitted)``; ``fitted`` is NaNs when the fit is degenerate.
    Used only for amplitude metrics (MAE/RMSE) on assembled, per-clip z-scored
    predictions -- Pearson/PSD/RR intervals are affine-invariant already.
    """
    p = np.asarray(pred, float)
    r = np.asarray(ref, float)
    n = min(p.size, r.size)
    p, r = p[:n], r[:n]
    ok = np.isfinite(p) & np.isfinite(r)
    if ok.sum() < 3:
        return np.nan, np.nan, np.full(n, np.nan)
    p, r = p[ok], r[ok]
    den = float(np.dot(p - p.mean(), p - p.mean()))
    if den < 1e-12:
        return np.nan, np.nan, np.full(n, np.nan)
    a = float(np.dot(p - p.mean(), r - r.mean()) / den)
    b = float(r.mean() - a * p.mean())
    return a, b, a * np.asarray(pred, float) + b
