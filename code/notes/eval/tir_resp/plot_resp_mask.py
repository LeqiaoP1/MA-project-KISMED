"""Plot a cached respiration session coloured by its validity mask.

Green = samples the artefact says are usable (``mask`` True). Grey = samples it
says are NOT (``mask`` False; the stored signal is ``NaN`` there, so no value is
invented for them -- the grey is a SPAN plus a per-sample ribbon at the bottom).

Contract: ``code/notes/design/RESP_session_preprocess.md`` §3.

Usage (from anywhere)::

    python code/notes/eval/tir_resp/plot_resp_mask.py --session M033_T9 \
        --resp_cache "$WORK_SCRATCH/resp_cache" \
        --out output/eval/resp_mask/M033_T9_resp_mask.png

    # pick the zoom window explicitly (seconds)
    ... --zoom 60,76
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

_CODE_DIR = Path(__file__).resolve().parents[3]      # code/
sys.path.insert(0, str(_CODE_DIR))

import matplotlib                                      # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402
from matplotlib.patches import Patch                   # noqa: E402

from data import resp_cache as rc                      # noqa: E402
from data import resp_preprocess as rpp                # noqa: E402
from data import tir_resp_dataset as trd               # noqa: E402

VALID_C = '#2e7d32'
INVALID_C = '#9e9e9e'
INVALID_BAND = '#e0e0e0'
RAW_VALID_C = '#bdbdbd'
RAW_BAD_C = '#3d3d3d'
GRAZE_C = '#ef6c00'
RAIL_C = '#c62828'


def find_cache_root(out_root: str, fs_out: float, session: str,
                    cache_key: str = '') -> Path:
    key = str(cache_key or '').strip()
    if key:
        root = Path(out_root) / key
        if not (root / rc.PARAMS_NAME).is_file():
            raise SystemExit(f'[mask-plot] no params.json under {root}')
        return root
    return Path(rc.find_cache(str(out_root), rc.eval_params(fs_out=float(fs_out)),
                              keys=rc.EVAL_COMPARE_KEYS,
                              require_session=str(session)))


def resp_runs(mask: np.ndarray, cause: np.ndarray, fs: float) -> list:
    """``[(t0, t1, 'valid'|'INVALID', cause_label)]`` on the output grid.

    One invalid STRETCH can mix causes (e.g. a dead channel that produces both
    flatline windows and sub-8 s fragments), so the label names the DOMINANT
    cause by sample count and appends the others -- reporting only the first
    sample's cause made a 72 s mixed hole look like pure ``short_run``.
    """
    code_of = {name: code for code, name in rpp.CAUSE_NAMES.items()}
    out = [(s / fs, e / fs, 'valid', '')
           for s, e in rpp.contiguous_runs(np.asarray(mask, dtype=bool))]
    for s, e in rpp.contiguous_runs(~np.asarray(mask, dtype=bool)):
        c = np.asarray(cause)[s:e]
        counts = {name: int((c == code).sum())
                  for name, code in code_of.items()
                  if code != rpp.CAUSE_VALID and (c == code).any()}
        if not counts:
            label = 'unknown'
        else:
            dom = max(counts, key=counts.get)
            rest = [n for n in counts if n != dom]
            label = dom + (('+' + '+'.join(rest)) if rest else '')
        out.append((s / fs, e / fs, 'INVALID', label))
    return sorted(out)


def _default_zoom(runs: list, total: float) -> tuple:
    bad = [(t1 - t0, t0, t1) for t0, t1, kind, _ in runs if kind == 'INVALID']
    if not bad:
        mid = total / 2.0
        return max(0.0, mid - 10.0), min(total, mid + 10.0)
    _, t0, t1 = max(bad)                       # widest hole
    pad = max(2.0, 2.0 * (t1 - t0))
    return max(0.0, t0 - pad), min(total, t1 + pad)


def draw_grid(ax, t: np.ndarray, sig: np.ndarray, mask: np.ndarray, runs: list,
              label_min_s: float, title: str) -> None:
    """Green waveform on valid spans, grey bands + a per-sample ribbon otherwise."""
    finite = sig[np.isfinite(sig)]
    lo = float(np.min(finite)) if finite.size else -1.0
    hi = float(np.max(finite)) if finite.size else 1.0
    span = max(hi - lo, 1e-6)
    ax.set_ylim(lo - 0.30 * span, hi + 0.18 * span)

    for t0, t1, kind, name in runs:
        if kind != 'INVALID':
            continue
        ax.axvspan(t0, t1, color=INVALID_BAND, alpha=0.85, lw=0, zorder=0)
        if (t1 - t0) >= label_min_s:
            # INSIDE the axes (va='top'), or the label lands on the panel title
            ax.text(0.5 * (t0 + t1), hi + 0.14 * span,
                    f'{name} · {t1 - t0:.1f} s', ha='center', va='top',
                    fontsize=7, color='#555555', zorder=5, clip_on=False)

    # the waveform itself: only where the artefact has a value (invalid = NaN)
    ax.plot(t, np.where(mask, sig, np.nan), color=VALID_C, lw=0.9, zorder=3,
            label='valid (mask True)')
    # per-sample validity ribbon, so a 1-sample break is still visible
    base = lo - 0.26 * span
    thick = 0.10 * span
    ax.fill_between(t, base, base + thick, where=mask, step='mid',
                    color=VALID_C, lw=0, zorder=4)
    ax.fill_between(t, base, base + thick, where=~mask, step='mid',
                    color=INVALID_C, lw=0, zorder=4)

    ax.set_ylabel('resp (V, run-mean removed)')
    ax.set_title(title, fontsize=10, loc='left')
    ax.grid(alpha=0.2, lw=0.5)


def draw_raw(ax, tr: np.ndarray, raw: np.ndarray, bad: np.ndarray,
             runs: list, sat_v: float, title: str,
             sat_min_s: float = 0.0) -> None:
    """The 1 kHz samples behind the grid, split into kept / KEPT-AT-RAIL / rejected.

    This is how an "invalid sample" is made visible: the artefact stores no
    value for it, but the raw file does -- light grey where the grid keeps the
    sample, dark where it rejects it, and ORANGE where it keeps a sample that
    sits at the rail. The last case is deliberate (design C1): a rail contact
    shorter than ``sat_min_s`` is not a saturation EVENT, so it stays valid --
    which is why the valid region of some sessions still contains -10 V samples.
    """
    for t0, t1, kind, _ in runs:
        if kind == 'INVALID':
            ax.axvspan(t0, t1, color=INVALID_BAND, alpha=0.85, lw=0, zorder=0)

    rail = (np.abs(raw) >= sat_v) if (sat_v and sat_v > 0) else np.zeros(
        raw.size, dtype=bool)
    finite = np.isfinite(raw)
    kept = ~bad & finite
    rej = bad & finite
    # NaN-BREAK each series: matplotlib connects consecutive points, so plotting
    # raw[rej] directly drew a straight line across the VALID region between two
    # disjoint holes (it looked like "rejected samples everywhere").
    # kept first (it is the dense one); alpha low so 1 kHz overplotting does not
    # read as "dark = rejected"
    ax.plot(tr, np.where(kept & ~rail, raw, np.nan), color=RAW_VALID_C, lw=0.4,
            alpha=0.8, zorder=2, label='raw sample kept')
    ax.plot(tr, np.where(kept & rail, raw, np.nan), color=GRAZE_C, lw=0.6,
            alpha=0.95, zorder=3,
            label=f'kept AT THE RAIL (graze < {sat_min_s:g} s, design C1)')
    ax.plot(tr, np.where(rej, raw, np.nan), color=RAW_BAD_C, lw=0.85, zorder=4,
            label='raw sample REJECTED (mask False)')
    if sat_v and sat_v > 0:
        for sgn in (-1.0, 1.0):
            ax.axhline(sgn * sat_v, color=RAIL_C, ls='--', lw=0.7, zorder=1)
        ax.text(0.002, sat_v, f' rail ±{sat_v:g} V', color=RAIL_C, fontsize=7,
                va='bottom', ha='left',
                transform=ax.get_yaxis_transform())

    ax.set_ylabel('raw Resp_Volts (V)')
    ax.set_title(title, fontsize=10, loc='left')
    ax.grid(alpha=0.2, lw=0.5)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', required=True, help='<SUBJECT>_<TASK>, e.g. M033_T9')
    p.add_argument('--resp_cache', default=os.environ.get('RESP_CACHE', ''),
                   help='cache OUT_ROOT (default $RESP_CACHE)')
    p.add_argument('--resp_cache_key', default='',
                   help='pin the cache DIRECTORY name instead of resolving it')
    p.add_argument('--fs_out', type=float, default=rpp.FS_OUT)
    p.add_argument('--zoom', default='',
                   help="zoom window 't0,t1' in seconds (default: widest hole)")
    p.add_argument('--no-raw', dest='raw', action='store_false',
                   help='omit the raw 1 kHz row(s) (kept vs rejected samples)')
    p.add_argument('--out', default='')
    p.add_argument('--dpi', type=int, default=130)
    args = p.parse_args(argv)

    if not args.resp_cache:
        raise SystemExit('[mask-plot] --resp_cache (or $RESP_CACHE) is required')
    root = find_cache_root(args.resp_cache, args.fs_out, args.session,
                           args.resp_cache_key)
    params = rc.read_params(root)
    sh = rc.read_shard(rc.shard_path(root, args.session))
    sig = np.asarray(sh['signal'], dtype=float)
    mask = np.asarray(sh['mask'], dtype=bool)
    cause = np.asarray(sh['cause'])
    fs = float(sh['fs_out'])
    fs_in = float(sh['fs_in'])
    t = np.arange(sig.size, dtype=float) / fs
    runs = resp_runs(mask, cause, fs)
    total = float(sig.size) / fs
    if str(args.zoom).strip():
        z0, z1 = (float(v) for v in str(args.zoom).split(','))
    else:
        z0, z1 = _default_zoom(runs, total)

    # the raw series behind the grid: which 1 kHz samples the grid REJECTS
    tr = raw = bad = None
    resp_file = str(sh.get('resp_file') or '')
    if args.raw:
        if resp_file and os.path.isfile(resp_file):
            raw = np.asarray(trd._load_1d(resp_file), dtype=float)
            tr = np.arange(raw.size, dtype=float) / fs_in
            k = np.minimum((np.arange(raw.size) * fs) // fs_in,
                           max(mask.size - 1, 0)).astype(np.int64)
            bad = ~mask[k] if mask.size else np.ones(raw.size, bool)
        else:
            print(f'[mask-plot] no readable raw file ({resp_file!r}); '
                  f'drawing the grid only')

    n_bad = sum(1 for r in runs if r[2] == 'INVALID')
    print(f'[mask-plot] {args.session}: {sig.size} samples = {total:.2f} s, '
          f'valid_frac {float(mask.mean()):.3f}, {n_bad} invalid stretch(es)')
    for t0, t1, kind, name in runs:
        print(f'    {t0:8.2f} -> {t1:8.2f} s  {t1 - t0:7.2f} s  '
              f'{"INVALID " + name.upper() if kind == "INVALID" else "valid"}')
    if raw is not None:
        railed = np.abs(raw) >= float(params['sat_v'])
        print(f'[mask-plot] raw {raw.size} samples @ {fs_in:g} Hz; '
              f'{int((bad & np.isfinite(raw)).sum())} rejected, '
              f'{int(((~bad) & railed & np.isfinite(raw)).sum())} KEPT at the '
              f'rail (short grazes, design C1)')
        for a, b in [(t0, t1) for t0, t1, kind, _ in runs if kind == 'INVALID']:
            sel = bad & (tr >= a) & (tr < b) & np.isfinite(raw)
            if sel.any():
                print(f'    span {a:7.2f}-{b:7.2f} s: '
                      f'{100.0 * float(np.mean(railed[sel])):.1f}% of its raw '
                      f'samples sit at the rail')

    zoom_runs = [r for r in runs if r[1] >= z0 and r[0] <= z1]
    if raw is not None:
        fig, axes = plt.subplots(
            2, 2, figsize=(15.0, 8.4), sharex='col',
            gridspec_kw={'width_ratios': [1.15, 1.0],
                         'height_ratios': [1.0, 0.72]})
        ax_g_full, ax_g_zoom = axes[0, 0], axes[0, 1]
        ax_r_full, ax_r_zoom = axes[1, 0], axes[1, 1]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(14.0, 7.6))
        ax_g_full, ax_g_zoom = axes[0], axes[1]
        ax_r_full = ax_r_zoom = None

    draw_grid(ax_g_full, t, sig, mask, runs, label_min_s=1.0,
              title=f'{args.session} — whole session: grid {fs:g} Hz '
                    f'({total:.1f} s, valid_frac {mask.mean():.3f})')
    ax_g_full.set_xlim(0.0, total)
    sel = (t >= z0) & (t <= z1)
    draw_grid(ax_g_zoom, t[sel], sig[sel], mask[sel], zoom_runs,
              label_min_s=0.0,
              title=f'zoom {z0:.1f}–{z1:.1f} s — grey band = mask hole')
    ax_g_zoom.set_xlim(z0, z1)

    if raw is not None:
        draw_raw(ax_r_full, tr, raw, bad, runs, float(params['sat_v']),
                 title=f'{args.session} — whole session: raw {fs_in:g} Hz '
                       f'(dark = samples the grid REJECTS)',
                 sat_min_s=float(params['sat_min_s']))
        ax_r_full.set_xlim(0.0, total)
        rsel = (tr >= z0) & (tr <= z1)
        draw_raw(ax_r_zoom, tr[rsel], raw[rsel], bad[rsel], zoom_runs,
                 float(params['sat_v']),
                 title=f'zoom {z0:.1f}–{z1:.1f} s — raw {fs_in:g} Hz',
                 sat_min_s=float(params['sat_min_s']))
        ax_r_zoom.set_xlim(z0, z1)
        ax_r_full.set_xlabel('session time (s)')
        ax_r_zoom.set_xlabel('session time (s)')
    else:
        ax_g_zoom.set_xlabel('session time (s)')

    key = root.name
    fig.suptitle(f'{args.session} · respiration validity mask · {key}',
                 fontsize=9, y=0.995)
    handles = [
        Line2D([], [], color=VALID_C, lw=1.4, label='grid sample valid'),
        Patch(facecolor=INVALID_BAND, edgecolor=INVALID_C,
              label='grid sample invalid (value is NaN)'),
    ]
    if raw is not None:
        handles += [
            Line2D([], [], color=RAW_VALID_C, lw=1.2, label='raw sample kept'),
            Line2D([], [], color=GRAZE_C, lw=1.2,
                   label=f'raw kept AT THE RAIL (graze < '
                         f'{float(params["sat_min_s"]):g} s, design C1)'),
            Line2D([], [], color=RAW_BAD_C, lw=1.4,
                   label='raw sample REJECTED'),
            Line2D([], [], color=RAIL_C, ls='--', lw=1.0,
                   label=f'rail threshold ±{float(params["sat_v"]):g} V'),
        ]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.978))
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out = args.out or os.path.join(
        'output', 'eval', 'resp_mask', f'{args.session}_resp_mask.png')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f'[mask-plot] wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
