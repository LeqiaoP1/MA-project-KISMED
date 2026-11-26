"""Presentable figure: Mode-1 RESP reconstruction from a frozen Stage-2 checkpoint.

One figure, six panels, for a slide:

    left column   three example clips -- ground truth vs. the VIRTUAL-SENSING
                  prediction (K_resp = 0: no respiration token was visible, the
                  waveform must come from the thermal face alone)
    top right     mean power spectral density of prediction and target, with the
                  0.10-0.60 Hz respiratory passband shaded
    middle right  estimated vs. reference breath rate (bpm), identity line and
                  the one-FFT-bin band, so a systematic rate error is visible
    bottom right  per-subject correlation (dot per held-out subject) with the
                  pooled median and its bootstrap CI

The numbers printed are the ADR-008 Criterion-2 diagnostics; the figure is a
*view* of them, never a replacement (see
``code/notes/eval/baselines/tir_resp/mode1_pretext_fold0.md``).

    cd code && python notes/eval/baselines/tir_resp/figure_resp_reconstruction.py \
        --checkpoint EXP_ROOT/fold0/stage2/checkpoints/checkpoint-0149.pth \
        --out ../presentation/figures/resp-reconstruction-mode1-fold0.png
"""
import argparse
import os
import sys

import numpy as np


def _find_code_dir(start: str) -> str:
    """Walk up to the ``code/`` root (the directory holding ``runners/``).

    The script lives UNDER ``code/notes/...`` so a fixed ``dirname`` depth would
    break on the next relocation; the marker files make the lookup explicit.
    """
    d = os.path.abspath(start)
    while True:
        if (os.path.isdir(os.path.join(d, 'runners'))
                and os.path.isfile(os.path.join(d, 'pytest.ini'))):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(
                f'could not locate code/ (a directory with runners/ and '
                f'pytest.ini) above {start}')
        d = parent


CODE = _find_code_dir(os.path.dirname(os.path.abspath(__file__)))
if CODE not in sys.path:
    sys.path.insert(0, CODE)

import matplotlib                                              # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402
import torch                                                   # noqa: E402

from runners.run_eval_recon import (                           # noqa: E402
    _band_peak_hz, _held_out_subjects, build_eval_dataset, build_model,
    get_args, load_checkpoint, pin_masking)

#: respiratory passband of the ADR-008 objective and of the evaluation protocol
BAND = (0.10, 0.60)

#: house colours: target is neutral, prediction is the accent
C_TRUE = '#4d4d4d'
C_PRED = '#1f77b4'
C_BAND = '#1f77b4'
C_WARN = '#d62728'


# --------------------------------------------------------------------------- #
def collect(a, max_clips: int = 0):
    """Run the Mode-1 probe and return per-clip predicted/target waveforms."""
    dev = torch.device(a.device if (a.device != 'cuda'
                                    or torch.cuda.is_available()) else 'cpu')
    ckpt = load_checkpoint(a.checkpoint)
    cargs = ckpt['args']

    if not a.partition:
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(a.checkpoint)))
        for cand in (os.path.join(os.path.dirname(run_dir), 'partition.json'),
                     os.path.join(run_dir, 'partition.json')):
            if os.path.isfile(cand):
                a.partition = cand
                break

    model = build_model(ckpt, dev)
    plan, caption = pin_masking(model, a)
    print(f'[fig] {a.checkpoint} (epoch {ckpt.get("epoch")})')
    print(f'[fig] masking: {caption}')

    held, _train, _src = _held_out_subjects(cargs, a.partition)
    ds, idx = build_eval_dataset(cargs, a, held)
    if max_clips:
        step = max(1, len(idx) // int(max_clips))
        idx = idx[::step][:int(max_clips)]
    print(f'[fig] clips: {len(idx)} over '
          f'{len({ds.entries[i].get("subject") for i in idx})} subject(s)')

    fs = float(model.fs)
    rows = []
    B = int(a.batch_size)
    for start in range(0, len(idx), B):
        chunk = idx[start:start + B]
        items = [ds[i] for i in chunk]
        batch = {k: torch.stack([it[k] for it in items]).to(dev).float()
                 for k in items[0]}
        gen = torch.Generator().manual_seed(int(a.seed))
        with torch.inference_mode():
            out = model(batch, plan=plan, placement_generator=gen)
        pred = out['preds']['resp'].detach().float().cpu().numpy()
        tgt = batch['resp'].detach().float().cpu().numpy().reshape(
            pred.shape[0], -1, model.sig_kernel)
        for b, i in enumerate(chunk):
            p = pred[b].reshape(-1)
            t = tgt[b].reshape(-1)
            r = (float(np.corrcoef(p, t)[0, 1])
                 if p.std() > 0 and t.std() > 0 else float('nan'))
            rows.append({
                'subject': str(ds.entries[i].get('subject', '')),
                'session': str(ds.entries[i].get('session', '')),
                'task': str(ds.entries[i].get('task', '')),
                'pred': p, 'true': t, 'r': r, 'fs': fs,
                'pred_bpm': 60.0 * _band_peak_hz(p, fs, *BAND),
                'true_bpm': 60.0 * _band_peak_hz(t, fs, *BAND),
            })
    return rows, fs, model


def _example_indices(rows, n=3):
    """One clip near the 95th, 50th and 5th percentile of per-clip ``r``.

    A figure that only showed the best clips would misrepresent the arm; the
    spread is the finding. Distinct subjects are preferred so the panels do not
    all describe one face; when a target percentile collides with an already
    used subject the search walks outward in the SAME direction (never back to
    the opposite tail), so a "weak" slot cannot silently become a strong clip.
    """
    finite = [k for k, r in enumerate(rows) if np.isfinite(r['r'])]
    if not finite:
        return []
    order = sorted(finite, key=lambda k: rows[k]['r'])      # ascending r
    picks, seen = [], set()
    for frac in (0.95, 0.50, 0.05):
        k0 = int(round(frac * (len(order) - 1)))
        k0 = min(len(order) - 1, max(0, k0))
        for d in range(len(order)):
            k = k0 - d if frac < 0.5 else k0 + d
            if not (0 <= k < len(order)):
                continue
            if rows[order[k]]['subject'] not in seen:
                picks.append((order[k], k / max(1, len(order) - 1)))
                seen.add(rows[order[k]]['subject'])
                break
    return picks[:n]


def _psd(x: np.ndarray, fs: float):
    """Hann periodogram zero-padded to ``4 * 2**ceil(log2 N)``, in z²/Hz.

    Same construction as the ADR-008 spectral term, so one bin is
    ``fs / N_fft = 0.0305 Hz`` (1.83 bpm) -- the resolution the rate panel
    warns about.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    n = 4 * int(2 ** int(np.ceil(np.log2(max(x.size, 2)))))
    w = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * w, n=n)) ** 2
    spec = spec / (float(fs) * float(np.sum(w * w)))      # PSD units
    return np.fft.rfftfreq(n, d=1.0 / float(fs)), spec


def _style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=11)


# --------------------------------------------------------------------------- #
def draw(rows, out_path, title, foot):
    picks = _example_indices(rows, 3)
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(14.0, 8.6), dpi=200)
    gs = gridspec.GridSpec(3, 2, width_ratios=[1.45, 1.0], wspace=0.24,
                           hspace=0.55, left=0.065, right=0.975, top=0.88,
                           bottom=0.135)

    # ---- left column: three reconstructions ----------------------------- #
    for row, (pick, frac) in enumerate(picks):
        k, name = pick, ('strong held-out clip' if frac >= 0.75 else
                         'weak held-out clip' if frac <= 0.25 else
                         'median held-out clip')
        ax = fig.add_subplot(gs[row, 0])
        r = rows[k]
        fs = r['fs']
        t = np.arange(r['true'].size) / fs
        ax.plot(t, r['true'], color=C_TRUE, lw=2.6, alpha=0.55,
                label='contact sensor (target)')
        ax.plot(t, r['pred'], color=C_PRED, lw=1.7,
                label='virtual sensing ($K_{resp}=0$)')
        ax.axhline(0.0, color='k', lw=0.5, alpha=0.35)
        ax.set_xlim(0, t[-1])
        ax.set_ylabel('respiration\n(z-score)', fontsize=11)
        ax.set_title(f'{name}   ·   subject {r["subject"]}, task {r["task"]}   ·   '
                     f'r = {r["r"]:+.2f}', fontsize=12, loc='left')
        _style(ax)
        if row == 0:
            ax.plot([], [], color=C_TRUE, lw=2.6, alpha=0.55)
        if row == 2:
            ax.set_xlabel('time within the 8 s clip (s)', fontsize=11)

    # ---- top right: mean PSD -------------------------------------------- #
    ax = fig.add_subplot(gs[0, 1])
    ps, pt = [], []
    for r in rows:
        if not np.isfinite(r['r']):
            continue
        f, p = _psd(r['pred'], r['fs'])
        _, q = _psd(r['true'], r['fs'])
        ps.append(p)
        pt.append(q)
    keep = (f > 0.05) & (f <= 1.2)
    f = f[keep]
    P_t = np.mean(pt, axis=0)[keep]
    P_p = np.mean(ps, axis=0)[keep]
    ax.axvspan(*BAND, color=C_BAND, alpha=0.12, lw=0)
    ax.plot(f, P_t, color=C_TRUE, lw=2.6, label='target')
    ax.plot(f, P_p, color=C_PRED, lw=2.0, label='prediction')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('frequency (Hz, log)', fontsize=11)
    ax.set_ylabel('mean PSD (z²/Hz)', fontsize=11)
    ax.set_title('power spectrum, shaded = 0.10–0.60 Hz passband',
                 fontsize=12, loc='left')
    _style(ax)

    # ---- middle right: rate scatter ------------------------------------- #
    ax = fig.add_subplot(gs[1, 1])
    tb = np.array([r['true_bpm'] for r in rows], dtype=float)
    pb = np.array([r['pred_bpm'] for r in rows], dtype=float)
    ok = np.isfinite(tb) & np.isfinite(pb)
    one_bin = 60.0 * (1000.0 / 32768.0)             # 1.83 bpm at L = 8000
    lo, hi = max(0.0, np.nanmin(tb[ok]) - 6), np.nanmax(tb[ok]) + 6
    ax.plot([lo, hi], [lo, hi], color='k', lw=1.0, ls='--', alpha=0.7)
    ax.fill_between([lo, hi], [lo - one_bin, hi - one_bin],
                    [lo + one_bin, hi + one_bin], color=C_WARN, alpha=0.08,
                    lw=0, label=f'±1 FFT bin ({one_bin:.1f} bpm)')
    ax.scatter(tb[ok], pb[ok], s=11, color=C_PRED, alpha=0.45, lw=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('reference breath rate (bpm)', fontsize=11)
    ax.set_ylabel('estimated rate (bpm)', fontsize=11)
    ax.set_title('breath rate: per-clip estimate vs. reference',
                 fontsize=12, loc='left')
    ax.legend(fontsize=9, frameon=False, loc='upper left')
    _style(ax)

    # ---- bottom right: per-subject correlation --------------------------- #
    ax = fig.add_subplot(gs[2, 1])
    by_subj = {}
    for r in rows:
        if np.isfinite(r['r']):
            by_subj.setdefault(r['subject'], []).append(r['r'])
    subs = sorted(by_subj, key=lambda s: float(np.median(by_subj[s])))
    med = np.array([float(np.median(by_subj[s])) for s in subs])
    rng = np.random.default_rng(0)
    ax.scatter(med, np.arange(med.size) + rng.normal(0, 0.06, med.size),
               s=26, color=C_PRED, alpha=0.85, lw=0)
    pooled = float(np.median(med))
    rng2 = np.random.default_rng(7)
    boot = np.median(med[rng2.integers(0, med.size, (2000, med.size))], axis=1)
    ax.axvline(pooled, color='k', lw=1.4, label=f'median {pooled:+.2f}')
    ax.axvspan(np.percentile(boot, 2.5), np.percentile(boot, 97.5),
               color='k', alpha=0.08, lw=0,
               label=f'95 % CI [{np.percentile(boot, 2.5):+.2f}, '
                     f'{np.percentile(boot, 97.5):+.2f}]')
    ax.axvline(0.0, color=C_WARN, lw=1.0, ls=':')
    ax.set_yticks([])
    ax.set_xlabel('per-subject correlation r (median over clips)', fontsize=11)
    ax.set_title(f'{med.size} held-out subjects', fontsize=12, loc='left')
    ax.legend(fontsize=9, frameon=False, loc='lower right')
    _style(ax)

    fig.suptitle(title, fontsize=16, y=0.975, x=0.065, ha='left')
    handles = [plt.Line2D([], [], color=C_TRUE, lw=2.6, alpha=0.55),
               plt.Line2D([], [], color=C_PRED, lw=2.0)]
    fig.legend(handles, ['contact sensor (target)',
                         'virtual sensing ($K_{resp}=0$)'],
               loc='lower center', bbox_to_anchor=(0.5, -0.005), ncol=2,
               fontsize=12, frameon=False)
    fig.text(0.065, 0.045, foot, fontsize=9.5, color='#555555', ha='left')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor='white')
    print(f'[fig] wrote {out_path}')
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--partition', default='')
    p.add_argument('--out', default=os.path.join(
        '..', 'presentation', 'figures', 'resp-reconstruction-mode1.png'))
    p.add_argument('--max_clips', default=0, type=int,
                   help='0 = the whole held-out role (recommended for the '
                        'aggregate panels)')
    p.add_argument('--batch_size', default=16, type=int)
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--device', default='cuda', type=str)
    p.add_argument('--title', default='Virtual sensing of respiration from '
                                     'thermal face video (fold 0, held-out)')
    a = p.parse_args(argv)

    # `pin_masking` reads the eval-runner's own namespace: build it from the
    # same flags the published report used, so the figure cannot describe a
    # different masking than the number it illustrates.
    eval_args = get_args(['--checkpoint', a.checkpoint,
                          '--partition', a.partition,
                          '--split', 'test',
                          '--resp_probe', 'hidden',
                          '--kept_resp', '0.0', '--kept_tir', '1.0',
                          '--batch_size', str(a.batch_size),
                          '--seed', str(a.seed), '--device', a.device])
    rows, fs, model = collect(eval_args, max_clips=a.max_clips)
    finite = np.array([r['r'] for r in rows if np.isfinite(r['r'])])
    foot = (f'{len(rows)} clips from the fold-0 test role (subject-disjoint)  ·  '
            f'mode 1: K_resp = 0, K_tir = 200  ·  '
            f'per-clip r mean {finite.mean():+.3f}, median '
            f'{np.median(finite):+.3f}  ·  rates quantised by '
            f'Δf = 1000/32768 Hz = 1.8 bpm\n'
            f'ADR-008 arm (alpha/beta/gamma = 1/2/1, lambda_resp 5.0, '
            f'lambda_align 0.5), epoch 149  ·  '
            f'{os.path.basename(os.path.dirname(os.path.dirname(a.checkpoint)))}'
            f'  ·  reproduced by scripts/project/figure_resp_reconstruction.py')
    return 0 if draw(rows, a.out, a.title, foot) else 1


if __name__ == '__main__':
    raise SystemExit(main())
