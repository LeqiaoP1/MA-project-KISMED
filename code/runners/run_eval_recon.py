"""Zero-shot RESPIRATION reconstruction from a FROZEN Stage-2 checkpoint.

No fine-tuning, no Stage-3: the checkpoint is used as it was saved (shared
encoder + per-stream decoders), the ``resp`` stream is masked with a PINNED
allocation of the constant budget, and the reconstruction is scored on the
MASKED positions against the dataset's own respiration target (the ROI lineage
z-scores each clip, so the target and the prediction live in the same per-clip
z space and Pearson is scale-invariant).

WHY THIS EXISTS -- the RGB2D question
-------------------------------------
"Does the extra 2D-texture stream improve respiration reconstruction?" is
answered with two complementary, controlled comparisons:

1. WITHIN-MODEL (paired).  Same weights, same clips, same seed; only the
   per-stream kept TOKENS change. ADR-002 fixes the total budget and the visual
   floor, so the smallest rgb2d allocation the policy can express is
   ``video_min_tubes`` (the floor exists so the encoder always has visual
   evidence); a physiological stream CAN be pinned to exactly 0, which is what
   makes the pure-virtual-sensing probe possible.
       --checkpoint <ckpt> --kept_rgb2d 0.50 --label rgb2d_half
       --checkpoint <ckpt> --kept_rgb2d 0.05 --label rgb2d_floor

2. ACROSS-MODEL (different training runs, same evaluation set).
   The tri-modal checkpoint vs the TIR+RESP twin checkpoint. ``--data_checkpoint``
   pins the DATASET (and therefore the clip set) to one checkpoint's arguments,
   so both arms score exactly the same clips; the 2-stream model simply ignores
   the extra ``rgb2d`` key of the batch.
       --checkpoint .../stage2_budget_tir_roi_resp_rgb2d_fold0/checkpoints/checkpoint-0149.pth
       --checkpoint .../stage2_hpc_tir_roi_resp_fold0/checkpoints/checkpoint-0149.pth \
           --data_checkpoint .../stage2_budget_tir_roi_resp_rgb2d_fold0/checkpoints/checkpoint-0149.pth

Metrics (per clip, then aggregated over clips; each clip is scored over
``--repeats`` independent mask draws and averaged FIRST, so a clip is never
weighted more than another):

* ``mse_masked``  -- masked-position MSE, the training objective (per-sample)
* ``r_masked``    -- Pearson over the MASKED samples only. The honest number:
                     those samples are never visible to the model.
* ``r_full``      -- Pearson over the assembled waveform (prediction at masked
                     positions, ground truth at visible ones). This is what a
                     downstream user sees, but it is inflated by the visible
                     part, so it is NOT the headline.
* ``mae``/``rmse`` -- masked-position error, z units.

Usage (from ``code/``)::

    python -u runners/run_eval_recon.py --checkpoint <ckpt> --device cuda \
        --label with_rgb2d --out out.json \\
        --kept_tir 0.10 --kept_rgb2d 0.50 --kept_resp 0.0 --repeats 5
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                      # noqa: E402
import torch                                            # noqa: E402
from core.multimae import build_pretraining_model        # noqa: E402
from runners.run_visualize_pretrain import _held_out_subjects  # noqa: E402


# --------------------------------------------------------------------------- #
# args
# --------------------------------------------------------------------------- #
def get_args(argv=None):
    p = argparse.ArgumentParser(
        description='Zero-shot RESP reconstruction from a frozen Stage-2 '
                    'checkpoint (no fine-tuning).')
    p.add_argument('--checkpoint', required=True,
                   help='Stage-2 checkpoint whose WEIGHTS are evaluated')
    p.add_argument('--data_checkpoint', default='',
                   help='optional: build the dataset (and thus the CLIP SET) '
                        'from THIS checkpoint\'s args instead. Use it to make '
                        'two different checkpoints score identical clips.')
    p.add_argument('--split', default='test', choices=('test', 'corpus'),
                   help="'test' (default) = the weight checkpoint's held-out "
                        'fold subjects')
    p.add_argument('--partition', default='',
                   help='the fold partition record (<exp_root>/fold<k>/'
                        'partition.json); default: discovered above the '
                        'checkpoint run dir')
    p.add_argument('--label', default='', help='name used in the report')
    p.add_argument('--out', default='', help='JSON report path (default: none)')
    # ---- masking of the evaluation clips (pinned, NOT the training policy) --
    # AMMM kept-ratio overrides: the ONLY mechanism that can mask a stream to
    # ZERO kept units (the ratio path clamps to n_units - 1, so 'mask 1.0' still
    # leaks one patch / one token). All streams are pinned, so no Dirichlet draw
    # happens and the masks of the OTHER streams are bit-identical between two
    # arms that differ only in one stream's value -> an exactly paired ablation.
    p.add_argument('--kept_tir', default=0.10, type=float,
                   help='fraction of the 36 TIR patches kept (0.10 == the '
                        'training mask ratio 0.90)')
    p.add_argument('--kept_resp', default=0.0, type=float,
                   help='fraction of the 50 resp tokens kept. 0.0 (default) = '
                        'the waveform is ENTIRELY hidden, so the reconstruction '
                        'can only come from the visual streams (the cross-modal '
                        'question); 0.5 reproduces the training draw and is '
                        'locally interpolatable (r~0.998) -- a reference only.')
    p.add_argument('--kept_rgb2d', default=0.50, type=float,
                   help='fraction of the 36 rgb2d patches kept. 0.0 = the 2D '
                        'stream contributes NO encoder token (the ablation). '
                        'Inert for a 2-stream checkpoint.')
    p.add_argument('--pin_single_carrier', default='none', type=str,
                   choices=('none', 'tir', 'rgb2d'),
                   help='ADR-006 §2.4 Regime 3: pin ONE visual stream to the '
                        'whole budget and every other stream to 0, bypassing '
                        'the 2-tube floor and the 25 % cap (which would '
                        'otherwise realise 225/125 and make the TIR-only / '
                        'RGB2D-only probe inexpressible). none (default) = '
                        'the clamped split of the --kept_* fractions. On a '
                        'ONE-visual checkpoint it is only meaningful as that '
                        'sole stream, where it is exactly Mode 1.')
    p.add_argument('--pin_core_tubes', default=-1, type=int,
                   help='ADR-007 §2.3: how many of the pinned kept tubes come '
                        'from the anatomical core A_core. -1 (default) = unset, '
                        'which lets an anatomy_aware checkpoint use its own '
                        'core_tubes_min and keeps a uniform checkpoint '
                        'core-free; >= 0 is forwarded explicitly and RAISES on a '
                        'uniform checkpoint (its core count would be '
                        'fabricated).')
    p.add_argument('--kept_bp', default=1.0, type=float,
                   help='fraction of the 250 bp tokens kept. 1.0 (default) '
                        'reproduces the legacy rule (a declared stream with no '
                        'flag keeps all, clamped to the 125-token physio cap); '
                        '0.0 gives BP no encoder token, so it cannot consume '
                        'the visual budget of a 4-stream checkpoint (P0-E). '
                        'Inert for checkpoints without a bp stream.')
    p.add_argument('--resp_probe', default='span',
                   choices=('span', 'random', 'hidden'),
                   help="how the waveform is hidden. 'random' keeps "
                        '--kept_resp scattered (the training condition: locally '
                        'interpolatable, r~0.998); \'span\' keeps the same '
                        'budget but as ONE contiguous block (not '
                        "interpolatable -- the informative probe); 'hidden' "
                        'masks it completely (no cross-modal prior at all).')
    p.add_argument('--span_seconds', default=4.0, type=float,
                   help='--resp_probe span: block length in seconds (converted '
                        'to whole signal tokens with the checkpoint geometry)')
    # ---- evaluation loop ---------------------------------------------------
    p.add_argument('--repeats', default=5, type=int,
                   help='independent mask draws per clip; the masks are seeded '
                        'as seed+r so two runs with the same seed see the SAME '
                        'masks (the paired ablation relies on this)')
    p.add_argument('--batch_size', default=16, type=int)
    p.add_argument('--preload', action='store_true', default=False,
                   help='decode every evaluation clip ONCE and reuse it across '
                        'the mask repeats. Same numbers (the batch is copied to '
                        'the device per step); costs ~10-20 MB of host RAM per '
                        'clip. Without it every repeat re-reads the caches.')
    p.add_argument('--max_clips', default=0, type=int,
                   help='cap the number of evaluated clips (0 = all); the '
                        'subset is drawn with --seed so it is reproducible')
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--subjects', default='', type=str,
                   help='comma-separated subject ids to restrict the probe to. '
                        'They must lie inside the evaluated role; the test '
                        'block is subject-disjoint by construction (ADR-001), '
                        "so naming a TRAIN subject here is refused rather "
                        "than silently scored.")
    # ---- input ablation (presence/absence of a visual stream's CONTENT) ----
    # The budget policy cannot express a zero-token visual stream (the floor is
    # mandatory), so presence/absence is tested by corrupting the stream's input
    # while keeping its token count, the masked plan and the placement seeds
    # IDENTICAL. 'roll' circularly shifts the time axis (keeps every frame and
    # the frame statistics, destroys alignment with respiration); 'zero' blanks
    # the stream. The same flags on TIR are the positive control that the probe
    # is sensitive to a visual stream's content.
    p.add_argument('--ablate_stream', default='', type=str,
                   help='visual stream whose INPUT is corrupted (default: none)')
    p.add_argument('--ablate_mode', default='roll', choices=('roll', 'zero'),
                   help="'roll' = circular time shift; 'zero' = blank frames")
    p.add_argument('--ablate_shift', default=0, type=int,
                   help='frames to roll; 0 = max(1, T//3)')
    p.add_argument('--device', default='cpu', type=str)
    p.add_argument('--pretext_metrics', action='store_true', default=False,
                   help='ADR-008 §3.2 pretext diagnostics: also score VARIANCE '
                        'RETENTION (Var(ŷ) over the masked samples of a clip, '
                        'z² units) and the SPECTRAL PEAK ERROR of ŷ vs y inside '
                        '--pretext_band (bpm). OFF by default, so every '
                        'pre-existing report is byte-identical.')
    p.add_argument('--pretext_band', default='0.10,0.60', type=str,
                   help='--pretext_metrics passband in Hz as "f_min,f_max" '
                        '(default 0.10,0.60 = the ADR-008 §2.2.3 respiratory '
                        'band, 6..36 bpm).')
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #
def load_checkpoint(path: str) -> Dict:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('model', 'args'):
        if key not in ckpt:
            raise SystemExit(f'{path}: not a Stage-2 checkpoint (no {key!r})')
    return ckpt


def build_model(ckpt: Dict, device: torch.device):
    cargs = ckpt['args']
    model = build_pretraining_model(cargs)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    if missing or unexpected:
        # A Stage-2 checkpoint always carries every encoder/decoder tensor; a
        # mismatch means the architecture changed under the weights.
        print(f'[eval] WARNING state_dict: {len(missing)} missing / '
              f'{len(unexpected)} unexpected')
        if missing:
            print('[eval]   missing:', missing[:5])
        if unexpected:
            print('[eval]   unexpected:', unexpected[:5])
    return model.to(device).eval()


def ablate_batch(batch: Dict, model, stream: str, mode: str, shift: int):
    """Corrupt ONE visual stream's input; return ``(batch, used_shift)``.

    Only the content changes: the tensor shape, the per-stream token count, the
    masked plan and the placement seeds are untouched, so an intact/ablated pair
    with the same ``--seed`` is exactly paired.
    """
    if stream not in model.visual:
        raise SystemExit(f'[eval] --ablate_stream {stream!r} is not a visual '
                         f'stream of this checkpoint (visual={model.visual})')
    x = batch[stream]
    if mode == 'zero':
        used = -1
        x = torch.zeros_like(x)
    else:
        t = int(x.shape[2])
        used = int(shift) or max(1, t // 3)
        x = torch.roll(x, shifts=used, dims=2)
    out = dict(batch)
    out[stream] = x
    return out, used


def pin_masking(model, a):
    """State the kept-token COUNTS of one reconstruction probe; return a caption.

    ADR-002 replaces the old per-stream ratio pins with ONE budget: the visible
    tokens of all streams must sum to ``model.n_budget`` and every visual stream
    keeps at least ``video_min_tubes``. The ``--kept_*`` fractions are therefore
    a REQUEST: they are converted to whole counts (visual streams to whole
    tubes, each clamped into its floor and cap) and whatever the budget still
    needs is added to, or taken from, the visual streams in declared order.
    A stream's floor is mandatory -- the policy cannot express "0 visual tubes",
    because the floor exists so the encoder always has visual evidence -- while
    a physiological stream CAN be pinned to exactly 0 (that is Mode 1).

    :return: ``(plan, caption)``; the realised counts are what the caption and
        the report record, never the requested fractions.
    """
    want = {}
    for s in model.streams:
        if s == 'tir':
            want[s] = float(a.kept_tir)
        elif s == 'rgb2d':
            want[s] = float(a.kept_rgb2d)
        elif s == 'resp':
            want[s] = 0.0 if a.resp_probe == 'hidden' else float(a.kept_resp)
        elif s == 'bp':
            want[s] = float(a.kept_bp)
        else:                      # a declared stream with no flag: keep it all
            want[s] = 1.0
        if not 0.0 <= want[s] <= 1.0:
            raise SystemExit(f'--kept_{s} must be in [0, 1]; got {want[s]}')

    app = model.apportioner
    floor = model.video_min_tubes
    counts: Dict[str, int] = {}
    pin_carrier = str(getattr(a, 'pin_single_carrier', 'none') or 'none')
    if pin_carrier != 'none':
        # ADR-006 §2.3 contract 3 / §2.4: one stream carries the WHOLE budget
        # and every other stream keeps nothing. The floor/cap clamp below is
        # bypassed on purpose -- it exists so a SPLIT always leaves visual
        # evidence, and a sole carrier already satisfies that.
        if pin_carrier not in model.visual:
            raise SystemExit(
                f'[eval] --pin_single_carrier {pin_carrier!r} is not a visual '
                f'stream of this checkpoint (visual={model.visual})')
        counts = {spec.name: (int(model.n_budget)
                              if spec.name == pin_carrier else 0)
                  for spec in model.specs}
    else:
        for spec in model.specs:
            k = int(round(want[spec.name] * spec.n_units))
            if spec.kind == 'visual':
                k = max(floor, min(k, app.video_cap_units[spec.name]))
                counts[spec.name] = k * int(spec.tokens_per_unit)
            else:
                counts[spec.name] = max(0, min(k, app.physio_cap_tokens[spec.name]))

    residual = int(model.n_budget) - sum(counts.values())
    for spec in model.specs:
        if residual == 0 or spec.kind != 'visual':
            continue
        lo = floor * int(spec.tokens_per_unit)
        hi = app.video_cap_units[spec.name] * int(spec.tokens_per_unit)
        room = hi - counts[spec.name] if residual > 0 \
            else counts[spec.name] - lo
        take = min(room, abs(residual))
        counts[spec.name] += take if residual > 0 else -take
        residual += -take if residual > 0 else take
    if residual:
        raise SystemExit(
            f'[eval] the requested masking cannot reach the constant budget '
            f'{model.n_budget}: {residual:+d} token(s) left after filling every '
            f'visual stream to its cap/floor ({counts}). Adjust --kept_* or '
            f'evaluate a checkpoint with another --n_budget.')
    plan = model.plan_from_counts(
        counts, kept_core=(None if int(getattr(a, 'pin_core_tubes', -1)) < 0
                           else int(a.pin_core_tubes)))

    # 1-D placement: contiguous spans remove the interpolate-from-neighbours
    # shortcut, which is the informative probe; the span length is a RUN setting
    # expressed in tokens (never a hidden default).
    if a.resp_probe == 'span':
        model.physio_placement = 'span'
        model.physio_span_tokens = max(
            1, int(round(float(a.span_seconds) * model.fs / model.sig_kernel)))
    else:
        model.physio_placement = 'scattered'

    kept = {s: counts[s] for s in model.streams}
    caption = ('KEPT ' + ' '.join(f'{s}={v}' for s, v in kept.items())
               + f' tokens  mode={plan.mode}  budget={model.n_budget}'
               + f'  placement={model.physio_placement}')
    if a.resp_probe == 'span':
        caption += f' (span {model.physio_span_tokens} tokens)'
    if getattr(plan, 'regime', None) == 'single':
        caption += f'  regime=single carrier={plan.carrier}'
    # ADR-007 §2.9: name the spatial policy and the realized split, so a figure
    # caption cannot leave the reader guessing whether the probe was stratified.
    _spatial = str(getattr(model, 'spatial_sampling', 'uniform') or 'uniform')
    caption += f'  spatial_sampling={_spatial}'
    if getattr(model, 'anatomy_aware', False) and plan.core_units:
        _name = model.visual[0]
        caption += (f"  regime={plan.regime} S_core={int(plan.core_units[_name])}"
                    f' S_periph={int(plan.periph_units[_name])}')
    return plan, caption


def build_eval_dataset(data_cargs, a, held: Optional[set]):
    """The held-out clip pool, built from ``data_cargs`` (never mutated).

    Uses the corpus-wide DIAGNOSTIC view: this probe selects clips by subject
    afterwards and produces no pooled number (see
    ``data.datasets.build_pretraining_corpus_dataset``).
    """
    import copy

    from data import build_pretraining_corpus_dataset

    cargs = copy.copy(data_cargs)
    ds = build_pretraining_corpus_dataset(cargs)
    idx = list(range(len(ds)))
    if held is not None:
        idx = [i for i in idx
               if str(ds.entries[i].get('subject', '')) in held]
    if not idx:
        raise SystemExit('[eval] no clip belongs to the evaluation set -- check '
                         '--split / the fold manifest / the caches')
    if a.max_clips and len(idx) > a.max_clips:
        g = torch.Generator().manual_seed(a.seed)
        order = torch.randperm(len(idx), generator=g).tolist()
        idx = [idx[k] for k in sorted(order[:a.max_clips])]
    return ds, idx


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r, or nan when either side is constant."""
    if x.size < 3:
        return float('nan')
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / den) if den > 0 else float('nan')


def _band_peak_hz(x: np.ndarray, fs: float, f_min: float,
                  f_max: float) -> float:
    """Dominant frequency of one waveform inside ``[f_min, f_max]`` (Hz).

    Same construction as ADR-008 §2.2.3: DC removal, Hann window, an rFFT
    zero-padded to ``4 * 2**ceil(log2 n)``. ``nan`` when the band is empty.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 4 or not np.isfinite(x).all():
        return float('nan')
    x = x - x.mean()
    n = 4 * int(2 ** int(np.ceil(np.log2(max(x.size, 2)))))
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size), n=n)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))
    sel = (freqs >= float(f_min)) & (freqs <= float(f_max))
    if not sel.any():
        return float('nan')
    return float(freqs[sel][int(np.argmax(spec[sel]))])


def _pretext_fields(p_m: np.ndarray, t_m: np.ndarray, pretext: Dict) -> Dict:
    """ADR-008 §3.2: variance retention and the spectral peak error.

    ``var_hat`` is ``Var(y_hat)`` over the MASKED samples of the clip (in Mode 1
    every sample is masked, so it is the whole predicted waveform). The ADR's
    collapse threshold is ``< 0.10`` and its success target ``>= 0.60``, against
    a target whose per-clip z-score gives ``Var(y) = 1``.
    """
    fs = float(pretext['fs'])
    f_min, f_max = pretext['band']
    f_hat = _band_peak_hz(p_m, fs, f_min, f_max)
    f_y = _band_peak_hz(t_m, fs, f_min, f_max)
    err = (float('nan') if not (np.isfinite(f_hat) and np.isfinite(f_y))
           else 60.0 * abs(f_hat - f_y))
    return {'var_hat': float(np.var(p_m)), 'var_y': float(np.var(t_m)),
            'peak_hat_hz': f_hat, 'peak_y_hz': f_y, 'peak_err_bpm': err}


def score_batch(out: Dict, batch: Dict, sig_kernel: int,
                pretext: Optional[Dict] = None) -> List[Dict]:
    """Per-clip metrics for the resp stream of one batch."""
    pred = out['preds']['resp'].detach().float().cpu().numpy()
    target = (batch['resp'].detach().float().cpu()
              .reshape(pred.shape[0], -1, sig_kernel).numpy())
    mask = out['masks']['resp'].detach().cpu().numpy().astype(bool)
    B = pred.shape[0]
    rows = []
    for b in range(B):
        m = mask[b]
        if m.sum() == 0:                       # 100 % visible: nothing to score
            rows.append({'n_masked': 0, 'mse': float('nan'),
                         'mae': float('nan'), 'rmse': float('nan'),
                         'r_masked': float('nan'), 'r_full': float('nan')})
            continue
        p_m = pred[b][m].reshape(-1)
        t_m = target[b][m].reshape(-1)
        d = p_m - t_m
        assembled = np.where(m[:, None], pred[b], target[b]).reshape(-1)
        rows.append({
            'n_masked': int(m.sum() * sig_kernel),
            'mse': float((d * d).mean()),
            'mae': float(np.abs(d).mean()),
            'rmse': float(np.sqrt((d * d).mean())),
            'r_masked': _pearson(p_m, t_m),
            'r_full': _pearson(assembled, target[b].reshape(-1)),
            **(_pretext_fields(p_m, t_m, pretext) if pretext else {}),
        })
    return rows


def mean_std(vals: List[float]) -> Dict[str, float]:
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return {'mean': float('nan'), 'median': float('nan'), 'std': float('nan'),
                'n': 0}
    return {'mean': float(a.mean()), 'median': float(np.median(a)),
            'std': float(a.std()), 'n': int(a.size)}


def subject_aggregate(per_clip: List[Dict], seed: int,
                      n_boot: int = 2000) -> Dict:
    """Median over each subject's clips, then a bootstrap over subjects.

    The evaluation protocol counts a SUBJECT as one independent observation, so
    this block takes the median of every subject's clip-level values first and
    then bootstraps the across-subject median. It is reported next to (not
    instead of) the clip-level summary, because the two answer different
    questions: the clip mean says how a random clip behaves, the subject median
    says how a random subject behaves.

    The headline protocol's extra per-session step is **not** applied here: this
    is a zero-shot clip probe, not the band-limited waveform metric, and a session
    contributes only a handful of 8 s windows. ``n_sessions`` is recorded so the
    imbalance can be judged.

    :param per_clip: rows carrying at least ``subject``/``session`` and the
        metric keys.
    :param seed: bootstrap RNG seed (deterministic report).
    :param n_boot: bootstrap resamples over subjects.
    """
    by_subject: Dict[str, List[Dict]] = {}
    for r in per_clip:
        by_subject.setdefault(str(r.get('subject', '')), []).append(r)
    rows = []
    for subj in sorted(by_subject):
        rs = by_subject[subj]
        sessions = sorted({str(r.get('session', '')) for r in rs})
        row = {'subject': subj, 'n_clips': len(rs), 'n_sessions': len(sessions)}
        for k in ('r_masked', 'r_full', 'mse', 'mae', 'rmse', 'var_hat',
                  'peak_err_bpm'):
            vals = np.asarray([r[k] for r in rs if k in r], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            row[f'median_{k}'] = float(np.median(vals)) if vals.size else float('nan')
        rows.append(row)

    def _boot(key: str) -> Dict[str, float]:
        vals = np.asarray([r[key] for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            return {'median': float(vals[0]) if vals.size else float('nan'),
                    'lo': float('nan'), 'hi': float('nan'), 'n_subjects': int(vals.size)}
        rng = np.random.default_rng(int(seed))
        idx = rng.integers(0, vals.size, size=(int(n_boot), vals.size))
        boot = np.median(vals[idx], axis=1)
        return {'median': float(np.median(vals)),
                'lo': float(np.percentile(boot, 2.5)),
                'hi': float(np.percentile(boot, 97.5)),
                'n_subjects': int(vals.size), 'n_boot': int(n_boot)}

    out = {'per_subject': rows,
           'r_masked_median': _boot('median_r_masked'),
           'r_full_median': _boot('median_r_full'),
           'mse_median': _boot('median_mse')}
    # ADR-008 §3.2 (opt-in): the two collapse diagnostics, aggregated the same
    # way (median per subject -> pooled median with a bootstrap CI).
    if rows and 'median_var_hat' in rows[0]:
        out['var_hat_median'] = _boot('median_var_hat')
        out['peak_err_bpm_median'] = _boot('median_peak_err_bpm')
    return out


# --------------------------------------------------------------------------- #
def main(argv=None):
    a = get_args(argv)
    dev = torch.device(a.device if (a.device != 'cuda' or
                                    torch.cuda.is_available()) else 'cpu')
    ckpt = load_checkpoint(a.checkpoint)
    cargs = ckpt['args']
    label = a.label or os.path.basename(os.path.dirname(
        os.path.dirname(os.path.abspath(a.checkpoint))))

    if not a.partition:
        # Stages write <exp_root>/fold<k>/partition.json and the run dirs under
        # it, so the record sits one level above <run_dir>.
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(a.checkpoint)))
        for cand in (os.path.join(os.path.dirname(run_dir), 'partition.json'),
                     os.path.join(run_dir, 'partition.json')):
            if os.path.isfile(cand):
                a.partition = cand
                break

    model = build_model(ckpt, dev)
    plan, caption = pin_masking(model, a)
    print(f'[eval] {label}')
    print(f'[eval] weights   : {a.checkpoint} (epoch {ckpt.get("epoch")})')
    print(f'[eval] streams   : {model.streams}  (2-stream ckpts ignore rgb2d)')
    print(f'[eval] masking   : {caption}')
    print(f'[eval] target_norm={getattr(cargs, "target_norm", "?")} '
          f'input_size={getattr(cargs, "input_size", "?")} '
          f'n_signal={model.n_signal} sig_kernel={model.sig_kernel}')
    if a.ablate_stream:
        probe = {'stream': a.ablate_stream, 'mode': a.ablate_mode,
                 'shift': int(a.ablate_shift)}
        print(f'[eval] INPUT ABLATION: {a.ablate_stream} {a.ablate_mode} '
              f'(shift={a.ablate_shift or "auto"}); plan and masks unchanged')
    else:
        probe = None

    # ---- data (from --data_checkpoint when given) -------------------------
    data_cargs = cargs
    if a.data_checkpoint:
        dck = load_checkpoint(a.data_checkpoint)
        data_cargs = dck['args']
        print(f'[eval] dataset   : from {a.data_checkpoint} '
              f'(data_set={getattr(data_cargs, "data_set", "?")})')
    held = None
    if a.split == 'test':
        held, train, src = _held_out_subjects(cargs, a.partition)
        print(f'[eval] split     : test -> fold {getattr(cargs, "fold", 0)}/'
              f'{getattr(cargs, "n_folds", 1)}, {len(held)} held-out subject(s) '
              f'({src}), {len(train)} train')
    if str(a.subjects or '').strip():
        wanted = {x.strip() for x in str(a.subjects).split(',') if x.strip()}
        if held is not None:
            outside = wanted - held
            if outside:
                raise SystemExit(
                    f'[eval] --subjects names {sorted(outside)}, which is not in '
                    f'the evaluated role ({len(held)} held-out subject(s)). The '
                    f'test block is subject-disjoint (ADR-001 I2); scoring a '
                    f'train subject here would not be a held-out probe.')
        held = wanted
        print(f'[eval] subjects  : {len(wanted)} -> {", ".join(sorted(wanted))}')
    ds, idx = build_eval_dataset(data_cargs, a, held)
    # ADR-008 §3.2 (opt-in): variance retention + spectral peak error.
    pretext = None
    if a.pretext_metrics:
        band = tuple(float(x) for x in str(a.pretext_band).split(','))
        if len(band) != 2 or not 0.0 < band[0] < band[1] <= model.fs / 2.0:
            raise SystemExit(
                f'[eval] --pretext_band {a.pretext_band!r} must be '
                f'"f_min,f_max" with 0 < f_min < f_max <= fs/2 = {model.fs/2.0}')
        pretext = {'fs': float(model.fs), 'band': band}
        print(f'[eval] pretext   : variance retention + spectral peak error '
              f'in [{band[0]:g}, {band[1]:g}] Hz (ADR-008 §3.2)')
    subjects = {str(ds.entries[i].get('subject', '')) for i in idx}
    sessions = {str(ds.entries[i].get('session', '')) for i in idx}
    print(f'[eval] clips     : {len(idx)} from {len(sessions)} session(s) / '
          f'{len(subjects)} subject(s); streams in a sample: '
          f'{sorted(ds[idx[0]].keys())}')

    # ---- loop -------------------------------------------------------------
    per_clip: List[Dict] = []
    B = int(a.batch_size)
    preloaded = [ds[i] for i in idx] if a.preload else None
    if preloaded is not None:
        print(f'[eval] preloaded {len(preloaded)} clip(s) into host RAM')
    for rep in range(max(1, int(a.repeats))):
        # identical masks across arms: the PLAN is stated above and the
        # PLACEMENT is seeded per repeat, so two runs with the same --seed see
        # the same kept indices.
        rep_rows: List[Dict] = []
        for start in range(0, len(idx), B):
            chunk = idx[start:start + B]
            if preloaded is None:
                items = [ds[i] for i in chunk]
            else:
                items = preloaded[start:start + B]
            batch = {k: torch.stack([it[k] for it in items]).to(dev).float()
                     for k in items[0]}
            used_shift = None
            if probe is not None:
                batch, used_shift = ablate_batch(
                    batch, model, probe['stream'], probe['mode'],
                    probe['shift'])
            gen = torch.Generator().manual_seed(int(a.seed) + rep)
            with torch.inference_mode():
                out = model(batch, plan=plan, placement_generator=gen)
            rows = score_batch(out, batch, model.sig_kernel, pretext=pretext)
            for row, i in zip(rows, chunk):
                entry = ds.entries[i]
                row['subject'] = str(entry.get('subject', ''))
                row['session'] = str(entry.get('session', ''))
                row['t_start'] = float(entry.get('t_start', float('nan')))
            rep_rows.extend(rows)
        # average the clip over its draws FIRST, then aggregate over clips
        if rep == 0:
            per_clip = [dict(r) for r in rep_rows]
        else:
            for acc, r in zip(per_clip, rep_rows):
                for k in ('mse', 'mae', 'rmse', 'r_masked', 'r_full'):
                    acc[k] = float(np.nanmean([acc[k], r[k]]))
                for k in ('var_hat', 'peak_err_bpm'):
                    if k in acc and k in r:
                        acc[k] = float(np.nanmean([acc[k], r[k]]))
        print(f'[eval] rep {rep + 1}/{a.repeats}: '
              f'r_masked mean {np.nanmean([r["r_masked"] for r in rep_rows]):+.4f}'
              f'  mse {np.nanmean([r["mse"] for r in rep_rows]):.5f}')

    # ---- aggregate --------------------------------------------------------
    summary = {k: mean_std([r[k] for r in per_clip])
               for k in ('mse', 'mae', 'rmse', 'r_masked', 'r_full')}
    if pretext is not None:
        for k in ('var_hat', 'peak_err_bpm'):
            summary[k] = mean_std([r[k] for r in per_clip if k in r])
    by_subject = subject_aggregate(per_clip, a.seed)
    print()
    print('=' * 68)
    print(f'{label}: {len(per_clip)} clip(s) x {a.repeats} draw(s)')
    for k in ('mse', 'rmse', 'mae'):
        s = summary[k]
        print(f'  {k:9s} mean {s["mean"]:.5f}  median {s["median"]:.5f} '
              f'(z units, masked positions)')
    for k in ('r_masked', 'r_full'):
        s = summary[k]
        print(f'  {k:9s} mean {s["mean"]:+.4f}  median {s["median"]:+.4f} '
              f'sd {s["std"]:.4f}  (n={s["n"]})')
    subj_r = by_subject['r_masked_median']
    print(f'  subject   median r_masked {subj_r["median"]:+.4f}  '
          f'[{subj_r["lo"]:+.4f}, {subj_r["hi"]:+.4f}] 95% CI  '
          f'({subj_r["n_subjects"]} subjects)')
    if pretext is not None:
        vr = by_subject.get('var_hat_median', {})
        pe = by_subject.get('peak_err_bpm_median', {})
        if vr:
            print(f'  subject   median Var(y_hat) {vr["median"]:.4f}  '
                  f'[{vr["lo"]:.4f}, {vr["hi"]:.4f}] 95% CI   '
                  f'(ADR-008 §3.2: <0.10 collapses, >=0.60 is the target)')
        if pe:
            print(f'  subject   median peak err {pe["median"]:.3f} bpm  '
                  f'[{pe["lo"]:.3f}, {pe["hi"]:.3f}] 95% CI   '
                  f'(ADR-008 §3.2: >3.0 bpm fails, <=1.5 is the target)')
    print('=' * 68)

    if a.out:
        with open(a.out, 'w') as fh:
            json.dump({'label': label, 'checkpoint': a.checkpoint,
                       'data_checkpoint': a.data_checkpoint,
                       'split': a.split, 'fold': getattr(cargs, 'fold', None),
                       'masking': caption, 'repeats': a.repeats,
                       'ablation': ({**probe, 'resolved_shift': used_shift}
                                    if probe is not None else None),
                       'n_clips': len(per_clip),
                       'n_sessions': len(sessions),
                       'n_subjects': len(subjects),
                       'summary': summary,
                       'subject_aggregate': by_subject,
                       'per_clip': per_clip}, fh, indent=2)
        print(f'[eval] wrote {a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
