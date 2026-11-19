"""Qualitative Stage-2 reconstruction figure (MultiMAE Figure-1 style).

Loads a trained Stage-2 checkpoint, draws the model's OWN masking plan for a
handful of samples, and renders one figure with, per modality, the classic
three panels:

    masked input | AMMM prediction | target

Rows are the modalities of this study -- ``TIR-ROI`` (video, shown as one frame
with the masked patches blacked out) and ``RESP`` (1-D waveform, masked samples
blanked) -- and the columns are 3 sample groups, exactly like Figure 1 of the
MultiMAE paper. As there, the prediction panel overlays the input on the
positions that were NOT masked (``mask == 0``), because the loss only scores the
masked ones.

The masking is NOT re-rolled by hand: ``MultiModalMAE.forward`` calls the run's
own policy (``masking: ammm`` or ``ratio``), so the figure shows what training
actually did, including a 100 %-masked stream when the Dirichlet draw gives one.

Usage (from ``code/``)::

    # real clips, random indices, CPU (login node has no GPU)
    python runners/run_visualize_pretrain.py \
        --checkpoint $WORK_SCRATCH/pretrain/stage2_ammm_hpc_tir_roi_resp_fold0/checkpoints/checkpoint-0299.pth \
        --samples 3 --out $WORK_SCRATCH/pretrain/stage2_ammm_hpc_tir_roi_resp_fold0/ammm_recon_figure.png

    # pure-noise sanity check of the architecture (no dataset touched)
    python runners/run_visualize_pretrain.py --checkpoint <ckpt> --source random

    # PARAMETERISE the masking of the test clips (instead of the trained draw):
    #   kept  : AMMM semantics -- fraction of units to KEEP (cap is raised if needed)
    #   ratio : legacy semantics -- fraction of units to MASK (no cap)
    python runners/run_visualize_pretrain.py --checkpoint <ckpt> \
        --mask_mode kept  --kept_tir 0.25 --kept_resp 0.50
    python runners/run_visualize_pretrain.py --checkpoint <ckpt> \
        --mask_mode ratio --mask_ratio_tir 0.90 --mask_ratio_resp 0.50

    # SAME subject+task as an earlier figure, different masking: pin the sessions
    # (one clip each, order kept; `--samples` is then ignored)
    python runners/run_visualize_pretrain.py --checkpoint <ckpt> \
        --sessions F053_T3,M012_T1,M058_T3 --mask_mode kept \
        --kept_tir 0.75 --kept_resp 0.75

    # ... and pin the EXACT clip the earlier figure showed (`@<t_start seconds>`)
    python runners/run_visualize_pretrain.py --checkpoint <ckpt> \
        --sessions F053_T3@54,M012_T1@8,M058_T3@40 --mask_mode kept \
        --kept_tir 0.25 --kept_resp 0.25

Geometry, streams, caches and the masking knobs all come from the CHECKPOINT
(``ckpt['args']``), so the figure can never disagree with the weights; only
``--mask_mode``/``--sessions``/``--seed`` change what is drawn for the TEST
clips.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                            # noqa: E402

from core.input_adapters import EPS as CLIP_EPS, zscore_clip   # noqa: E402
from core.multimae import _EPS as CUBE_EPS, build_pretraining_model  # noqa: E402


def get_args(argv=None):
    p = argparse.ArgumentParser(
        description='Render the masked-input / prediction / target figure of a '
                    'Stage-2 multimodal-MAE checkpoint.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True,
                   help='Stage-2 checkpoint (e.g. .../checkpoints/checkpoint-0299.pth)')
    p.add_argument('--out', default='', help='output PNG (default: beside the checkpoint)')
    p.add_argument('--samples', default=3, type=int, help='number of sample columns')
    p.add_argument('--source', default='dataset', choices=('dataset', 'random'),
                   help="'dataset' = random REAL clips (default); 'random' = noise "
                        "tensors, an architecture sanity check only")
    p.add_argument('--seed', default=0, type=int, help='RNG seed (sample choice + masks)')
    p.add_argument('--frame', default=-1, type=int,
                   help='which frame of the clip to show (default -1 = middle)')
    p.add_argument('--split', default='test', choices=('test', 'corpus'),
                   help="'test' (default) = clips from the checkpoint fold's "
                        'HELD-OUT subjects; \'corpus\' = any clip')
    p.add_argument('--sessions', default='',
                   help='comma-separated session tags (<SUBJECT>_<TASK>, e.g. '
                        'F053_T3,M012_T1) drawn ONE clip each, in this order. '
                        'Append @<t_start seconds> (F053_T3@54) to pin the EXACT '
                        'clip an earlier figure used. Used to re-show the same '
                        'subject+task under a different masking; overrides '
                        '--samples. The tag must exist in the chosen split (so '
                        '--split test still enforces held-out subjects).')
    p.add_argument('--partition', default='',
                   help='the fold partition record (<exp_root>/fold<k>/'
                        'partition.json); default: discovered above the '
                        'checkpoint run dir')
    p.add_argument('--max_sessions', default=0, type=int,
                   help='cap discovery (0 = no cap). A small cap speeds the build '
                        'up but can EXCLUDE the fold test subjects, so --split test '
                        'needs 0 unless you know they are inside the cap')
    p.add_argument('--device', default='cpu', type=str)
    p.add_argument('--title', default='', type=str)
    # ---- masking of the TEST clips (not the training policy) -------------- #
    # 'plan'  = whatever the checkpoint trained with (Dirichlet draw per step)
    # 'kept'  = AMMM semantics, PINNED per stream: kept_ratio_override
    # 'ratio' = legacy semantics, PINNED per stream: mask_ratio_<stream>
    p.add_argument('--mask_mode', default='plan', choices=('plan', 'kept', 'ratio'),
                   help="'plan' = the checkpoint's own draw (default); 'kept' = pin "
                        'the AMMM KEPT ratio per stream; \'ratio\' = pin the legacy '
                        'MASK ratio per stream')
    p.add_argument('--kept_tir', default=0.25, type=float,
                   help='--mask_mode kept: fraction of the 36 TIR patches to KEEP')
    p.add_argument('--kept_resp', default=0.5, type=float,
                   help='--mask_mode kept: fraction of the 50 RESP tokens to KEEP')
    p.add_argument('--mask_ratio_tir', default=0.9, type=float,
                   help='--mask_mode ratio: fraction of the 36 TIR patches to MASK')
    p.add_argument('--mask_ratio_resp', default=0.5, type=float,
                   help='--mask_mode ratio: fraction of the 50 RESP tokens to MASK')
    return p.parse_args(argv)


def apply_mask_mode(model, a) -> str:
    """Override the model's masking policy for the test clips. Returns a caption.

    ``kept`` uses the AMMM override path (``kept_ratio_override``), which is used
    VERBATIM and is exempt from ``min_kept``; a pin above the run's kept cap also
    raises that cap, otherwise ``plan_step`` refuses it (``0 <= r <= cap``).
    ``ratio`` switches to the legacy per-stream ``mask_ratio_<s>`` engine, which
    has no cap, so it is the most direct way to ask for e.g. 90 % / 50 % masked.
    """
    if a.mask_mode == 'plan':
        return 'checkpoint policy ({})'.format(getattr(model, 'masking', '?'))

    if a.mask_mode == 'kept':
        model.masking = 'ammm'
        pins = {'tir': float(a.kept_tir), 'resp': float(a.kept_resp)}
        caps = dict(getattr(model, 'ammm_caps', {}) or {})
        for s, v in pins.items():
            if not 0.0 <= v <= 1.0:
                raise SystemExit(f'--kept_{s} must be in [0, 1]; got {v}')
            if v > caps.get(s, 1.0):
                print('[viz] raising {} KEPT cap {:.2f} -> {:.2f} to allow the pin'
                      .format(s, caps.get(s, 1.0), v))
                caps[s] = v
        model.ammm_caps = caps
        model.ammm_overrides = pins
        model.ammm_min_kept = 0
        return 'pinned KEPT tir={:.2f} resp={:.2f}'.format(pins['tir'], pins['resp'])

    model.masking = 'ratio'
    ratios = {'tir': float(a.mask_ratio_tir), 'resp': float(a.mask_ratio_resp)}
    for s, v in ratios.items():
        if not 0.0 <= v <= 1.0:
            raise SystemExit(f'--mask_ratio_{s} must be in [0, 1]; got {v}')
    model.mask_ratios = ratios
    return 'pinned MASK tir={:.2f} resp={:.2f}'.format(ratios['tir'],
                                                      ratios['resp'])


# --------------------------------------------------------------------------- #
# token <-> pixel / sample helpers (exact inverses of the model's own views)
# --------------------------------------------------------------------------- #
def _tokens_to_video(tokens: torch.Tensor, model) -> torch.Tensor:
    """``[B, N, C*t*ph*pw] -> [B, C, T, H, W]`` (inverse of ``_targets``)."""
    t, ph, pw = model.tubelet
    B, N, flat = tokens.shape
    C = flat // (t * ph * pw)
    Gh, Gw = model.grid_h, model.grid_w
    Gt = N // (Gh * Gw)
    v = tokens.view(B, Gt, Gh, Gw, C, t, ph, pw)
    return v.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
        B, C, Gt * t, Gh * ph, Gw * pw)


def _cube_stats(xin: torch.Tensor, model):
    """Clip-normalised video -> ``(cubes [B,N,flat], mean, var)`` per CUBE.

    Mirrors ``MultiModalMAE._targets``: a masked cube is standardised with its
    OWN mean/var, so this is the map to invert before displaying a prediction.
    """
    t, ph, pw = model.tubelet
    B, C, T, H, W = xin.shape
    Gt, Gh, Gw = T // t, H // ph, W // pw
    x = xin.view(B, C, Gt, t, Gh, ph, Gw, pw)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    cubes = x.view(B, Gt * Gh * Gw, -1)
    mean = cubes.mean(dim=-1, keepdim=True)
    var = cubes.var(dim=-1, unbiased=False, keepdim=True)
    return cubes, mean, var


def reconstruct_visual(raw: torch.Tensor, pred: torch.Tensor,
                       mask: torch.Tensor, model) -> torch.Tensor:
    """Prediction in PIXEL space with the unmasked positions taken from input.

    :param raw: ``[B,C,T,H,W]`` dataset tensor (``[0, 1]``).
    :param pred: ``[B,N,flat]`` model output (cube-z-scored space).
    :param mask: ``[B,N]``, 1 = masked (= to reconstruct).
    """
    xin = zscore_clip(raw)                       # what the encoder sees
    cubes, mean, var = _cube_stats(xin, model)
    rec_norm = torch.where(mask.unsqueeze(-1).bool(),
                           pred * torch.sqrt(var + CUBE_EPS) + mean, cubes)
    rec = _tokens_to_video(rec_norm, model)
    # undo the clip z-score to get back to pixel units
    mu = raw.mean(dim=(2, 3, 4), keepdim=True)
    sd = torch.sqrt(raw.var(dim=(2, 3, 4), unbiased=False, keepdim=True)
                    + CLIP_EPS)
    return (rec * sd + mu).clamp(0.0, 1.0)


def _frame_axes(model):
    """Spatial patch grid (Gh, Gw) and patch size (ph, pw)."""
    _, ph, pw = model.tubelet
    return model.grid_h, model.grid_w, ph, pw


def _blank_masked_patches(img: torch.Tensor, mask_spatial: torch.Tensor,
                          model, level: float = 0.06) -> torch.Tensor:
    """Grey out the spatially masked patches of ``img`` ``[C,H,W]``."""
    Gh, Gw, ph, pw = _frame_axes(model)
    out = img.clone()
    m = mask_spatial.view(Gh, Gw)
    for i in range(Gh):
        for j in range(Gw):
            if bool(m[i, j]):
                out[:, i * ph:(i + 1) * ph, j * pw:(j + 1) * pw] = level
    return out


def _to_rgb(img: torch.Tensor) -> "torch.Tensor":
    """[C,H,W] in [0,1] -> [H,W,3] displayable numpy."""
    arr = img.detach().cpu().float().clamp(0, 1)
    if arr.shape[0] == 1:
        arr = arr.repeat(3, 1, 1)
    return arr.permute(1, 2, 0).numpy()


def _recorded_subjects(value) -> set:
    """The checkpoint's train-subject list, whichever form it was saved in.

    ``run_pretrain`` overwrites ``args.subjects`` with the resolved train list,
    and a YAML round-trip can leave it as the STRING of that list -- parse it.
    """
    import ast
    if isinstance(value, (list, tuple, set)):
        return {str(v).strip(" '\"") for v in value if str(v).strip()}
    text = str(value or '').strip()
    if text.startswith('['):
        try:
            parsed = ast.literal_eval(text)
            return {str(v).strip(" '\"") for v in parsed}
        except (ValueError, SyntaxError):
            text = text.strip('[]')
    return {t.strip(" '\"") for t in text.split(',') if t.strip(" '\"")}


def _pin_session_indices(pool, sessions, want, seed, starts=None):
    """One clip per requested session, in the ORDER given.

    A session holds many overlapping windows (a clip is 8 s at a 2 s hop), so a
    tag still needs a clip choice. By default it is a SEEDED pick inside that
    session's pool entries; append ``@<t_start seconds>`` to a tag
    (``F053_T3@54``) to pin the EXACT window instead -- that is how a new masking
    setting is rendered on the SAME clip an earlier figure showed.

    :param pool: candidate dataset indices, already filtered to the split.
    :param sessions: ``{dataset index: session tag}`` for every pool entry.
    :param want: session specs to pin, in display order (``TAG`` or ``TAG@t``).
    :param seed: RNG seed (same seed -> same clip per session).
    :param starts: ``{dataset index: t_start}``; required only for a ``TAG@t``
        spec.
    :returns: one dataset index per spec, in ``want`` order.
    :raises SystemExit: an empty request, a tag with no clip in the pool, or a
        ``@t`` that matches no window of that session (the available starts are
        listed).
    """
    if not want:
        raise SystemExit('[viz] --sessions was given but lists no session tag')
    by_sess = {}
    for i in pool:
        by_sess.setdefault(str(sessions.get(i, '')), []).append(i)
    g = torch.Generator().manual_seed(int(seed))
    picks = []
    for spec in want:
        tag, _, raw_t = str(spec).partition('@')
        tag = tag.strip()
        cand = sorted(by_sess.get(tag, []))
        if not cand:
            raise SystemExit(
                '[viz] --sessions: no clip in this pool for {!r} (pool has {} '
                'session(s), e.g. {})'.format(tag, len(by_sess), sorted(by_sess)[:3]))
        if not raw_t.strip():
            picks.append(cand[int(torch.randperm(len(cand), generator=g)[0])])
            continue
        if starts is None:
            raise SystemExit('[viz] --sessions {}: @t_start needs the clip '
                             'starts'.format(spec))
        try:
            want_t = float(raw_t)
        except ValueError:
            raise SystemExit('[viz] --sessions {}: t_start is not a number'
                             .format(spec))
        hit = [i for i in cand
               if abs(float(starts.get(i, float('nan'))) - want_t) < 1e-3]
        if not hit:
            have = sorted({float(starts.get(i, float('nan'))) for i in cand})
            raise SystemExit(
                '[viz] --sessions {}: {} has no clip at t_start={:g}s; available: '
                '{}'.format(spec, tag, want_t, have))
        picks.append(hit[0])
    return picks


def _held_out_subjects(cargs, partition_path=None):
    """``(test, train, source)`` subject sets of the checkpoint's fold.

    AUTHORITATIVE SOURCE = the fold's write-once ``partition.json``, read with
    :func:`protocol.load_plan`. Re-deriving the partition here is NOT a
    fallback: the trainer partitions the *discovered usable* subject universe
    (139 on BP4D+), while the dataset entries hold the *clip-surviving*
    subjects (133). A different universe re-splits EVERY fold, so partitioning
    ``ds.entries`` gives the wrong held-out block -- measured: 15 of 27 "unseen"
    subjects were in the checkpoint's train list (ADR-001 I2).
    """
    from protocol import ProtocolError, load_plan
    if not partition_path or not os.path.isfile(str(partition_path)):
        raise SystemExit(
            '[viz] no partition record for this checkpoint. Pass '
            '--partition <exp_root>/fold<k>/partition.json. The old '
            're-derive-from-entries fallback was removed: it partitions a '
            'different subject universe and therefore re-splits every fold '
            '(ADR-001 I2).')
    try:
        plan = load_plan(str(partition_path))
    except ProtocolError as exc:
        raise SystemExit(f'[viz] {exc}') from None
    return (set(plan.roles['test']), set(plan.roles['train']),
            os.path.basename(str(partition_path)))


# --------------------------------------------------------------------------- #
def main(argv=None):
    args = get_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    if args.sessions and args.source == 'random':
        raise SystemExit('[viz] --sessions selects REAL clips; drop --source random')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dev = torch.device(args.device if (args.device != 'cuda'
                                       or torch.cuda.is_available()) else 'cpu')
    if not args.partition:
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
        for cand in (os.path.join(os.path.dirname(run_dir), 'partition.json'),
                     os.path.join(run_dir, 'partition.json')):
            if os.path.isfile(cand):
                args.partition = cand
                break
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cargs = ckpt['args']
    print(f'[viz] checkpoint {args.checkpoint} (epoch {ckpt.get("epoch")})')
    print(f'[viz] masking={getattr(cargs, "masking", "ratio")} '
          f'target_norm={getattr(cargs, "target_norm", "?")} '
          f'streams={getattr(cargs, "streams", "?")} '
          f'input_size={getattr(cargs, "input_size", "?")}')

    model = build_pretraining_model(cargs)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    if missing or unexpected:
        print(f'[viz] state_dict: {len(missing)} missing / '
              f'{len(unexpected)} unexpected')
    model.to(dev).eval()
    mask_caption = apply_mask_mode(model, args)
    print('[viz] masking for the test clips: ' + mask_caption)

    # ------------------------------------------------------------------ #
    # data
    # ------------------------------------------------------------------ #
    B = int(args.samples)
    if args.source == 'random':
        print('[viz] WARNING: --source random feeds NOISE tensors -- the panels '
              'are an architecture check, not a reconstruction result.')
        T = int(model.num_frames)
        S = int(model.input_size)
        batch = {'tir': torch.rand(B, model.stream_channels['tir'], T, S, S),
                 'resp': torch.randn(B, 1, model.n_signal * model.sig_kernel)}
        tags = ['noise {}'.format(i) for i in range(B)]
    else:
        from data import build_pretraining_corpus_dataset
        if args.sessions and args.split == 'corpus':
            print('[viz] NOTE: --split corpus with --sessions draws the clip from '
                  'ANY subject, including the checkpoint TRAIN subjects')
        # The checkpoint carries the RESOLVED run arguments. This is the
        # corpus-wide DIAGNOSTIC view (it can see the test block), so no figure
        # produced here is a protocol result; the held-out set is selected
        # afterwards from the run's partition.json (see _held_out_subjects).
        cargs.max_sessions = int(args.max_sessions) or None
        ds = build_pretraining_corpus_dataset(cargs)
        n = len(ds)
        if n == 0:
            raise SystemExit('[viz] the dataset is empty -- check the caches/raw_root')

        pool = list(range(n))
        if args.split == 'test':
            held, train, src = _held_out_subjects(cargs, args.partition)
            pool = [i for i in pool
                    if str(ds.entries[i].get('subject', '')) in held]
            print('[viz] --split test: fold {}/{} holds out {} subject(s) '
                  '(list from {}); {} of {} clips come from them'.format(
                      getattr(cargs, 'fold', 0), getattr(cargs, 'n_folds', 1),
                      len(held), src, len(pool), n))
            if not pool:
                raise SystemExit(
                    '[viz] no clip belongs to the fold test subjects -- rerun '
                    'with --max_sessions 0 (the default) or check raw_root')
            overlap = held & train
            if overlap:
                raise SystemExit(
                    f'[viz] fold contract VIOLATED: {len(overlap)} "unseen" '
                    f'subject(s) are in the checkpoint train list '
                    f'(e.g. {sorted(overlap)[:3]})')
            found = {str(ds.entries[i].get('subject', '')) for i in pool}
            print('[viz] verified: those clips come from {} held-out subject(s), '
                  'none of them in the {}-subject train list'.format(
                      len(found), len(train)))

        if args.sessions:
            want = [t.strip() for t in args.sessions.split(',') if t.strip()]
            idx = _pin_session_indices(
                pool, {i: str(ds.entries[i].get('session', '')) for i in pool},
                want, args.seed,
                starts={i: float(ds.entries[i].get('t_start', float('nan')))
                        for i in pool})
            B = len(idx)
            print('[viz] --sessions pinned {} session(s), 1 clip each, order kept '
                  '(--samples ignored): {}'.format(
                      len(idx), ', '.join('{}@{:g}s'.format(
                          ds.entries[i].get('session'),
                          float(ds.entries[i].get('t_start', float('nan'))))
                          for i in idx)))
        else:
            g = torch.Generator().manual_seed(args.seed)
            order = torch.randperm(len(pool), generator=g).tolist()
            idx = [pool[k] for k in order[:B]]
        items = [ds[i] for i in idx]
        batch = {s: torch.stack([it[s] for it in items]) for s in items[0]}
        tags = []
        for i in idx:
            e = ds.entries[i] if hasattr(ds, 'entries') else {}
            tags.append(str(e.get('session', 'sample {}'.format(i))))

    batch = {s: v.to(dev).float() for s, v in batch.items()}
    print('[viz] batch ' + ', '.join(f'{s} {tuple(v.shape)}' for s, v in batch.items()))

    with torch.no_grad():
        out = model(batch)
    plan = getattr(model, '_last_mask_plan', None)
    kept = out.get('kept', {})
    print('[viz] plan: ' + (', '.join(f'kept_{s}={k}' for s, k in kept.items())
                            or 'ratio masking (no plan)'))

    masks, preds = out['masks'], out['preds']

    # ------------------------------------------------------------------ #
    # figure: 2 rows (modalities) x 3 groups x [masked | pred | target]
    # ------------------------------------------------------------------ #
    ngroups = B
    fig = plt.figure(figsize=(3.15 * 3 * ngroups, 5.4), constrained_layout=True)
    widths = []
    for _ in range(ngroups):
        widths += [1, 1, 1, 0.18]
    gs = fig.add_gridspec(2, len(widths), width_ratios=widths)

    def axes_for(row, g):
        base = g * 4
        return [fig.add_subplot(gs[row, base + k]) for k in range(3)]

    frame = args.frame
    sig_kernel = int(model.sig_kernel)

    for g in range(ngroups):
        # ---------------- TIR-ROI (video) ----------------------------- #
        ax_in, ax_pr, ax_gt = axes_for(0, g)
        raw = batch['tir'][g:g + 1]
        if frame < 0:
            fi = raw.shape[2] // 2
        else:
            fi = min(frame, raw.shape[2] - 1)
        m_tir = masks['tir'][g]                        # [Gt*Gh*Gw]
        Gt = m_tir.numel() // (model.grid_h * model.grid_w)
        m_spatial = m_tir.view(Gt, model.grid_h, model.grid_w)[fi // model.tubelet[0]]
        rec = reconstruct_visual(raw, preds['tir'][g:g + 1], masks['tir'][g:g + 1],
                                 model)
        gt_frame = raw[0, :, fi]
        ax_in.imshow(_to_rgb(_blank_masked_patches(gt_frame, m_spatial, model)))
        ax_pr.imshow(_to_rgb(rec[0, :, fi]))
        ax_gt.imshow(_to_rgb(gt_frame))
        kept_tir = int((m_spatial == 0).sum())
        ax_in.set_title('{} | masked input\nkept {}/{} patches'.format(
            tags[g], kept_tir, m_spatial.numel()), fontsize=9)
        ax_pr.set_title('prediction\n(input overlaid where kept)', fontsize=9)
        ax_gt.set_title('target\nTIR-ROI frame {} / {}'.format(fi, raw.shape[2] - 1),
                        fontsize=9)

        # ---------------- RESP (1-D) ---------------------------------- #
        ax_in, ax_pr, ax_gt = axes_for(1, g)
        m_resp = masks['resp'][g].repeat_interleave(sig_kernel)   # [seq_len]
        tgt = batch['resp'][g, 0].detach().cpu()
        prd = preds['resp'][g].reshape(-1).detach().cpu()
        x = np.arange(tgt.numel())
        vis = (m_resp == 0).cpu()
        masked_y = tgt.clone()
        masked_y[~vis] = float('nan')                 # blank the masked samples
        ax_in.plot(x, masked_y.numpy(), lw=1.0, color='tab:blue')
        ax_in.set_title('{} | masked input\nkept {} / {} samples'.format(
            tags[g], int(vis.sum()), vis.numel()), fontsize=9)
        if int(vis.sum()) == 0:
            # 100 % masked: nothing to draw, and an autoscaled empty axes would
            # show a meaningless +-0.04 range -- say it instead.
            ax_in.text(.5, .5, '100 % masked\n(no input)', ha='center',
                       va='center', transform=ax_in.transAxes, fontsize=9)
        ax_pr.plot(x, prd.numpy(), lw=1.2, color='tab:red', label='prediction')
        ax_pr.plot(x[vis], tgt[vis].numpy(), '.', ms=2.5, color='k',
                   label='kept input')
        ax_pr.legend(fontsize=7, loc='upper right')
        ax_pr.set_title('prediction (red) + kept input (dots)', fontsize=9)
        ax_gt.plot(x, tgt.numpy(), lw=1.2, color='tab:green')
        ax_gt.set_title('target\nRESP (z-scored clip)', fontsize=9)
        # one shared y-scale so the three panels are comparable, and it stays
        # sensible when the input panel is empty
        lo, hi = float(tgt.min()), float(tgt.max())
        pad = 0.08 * max(hi - lo, 1e-6)
        for ax in (ax_in, ax_pr, ax_gt):
            ax.set_xlim(0, tgt.numel() - 1)
            ax.set_ylim(lo - pad, hi + pad)

        for ax in (ax_in, ax_pr, ax_gt):
            ax.set_xlabel('sample (100 Hz)', fontsize=8)
            ax.tick_params(labelsize=7)
        ax_in.set_ylabel('resp (z)', fontsize=8)

    fig.suptitle((args.title or ('{} -- masked input / prediction / target'
                                 .format(os.path.basename(os.path.dirname(
                                     os.path.dirname(args.checkpoint))))))
                 + '   [' + mask_caption + '; clips: fold {}/{} {} subjects]'.format(
                     getattr(cargs, 'fold', 0), getattr(cargs, 'n_folds', 1),
                     'HELD-OUT' if args.split == 'test' and
                     args.source == 'dataset' else 'any'),
                 fontsize=12)
    # row labels
    fig.text(0.002, 0.73, 'TIR-ROI', rotation=90, va='center', fontsize=11)
    fig.text(0.002, 0.27, 'RESP', rotation=90, va='center', fontsize=11)

    out_png = args.out or os.path.join(os.path.dirname(args.checkpoint),
                                      'pretrain_reconstruction.png')
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print('[viz] wrote {}'.format(out_png))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
