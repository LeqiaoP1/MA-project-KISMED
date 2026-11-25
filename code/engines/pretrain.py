"""Masked pre-training loop.

This mirrors the *shape* of VideoMAE's
``tmp/videomae/engine_for_pretraining.py`` and MultiMAE's
``run_pretraining_multimae.py`` training loop.

The batch-level reconstruction step is thesis-specific. Reference recipe
(VideoMAE): the dataset yields ``(samples, labels, index, bool_masked_pos)``;
the model implements ``forward(samples, bool_masked_pos)`` returning
``(pred, target)`` token tensors, and the masked loss is::

    loss = criterion(pred, target, bool_masked_pos)   # see core/criterion.py

Unpack and adapt inside the loop below where marked.
"""
import contextlib
import inspect
from typing import Iterable, Optional

import torch

from utils import (MetricLogger, SmoothedValue, autocast_context,
                   scaler_enabled)


def _accepts_kw(fn, name: str) -> bool:
    """True when ``fn`` accepts the keyword ``name`` (real model vs stub).

    ADR-008 adds an optional ``step`` keyword for the gradient-balance probe;
    several contract tests pass small duck-typed models whose ``forward``
    predates it. Introspecting the signature keeps both call shapes working
    without wrapping the forward in a ``try/except TypeError``, which would also
    swallow real argument errors.

    A ``**kwargs`` signature (e.g. ``DistributedDataParallel.forward``) is
    accepted, and an un-introspectable callable is assumed to support the
    documented signature.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True
    if name in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in params.values())


def _accepts_epoch(fn) -> bool:
    """True when ``fn`` takes an ``epoch`` keyword (real model vs stub).

    ADR-005 §2.7 requires the epoch to reach the policy, but several contract
    tests pass small duck-typed models whose ``forward`` predates that
    argument. Introspecting the signature keeps both call shapes working
    without wrapping the forward in a ``try/except TypeError``, which would
    also swallow real argument errors.

    A ``**kwargs`` signature (e.g. ``DistributedDataParallel.forward``) is
    accepted, and an un-introspectable callable is assumed to support the
    documented signature.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True
    if 'epoch' in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in params.values())


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, loss_scaler=None, max_norm: float = 0.0,
                    model_ema=None, log_writer=None,
                    lr_schedule_values=None, wd_schedule_values=None,
                    update_freq: int = 1,
                    start_steps: Optional[int] = None,
                    amp_config=None,
                    mask_base: int = 42, steps_per_epoch: int = 0,
                    rank: int = 0):
    """One pre-training epoch.

    ``mask_base`` / ``steps_per_epoch`` / ``rank`` drive the ADR-005 mask RNG:
    every step gets a COUNTS generator from a rank-independent key (so all
    ranks derive the same plan without a collective) and a PLACEMENT generator
    that includes the rank (so the samples differ across ranks). The global RNG
    is left to the model's own dropout and is never used for masking, which is
    what keeps the counts identical across DDP ranks.

    ``epoch`` also selects the Mode-1 probability of the ADR-005 §2.7
    curriculum (``p(e)``); with ``virtual_sensing_warmup_epochs = 0`` the
    threshold is the constant target and the draw is unchanged.
    """
    from core.budget_masking import MODE_VIRTUAL_SENSING, REGIME_SINGLE
    from core.mask_placement import counts_generator, placement_generator
    model.train(True)
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    use_scaler = scaler_enabled(amp_config)   # False for --amp_dtype none/bf16
    # Computed once: the epoch only reaches models that declare it.
    epoch_kwarg = _accepts_epoch(model.forward)
    # ADR-008 §2.7: the global step feeds the opt-in gradient-balance probe.
    step_kwarg = _accepts_kw(model.forward, 'step')

    optimizer.zero_grad()
    for data_iter_step, batch in enumerate(
            metric_logger.log_every(data_loader, 10, header)):
        # Multimodal MAE: the batch is a dict {stream: tensor}. A plain tensor
        # batch (single-stream placeholder model) is still accepted.
        step = data_iter_step // update_freq
        # index into the GLOBAL step schedule (start_steps = epoch * n_steps)
        sched_idx = step if start_steps is None else start_steps + step

        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    # multiply by the group's layer-wise LR scale (1.0 when
                    # layer decay is off): the schedule is ONE curve, so without
                    # this a per-group lr_scale would be overwritten every step.
                    param_group['lr'] = (lr_schedule_values[sched_idx]
                                         * param_group.get('lr_scale', 1.0))
                if (wd_schedule_values is not None
                        and param_group['weight_decay'] > 0):
                    param_group['weight_decay'] = wd_schedule_values[sched_idx]

        final_step = (data_iter_step + 1) % update_freq == 0
        with contextlib.ExitStack() as stack:
            # autocast the forward/loss: a no-op unless --amp_dtype fp16/bf16
            stack.enter_context(autocast_context(amp_config))
            # DDP: skip the gradient all-reduce on accumulation micro-steps
            # (a no-op for update_freq == 1, i.e. every existing config)
            if not final_step and hasattr(model, 'no_sync'):
                stack.enter_context(model.no_sync())

            if isinstance(batch, dict):
                x = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                # global step index -> reproducible, rank-independent plan
                global_step = (epoch * max(1, int(steps_per_epoch))
                               + data_iter_step)
                out = model(
                    x,
                    counts_generator=counts_generator(
                        mask_base, epoch, global_step),
                    placement_generator=placement_generator(
                        mask_base, epoch, global_step, rank=rank),
                    **({'epoch': epoch} if epoch_kwarg else {}),
                    **({'step': global_step} if step_kwarg else {}))
                loss = out['loss'] / update_freq
            else:
                samples = batch[0].to(device, non_blocking=True)
                out = model(samples)
                loss = (out['loss'] if isinstance(out, dict) else out) / update_freq

            if loss_scaler is None:
                loss.backward()
                if max_norm is not None and max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                if final_step:
                    optimizer.step()
                    optimizer.zero_grad()
                grad_norm = None
            else:
                grad_norm = loss_scaler(
                    loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(),
                    update_grad=final_step,
                    scaler_enabled=use_scaler)
                if final_step:
                    optimizer.zero_grad()

        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss.item() * update_freq)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        if grad_norm is not None:
            metric_logger.update(grad_norm=grad_norm.item())
        # Per-stream diagnostics for the multimodal MAE: the RAW masked MSE of
        # every stream (the weighted contributions would hide which stream
        # actually moved). A placeholder/single-stream model simply logs
        # nothing extra.
        if isinstance(out, dict):
            for name, value in (out.get('losses_mse') or {}).items():
                metric_logger.update(**{f'mse_{name}': value})
            # ADR-005 §2.8 plan diagnostics: the sampled mode, the contract-4
            # RESOLVED mode, the active p(e), and the kept TOKEN and UNIT count
            # per stream. Logged every step because they are a property of the
            # step's draw, not of the run; the key set is a function of the
            # config (every declared stream appears on every step).
            kept = out.get('kept') or {}
            if kept:
                metric_logger.update(**{f'kept_{name}': value
                                        for name, value in kept.items()})
            kept_units = out.get('kept_units') or {}
            if kept_units:
                metric_logger.update(**{f'kept_units_{name}': value
                                        for name, value in kept_units.items()})
            # ADR-007 §2.9: the stratified split, in TUBES. Keyed per declared
            # stream and emitted on every step (0 for a signal stream and for
            # the whole run under the ``uniform`` ablation), so the epoch mean
            # of `kept_core_<s>` IS the advertised S_core.
            kept_core = out.get('kept_core') or {}
            if kept_core:
                metric_logger.update(**{f'kept_core_{name}': value
                                        for name, value in kept_core.items()})
            kept_periph = out.get('kept_periph') or {}
            if kept_periph:
                metric_logger.update(**{f'kept_periph_{name}': value
                                        for name, value in kept_periph.items()})
            if 'mode' in out:
                metric_logger.update(mode=out['mode'])
            if 'mode_resolved' in out:
                # The epoch mean of `mode` is 2 - f, NOT the Mode-1 fraction f
                # that ADR-005 §2.8 asks for; this 0/1 indicator averages to f.
                resolved = int(out['mode_resolved'])
                metric_logger.update(mode_resolved=resolved)
                metric_logger.update(
                    is_virtual=int(resolved == MODE_VIRTUAL_SENSING))
            if 'p_virtual' in out:
                metric_logger.update(p_virtual=out['p_virtual'])
            # ADR-006 §2.8: the step class. ``is_single`` averages to the
            # realized single-carrier fraction, and the per-carrier indicators
            # are emitted for EVERY declared visual stream on EVERY step (a key
            # set that depended on the sampled regime would desynchronise a
            # distributed run -- ADR-005 contract 9).
            if 'regime' in out:
                metric_logger.update(is_single=int(out['regime'] == REGIME_SINGLE))
                carrier = out.get('carrier')
                for name in getattr(model, 'visual', []) or []:
                    metric_logger.update(
                        **{f'is_single_{name}': int(carrier == name)})
            if 'p_single' in out:
                metric_logger.update(p_single=out['p_single'])
            if 'n_batch' in out:
                metric_logger.update(n_batch=out['n_batch'])
            # ADR-008 §2.7: the composite objective terms and the InfoNCE
            # labels. Present iff the CONFIG enables an objective term, so the
            # key set never depends on the sampled mode or the rank.
            objective = out.get('objective') or {}
            if objective:
                metric_logger.update(
                    **{k: float(v) for k, v in objective.items()})
            # ... and the opt-in gradient-balance probe, which exists only on
            # `global_step % grad_balance_freq == 0`. `step` is global, so
            # every rank adds the same keys on the same steps.
            for key, value in (out.get('diagnostics') or {}).items():
                if value is not None:
                    metric_logger.update(**{key: float(value)})

        # Per-step loss series for the optional writer (W&B / TensorBoard).
        # The step axis is the GLOBAL optimizer step, the same axis the runner
        # uses for its epoch and validation points: mixing a per-epoch axis
        # with a per-step one makes the backend reject the run as
        # non-monotonic. ``log_writer`` is None (no-op) unless a logger is
        # passed, so a run without --log_wandb behaves exactly as before.
        if log_writer is not None:
            log_writer.add_scalar('train/loss', loss.item() * update_freq,
                                  sched_idx)

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _merge_sums(parts: Iterable[dict]) -> dict:
    """Add per-rank ``{key: batch_n-weighted sum}`` dicts key by key.

    Split out of :func:`evaluate_pretrain` because it is the part of the DDP
    aggregation that can be wrong without any shape error: keys missing on one
    rank (a mask mode that rank never drew) must contribute 0, not raise.
    """
    out: dict = {}
    for part in parts:
        for key, value in (part or {}).items():
            out[key] = out.get(key, 0.0) + float(value)
    return out


@torch.inference_mode()
def evaluate_pretrain(model: torch.nn.Module, data_loader: Iterable,
                      device: torch.device, *, mask_base: int,
                      epoch: int = 0) -> dict:
    """One deterministic validation pass (ADR-001 I6, Criterion 2).

    The recipe is fixed here rather than left to the caller, because a
    validation curve is only comparable across epochs if every stochastic part
    of the forward path is pinned:

    * ``model.eval()`` and ``torch.inference_mode()`` -- no dropout, no graph;
    * **no autocast**: the pass runs in FP32 so two evaluations of one
      checkpoint compare equal bit-for-bit;
    * the mask generators are re-created per batch as
      ``protocol.val_mask_seed(mask_base, batch_idx)`` -- the same sequence of
      modes, counts and placements every epoch **within a phase**, different
      across batches. The epoch slot of the generator stays ``0`` on purpose:
      only the THRESHOLD ``p(epoch)`` of ADR-005 §2.7 moves, so the draw
      sequence is replayed, not resampled. No
      per-stream fraction is hand-pinned (ADR-002 §6): the policy replays a
      deterministic DRAW sequence instead;
    * the caller supplies an ORDERED, single-worker loader (``shuffle=False``,
      ``drop_last=False``, ``num_workers=0``).

    DDP: the loader is SHARDED, so each rank scores only its own slice of the
    validation role. The per-rank ``(loss_sum, n_samples, components, kept,
    mode, resolved)`` payloads are therefore all-gathered before averaging;
    without that, the reported ``val_loss`` silently depended on the WORLD SIZE
    (a 4-rank run reported a quarter of the val set as if it were the whole).

    :returns: ``{'val_loss', 'components': {'mse_<stream>': …}, 'kept':
        {'kept_<stream>': …}, 'mode': {'mode1': …, 'mode2': …},
        'resolved': {'mode1': …, 'mode2': …}, 'single': …, 'stratification':
        {'core_tubes_<stream>': …, 'periph_tubes_<stream>': …},
        'objective': {'resp_mse', 'resp_pearson', 'resp_spectral',
        'loss_align', 'align_active'} — ADR-008 §2.7, EMPTY unless the config
        enables an objective term; a sibling of ``components``, which stays the
        MSE component set,
        'n_samples', 'n_batches', 'mask_base', 'pinned'}`` — ``mode`` counts
        the DRAWN modes, ``resolved`` the contract-4 modes, so their Mode-1
        shares differ by the ``K_tilde = 0`` draws (ADR-005 §2.8).
        ``stratification`` holds the ADR-007 tube counts in TUBES (mean
        ``S_core`` / ``S_periph``); it is a SIBLING of ``components``, which
        stays the MSE component set.
    """
    from core.budget_masking import REGIME_SINGLE
    from core.mask_placement import counts_generator, placement_generator
    from protocol import val_mask_seed

    was_training = model.training
    model.eval()
    epoch_kwarg = _accepts_epoch(model.forward)
    total_loss = 0.0
    n_samples = 0
    n_batches = 0
    components: dict = {}
    kept_totals: dict = {}
    mode_totals: dict = {}
    resolved_totals: dict = {}
    single_totals: dict = {}
    strat_totals: dict = {}
    objective_totals: dict = {}

    for batch_idx, batch in enumerate(data_loader):
        seed = int(val_mask_seed(mask_base, batch_idx))
        if isinstance(batch, dict):
            x = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(x,
                        counts_generator=counts_generator(seed, 0, batch_idx),
                        placement_generator=placement_generator(
                            seed, 0, batch_idx, rank=0),
                        **({'epoch': epoch} if epoch_kwarg else {}))
        else:
            samples = batch[0].to(device, non_blocking=True)
            out = model(samples)
        if not isinstance(out, dict) or 'loss' not in out:
            raise SystemExit(
                '[val] the model returned no masked loss; evaluate_pretrain '
                'needs the multimodal MAE forward, not a placeholder model.')
        batch_n = int(next(iter(out['masks'].values())).shape[0])
        total_loss += float(out['loss']) * batch_n
        n_samples += batch_n
        n_batches += 1
        for name, value in (out.get('losses_mse') or {}).items():
            components.setdefault(f'mse_{name}', 0.0)
            components[f'mse_{name}'] += float(value) * batch_n
        for name, value in (out.get('kept') or {}).items():
            kept_totals.setdefault(f'kept_{name}', 0.0)
            kept_totals[f'kept_{name}'] += float(value) * batch_n
        # ADR-007 §2.9: tube counts, kept OUT of ``components`` (which is the
        # MSE component set — ``test_stage2_protocol`` pins that) and out of
        # ``kept`` (which is in TOKENS), so the units of each dict stay honest.
        for name, value in (out.get('kept_core') or {}).items():
            strat_totals.setdefault(f'core_tubes_{name}', 0.0)
            strat_totals[f'core_tubes_{name}'] += float(value) * batch_n
        for name, value in (out.get('kept_periph') or {}).items():
            strat_totals.setdefault(f'periph_tubes_{name}', 0.0)
            strat_totals[f'periph_tubes_{name}'] += float(value) * batch_n
        # ADR-008 §2.7: the objective summary block. A SIBLING of `components`
        # (which stays the MSE component set -- ``test_stage2_protocol`` pins
        # that), sample-weighted like everything else here. Empty for a run
        # whose config enables no objective term, so a neutral validation pass
        # returns exactly the pre-ADR-008 dict.
        for name, value in (out.get('objective') or {}).items():
            objective_totals.setdefault(name, 0.0)
            objective_totals[name] += float(value) * batch_n
        if 'mode' in out:
            mode_totals.setdefault(f'mode{int(out["mode"])}', 0.0)
            mode_totals[f'mode{int(out["mode"])}'] += batch_n
        if 'mode_resolved' in out:
            resolved_totals.setdefault(f'mode{int(out["mode_resolved"])}', 0.0)
            resolved_totals[f'mode{int(out["mode_resolved"])}'] += batch_n
        # ADR-006 §2.8: the realized step-class shares of the validation pass.
        # Sample-weighted, so the payload is an average over the corpus and not
        # over the batches.
        if 'regime' in out:
            single_totals.setdefault('single', 0.0)
            single_totals['single'] += float(out['regime'] == REGIME_SINGLE) * batch_n
            carrier = out.get('carrier')
            for name in getattr(model, 'visual', []) or []:
                single_totals.setdefault(name, 0.0)
                single_totals[name] += float(carrier == name) * batch_n

    model.train(was_training)
    if n_samples == 0:
        raise SystemExit(
            '[val] the validation loader produced no sample; an empty '
            'validation pass cannot produce a criterion value (I7).')

    # DDP: fold every rank's slice into the corpus-wide value. The payload is a
    # picklable dict, so dict KEYS may differ between ranks (e.g. only one rank
    # drew mode 2); _merge_sums adds them key-by-key and missing keys count 0.
    payload = {
        'loss_sum': total_loss, 'n_samples': n_samples,
        'n_batches': n_batches, 'components': components,
        'kept': kept_totals, 'mode': mode_totals,
        'resolved': resolved_totals,
        'single': single_totals, 'stratification': strat_totals,
        'objective': objective_totals,
    }
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, payload)
        total_loss = sum(float(g['loss_sum']) for g in gathered)
        n_samples = sum(int(g['n_samples']) for g in gathered)
        n_batches = sum(int(g['n_batches']) for g in gathered)
        components = _merge_sums([g['components'] for g in gathered])
        kept_totals = _merge_sums([g['kept'] for g in gathered])
        mode_totals = _merge_sums([g['mode'] for g in gathered])
        resolved_totals = _merge_sums([g.get('resolved') or {}
                                       for g in gathered])
        single_totals = _merge_sums([g.get('single') or {}
                                     for g in gathered])
        strat_totals = _merge_sums([g.get('stratification') or {}
                                   for g in gathered])
        objective_totals = _merge_sums([g.get('objective') or {}
                                       for g in gathered])
        if n_samples == 0:
            raise SystemExit(
                '[val] the gathered validation pass contains no sample across '
                'any rank; the criterion value is undefined (I7).')

    return {
        'val_loss': total_loss / n_samples,
        'components': {k: v / n_samples for k, v in components.items()},
        'kept': {k: v / n_samples for k, v in kept_totals.items()},
        'mode': {k: v / n_samples for k, v in mode_totals.items()},
        'resolved': {k: v / n_samples for k, v in resolved_totals.items()},
        'single': {k: v / n_samples for k, v in single_totals.items()},
        'stratification': {k: v / n_samples for k, v in strat_totals.items()},
        'objective': {k: v / n_samples for k, v in objective_totals.items()},
        'n_samples': n_samples,
        'n_batches': n_batches,
        'mask_base': int(mask_base),
        'pinned': {},
    }
