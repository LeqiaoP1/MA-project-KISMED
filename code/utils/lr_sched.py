"""Learning-rate schedules.

Port of MultiMAE / VideoMAE ``cosine_scheduler`` (the reference run scripts
build a *step-level* schedule and hand it to the training loop, which then sets
``param_group['lr']`` every optimizer step):

    lr_schedule_values = cosine_scheduler(args.lr, args.min_lr, args.epochs,
                                          steps_per_epoch,
                                          warmup_epochs=args.warmup_epochs)

Returns a flat numpy array of length ``epochs * niter_per_ep``: linear warmup
from ``start_warmup_value`` up to ``base_value`` over ``warmup_epochs``, then a
cosine decay from ``base_value`` down to ``final_value``.
"""
import math

import numpy as np

__all__ = ['cosine_scheduler']


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep,
                     warmup_epochs=0, start_warmup_value=0.0, warmup_steps=-1):
    """Step-level cosine LR schedule with optional linear warmup."""
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print('Set warmup steps = %d' % warmup_iters)

    warmup_schedule = np.array([])
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value,
                                      warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array([
        final_value + 0.5 * (base_value - final_value)
        * (1 + math.cos(math.pi * i / len(iters)))
        for i in iters
    ]) if len(iters) > 0 else np.array([])

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep, \
        f'{len(schedule)} != {epochs * niter_per_ep}'
    return schedule
