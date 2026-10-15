"""Pure helpers in ``utils``: the LR schedule, top-k accuracy, pos-embeddings.

These are the small functions a training run leans on every step, and they were
entirely untested. All three are CPU-only.

The ``cosine_scheduler`` cases pin the trap that cost two runs: the schedule is
REBUILT from ``epochs`` on every launch (it is not restored with a checkpoint),
and it asserts ``len(schedule) == epochs * niter_per_ep`` -- so a tiny ``epochs``
combined with a config's ``warmup_epochs`` aborts BEFORE any step runs.
"""
import math

import numpy as np
import pytest
import torch

from utils.lr_sched import cosine_scheduler
from utils.metrics import accuracy
from utils.pos_embed import (get_1d_sincos_pos_embed, get_2d_sincos_pos_embed,
                             get_3d_sincos_pos_embed)


# --------------------------------------------------------------------------- #
# cosine_scheduler
# --------------------------------------------------------------------------- #
def test_cosine_scheduler_has_one_value_per_step(capsys):
    s = cosine_scheduler(1e-3, 1e-5, epochs=3, niter_per_ep=4)
    assert s.shape == (12,)
    assert 'warmup steps' in capsys.readouterr().out


def test_cosine_scheduler_decays_from_base_to_final():
    s = cosine_scheduler(1e-3, 1e-5, epochs=10, niter_per_ep=5)
    assert s[0] == pytest.approx(1e-3)
    # the cosine only REACHES final at i == len, so the last step is just above
    assert s[-1] == pytest.approx(1e-5, abs=1e-6)
    assert np.all(np.diff(s) <= 0), 'the schedule must never increase'


def test_cosine_scheduler_follows_the_cosine_in_its_middle():
    """Not a straight line: a cosine decays SLOWER at the start.

    The step index is divided by the length of the decay block (not by
    ``epochs * niter_per_ep - 1``), so the last value is only approached.
    """
    n = 101
    s = cosine_scheduler(1.0, 0.0, epochs=1, niter_per_ep=n)
    assert s[25] == pytest.approx(0.5 * (1 + math.cos(math.pi * 25 / n)))
    assert s[25] > (1.0 - 25 / 100) + 0.05, 'a linear ramp would give 0.75'


def test_cosine_scheduler_ramps_linearly_through_the_warmup():
    base, start = 1e-3, 0.0
    s = cosine_scheduler(base, 1e-5, epochs=10, niter_per_ep=10, warmup_epochs=2)
    assert s.shape == (100,)
    # 20 warmup steps: 0 -> base, reached AT index 19 (linspace endpoint)
    assert s[0] == pytest.approx(start)
    assert s[19] == pytest.approx(base)
    assert np.all(np.diff(s[:20]) > 0)
    assert s[-1] == pytest.approx(1e-5, abs=1e-6)


def test_cosine_scheduler_warmup_steps_overrides_warmup_epochs():
    s = cosine_scheduler(1.0, 0.0, epochs=10, niter_per_ep=10,
                         warmup_epochs=5, warmup_steps=3)
    assert s[2] == pytest.approx(1.0)
    assert np.all(np.diff(s[3:]) <= 0)


def test_cosine_scheduler_aborts_when_warmup_swallows_the_budget():
    """THE trap: a 1-epoch bench against a config's ``warmup_epochs``.

    ``warmup_iters`` exceeds the total step count, so ``iters`` is empty and the
    length assertion fires -- before a single optimiser step runs.
    """
    with pytest.raises(AssertionError, match=r'1000 != 100'):
        cosine_scheduler(1e-3, 1e-5, epochs=1, niter_per_ep=100, warmup_epochs=10)


# --------------------------------------------------------------------------- #
# accuracy
# --------------------------------------------------------------------------- #
def test_accuracy_top1_and_topk():
    logits = torch.tensor([[0.9, 0.1, 0.0],      # argmax 0, target 0 -> hit
                           [0.1, 0.8, 0.1],      # argmax 1, target 1 -> hit
                           [0.2, 0.3, 0.5],      # argmax 2, target 1 -> top-2 only
                           [0.4, 0.3, 0.3]])     # argmax 0, target 0 -> hit
    target = torch.tensor([0, 1, 1, 0])
    top1, top2 = accuracy(logits, target, topk=(1, 2))
    assert float(top1) == pytest.approx(75.0)     # row 2 (0-based) misses top-1
    assert float(top2) == pytest.approx(100.0)    # ...but its target is top-2


def test_accuracy_is_a_float_tensor_of_percent():
    out = accuracy(torch.eye(3), torch.arange(3))
    assert len(out) == 1
    assert float(out[0]) == pytest.approx(100.0)


def test_accuracy_does_not_build_a_graph():
    logits = torch.randn(4, 5, requires_grad=True)
    out = accuracy(logits, torch.zeros(4, dtype=torch.long))
    assert not out[0].requires_grad


# --------------------------------------------------------------------------- #
# positional embeddings
# --------------------------------------------------------------------------- #
def test_1d_sincos_pos_embed_shape_and_range():
    pe = get_1d_sincos_pos_embed(16, 7)
    assert pe.shape == (7, 16)
    assert np.abs(pe).max() <= 1.0


def test_2d_sincos_pos_embed_shape_with_and_without_cls():
    assert get_2d_sincos_pos_embed(16, 3).shape == (9, 16)
    assert get_2d_sincos_pos_embed(16, 3, cls_token=True).shape == (10, 16)


def test_2d_cls_token_row_is_zero():
    pe = get_2d_sincos_pos_embed(8, 2, cls_token=True)
    np.testing.assert_array_equal(pe[0], np.zeros(8))


def test_3d_sincos_pos_embed_takes_a_t_h_w_grid_size():
    assert get_3d_sincos_pos_embed(24, (2, 3, 4)).shape == (24, 24)
    assert get_3d_sincos_pos_embed(24, (2, 3, 4), cls_token=True).shape == (25, 24)


def test_pos_embeddings_are_deterministic():
    for fn, args in ((get_1d_sincos_pos_embed, (8, 5)),
                     (get_2d_sincos_pos_embed, (8, 4)),
                     (get_3d_sincos_pos_embed, (24, (2, 2, 2)))):
        assert np.array_equal(fn(*args), fn(*args))
