"""Contract tests for the mask placement layer (ADR-002, Phase 2).

Design: ``code/notes/design/ADR-002-asymmetric-multimodal-masking.md``.
Plan + gate ids: ``code/notes/plan/ADR-002-constant-budget-masking-plan.md``
(Phase 2 and §8 G10a/G17).
"""
import pytest
import torch

from core.mask_placement import (SCATTERED, SPAN, counts_generator,
                                 place_signal, place_video,
                                 placement_generator, step_generators)

N_SPATIAL = 36
T_TOKENS = 25


# --------------------------------------------------------------------------- #
# place_video — a kept unit is a whole temporal column
# --------------------------------------------------------------------------- #
def test_video_placement_keeps_whole_temporal_columns():
    B, k = 4, 5
    idx = place_video(k, N_SPATIAL, T_TOKENS, batch_size=B,
                      generator=torch.Generator().manual_seed(0))
    assert idx.shape == (B, k * T_TOKENS)
    for row in idx:
        assert row.unique().numel() == row.numel()          # no duplicates
        assert torch.equal(row, row.sort().values)          # ascending
        spatial, temporal = row % N_SPATIAL, row // N_SPATIAL
        assert temporal.unique().numel() == T_TOKENS        # every tube column
        assert spatial.unique().numel() == k                # exactly k patches
        assert torch.equal(temporal.unique(),
                           torch.arange(T_TOKENS, dtype=torch.long))


def test_video_placement_differs_across_samples_of_one_batch():
    idx = place_video(5, N_SPATIAL, T_TOKENS, batch_size=8,
                      generator=torch.Generator().manual_seed(1))
    distinct = {tuple(row.tolist()) for row in idx}
    assert len(distinct) == 8


def test_zero_kept_yields_an_empty_index_set():
    idx = place_video(0, N_SPATIAL, T_TOKENS, batch_size=3)
    assert idx.shape == (3, 0)


def test_video_placement_rejects_out_of_range_and_ragged_counts():
    with pytest.raises(ValueError, match='exceeds'):
        place_video(N_SPATIAL + 1, N_SPATIAL, T_TOKENS)
    with pytest.raises(ValueError, match='ONE count per step'):
        place_video([2, 3], N_SPATIAL, T_TOKENS)
    with pytest.raises(ValueError, match='must be >= 0'):
        place_video(-1, N_SPATIAL, T_TOKENS)


# --------------------------------------------------------------------------- #
# place_signal — scattered and span
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('mode,span', [(SCATTERED, None), (SPAN, 10)])
def test_signal_placement_returns_exactly_the_requested_count(mode, span):
    B, k, n = 4, 30, 250
    idx = place_signal(k, n, generator=torch.Generator().manual_seed(2),
                       mode=mode, span_tokens=span, batch_size=B)
    assert idx.shape == (B, k)
    for row in idx:
        assert row.unique().numel() == k
        assert torch.equal(row, row.sort().values)
        assert int(row.min()) >= 0 and int(row.max()) < n


def test_scattered_signal_placement_is_not_contiguous():
    idx = place_signal(50, 250, generator=torch.Generator().manual_seed(3),
                       mode=SCATTERED, batch_size=32)
    assert not any(torch.all(torch.diff(row) == 1) for row in idx)


def test_span_placement_forms_contiguous_blocks_of_the_requested_length():
    k, n, span = 30, 250, 10
    idx = place_signal(k, n, generator=torch.Generator().manual_seed(4),
                       mode=SPAN, span_tokens=span, batch_size=8)
    for row in idx:
        breaks = torch.where(torch.diff(row) != 1)[0] + 1
        bounds = torch.cat([torch.zeros(1, dtype=torch.long), breaks,
                            torch.tensor([row.numel()])])
        runs = [row[int(bounds[i]):int(bounds[i + 1])]
                for i in range(len(bounds) - 1)]
        assert sum(r.numel() for r in runs) == k
        assert all(r.numel() <= span for r in runs)
        assert len(runs) == -(-k // span)        # ceil(k / span) blocks


def test_span_placement_needs_a_span_length():
    with pytest.raises(ValueError, match='span_tokens'):
        place_signal(10, 250, mode=SPAN)


def test_signal_placement_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match='mode must be one of'):
        place_signal(10, 250, mode='blocky')


def test_zero_kept_signal_is_an_empty_index_set():
    assert place_signal(0, 250, batch_size=2).shape == (2, 0)


# --------------------------------------------------------------------------- #
# G10a — counts -> placement equality, placements differ per sample
# --------------------------------------------------------------------------- #
def test_placement_counts_equal_the_plan_counts_and_vary_per_sample():
    plan_counts = {'tir': 125, 'rgb2d': 100, 'resp': 25}
    gen = torch.Generator().manual_seed(5)
    vid = {name: place_video(tokens // T_TOKENS, N_SPATIAL, T_TOKENS,
                             generator=gen, batch_size=8)
           for name, tokens in plan_counts.items() if name != 'resp'}
    sig = place_signal(plan_counts['resp'], 250, generator=gen, batch_size=8)
    for name, idx in vid.items():
        assert idx.shape[1] == plan_counts[name]
        assert len({tuple(row.tolist()) for row in idx}) == 8
    assert sig.shape[1] == plan_counts['resp']
    assert len({tuple(row.tolist()) for row in sig}) == 8
    assert sum(i.shape[1] for i in vid.values()) + sig.shape[1] == 250


# --------------------------------------------------------------------------- #
# G17 — rank-distinct, collision-free placement seeds
# --------------------------------------------------------------------------- #
def test_counts_generators_agree_across_ranks_and_differ_across_steps():
    a = counts_generator(42, epoch=0, step=7)
    b = counts_generator(42, epoch=0, step=7)
    c = counts_generator(42, epoch=0, step=8)
    d = counts_generator(43, epoch=0, step=7)
    x = torch.rand(8, generator=a)
    assert torch.equal(x, torch.rand(8, generator=b))
    assert not torch.equal(x, torch.rand(8, generator=c))
    assert not torch.equal(x, torch.rand(8, generator=d))


def test_placement_seeds_are_distinct_for_every_rank_and_step():
    seen = {}
    for rank in range(4):
        for step in range(500):
            g = placement_generator(42, epoch=3, step=step, rank=rank)
            key = int(torch.randint(0, 2 ** 62, (1,), dtype=torch.int64,
                                    generator=g).item())
            assert key not in seen or seen[key] == (rank, step), key
            seen[key] = (rank, step)
    assert len(seen) == 4 * 500


def test_step_generators_return_two_independent_streams():
    counts, place = step_generators(42, epoch=1, step=2, rank=3)
    assert not torch.equal(torch.rand(4, generator=counts),
                           torch.rand(4, generator=place))
