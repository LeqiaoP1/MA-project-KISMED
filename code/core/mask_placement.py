"""Mask placement for the constant-budget policy (ADR-002, Phase 2).

The apportionment engine (:mod:`core.budget_masking`) decides HOW MANY tokens
each stream keeps. This module decides WHICH ones, per sample, and never looks
at the policy: it receives a count and returns the flattened encoder indices of
the kept tokens.

Two placements exist, matching the two kinds of masking unit:

* :func:`place_video` — a visual unit is one spatial tube, i.e. one spatial
  patch together with all of its temporal positions. Keeping a unit adds a
  whole temporal column, so the masked video stays a dense rectangle with no
  padding.
* :func:`place_signal` — a physiological unit is one temporal token. The
  indices are either ``'scattered'`` (uniform without replacement) or
  ``'span'`` (contiguous blocks). The choice is a run setting and is recorded
  with the run.
* :func:`place_video_stratified` — ADR-007 §2.3: the same tube placement, but
  drawn from TWO strata (an anatomical core subgrid and its periphery) so at
  least one core tube survives in every sample.

Counts are identical for every sample of a step (ADR-002 contract 6), so both
functions produce a rectangular ``[B, k]`` index tensor; a per-sample count that
differs from its siblings is rejected rather than padded, because padding would
break the constant-budget contract the policy exists to provide.

Seeding
-------
:func:`counts_generator` and :func:`placement_generator` derive a generator
from a 64-bit mix of ``(base_seed, epoch, step[, rank])``:

* **counts** omit the rank, so every DDP rank independently derives the SAME
  plan — no collective, no serialisation, no host synchronisation;
* **placement** include the rank, so the samples differ across ranks while
  staying reproducible for a given ``(seed, rank, epoch, step)``.

The mix is a hash, not an arithmetic stride: ``rank * 10000 + step`` would
collide once the run exceeds 10 000 steps, while a 64-bit mix does not.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch

__all__ = ['SCATTERED', 'SPAN', 'PLACEMENT_MODES', 'place_video',
           'place_video_stratified', 'place_signal', 'counts_generator',
           'placement_generator', 'step_generators']

#: scattered placement: uniform without replacement over the token axis.
SCATTERED = 'scattered'
#: span placement: contiguous blocks of (approximately) ``span_tokens``.
SPAN = 'span'
PLACEMENT_MODES: Tuple[str, str] = (SCATTERED, SPAN)

_MASK64 = (1 << 64) - 1
_I63 = (1 << 63) - 1
_COUNTS_TAG = 0x00C0_0000_0000_0001
_PLACE_TAG = 0x00C0_0000_0000_0002


# --------------------------------------------------------------------------- #
# seeding
# --------------------------------------------------------------------------- #
def _mix(x: int) -> int:
    """SplitMix64 finaliser: a cheap, well-spread 64-bit mix."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (x ^ (x >> 31)) & _MASK64


def _seed(base_seed: int, tag: int, *parts: int) -> int:
    """Positive int64 seed from ``(base_seed, tag, *parts)``."""
    h = _mix(int(base_seed) & _MASK64)
    h = _mix(h ^ (int(tag) & _MASK64))
    for part in parts:
        h = _mix(h ^ (int(part) & _MASK64))
    return h & _I63


def counts_generator(base_seed: int, epoch: int, step: int,
                     device: Union[str, torch.device] = 'cpu'
                     ) -> torch.Generator:
    """RNG of the step's PLAN — rank-independent, so all ranks agree."""
    g = torch.Generator(device=device)
    g.manual_seed(_seed(base_seed, _COUNTS_TAG, int(epoch), int(step)))
    return g


def placement_generator(base_seed: int, epoch: int, step: int, rank: int,
                        device: Union[str, torch.device] = 'cpu'
                        ) -> torch.Generator:
    """RNG of the step's PLACEMENT — distinct per rank, replayable."""
    g = torch.Generator(device=device)
    g.manual_seed(_seed(base_seed, _PLACE_TAG, int(epoch), int(step),
                        int(rank)))
    return g


def step_generators(base_seed: int, epoch: int, step: int, rank: int = 0,
                    device: Union[str, torch.device] = 'cpu'):
    """``(counts_generator, placement_generator)`` of one step."""
    return (counts_generator(base_seed, epoch, step, device=device),
            placement_generator(base_seed, epoch, step, rank, device=device))


# --------------------------------------------------------------------------- #
# count normalisation
# --------------------------------------------------------------------------- #
def _uniform_counts(counts, n_max: int, batch_size: Optional[int],
                    what: str) -> Tuple[int, int]:
    """``(k, B)`` from an int or a per-sample sequence of identical counts.

    A step's count is one scalar shared by the whole batch (ADR-002 contract 6),
    so a sequence with differing entries is an error: the encoder input would
    stop being a dense ``[B, n_budget, D]`` tensor.
    """
    if isinstance(counts, torch.Tensor):
        values = [int(v) for v in counts.flatten().tolist()]
    elif isinstance(counts, (list, tuple)):
        values = [int(v) for v in counts]
    else:
        b = 1 if batch_size is None else int(batch_size)
        if b < 1:
            raise ValueError(f'{what}: batch_size must be >= 1; got {b}.')
        values = [int(counts)] * b
    if not values:
        values = [0]
    if batch_size is not None and len(values) != int(batch_size):
        raise ValueError(
            f'{what}: got {len(values)} per-sample count(s) but batch_size '
            f'{int(batch_size)}.')
    if any(v < 0 for v in values):
        raise ValueError(f'{what}: counts must be >= 0; got {values}.')
    if len(set(values)) > 1:
        raise ValueError(
            f'{what}: the batch must share ONE count per step (ADR-002 '
            f'contract 6); got {sorted(set(values))}. Sharing counts keeps the '
            f'encoder input dense; placements are what vary per sample.')
    k = values[0]
    if k > int(n_max):
        raise ValueError(
            f'{what}: count {k} exceeds the {int(n_max)} available unit(s).')
    return k, len(values)


# --------------------------------------------------------------------------- #
# placement
# --------------------------------------------------------------------------- #
def place_video(counts, n_units: int, tokens_per_unit: int,
                generator: Optional[torch.Generator] = None,
                batch_size: Optional[int] = None,
                device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
    """Indices of the kept VENETIAN-BLIND columns of a visual stream.

    :param counts: kept spatial tubes, one scalar for the whole batch (an int,
        or a per-sample sequence of identical values).
    :param n_units: spatial patches ``Gh*Gw``.
    :param tokens_per_unit: tokens per tube, i.e. the tubelet count ``Gt``.
    :param generator: optional seeded RNG (placement only).
    :param batch_size: number of samples when ``counts`` is a scalar.
    :param device: device of the returned index tensor.
    :return: ``[B, k*tokens_per_unit]`` long tensor of kept encoder indices,
        ascending per row. Encoder index ``t*n_units + s`` — the flatten order
        of ``[Gt, Gh*Gw]`` used by the tubelet adapter.
    """
    n_units = int(n_units)
    tpu = int(tokens_per_unit)
    if n_units < 1 or tpu < 1:
        raise ValueError(
            f'place_video: n_units and tokens_per_unit must be >= 1; got '
            f'{n_units}, {tpu}.')
    k, b = _uniform_counts(counts, n_units, batch_size, 'place_video')
    if k == 0:
        return torch.zeros(b, 0, dtype=torch.long, device=device)
    # draw on CPU so the SAME generator state reproduces the same placement on
    # every device, then move only the (small) index tensor
    perm = torch.rand(b, n_units, generator=generator).argsort(dim=1)
    spatial = perm[:, :k]                              # [B, k]
    offsets = (torch.arange(tpu, dtype=torch.long).view(1, tpu, 1) * n_units)
    idx = spatial.view(b, 1, k) + offsets              # [B, tpu, k]
    return idx.reshape(b, tpu * k).sort(dim=1).values.to(device)


def place_video_stratified(counts_core, counts_periph, n_units: int,
                           tokens_per_unit: int, core_indices: Sequence[int],
                           generator: Optional[torch.Generator] = None,
                           batch_size: Optional[int] = None,
                           device: Union[str, torch.device] = 'cpu'
                           ) -> torch.Tensor:
    """Kept VENETIAN-BLIND columns drawn from two spatial strata (ADR-007 §2.3).

    The policy splits a step's kept-tube budget into ``S_core`` tubes drawn from
    the anatomical core :math:`\\mathcal{A}_{core}` and ``S_periph = S_tir -
    S_core`` drawn from its complement. Drawing them as ONE uniform set would
    miss a 4-tube core on 35-61 % of samples (ADR-007 §1.2 pressure 2), so the
    two strata are sampled independently and, within each, without replacement.

    Both counts are per-STEP scalars shared by the batch (ADR-002 contract 6);
    only the chosen indices vary per sample. ``counts_core >= 1`` is required,
    which is what makes ``P(core hit) = 1`` rather than a probability.

    RNG order is fixed and documented: the core draw comes first, then the
    periphery draw, and NEITHER consumes a sample when its count is 0 — so a
    caller can reason about the generator stream from the counts alone.

    :param counts_core: kept tubes from ``A_core`` (one scalar, or a per-sample
        sequence of identical values). Must be >= 1.
    :param counts_periph: kept tubes from ``A_periph``.
    :param n_units: spatial patches ``Gh*Gw``.
    :param tokens_per_unit: tokens per tube, i.e. the tubelet count ``Gt``.
    :param core_indices: the ``A_core`` tube indices on the ``Gh*Gw`` grid, in
        conv flatten order (``h*G_w + w``), as resolved by
        :class:`core.budget_masking.ConstantBudgetApportioner`.
    :param generator: optional seeded RNG (placement only).
    :param batch_size: number of samples when a count is a scalar.
    :param device: device of the returned index tensor.
    :return: ``[B, (k_core + k_periph) * tokens_per_unit]`` long tensor of kept
        encoder indices, ascending per row.
    """
    n_units = int(n_units)
    tpu = int(tokens_per_unit)
    if n_units < 1 or tpu < 1:
        raise ValueError(
            f'place_video_stratified: n_units and tokens_per_unit must be '
            f'>= 1; got {n_units}, {tpu}.')
    core = tuple(int(i) for i in core_indices)
    if not core:
        raise ValueError(
            'place_video_stratified: core_indices is empty; the anatomical '
            'core must contain at least one tube (ADR-007 §2.6 rule 5b).')
    if len(set(core)) != len(core):
        raise ValueError(
            f'place_video_stratified: core_indices has duplicates: {core}.')
    if min(core) < 0 or max(core) >= n_units:
        raise ValueError(
            f'place_video_stratified: core_indices must lie in '
            f'[0, {n_units - 1}]; got {sorted(core)}.')
    core_set = set(core)
    periph = tuple(i for i in range(n_units) if i not in core_set)
    if not periph:
        raise ValueError(
            f'place_video_stratified: A_periph is empty (A_core covers all '
            f'{n_units} tubes), so no peripheral context could be kept.')

    k_core, b = _uniform_counts(counts_core, len(core), batch_size,
                                'place_video_stratified(counts_core)')
    if k_core < 1:
        raise ValueError(
            'place_video_stratified: counts_core must be >= 1 — a stratified '
            'step always keeps at least one core tube (ADR-007 §2.3); got 0.')
    k_periph, _ = _uniform_counts(counts_periph, len(periph), b,
                                  'place_video_stratified(counts_periph)')

    core_t = torch.as_tensor(core, dtype=torch.long)
    periph_t = torch.as_tensor(periph, dtype=torch.long)
    perm_core = torch.rand(b, len(core), generator=generator).argsort(dim=1)
    spatial_core = core_t[perm_core[:, :k_core]]            # [B, k_core]
    if k_periph:
        perm_periph = torch.rand(b, len(periph),
                                 generator=generator).argsort(dim=1)
        spatial_periph = periph_t[perm_periph[:, :k_periph]]  # [B, k_periph]
        spatial = torch.cat([spatial_core, spatial_periph], dim=1)
    else:
        spatial = spatial_core
    k = k_core + k_periph

    offsets = (torch.arange(tpu, dtype=torch.long).view(1, tpu, 1) * n_units)
    idx = spatial.view(b, 1, k) + offsets                   # [B, tpu, k]
    idx = idx.reshape(b, tpu * k).sort(dim=1).values
    # Per-row contract: exactly k distinct columns, and exactly k_core*T of the
    # kept ENCODER indices inside A_core (``idx % n_units`` recovers the spatial
    # position, which is what makes "P(core hit) = 1" checkable, not assumed).
    if idx.shape[1] and not bool(torch.all(idx[:, 1:] > idx[:, :-1])):
        raise RuntimeError(
            'place_video_stratified: a sample received a duplicate kept index '
            '-- the two strata must be disjoint (ADR-007 §2.3).')
    core_tokens = torch.isin(idx % n_units, core_t).sum(dim=1)
    if not bool(torch.all(core_tokens == k_core * tpu)):
        raise RuntimeError(
            f'place_video_stratified: every sample must keep exactly '
            f'{k_core} core tube(s) = {k_core * tpu} core token(s); got '
            f'{core_tokens.tolist()}.')
    return idx.to(device)


def place_signal(counts, n_units: int,
                 generator: Optional[torch.Generator] = None,
                 mode: str = SCATTERED,
                 span_tokens: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
    """Indices of the kept temporal tokens of a physiological stream.

    :param counts: kept tokens, one scalar for the whole batch.
    :param n_units: temporal tokens ``seq_len // sig_kernel``.
    :param generator: optional seeded RNG (placement only).
    :param mode: :data:`SCATTERED` or :data:`SPAN`.
    :param span_tokens: span placement only — length of one contiguous block.
        The exact kept count still comes from ``counts``: the tokens are split
        over ``ceil(k/span_tokens)`` near-equal blocks, so the count is never
        rounded.
    :param batch_size: number of samples when ``counts`` is a scalar.
    :param device: device of the returned index tensor.
    :return: ``[B, k]`` long tensor of kept token indices, ascending per row.
    """
    n_units = int(n_units)
    if n_units < 1:
        raise ValueError(f'place_signal: n_units must be >= 1; got {n_units}.')
    mode = str(mode or SCATTERED).strip().lower()
    if mode not in PLACEMENT_MODES:
        raise ValueError(
            f'place_signal: mode must be one of {PLACEMENT_MODES}; got '
            f'{mode!r}.')
    k, b = _uniform_counts(counts, n_units, batch_size, 'place_signal')
    if k == 0:
        return torch.zeros(b, 0, dtype=torch.long, device=device)
    if mode == SCATTERED:
        perm = torch.rand(b, n_units, generator=generator).argsort(dim=1)
        return perm[:, :k].sort(dim=1).values.to(device)

    # ---- span placement: contiguous blocks, exact count ------------------ #
    span = int(span_tokens or 0)
    if span < 1:
        raise ValueError(
            f"place_signal: mode='span' needs span_tokens >= 1; got "
            f'{span_tokens!r}.')
    n_blocks = int(math.ceil(k / span))
    base, extra = divmod(k, n_blocks)
    sizes = [base + (1 if i < extra else 0) for i in range(n_blocks)]
    slack = n_units - k
    cuts = torch.rand(b, n_blocks + 1, generator=generator)
    cuts = cuts / cuts.sum(dim=1, keepdim=True)
    gaps = (cuts * slack).long()
    gaps[:, -1] += slack - gaps.sum(dim=1)     # floor() loses at most n_blocks

    rows = []
    for b_i in range(b):
        pos = 0
        parts = []
        for i, size in enumerate(sizes):
            pos += int(gaps[b_i, i])
            parts.append(torch.arange(pos, pos + size, dtype=torch.long))
            pos += size
        rows.append(torch.cat(parts))
    return torch.stack(rows, dim=0).to(device)
