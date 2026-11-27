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
  indices are ``'scattered'`` (uniform without replacement), ``'span'``
  (contiguous KEPT blocks) or ``'masked_span'`` (contiguous MASKED gaps of
  bounded length, ADR-017). The choice is a run setting and is recorded with
  the run.
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

__all__ = ['SCATTERED', 'SPAN', 'MASKED_SPAN', 'PLACEMENT_MODES', 'place_video',
           'place_video_stratified', 'place_signal', 'admissible_gap_counts',
           'validate_physio_placement', 'counts_generator',
           'placement_generator', 'step_generators']

#: scattered placement: uniform without replacement over the token axis.
SCATTERED = 'scattered'
#: span placement: contiguous blocks of (approximately) ``span_tokens``.
SPAN = 'span'
#: masked-span placement: contiguous MASKED runs of bounded length (ADR-017).
MASKED_SPAN = 'masked_span'
PLACEMENT_MODES: Tuple[str, ...] = (SCATTERED, SPAN, MASKED_SPAN)

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
# masked-span support (ADR-017)
# --------------------------------------------------------------------------- #
def admissible_gap_counts(masked: int, kept: int, gap_min: int,
                          gap_max: int) -> Tuple[int, ...]:
    r"""Gap counts ``n`` that can tile ``masked`` tokens (ADR-017 §2.1).

    .. math::

        \mathcal{N}(M, K) = \{\, n \ge 1 : n\,g_\text{min} \le M \le
        n\,g_\text{max},\;\; n - 1 \le K \,\}

    The ``n - 1 <= kept`` term is what guarantees at least one kept token
    between consecutive gaps (ADR-017 I3): without it two gaps could be
    placed edge to edge and merge into one masked run longer than
    ``gap_max``.

    :param masked: ``M``, the number of tokens to hide.
    :param kept: ``K``, the number of tokens that survive.
    :param gap_min: shortest admissible masked run, in tokens.
    :param gap_max: longest admissible masked run, in tokens.
    :return: the admissible ``n``, ascending; empty when the bounds cannot
        tile ``masked`` (or when ``masked == 0``, where no gap exists).
    """
    m, k = int(masked), int(kept)
    lo, hi = int(gap_min), int(gap_max)
    if lo < 1 or hi < lo:
        raise ValueError(
            f'admissible_gap_counts: need 1 <= gap_min <= gap_max; got '
            f'{lo}, {hi}.')
    if m < 0 or k < 0:
        raise ValueError(
            f'admissible_gap_counts: M and K must be >= 0; got {m}, {k}.')
    if m == 0:
        return ()
    n_lo = max(1, -(-m // hi))        # ceil(M / g_max): n gaps must cover M
    n_hi = min(m // lo, k + 1)        # floor(M / g_min), and n - 1 <= K
    if n_hi < n_lo:
        return ()
    return tuple(range(n_lo, n_hi + 1))


def validate_physio_placement(*, mode: str, n_tokens: int,
                              span_tokens: int = 0,
                              gap_min_tokens: int = 0,
                              gap_max_tokens: int = 0,
                              n_streams: int = 1,
                              kept_counts: Sequence[int] = (),
                              candidate_counts: Sequence[int] = ()) -> None:
    """Refuse a mis-paired or infeasible 1-D placement (ADR-017 §1.1-§2.2, I6).

    ONE validator, called by the model constructor AND by the reconstruction
    probe. The probe overrides ``model.physio_placement`` after the model was
    built, so a check living only in ``__init__`` would never see it — and
    ADR-017 R6 asks for a refusal before the first step, not mid-run.

    The knob pairing is symmetric on purpose: a non-zero ``span_tokens`` under
    ``masked_span`` is rejected as loudly as a gap bound under ``span``, so no
    value can sit in a config looking authoritative while being ignored.

    :param mode: ``scattered``, ``span`` or ``masked_span``.
    :param n_tokens: ``N_s``, the number of respiration tokens of one stream.
    :param span_tokens: kept-block length; must be 0 unless ``mode='span'``.
    :param gap_min_tokens: ``g_min``; must be 0 unless ``mode='masked_span'``.
    :param gap_max_tokens: ``g_max``; must be 0 unless ``mode='masked_span'``.
    :param n_streams: declared physiological streams. ``masked_span`` is
        defined on ONE token axis (ADR-017 §1.1).
    :param kept_counts: kept counts to check for feasibility — the
        apportioner's Mode-2 set, or a single pinned probe count. Empty only
        validates the pairing (the counts are then not known yet).
    :param candidate_counts: counts to NAME as feasible in the failure message,
        when they differ from the ones being checked. A single-point probe
        passes its pinned count as ``kept_counts`` and the arm's Mode-2 set
        here, so an override can be judged on the point it actually scores.
    :raises ValueError: on any violation, naming the numbers.
    """
    mode = str(mode or SCATTERED).strip().lower()
    if mode not in PLACEMENT_MODES:
        raise ValueError(
            f'validate_physio_placement: mode must be one of '
            f'{PLACEMENT_MODES}; got {mode!r}.')
    span = int(span_tokens or 0)
    g_min = int(gap_min_tokens or 0)
    g_max = int(gap_max_tokens or 0)
    n_tok = int(n_tokens)
    if mode == MASKED_SPAN:
        if int(n_streams) != 1:
            raise ValueError(
                'validate_physio_placement: masked_span is defined on one '
                f'token axis (ADR-017 §1.1); got {int(n_streams)} '
                'physiological stream(s).')
        if span:
            raise ValueError(
                'validate_physio_placement: span_tokens must be 0 with '
                f'masked_span (the knobs are exclusive); got {span}.')
        if not 1 <= g_min <= g_max <= n_tok:
            raise ValueError(
                'validate_physio_placement: masked_span needs '
                f'1 <= gap_min <= gap_max <= {n_tok}; got {g_min}, {g_max}.')
    elif g_min or g_max:
        raise ValueError(
            'validate_physio_placement: gap bounds are only meaningful with '
            f"mode='{MASKED_SPAN}'; got mode={mode!r} with "
            f'gap_min={g_min}, gap_max={g_max}.')
    elif mode == SPAN:
        if span < 1:
            raise ValueError(
                "validate_physio_placement: mode='span' needs span_tokens "
                f'>= 1; got {span}.')
    elif span:
        # Symmetric exclusivity (ADR-017 §2.2): a span length left behind under
        # another placement would look authoritative while being ignored.
        raise ValueError(
            'validate_physio_placement: span_tokens must be 0 unless '
            f"mode='{SPAN}' (the knobs are exclusive); got mode={mode!r} with "
            f'span_tokens={span}.')
    if mode != MASKED_SPAN:
        return
    counts = [int(k) for k in kept_counts]
    if not counts:
        return
    # The counts that get CHECKED and the counts NAMED in the failure message
    # are separable on purpose: a single-point probe (one pinned K) may be
    # feasible even when the arm's other Mode-2 counts are not, and the message
    # still has to tell the reader which counts those bounds WOULD tile.
    candidates = [int(c) for c in candidate_counts] or counts
    for k in counts:
        m = n_tok - k
        if m <= 0:
            continue                       # every token kept: no gap at all
        if not admissible_gap_counts(m, k, g_min, g_max):
            feasible = [c for c in candidates
                        if c < n_tok and admissible_gap_counts(
                            n_tok - c, c, g_min, g_max)]
            raise ValueError(
                f'validate_physio_placement: gaps [{g_min}, {g_max}] cannot '
                f'tile M={m} (K={k}, N_s={n_tok}): no n satisfies '
                f'n*{g_min} <= {m} <= n*{g_max} with n - 1 <= {k} (ADR-017 '
                f'I6). Kept counts these bounds CAN tile: {feasible}.')


def _draw_gap_lengths(masked: int, n: int, gap_min: int, gap_max: int,
                      generator: Optional[torch.Generator]) -> list:
    """ADR-017 §2.1 step 3: ``n`` gap lengths summing to ``masked``.

    The excesses are drawn SEQUENTIALLY (a bounded uniform, the last gap
    taking the remainder), then the whole vector is permuted so clip position
    does not bias gap length. The procedure is deliberately not a uniform
    composition — the ADR pins it because the training/probe match (R7) must
    reproduce the same structure.

    :return: ``n`` gap lengths, each in ``[gap_min, gap_max]``, summed to
        ``masked``.
    """
    width = int(gap_max) - int(gap_min)
    remaining = int(masked) - int(n) * int(gap_min)
    excess = []
    for i in range(int(n) - 1):
        later = (int(n) - 1) - i          # gaps still to draw after this one
        lo = max(0, remaining - later * width)
        hi = min(width, remaining)
        draw = int(torch.randint(lo, hi + 1, (1,),
                                 generator=generator).item())
        excess.append(draw)
        remaining -= draw
    excess.append(remaining)              # the last gap takes what is left
    order = torch.randperm(int(n), generator=generator).tolist()
    return [int(gap_min) + excess[j] for j in order]


def _compose_kept_blocks(kept: int, n: int, rank: torch.Tensor) -> list:
    """ADR-017 §2.1 step 4: ``n + 1`` kept blocks from a rank permutation.

    Stars-and-bars: with ``X = K - (n - 1)`` items in ``n + 1`` bins, the
    ``n`` bar positions are a uniform subset of ``X + n = K + 1`` slots. The
    caller supplies the RANK permutation of a ``[b, K + 1]`` uniform draw, so
    its prefix is a uniform ``n``-subset and the width is independent of ``n``.

    :param kept: ``K``, the kept token count.
    :param n: the number of gaps this sample will place.
    :param rank: one row of the batched rank permutation.
    :return: ``b_0 .. b_n``; the interior blocks are ``>= 1``, so two gaps can
        never touch.
    """
    bars = sorted(int(x) for x in rank[:int(n)].tolist())
    blocks = [bars[0]]
    for j in range(1, int(n)):
        blocks.append(1 + (bars[j] - bars[j - 1] - 1))
    blocks.append(int(kept) - bars[-1])
    return blocks


def _assemble_masked_span(blocks, gaps, n_units: int, kept: int,
                          gap_min: int, gap_max: int) -> torch.Tensor:
    """Interleave kept blocks and masked gaps into ascending kept indices."""
    idx: list = []
    pos = 0
    pos += blocks[0]
    idx.extend(range(0, blocks[0]))
    for i, gap in enumerate(gaps):
        pos += int(gap)
        start = pos
        pos += blocks[i + 1]
        idx.extend(range(start, pos))
    out = torch.tensor(idx, dtype=torch.long)
    # cheap post-conditions (ADR-017 I1-I3): they cost O(N_s) per sample and
    # turn a silent placement bug into a loud one at the source.
    if out.numel() != int(kept):
        raise RuntimeError(
            f'masked_span: assembled {out.numel()} kept token(s), expected '
            f'{int(kept)} (N_s={n_units}).')
    if out.numel() > 1 and not bool(torch.all(out[1:] > out[:-1])):
        raise RuntimeError(
            'masked_span: assembled indices are not strictly ascending.')
    if pos != int(n_units):
        raise RuntimeError(
            f'masked_span: assembled {pos} token slot(s), expected {n_units}.')
    keep = torch.zeros(int(n_units), dtype=torch.bool)
    keep[out] = True
    runs, run = [], 0
    for v in (~keep).tolist() + [False]:
        if v:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if len(runs) != len(gaps):
        raise RuntimeError(
            f'masked_span: {len(runs)} masked run(s) for {len(gaps)} drawn '
            'gap(s); a kept block collapsed to zero width (ADR-017 I3).')
    if any(not int(gap_min) <= r <= int(gap_max) for r in runs):
        raise RuntimeError(
            f'masked_span: masked runs {runs} fall outside '
            f'[{int(gap_min)}, {int(gap_max)}] tokens.')
    return out


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
                 gap_min_tokens: Optional[int] = None,
                 gap_max_tokens: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
    """Indices of the kept temporal tokens of a physiological stream.

    :param counts: kept tokens, one scalar for the whole batch.
    :param n_units: temporal tokens ``seq_len // sig_kernel``.
    :param generator: optional seeded RNG (placement only). Drawn on CPU.
    :param mode: :data:`SCATTERED`, :data:`SPAN` or :data:`MASKED_SPAN`.
    :param span_tokens: span placement only — length of one contiguous block.
        The exact kept count still comes from ``counts``: the tokens are split
        over ``ceil(k/span_tokens)`` near-equal blocks, so the count is never
        rounded. Must be unset for the other modes.
    :param gap_min_tokens: masked-span placement only — ``g_min``, the shortest
        masked run (ADR-017). Must be unset for the other modes.
    :param gap_max_tokens: masked-span placement only — ``g_max``, the longest
        masked run (ADR-017). Must be unset for the other modes.
    :param batch_size: number of samples when ``counts`` is a scalar.
    :param device: device of the returned index tensor.
    :return: ``[B, k]`` long tensor of kept token indices, ascending per row.

    The knob pairing is validated BEFORE the ``k == 0`` return: a Mode-1 step
    must not be able to hide a misconfigured tuple until the run's first
    Mode-2 step (ADR-017 R6). The placement branches themselves sit AFTER the
    return, so Mode 1 still consumes no random number (ADR-017 I4 / R4).
    """
    n_units = int(n_units)
    if n_units < 1:
        raise ValueError(f'place_signal: n_units must be >= 1; got {n_units}.')
    mode = str(mode or SCATTERED).strip().lower()
    if mode not in PLACEMENT_MODES:
        raise ValueError(
            f'place_signal: mode must be one of {PLACEMENT_MODES}; got '
            f'{mode!r}.')
    span = int(span_tokens or 0)
    gap_min = int(gap_min_tokens or 0)
    gap_max = int(gap_max_tokens or 0)
    if mode == MASKED_SPAN:
        if span:
            raise ValueError(
                'place_signal: span_tokens must be 0 with masked_span (the '
                f'knobs are exclusive); got {span_tokens!r}.')
        if not 1 <= gap_min <= gap_max <= n_units:
            raise ValueError(
                'place_signal: masked_span needs '
                f'1 <= gap_min_tokens <= gap_max_tokens <= {n_units}; got '
                f'{gap_min_tokens!r}, {gap_max_tokens!r}.')
    elif gap_min or gap_max:
        raise ValueError(
            'place_signal: gap bounds are only meaningful with '
            f"mode='{MASKED_SPAN}'; got mode={mode!r} with "
            f'gap_min_tokens={gap_min_tokens!r}, '
            f'gap_max_tokens={gap_max_tokens!r}.')
    elif mode == SPAN:
        if span < 1:
            raise ValueError(
                f"place_signal: mode='span' needs span_tokens >= 1; got "
                f'{span_tokens!r}.')
    elif span:
        raise ValueError(
            'place_signal: span_tokens must be 0 unless '
            f"mode='{SPAN}' (the knobs are exclusive); got mode={mode!r} with "
            f'span_tokens={span_tokens!r}.')
    k, b = _uniform_counts(counts, n_units, batch_size, 'place_signal')
    if k == 0:
        return torch.zeros(b, 0, dtype=torch.long, device=device)
    if mode == SCATTERED:
        perm = torch.rand(b, n_units, generator=generator).argsort(dim=1)
        return perm[:, :k].sort(dim=1).values.to(device)

    # ---- masked_span: bounded masked gaps, ADR-017 ---------------------- #
    if mode == MASKED_SPAN:
        masked = n_units - k
        if masked == 0:                   # every token kept: nothing to hide
            every = torch.arange(n_units, dtype=torch.long).repeat(b, 1)
            return every.to(device)
        choices = admissible_gap_counts(masked, k, gap_min, gap_max)
        if not choices:
            raise ValueError(
                f'place_signal: no gap count can tile M={masked} (K={k}) '
                f'with bounds [{gap_min}, {gap_max}] (ADR-017 I6).')
        # Draw order (ADR-017 §2.1, pinned): per sample n -> the sequential
        # gap excesses -> the gap permutation; then ONE batched composition
        # draw, whose width K + 1 does not depend on n.
        drawn = []
        for _ in range(b):
            pick = int(torch.randint(len(choices), (1,),
                                     generator=generator).item())
            n = int(choices[pick])
            drawn.append((n, _draw_gap_lengths(masked, n, gap_min, gap_max,
                                               generator)))
        rank = torch.rand(b, k + 1, generator=generator).argsort(dim=1)
        rows = [
            _assemble_masked_span(
                _compose_kept_blocks(k, n, rank[b_i]), gaps,
                n_units, k, gap_min, gap_max)
            for b_i, (n, gaps) in enumerate(drawn)
        ]
        return torch.stack(rows, dim=0).to(device)

    # ---- span placement: contiguous blocks, exact count ------------------ #
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
