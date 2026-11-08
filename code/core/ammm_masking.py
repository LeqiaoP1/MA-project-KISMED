"""Stream-agnostic masking policy for Asymmetric Multi-Modal Masking (AMMM).

This module answers exactly ONE question: *how many units does each stream keep
in this step*. It knows nothing about modality names, tensor shapes, tubelets or
token grids — the caller passes one :class:`StreamSpec` per declared stream and
receives counts only. Mask *placement* (tube masking for video, scattered/span
masking for 1-D) stays in :class:`core.multimae.MultiModalMAE`, which is the only
place that knows the token grid.

Normative rules
---------------
``code/notes/design/Asymmetric_MultiModal_Masking.md`` §3 (declared-set
precondition, batch-level availability, Dirichlet over the DRAWN streams,
per-kind caps, ``[0, limit]`` clamp, global-visibility guard).
Build order, invariants and gates: ``code/notes/plan/Asymmetric_MultiModal_Masking_plan.md``
(Phase 1, G1-G20).

Rules, in order
---------------
1. **dropout** — each declared stream is dropped with probability ``dropout_p``.
2. **co-existence fallback** — if no ``visual`` (or no ``signal``) stream
   survives, the FIRST declared stream of that kind is re-activated. The target
   provably exists because the declared-set precondition is validated here.
3. **Dirichlet over the drawn streams only** —
   ``drawn = active - overrides``. ``|drawn| >= 2`` uses ``Dir(alpha_drawn)``;
   ``|drawn| == 1`` uses ``Uniform(0, 1)`` (a 1-D Dirichlet is a point mass at
   1.0 and would pin that stream to its cap every step); ``|drawn| == 0`` draws
   nothing at all (an empty Dirichlet is undefined).
4. **asymmetric mapping** — ``r = min(u, cap)``.
5. **discretise** — ``clamp(round_half_up(r * n_units), lo, limit)`` with
   ``limit = n_units`` for a signal stream and
   ``floor(cap * n_units)`` for a visual one. ``lo = min_kept`` for a DRAWN
   stream and ``0`` for an OVERRIDDEN one: an explicit
   ``kept_ratio_override: {tir: 0.0}`` must survive ``min_kept: 1``.
6. **global-visibility guard** — if every active stream rounds to 0, one unit is
   granted so ``sum(kept) >= 1`` and the encoder never receives an empty
   ``[B, 0, D]`` sequence. The recipient is chosen by a total rule: largest
   ``lambda``, else largest pinned override ratio, else the first active stream
   in declared order.

Determinism
-----------
Every draw goes through the optional ``torch.Generator``, so a caller can
isolate mask RNG from model RNG and a test can replay a sequence. Ratios are
drawn on CPU in ``float64`` (the count is a handful of scalars).

The module returns **counts**, never masks: two consecutive steps may share the
counts of a stream while their placements differ, and the counts of a step are
identical for every sample of the batch (dense-gather contract).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch

__all__ = ['VISUAL', 'SIGNAL', 'KINDS', 'StreamSpec', 'MaskPlan', 'plan_step']

#: mask-kind tags. The visual/signal taxonomy itself is owned by
#: ``core.multimae`` (``_VISUAL_STREAMS`` / ``_SIGNAL_STREAMS``); this module
#: only needs the two CATEGORIES, never the modality names.
VISUAL = 'visual'
SIGNAL = 'signal'
KINDS: Tuple[str, str] = (VISUAL, SIGNAL)

_F64 = torch.float64


@dataclass(frozen=True)
class StreamSpec:
    """One declared input stream, described by what the masker needs.

    :param name: the run's own stream name (``'tir'``, ``'resp'``, ...).
    :param kind: :data:`VISUAL` or :data:`SIGNAL`.
    :param n_units: number of masking units — spatial patches for a visual
        stream (``Gh*Gw``, the tube-mask grid), temporal tokens for a signal
        stream (``seq_len // sig_kernel``).
    """

    name: str
    kind: str
    n_units: int


@dataclass(frozen=True)
class MaskPlan:
    """Counts for one training step (identical for every sample of the batch).

    :param active: streams present in this step, in DECLARED order. This is the
        only source of truth for activity: the mask tensors cannot express it
        (an active 0-kept stream also has an all-ones mask).
    :param kept_units: per active stream, how many masking units survive. ``0``
        is legal (the stream is 100 % masked).
    :param r: the realised kept RATIO per active stream, after the cap.
    :param dropped: the streams the dropout draw removed, BEFORE the
        co-existence fallback re-activated any of them.
    """

    active: Tuple[str, ...]
    kept_units: Dict[str, int] = field(default_factory=dict)
    r: Dict[str, float] = field(default_factory=dict)
    dropped: Tuple[str, ...] = ()

    @property
    def kept_total(self) -> int:
        """Total visible units over all active streams (>= 1, by the guard)."""
        return int(sum(self.kept_units.values()))

    def masked_units(self, specs: Sequence[StreamSpec]) -> Dict[str, int]:
        """``n_units - kept`` per active stream (0 when fully kept)."""
        n = {s.name: int(s.n_units) for s in specs}
        return {name: n[name] - kept for name, kept in self.kept_units.items()}


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
def _validate_specs(specs: Sequence[StreamSpec]) -> None:
    """Enforce the declared-set precondition (design §3)."""
    if not specs:
        raise ValueError('AMMM plan_step: no streams declared.')
    names = [s.name for s in specs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f'AMMM plan_step: duplicate stream name(s) {dupes}.')
    for s in specs:
        if s.kind not in KINDS:
            raise ValueError(
                f'AMMM plan_step: stream {s.name!r} has kind {s.kind!r}; '
                f'expected one of {KINDS}.')
        if int(s.n_units) < 1:
            raise ValueError(
                f'AMMM plan_step: stream {s.name!r} has n_units='
                f'{s.n_units}; every stream needs at least one masking unit.')
    for kind in KINDS:
        if not any(s.kind == kind for s in specs):
            have = [s.name for s in specs]
            raise ValueError(
                f'AMMM plan_step: declared set needs >= 1 {kind} stream and '
                f'>= 1 {"signal" if kind == VISUAL else "visual"} stream '
                f'(design §3); got {have}. Video-only or signal-only runs are '
                f'invalid, and the co-existence fallback cannot repair them.')


def _per_stream(value, specs: Sequence[StreamSpec], what: str) -> Dict[str, float]:
    """Normalise a scalar-or-mapping argument to ``{stream: float}``."""
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - {s.name for s in specs})
        if unknown:
            raise ValueError(
                f'AMMM plan_step: {what} names undeclared stream(s) {unknown}; '
                f'declared: {[s.name for s in specs]}.')
        out = {s.name: float(value[s.name]) for s in specs if s.name in value}
        missing = [s.name for s in specs if s.name not in out]
        if missing:
            raise ValueError(
                f'AMMM plan_step: {what} is missing declared stream(s) '
                f'{missing}.')
        return out
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f'AMMM plan_step: {what} must be a number or a mapping; got '
            f'{value!r}')
    return {s.name: scalar for s in specs}


def _validate_caps(caps: Mapping[str, float], specs: Sequence[StreamSpec]) -> None:
    """Every cap must allow at least one kept unit (else the guard cannot fire)."""
    for s in specs:
        cap = float(caps[s.name])
        if not 0.0 <= cap <= 1.0:
            raise ValueError(
                f'AMMM plan_step: cap for {s.name!r} is {cap}; expected '
                f'0.0 <= cap <= 1.0.')
        if _limit(s.kind, cap, int(s.n_units)) < 1:
            raise ValueError(
                f'AMMM plan_step: cap {cap} on {s.name!r} (kind={s.kind}, '
                f'n_units={s.n_units}) allows 0 kept units, so no assignment '
                f'can satisfy the global-visibility guard. Raise the cap to at '
                f'least {1.0 / int(s.n_units):.4f}.')


def _validate_overrides(overrides: Mapping[str, float],
                        specs: Sequence[StreamSpec],
                        caps: Mapping[str, float]) -> None:
    """Overrides are used verbatim: no silent clamping and no unknown names."""
    declared = {s.name for s in specs}
    unknown = sorted(set(overrides) - declared)
    if unknown:
        raise ValueError(
            f'AMMM plan_step: kept_ratio_override names undeclared stream(s) '
            f'{unknown}; declared: {sorted(declared)}.')
    for name, value in overrides.items():
        value = float(value)
        cap = float(caps[name])
        if not 0.0 <= value <= cap:
            raise ValueError(
                f'AMMM plan_step: kept_ratio_override[{name!r}] = {value} is '
                f'outside [0.0, {cap:g}] (the per-stream kept cap). Use a '
                f'value inside the cap, or raise the cap deliberately.')


# --------------------------------------------------------------------------- #
# sampling / arithmetic helpers
# --------------------------------------------------------------------------- #
def _limit(kind: str, cap: float, n_units: int) -> int:
    """Maximum kept units for a stream: the cap for video, the count for 1-D."""
    if kind == SIGNAL:
        return int(n_units)
    return min(int(n_units), int(math.floor(float(cap) * int(n_units))))


def _round_half_up(x: float) -> int:
    """``floor(x + 0.5)`` — the design's ``⌊·⌉``. Not banker's rounding."""
    return int(math.floor(float(x) + 0.5))


def _uniform(n: int, generator: Optional[torch.Generator]) -> torch.Tensor:
    return torch.rand(n, dtype=_F64, generator=generator)


def _sample_dirichlet(alphas: Sequence[float],
                      generator: Optional[torch.Generator]) -> torch.Tensor:
    """Draw one Dirichlet sample with a generator-aware RNG where possible.

    ``alpha == 1`` (the design's value) is sampled exactly as normalised
    ``Exp(1)`` variates, which honours ``generator``. Any other concentration
    falls back to ``torch.distributions.Gamma`` — PyTorch exposes no
    generator-aware gamma sampler — so a seed derived from ``generator`` is used
    and the global CPU RNG state is restored afterwards. That path exists for
    ablations; the default path is fully deterministic.
    """
    a = torch.tensor(list(alphas), dtype=_F64)
    if bool(torch.all(a == 1.0)):
        tiny = torch.finfo(_F64).tiny
        g = -torch.log(_uniform(a.numel(), generator).clamp_min(tiny))
        return g / g.sum()
    seed = int(torch.randint(0, 2 ** 62, (1,), dtype=torch.int64,
                             generator=generator).item())
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        g = torch.distributions.Gamma(a, torch.ones_like(a)).sample()
    finally:
        torch.random.set_rng_state(state)
    return g / g.sum()


def _first_of_kind(specs: Sequence[StreamSpec], kind: str) -> StreamSpec:
    for s in specs:
        if s.kind == kind:
            return s
    raise ValueError(  # unreachable after _validate_specs
        f'AMMM plan_step: no {kind} stream declared.')


def _guard_recipient(active: Sequence[StreamSpec],
                     drawn: Mapping[str, float],
                     overrides: Mapping[str, float]) -> str:
    """Who gets the single unit when the whole step would be invisible.

    Total by construction (design §3.2.3): rank by the drawn ``lambda``, else by
    the pinned override ratio, else fall back to the first active stream in
    declared order. Iterating in declared order makes ties resolve to the first.
    """
    best_name, best_score = active[0].name, None
    for s in active:
        score = drawn.get(s.name, overrides.get(s.name))
        score = -1.0 if score is None else float(score)
        if best_score is None or score > best_score:
            best_name, best_score = s.name, score
    return best_name


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def plan_step(
    specs: Sequence[StreamSpec],
    *,
    caps: Union[Mapping[str, float], float],
    alpha: Union[Mapping[str, float], float] = 1.0,
    dropout_p: float = 0.0,
    min_kept: int = 0,
    overrides: Optional[Mapping[str, float]] = None,
    generator: Optional[torch.Generator] = None,
) -> MaskPlan:
    """Choose how many units each declared stream keeps in this step.

    :param specs: one :class:`StreamSpec` per declared stream, in DECLARED
        order (that order defines the fallback and tie-break priority).
    :param caps: per-stream kept-ratio cap, e.g. ``{'tir': 0.25, 'resp': 1.0}``
        (a scalar applies to every stream). The caller builds it from the
        per-kind defaults + per-stream overrides.
    :param alpha: Dirichlet concentration per stream (scalar or mapping).
    :param dropout_p: probability that a stream is dropped from the step.
    :param min_kept: lower bound for DRAWN streams only (``0`` default — a
        stream may be 100 % masked). Overridden streams are exempt.
    :param overrides: ``kept_ratio_override`` — fixed ratios used verbatim,
        outside the Dirichlet, exempt from ``min_kept``.
    :param generator: optional RNG; pass one for reproducibility.
    :return: :class:`MaskPlan` with counts (never masks).
    :raises ValueError: invalid declared set, cap, override or probability.
    """
    specs = tuple(specs)
    _validate_specs(specs)
    caps_map = _per_stream(caps, specs, 'caps')
    alpha_map = _per_stream(alpha, specs, 'alpha')
    overrides = {str(k): float(v) for k, v in (overrides or {}).items()}
    _validate_caps(caps_map, specs)
    _validate_overrides(overrides, specs, caps_map)
    if not 0.0 <= float(dropout_p) <= 1.0:
        raise ValueError(
            f'AMMM plan_step: dropout_p must be in [0, 1]; got {dropout_p!r}.')
    if int(min_kept) < 0:
        raise ValueError(
            f'AMMM plan_step: min_kept must be >= 0; got {min_kept!r}.')

    # 1. dropout (one vector per STEP, shared by every sample of the batch)
    keep = {s.name: True for s in specs}
    if dropout_p > 0.0:
        u = _uniform(len(specs), generator)
        for i, s in enumerate(specs):
            keep[s.name] = bool(float(u[i]) >= float(dropout_p))
    dropped = tuple(s.name for s in specs if not keep[s.name])

    # 2. co-existence fallback: re-activate the FIRST declared stream per kind
    for kind in KINDS:
        if not any(keep[s.name] for s in specs if s.kind == kind):
            first = _first_of_kind(specs, kind)
            keep[first.name] = True
    active = [s for s in specs if keep[s.name]]     # declared order preserved

    # 3. Dirichlet over the DRAWN streams only
    drawn = [s for s in active if s.name not in overrides]
    vals: Dict[str, float] = {}
    if len(drawn) >= 2:
        lam = _sample_dirichlet([alpha_map[s.name] for s in drawn], generator)
        vals = {s.name: float(lam[i]) for i, s in enumerate(drawn)}
    elif len(drawn) == 1:
        # A 1-D Dirichlet is a point mass at 1.0 -> Uniform(0, 1) instead.
        vals = {drawn[0].name: float(_uniform(1, generator)[0])}

    # 4./5. asymmetric mapping + discretisation
    kept_units: Dict[str, int] = {}
    r_out: Dict[str, float] = {}
    for s in active:
        u_s = vals[s.name] if s.name in vals else overrides[s.name]
        r_s = min(float(u_s), float(caps_map[s.name]))
        limit = _limit(s.kind, float(caps_map[s.name]), int(s.n_units))
        lo = 0 if s.name in overrides else int(min_kept)
        kept = _round_half_up(r_s * int(s.n_units))
        kept_units[s.name] = max(lo, min(kept, limit))
        r_out[s.name] = r_s

    # 6. global-visibility guard: the encoder input must never be empty
    if sum(kept_units.values()) == 0:
        recipient = _guard_recipient(active, vals, overrides)
        kept_units[recipient] = max(kept_units[recipient], 1)

    return MaskPlan(active=tuple(s.name for s in active),
                    kept_units=kept_units, r=r_out, dropped=dropped)
