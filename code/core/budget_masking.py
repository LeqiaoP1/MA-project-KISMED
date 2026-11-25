"""Constant-budget apportionment engine (ADR-002).

This module answers exactly ONE question: *how many tokens does each declared
stream keep in this step, and in which of the two modes*. It holds no tensors,
no modality data and no token grids — the caller passes one :class:`StreamSpec`
per declared stream and receives a :class:`BudgetPlan`. Mask *placement*
(counts -> per-sample indices) is :mod:`core.mask_placement`.

Design of record: ``code/notes/design/ADR-005-asymmetric-multimodal-masking-3streams.md``
(it supersedes ADR-002 and restates the arithmetic normatively).
Plan and gate ids: ``code/notes/plan/features/ADR-005-asymmetric-multimodal-masking-3streams-plan.md``.

The policy in one paragraph
---------------------------
Every step declares a mode. **Mode 1 (pure virtual sensing)** keeps no
physiological token at all, so the two visual streams supply the whole budget;
**Mode 2 (waveform inpainting)** draws each physiological stream's kept count
independently, aligns the total to a whole number of visual tubes, and gives
the visual streams what remains. Either way the visible tokens of all streams
sum to ``n_budget`` exactly, so the encoder input has a constant length and
carries no ``[MASK]`` placeholders and no padding.

The Mode-1 probability is a schedule, not a constant (ADR-005 §2.7):
``p(e) = p_warm`` for epochs ``e < E_warm`` and ``p_target`` from ``E_warm``
onwards, through :meth:`ConstantBudgetApportioner.p_at`. ``E_warm = 0`` — the
compatibility default — makes ``p(e)`` the constant ``p_target`` again, which
is what keeps every pre-ADR-005 configuration bit-identical.

Rules, in order
---------------
1. **mode** — one draw per step with ``p(e)`` (see :meth:`p_at`) for Mode 1.
2. **Mode 1** — ``total = n_budget // tokens_per_unit`` tubes are split
   uniformly inside both the per-stream tube cap ``M`` and the floor ``m``.
3. **Mode 2** — each physiological stream draws ``K_p`` uniformly on
   ``[0, C_p]``.
4. **alignment** — with ``R = (n_budget - sum K_p) mod tokens_per_unit``, ``R``
   is added to the FIRST declared physiological stream when that stays inside
   its cap, otherwise to the next one, otherwise ``tokens_per_unit - R`` is
   subtracted from the first. The visual residual is then a whole number of
   tubes.
5. **Mode 2 visual split** — the residual tube count is split with the same
   clamped-uniform rule as Mode 1.
6. **assertions** — the exact sum, the per-stream caps and floors and the tube
   multiple are re-checked before the plan is returned.

Feasibility is a property of the *numbers*, not of a draw, so it is validated
once at construction (see :meth:`ConstantBudgetApportioner._validate`). An
infeasible tuple raises :class:`ValueError` naming the failing condition and
the numbers behind it.

Determinism
-----------
Every draw goes through an optional ``torch.Generator``, so a caller can give
the policy its own RNG, replay a step sequence from a seed, and keep the counts
identical across ranks (they are a function of the seed alone — see
:func:`core.mask_placement.counts_generator`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

__all__ = ['VISUAL', 'SIGNAL', 'KINDS', 'MODES', 'REGIMES',
           'REGIME_INPAINTING', 'REGIME_BI_VISUAL', 'REGIME_SINGLE',
           'REGIME_SINGLE_VISUAL', 'SPATIAL_UNIFORM', 'SPATIAL_ANATOMY_AWARE',
           'SPATIAL_SAMPLING', 'SPATIAL_POLICY_SINGLE_VISUAL',
           'SPATIAL_POLICY_DUAL_VISUAL',
           'StreamSpec', 'BudgetPlan', 'ConstantBudgetApportioner']

#: stream-kind tags. The visual/signal taxonomy itself is owned by
#: ``core.multimae`` (``_VISUAL_STREAMS`` / ``_SIGNAL_STREAMS``); this module
#: only needs the two CATEGORIES, never the modality names.
VISUAL = 'visual'
SIGNAL = 'signal'
KINDS: Tuple[str, str] = (VISUAL, SIGNAL)

#: the two operating modes of the policy (ADR-002 §2 contract 4).
MODE_VIRTUAL_SENSING = 1
MODE_INPAINTING = 2
MODES: Tuple[int, int] = (MODE_VIRTUAL_SENSING, MODE_INPAINTING)

#: ADR-006 step classes. These REFINE the modes above, they do not replace
#: them: a single-carrier step is still ``MODE_VIRTUAL_SENSING`` (no
#: physiological token reaches the encoder), so ``mode`` and ``mode_by_counts``
#: keep their ADR-005 meaning and every existing assertion stays valid.
REGIME_INPAINTING = 'inpainting'      # K_resp > 0; the visual(s) carry the rest
REGIME_BI_VISUAL = 'bi_visual'        # K_resp = 0; both visuals carry tokens
REGIME_SINGLE = 'single'              # K_resp = 0; ONE visual carries all of N
#: ADR-007 (one-visual) step class. Neither existing sensor-absent label names a
#: run with exactly ONE visual stream: ``REGIME_BI_VISUAL`` is "BOTH visuals
#: carry tokens" and ``REGIME_SINGLE`` is "one of SEVERAL carries everything".
#: A one-visual step is therefore :data:`REGIME_SINGLE_VISUAL` when no
#: physiological token survives, and :data:`REGIME_INPAINTING` when some do —
#: which keeps "regime == inpainting iff a signal stream keeps > 0 tokens" true
#: in BOTH stream topologies.
REGIME_SINGLE_VISUAL = 'single_visual'
REGIMES: Tuple[str, ...] = (REGIME_INPAINTING, REGIME_BI_VISUAL,
                            REGIME_SINGLE, REGIME_SINGLE_VISUAL)

#: ADR-007 §2.3/§2.10: how the kept spatial TUBES of a visual stream are drawn.
#: ``uniform`` is the pre-ADR-007 behaviour (uniform without replacement over the
#: whole ``Gh*Gw`` grid) and stays the default, so no existing run moves.
SPATIAL_UNIFORM = 'uniform'
SPATIAL_ANATOMY_AWARE = 'anatomy_aware'   # stratified A_core + A_periph draw
SPATIAL_SAMPLING: Tuple[str, str] = (SPATIAL_UNIFORM, SPATIAL_ANATOMY_AWARE)

#: Which feasibility block a tuple is validated against, RESOLVED from the
#: declared stream set (ADR-007 decision D1) — never passed in, so a caller
#: cannot select a policy its stream set does not support. Recorded in logs.
SPATIAL_POLICY_DUAL_VISUAL = 'dual_visual'      # two visuals (ADR-002..ADR-006)
SPATIAL_POLICY_SINGLE_VISUAL = 'single_visual'  # one visual (ADR-007)

_F64 = torch.float64


@dataclass(frozen=True)
class StreamSpec:
    """One declared input stream, described by what the masker needs.

    :param name: the run's own stream name (``'tir'``, ``'resp'``, ...).
    :param kind: :data:`VISUAL` or :data:`SIGNAL`.
    :param n_units: number of masking units — spatial tubes for a visual
        stream (``Gh*Gw``), temporal tokens for a signal stream
        (``seq_len // sig_kernel``).
    :param tokens_per_unit: how many encoder tokens one masking unit carries.
        A visual unit is a full temporal column, so this is the tubelet count
        ``Gt``; a physiological unit IS one token, so it is 1.
    :param spatial_grid: the 2-D patch grid ``(G_h, G_w)`` of a VISUAL stream.
        ``None`` on a signal stream, and optional on a visual one because only
        the anatomy-aware policy (ADR-007 §2.3) needs the row/column layout:
        ``n_units`` alone cannot resolve ``[0, G_h-1] x [0, G_w-1]``. When
        given it must satisfy ``G_h * G_w == n_units``.
    """

    name: str
    kind: str
    n_units: int
    tokens_per_unit: int = 1
    spatial_grid: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(
                f'StreamSpec({self.name!r}): kind must be one of {KINDS}; '
                f'got {self.kind!r}.')
        if int(self.n_units) < 1:
            raise ValueError(
                f'StreamSpec({self.name!r}): n_units must be >= 1; got '
                f'{self.n_units!r}.')
        if int(self.tokens_per_unit) < 1:
            raise ValueError(
                f'StreamSpec({self.name!r}): tokens_per_unit must be >= 1; '
                f'got {self.tokens_per_unit!r}.')
        if self.spatial_grid is not None:
            if len(tuple(self.spatial_grid)) != 2:
                raise ValueError(
                    f'StreamSpec({self.name!r}): spatial_grid must be a '
                    f'(G_h, G_w) pair; got {self.spatial_grid!r}.')
            gh, gw = (int(x) for x in self.spatial_grid)
            if gh < 1 or gw < 1:
                raise ValueError(
                    f'StreamSpec({self.name!r}): spatial_grid entries must be '
                    f'>= 1; got {self.spatial_grid!r}.')
            if gh * gw != int(self.n_units):
                raise ValueError(
                    f'StreamSpec({self.name!r}): spatial_grid {gh}x{gw} = '
                    f'{gh * gw} does not match n_units {int(self.n_units)}.')

    @property
    def n_tokens(self) -> int:
        """Encoder tokens this stream can contribute when fully visible."""
        return int(self.n_units) * int(self.tokens_per_unit)


@dataclass(frozen=True)
class BudgetPlan:
    r"""Counts for one training step (identical for every sample of a batch).

    :param mode: :data:`MODE_VIRTUAL_SENSING` or :data:`MODE_INPAINTING` —
        the mode the policy DREW (with ``p(e)``). Kept for compatibility and
        for replay: it is what the sampler decided, not what the counts say.
    :param kept_tokens: per declared stream, how many encoder tokens survive.
    :param kept_units: the same allocation in masking units — kept tubes for a
        visual stream, kept tokens for a physiological one.
    :param mode_by_counts: the RESOLVED mode of ADR-005 contract 4 —
        :data:`MODE_VIRTUAL_SENSING` iff every signal stream keeps 0 tokens,
        else :data:`MODE_INPAINTING`. Differs from ``mode`` on exactly the
        Mode-2 draw with ``K_tilde = 0`` (1 of 126), which contract 4 relabels.
    :param regime: the ADR-006 step class — :data:`REGIME_INPAINTING`,
        :data:`REGIME_BI_VISUAL` or :data:`REGIME_SINGLE`. ADDITIVE: it refines
        ``mode_by_counts`` and changes no existing field.
    :param carrier: the sole visual stream of a :data:`REGIME_SINGLE` plan;
        ``None`` on every other regime.
    :param core_units: ADR-007 anatomy-aware ONLY — per declared stream, how
        many of the kept visual tubes were drawn from the anatomical core
        :math:`\mathcal{A}_{core}`. ``None`` under the ``uniform`` policy, so a
        plan that never stratified cannot claim a core count.
    :param periph_units: the same allocation in :math:`\mathcal{A}_{periph}`;
        invariant ``core_units[s] + periph_units[s] == kept_units[s]``.
    """

    mode: int
    kept_tokens: Dict[str, int]
    kept_units: Dict[str, int]
    mode_by_counts: int
    regime: str = REGIME_BI_VISUAL
    carrier: Optional[str] = None
    core_units: Optional[Dict[str, int]] = None
    periph_units: Optional[Dict[str, int]] = None

    def __post_init__(self):
        if int(self.mode) not in MODES:
            raise ValueError(
                f'BudgetPlan: mode must be one of {MODES}; got {self.mode!r}.')
        if int(self.mode_by_counts) not in MODES:
            raise ValueError(
                f'BudgetPlan: mode_by_counts must be one of {MODES}; got '
                f'{self.mode_by_counts!r}.')
        if self.regime not in REGIMES:
            raise ValueError(
                f'BudgetPlan: regime must be one of {REGIMES}; got '
                f'{self.regime!r}.')
        if self.regime == REGIME_SINGLE:
            if not self.carrier:
                raise ValueError(
                    'BudgetPlan: a single-carrier plan must name its carrier.')
        elif self.carrier is not None:
            raise ValueError(
                f'BudgetPlan: carrier is only meaningful on a '
                f'{REGIME_SINGLE!r} plan; got regime {self.regime!r} with '
                f'carrier {self.carrier!r}.')
        # ADR-007 §2.4(5): the stratified split is all-or-nothing. A plan that
        # carries ONE of the two halves would let a caller report a core count
        # without a periphery (or vice versa) and silently break the
        # placement contract.
        if (self.core_units is None) != (self.periph_units is None):
            raise ValueError(
                f'BudgetPlan: core_units and periph_units must be given '
                f'together (got core={self.core_units!r}, '
                f'periph={self.periph_units!r}).')
        if self.core_units is not None:
            # Symmetry FIRST: the per-stream loop below reads the other dict with
            # ``.get(name, 0)``, so an asymmetric pair would be reported as a sum
            # mismatch rather than as the missing key it really is.
            if set(self.periph_units) != set(self.core_units):
                raise ValueError(
                    f'BudgetPlan: core_units and periph_units must name the '
                    f'same streams; got {sorted(self.core_units)} vs '
                    f'{sorted(self.periph_units)}.')
            # Only the STRATIFIED streams carry a split (the visual one(s)); a
            # signal stream has no spatial grid and is absent from both dicts.
            for name, core in self.core_units.items():
                if name not in self.kept_units:
                    raise ValueError(
                        f'BudgetPlan: core_units names undeclared stream '
                        f'{name!r}; declared {sorted(self.kept_units)}.')
                core = int(core)
                periph = int(self.periph_units.get(name, 0))
                if core < 0 or periph < 0:
                    raise ValueError(
                        f'BudgetPlan: stream {name!r} has a negative '
                        f'stratified count (core={core}, periph={periph}).')
                if core + periph != int(self.kept_units[name]):
                    raise ValueError(
                        f'BudgetPlan: stream {name!r} keeps '
                        f'{int(self.kept_units[name])} unit(s) but the '
                        f'stratified split is {core} core + {periph} '
                        f'periphery.')

    @property
    def total_tokens(self) -> int:
        """Visible encoder tokens over all streams (the constant budget)."""
        return int(sum(self.kept_tokens.values()))

    def kept(self, name: str) -> int:
        """Kept tokens of one stream (0 when the stream was not declared)."""
        return int(self.kept_tokens.get(name, 0))


class ConstantBudgetApportioner:
    r"""Draw one kept-token count per declared stream per step (ADR-002 §2).

    :param streams: one :class:`StreamSpec` per declared stream, in DECLARED
        order. The order defines the alignment priority of the physiological
        streams (the first one absorbs the modulo remainder first).
    :param n_budget: constant number of visible encoder tokens per step.
    :param video_cap_ratio: kept-ratio ceiling per visual stream; the cap in
        tubes is ``floor(video_cap_ratio * n_units)``.
    :param physio_cap_ratio: kept-ratio ceiling per physiological stream; the
        cap in tokens is ``floor(physio_cap_ratio * n_units)``.
    :param video_min_tubes: kept-tube floor per visual stream.
    :param p_virtual_sensing: Mode-1 probability from ``E_warm`` onwards
        (``p_target`` in ADR-005 §2.9).
    :param virtual_sensing_warmup_p: Mode-1 probability strictly below
        ``E_warm`` (``p_warm``). While the curriculum is active it must not
        exceed ``p_virtual_sensing`` (a decreasing schedule is rejected);
        with ``E_warm = 0`` it is never read and therefore unconstrained.
    :param virtual_sensing_warmup_epochs: number of leading epochs at
        ``p_warm`` (``E_warm``). ``0`` disables the curriculum and is the
        compatibility default — the shipped arm's ``50`` lives in its config.
    :param spatial_sampling: ADR-007 §2.3 — ``'uniform'`` (the pre-ADR-007 draw,
        and the neutral default) or ``'anatomy_aware'`` (stratified
        :math:`\mathcal{A}_{core}` + :math:`\mathcal{A}_{periph}` placement, only
        with exactly ONE visual stream).
    :param core_row_bounds: inclusive row span of :math:`\mathcal{A}_{core}` on
        the visual patch grid.
    :param core_col_bounds: inclusive column span of :math:`\mathcal{A}_{core}`.
    :param core_tubes_min: ``S_core^min``, kept core tubes per step.
    :param core_tubes_max: ``S_core^max``, kept core tubes per step.
    :param periph_min_tubes: ``m_periph`` — the peripheral floor of §2.6(5).
    :param physio_draw_min: lower bound of the Mode-2 physiological draw. ``0``
        keeps the ADR-002..006 draw (and its ``K_tilde = 0`` fall-through) and
        is the neutral default; ``1`` implements ADR-007 §2.5
        (``Uniform{1..C_p}``, so no Mode-2 step can leak into Mode 1).
    :param generator: optional default RNG for :meth:`sample_step`.
    """

    def __init__(self, streams: Sequence[StreamSpec], n_budget: int = 350,
                 video_cap_ratio: float = 0.25,
                 physio_cap_ratio: float = 0.50,
                 video_min_tubes: int = 2,
                 p_virtual_sensing: float = 0.50,
                 virtual_sensing_warmup_p: float = 0.10,
                 virtual_sensing_warmup_epochs: int = 0,
                 single_carrier_tubes: int = 0,
                 p_single_modality: float = 0.0,
                 single_modality_warmup_p: float = 0.0,
                 spatial_sampling: str = SPATIAL_UNIFORM,
                 core_row_bounds: Sequence[int] = (2, 3),
                 core_col_bounds: Sequence[int] = (2, 3),
                 core_tubes_min: int = 1,
                 core_tubes_max: int = 2,
                 periph_min_tubes: int = 1,
                 physio_draw_min: int = 0,
                 generator: Optional[torch.Generator] = None):
        self.streams: Tuple[StreamSpec, ...] = tuple(streams)
        self.n_budget = int(n_budget)
        self.video_cap_ratio = float(video_cap_ratio)
        self.physio_cap_ratio = float(physio_cap_ratio)
        self.video_min_tubes = int(video_min_tubes)
        self.p_virtual_sensing = float(p_virtual_sensing)
        self.virtual_sensing_warmup_p = float(virtual_sensing_warmup_p)
        self.virtual_sensing_warmup_epochs = int(virtual_sensing_warmup_epochs)
        # ADR-006 §2.7: the single-carrier probability WITHIN the sensor-absent
        # class, and its warm-up value (Phase I has no single-carrier step).
        self.p_single_modality = float(p_single_modality)
        self.single_modality_warmup_p = float(single_modality_warmup_p)
        # ADR-007 §2.3/§2.5: stratified spatial placement and the Mode-2 draw
        # floor. Both default to the pre-ADR-007 behaviour.
        self.spatial_sampling = str(
            spatial_sampling or SPATIAL_UNIFORM).strip().lower()
        self.core_row_bounds: Tuple[int, ...] = tuple(
            int(x) for x in core_row_bounds)
        self.core_col_bounds: Tuple[int, ...] = tuple(
            int(x) for x in core_col_bounds)
        self.core_tubes_min = int(core_tubes_min)
        self.core_tubes_max = int(core_tubes_max)
        self.periph_min_tubes = int(periph_min_tubes)
        self.physio_draw_min = int(physio_draw_min)
        self.generator = generator

        self.visual: Tuple[StreamSpec, ...] = tuple(
            s for s in self.streams if s.kind == VISUAL)
        self.signal: Tuple[StreamSpec, ...] = tuple(
            s for s in self.streams if s.kind == SIGNAL)
        self.n_visual = len(self.visual)
        #: ADR-007 D1: the policy is RESOLVED from the stream set, never passed
        #: in, so a caller cannot ask for a policy its streams do not support.
        self.policy: str = (SPATIAL_POLICY_SINGLE_VISUAL
                            if self.n_visual == 1
                            else SPATIAL_POLICY_DUAL_VISUAL)
        self.anatomy_aware: bool = self.spatial_sampling == SPATIAL_ANATOMY_AWARE

        #: tokens per visual tube (the ADR's ``T``), shared by both visual
        #: streams -- one patch grid means one tube layout.
        self.tokens_per_unit = (int(self.visual[0].tokens_per_unit)
                                if self.visual else 0)
        # ADR-006 §2.4: ``S_v`` — the allocation of a SOLE carrier. ``0`` means
        # DERIVE (the whole budget); any other value must equal that derivation
        # and ``_validate`` rejects everything else, so this is an assertion
        # expressed as an option (the ``spatial_patches`` pattern).
        self.single_carrier_tubes = (
            int(single_carrier_tubes)
            or (self.n_budget // self.tokens_per_unit
                if self.tokens_per_unit else 0))
        #: per-stream cap in TUBES (``M_v``) for visual streams.
        #: DUAL visual streams: the kept-ratio ceiling of ADR-002,
        #: ``floor(video_cap_ratio * n_units)``.
        #: SINGLE visual stream (ADR-007 D2): the allocation is fixed by the
        #: budget, so the cap IS the Mode-1 allocation ``S_max = N / T`` and
        #: ``video_cap_ratio`` becomes the *ceiling* on the kept fraction,
        #: checked in :meth:`_validate`. Reusing the floored ratio here would
        #: admit 9 tubes (floor(0.25*36)) and break the §2.5 {7,6,5,4} set.
        self.video_cap_units: Dict[str, int] = {}
        if self.n_visual == 1 and self.tokens_per_unit:
            self.video_cap_units = {
                self.visual[0].name: self.n_budget // self.tokens_per_unit}
        elif self.n_visual > 1:
            self.video_cap_units = {
                s.name: int(math.floor(self.video_cap_ratio * int(s.n_units)))
                for s in self.visual}
        #: per-stream cap in TOKENS (``C_p``) for physiological streams.
        self.physio_cap_tokens: Dict[str, int] = {
            s.name: int(math.floor(self.physio_cap_ratio * int(s.n_units)))
            for s in self.signal}

        # ADR-007 §2.4(2)(3): the two bounds of the visual allocation, published
        # for the manifest. ``None`` under the dual-visual policy, where the
        # allocation is a split rather than a single count.
        self.s_tir_min: Optional[int] = None
        self.s_tir_max: Optional[int] = None
        if self.n_visual == 1 and self.tokens_per_unit and self.signal:
            physio_total = sum(self.physio_cap_tokens.values())
            self.s_tir_max = self.n_budget // self.tokens_per_unit
            self.s_tir_min = (self.n_budget - physio_total) // self.tokens_per_unit

        # ADR-007 §2.3: the anatomical core/periphery partition, resolved from
        # the 2-D patch grid (``StreamSpec.spatial_grid``). Populated only under
        # ``anatomy_aware``; the placement code consumes ``core_indices``.
        self.core_coords: Tuple[Tuple[int, int], ...] = ()
        self.core_indices: Tuple[int, ...] = ()
        self.periph_indices: Tuple[int, ...] = ()

        self._validate()

    # ------------------------------------------------------------------ #
    # construction-time feasibility
    # ------------------------------------------------------------------ #
    def _validate(self) -> None:
        """Reject an infeasible tuple, naming the condition and the numbers."""
        names = [s.name for s in self.streams]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(
                f'ConstantBudgetApportioner: duplicate stream name(s) {dupes}.')
        n_visual, n_signal = len(self.visual), len(self.signal)
        # ADR-007 D1: the stream set SELECTS the policy. ONE visual stream is
        # the ADR-007 two-stream policy (a sole visual must carry the whole
        # Mode-1 budget out of its own grid); TWO is every ADR-002..ADR-006 arm.
        if n_visual not in (1, 2):
            raise ValueError(
                f'ConstantBudgetApportioner: the declared set needs one or two '
                f'visual streams (ADR-002 contract 5 / ADR-007 §2.1); got '
                f'{n_visual} {[s.name for s in self.visual]} of {names}. Zero '
                f'visual streams cannot supply the Mode-1 budget and three '
                f'cannot survive physiological saturation.')
        if not 1 <= n_signal <= 2:
            raise ValueError(
                f'ConstantBudgetApportioner: the declared set needs one or two '
                f'physiological streams (ADR-002 contract 5); got {n_signal} '
                f'{[s.name for s in self.signal]} of {names}.')
        if n_visual == 1 and n_signal != 1:
            raise ValueError(
                f'ConstantBudgetApportioner: the ADR-007 one-visual policy '
                f'declares exactly ONE physiological stream (its canonical set '
                f'is {{tir, resp}}, ADR-007 §2.1); got {n_signal} '
                f'{[s.name for s in self.signal]}. The Mode-2 apportionment of '
                f'§2.5 and the C_p of §2.4(2) are defined for a single '
                f'physiological stream.')

        tpu = {s.tokens_per_unit for s in self.visual}
        if len(tpu) != 1:
            per_stream = ', '.join(f'{s.name!r}: {s.tokens_per_unit}'
                                   for s in self.visual)
            raise ValueError(
                f'ConstantBudgetApportioner: all visual streams must share '
                f'one tube layout (tokens_per_unit); got {{{per_stream}}}.')
        if not 0.0 < self.video_cap_ratio <= 1.0:
            raise ValueError(
                f'ConstantBudgetApportioner: video_cap_ratio must be in '
                f'(0, 1]; got {self.video_cap_ratio!r}.')
        if not 0.0 < self.physio_cap_ratio <= 1.0:
            raise ValueError(
                f'ConstantBudgetApportioner: physio_cap_ratio must be in '
                f'(0, 1]; got {self.physio_cap_ratio!r}.')
        if not 0.0 <= self.p_virtual_sensing <= 1.0:
            raise ValueError(
                f'ConstantBudgetApportioner: p_virtual_sensing must be in '
                f'[0, 1]; got {self.p_virtual_sensing!r}.')
        if not 0.0 <= self.virtual_sensing_warmup_p <= 1.0:
            raise ValueError(
                f'ConstantBudgetApportioner: virtual_sensing_warmup_p must be '
                f'in [0, 1]; got {self.virtual_sensing_warmup_p!r}.')
        if self.virtual_sensing_warmup_epochs < 0:
            raise ValueError(
                f'ConstantBudgetApportioner: virtual_sensing_warmup_epochs '
                f'must be >= 0 (0 disables the curriculum); got '
                f'{self.virtual_sensing_warmup_epochs!r}.')
        # The ordering is only meaningful when Phase I exists. With E_warm = 0
        # the warm-up value is never read, and rejecting it would make an
        # ADR-002 configuration with p_target = 0.0 (a legal value) fail merely
        # because the warm-up key kept its 0.10 default -- i.e. it would break
        # the neutrality guarantee of DoD 1.
        if (self.virtual_sensing_warmup_epochs > 0
                and self.virtual_sensing_warmup_p > self.p_virtual_sensing):
            raise ValueError(
                f'ConstantBudgetApportioner: virtual_sensing_warmup_p '
                f'{self.virtual_sensing_warmup_p!r} exceeds p_virtual_sensing '
                f'{self.p_virtual_sensing!r}; the schedule of ADR-005 §2.7 '
                f'would be decreasing. Lower the warm-up value, raise the '
                f'target, or set virtual_sensing_warmup_epochs=0 to disable '
                f'the curriculum.')
        # ADR-006 §2.7: same two checks for the single-carrier schedule. The
        # ordering is read only while Phase I exists, for the same reason as
        # above -- with E_warm = 0 a neutral ADR-005 config must stay legal.
        if not 0.0 <= self.p_single_modality <= 1.0:
            raise ValueError(
                f'ConstantBudgetApportioner: p_single_modality must be in '
                f'[0, 1]; got {self.p_single_modality!r}.')
        if not 0.0 <= self.single_modality_warmup_p <= 1.0:
            raise ValueError(
                f'ConstantBudgetApportioner: single_modality_warmup_p must be '
                f'in [0, 1]; got {self.single_modality_warmup_p!r}.')
        # ADR-006 §2.7/§2.9 validates this ordering UNCONDITIONALLY. It is not
        # given the ``E_warm > 0`` exemption that ``virtual_sensing_warmup_p``
        # needs: that value defaults to 0.10, so the exemption there protects a
        # legal ``p_target = 0.0`` configuration, whereas this one defaults to
        # 0.00 and therefore cannot make a neutral configuration infeasible.
        if self.single_modality_warmup_p > self.p_single_modality:
            raise ValueError(
                f'ConstantBudgetApportioner: single_modality_warmup_p '
                f'{self.single_modality_warmup_p!r} exceeds p_single_modality '
                f'{self.p_single_modality!r}; the ADR-006 §2.7 schedule would '
                f'be decreasing (admissible: [0, p_single_modality]).')
        if self.n_budget < 1:
            raise ValueError(
                f'ConstantBudgetApportioner: n_budget must be >= 1; got '
                f'{self.n_budget!r}.')

        # ---- ADR-007 §2.3/§2.5 parameter guards (all resolved-time) -------- #
        if self.spatial_sampling not in SPATIAL_SAMPLING:
            raise ValueError(
                f'ConstantBudgetApportioner: spatial_sampling must be one of '
                f'{SPATIAL_SAMPLING}; got {self.spatial_sampling!r}.')
        if self.anatomy_aware and n_visual != 1:
            raise ValueError(
                f'ConstantBudgetApportioner: spatial_sampling='
                f'{SPATIAL_ANATOMY_AWARE!r} needs exactly ONE visual stream -- '
                f'the core subgrid indexes a single ROI patch grid; got '
                f'{[s.name for s in self.visual]}. Use {SPATIAL_UNIFORM!r} for '
                f'a multi-visual arm.')
        if self.p_single_modality > 0.0 and n_visual == 1:
            raise ValueError(
                f'ConstantBudgetApportioner: p_single_modality='
                f'{self.p_single_modality!r} > 0 needs TWO visual streams (one '
                f'to carry, one to zero), but the declared set has a sole '
                f'visual stream {[s.name for s in self.visual]}. Keep it at '
                f'0.0 on an ADR-007 arm.')
        if self.physio_draw_min < 0:
            raise ValueError(
                f'ConstantBudgetApportioner: physio_draw_min must be >= 0 '
                f'(0 = the ADR-002..006 draw); got {self.physio_draw_min!r}.')
        if self.physio_cap_tokens and self.physio_draw_min > min(
                self.physio_cap_tokens.values()):
            raise ValueError(
                f'ConstantBudgetApportioner: physio_draw_min '
                f'{self.physio_draw_min} exceeds the physiological cap '
                f'{self.physio_cap_tokens} (the draw must stay inside C_p, '
                f'ADR-007 §2.6 rule 2).')
        if (self.physio_draw_min > 0 and n_visual > 1
                and self.physio_cap_tokens
                and min(self.physio_cap_tokens.values()) < self.tokens_per_unit):
            # ADR-007 §2.5: with the non-zero floor the whole-tube alignment
            # must be able to land on a real draw, which needs C_p >= T. On the
            # TWO-visual topology the alignment is on the SUM of the signals and
            # cannot be repaired by clamping one stream, so the tuple is
            # rejected here. The one-visual policy reports the same condition
            # as rule 2 inside :meth:`_validate_single_visual`, which keeps the
            # documented 1 -> 2 -> 3 ... order intact.
            raise ValueError(
                f'ConstantBudgetApportioner: physio_draw_min='
                f'{self.physio_draw_min} > 0 needs every physiological cap to '
                f'reach T={self.tokens_per_unit} tokens, so the whole-tube '
                f'alignment has a legal target; got C_p='
                f'{self.physio_cap_tokens} (ADR-007 §2.5/§2.6 rule 2).')

        t = self.tokens_per_unit
        m = self.video_min_tubes
        caps = self.video_cap_units
        if n_visual == 1:
            # ADR-007 §2.6: the one-visual policy has its own feasibility block,
            # evaluated in TABLE ORDER (1 -> 2 -> 3 -> 4 -> 5 -> 5b -> 6) so a
            # multi-violating tuple always reports the first condition.
            self._validate_single_visual(t, m)
            return
        m_lo = min(caps.values())          # tightest visual cap, in tubes
        if m < 1:
            raise ValueError(
                f'ConstantBudgetApportioner: video_min_tubes must be >= 1; '
                f'got {m!r}.')
        for name, cap in caps.items():
            if cap < m:
                raise ValueError(
                    f'ConstantBudgetApportioner: video_cap_ratio '
                    f'{self.video_cap_ratio} gives stream {name!r} a cap of '
                    f'{cap} tube(s), below video_min_tubes {m}. Raise the cap '
                    f'or lower the floor.')

        # 1. the budget must divide into whole visual tubes
        if self.n_budget % t:
            raise ValueError(
                f'ConstantBudgetApportioner: feasibility condition 1 failed — '
                f'n_budget {self.n_budget} is not a multiple of tokens_per_unit '
                f'T={t} (remainder {self.n_budget % t}), so Mode 1 cannot map '
                f'the whole budget onto whole visual tubes.')

        # ADR-006 §2.6 rules 4-5: a SOLE carrier must be able to hold the whole
        # budget in its own tube grid, and ``single_carrier_tubes`` must agree
        # with the derivation (pass 0 to derive it).
        total_units = self.n_budget // t
        if self.single_carrier_tubes != total_units:
            raise ValueError(
                f'ConstantBudgetApportioner: single_carrier_tubes '
                f'{self.single_carrier_tubes} must equal n_budget/T = '
                f'{self.n_budget}/{t} = {total_units} (ADR-006 §2.6 rule 4; '
                f'pass 0 to derive it). A single-carrier step keeps either the '
                f'whole budget or nothing, so this is an assertion, not a '
                f'knob.')
        carrier_rooms = {s.name: int(s.n_units) for s in self.visual}
        if not 1 <= self.single_carrier_tubes <= min(carrier_rooms.values()):
            raise ValueError(
                f'ConstantBudgetApportioner: the sole carrier needs '
                f'{self.single_carrier_tubes} tube(s) but the visual streams '
                f'have {carrier_rooms} (ADR-006 §2.6 rule 5).')
        if self.p_single_modality > 0.0 and len(self.visual) != 2:
            raise ValueError(
                f'ConstantBudgetApportioner: p_single_modality='
                f'{self.p_single_modality!r} > 0 needs exactly TWO visual '
                f'streams (one to carry, one to zero); got '
                f'{[s.name for s in self.visual]} (ADR-006 §2.6 rule 6).')

        # 2. Mode 1 must be splittable inside both the floor and the cap
        total_units = self.n_budget // t
        if not 2 * m <= total_units <= 2 * m_lo:
            raise ValueError(
                f'ConstantBudgetApportioner: feasibility condition 2 failed — '
                f'n_budget/T = {self.n_budget}/{t} = {total_units} tube(s) '
                f'must lie in [2*video_min_tubes, 2*M] = '
                f'[{2 * m}, {2 * m_lo}] (M = the tightest visual cap, '
                f'{caps}).')

        # 3. physiological saturation must leave both visual floors
        p_caps = self.physio_cap_tokens
        physio_total = sum(p_caps.values())
        leftover = self.n_budget - physio_total
        need = 2 * m * t
        if leftover < need:
            raise ValueError(
                f'ConstantBudgetApportioner: feasibility condition 3 failed — '
                f'physiological saturation keeps {physio_total} token(s) '
                f'({p_caps}), leaving {leftover} visual token(s), but the two '
                f'{m}-tube floors need {need}. Lower physio_cap_ratio, lower '
                f'video_min_tubes or raise n_budget.')

    def _resolve_core_grid(self, spec: StreamSpec) -> None:
        """Resolve :math:`\\mathcal{A}_{core}` / :math:`\\mathcal{A}_{periph}`.

        ADR-007 §2.3 partitions the ``G_h x G_w`` patch grid into the anatomical
        core subgrid ``core_row_bounds x core_col_bounds`` and its complement.
        Tube indices use the conv/tubelet flatten order ``h * G_w + w`` -- the
        same order ``core.mask_placement.place_video`` expands to encoder
        indices -- so a resolved core is directly consumable by the placement.
        """
        grid = spec.spatial_grid
        if grid is None:
            raise ValueError(
                f'ConstantBudgetApportioner: spatial_sampling='
                f'{SPATIAL_ANATOMY_AWARE!r} needs StreamSpec.spatial_grid on '
                f'the visual stream {spec.name!r} -- the anatomical core is a '
                f'(row, column) subgrid, and n_units={int(spec.n_units)} alone '
                f'cannot resolve it. Pass spatial_grid=(G_h, G_w) from the '
                f'tokenizer geometry.')
        gh, gw = (int(x) for x in grid)
        rows = tuple(int(x) for x in self.core_row_bounds)
        cols = tuple(int(x) for x in self.core_col_bounds)
        if len(rows) != 2 or len(cols) != 2:
            raise ValueError(
                f'ConstantBudgetApportioner: core_row_bounds / '
                f'core_col_bounds must be (lo, hi) pairs; got {rows}, {cols}.')
        r0, r1 = rows
        c0, c1 = cols
        if not 0 <= r0 <= r1 <= gh - 1:
            raise ValueError(
                f'ConstantBudgetApportioner: core_row_bounds {rows} must lie '
                f'in [0, G_h-1] = [0, {gh - 1}] with lo <= hi (ADR-007 §2.3).')
        if not 0 <= c0 <= c1 <= gw - 1:
            raise ValueError(
                f'ConstantBudgetApportioner: core_col_bounds {cols} must lie '
                f'in [0, G_w-1] = [0, {gw - 1}] with lo <= hi (ADR-007 §2.3).')
        coords = tuple((r, c) for r in range(r0, r1 + 1)
                       for c in range(c0, c1 + 1))
        indices = tuple(r * gw + c for r, c in coords)
        core_set = set(indices)
        periph = tuple(i for i in range(int(spec.n_units))
                       if i not in core_set)
        # §2.6 rule 6: disjoint partition covering the whole grid.
        if core_set & set(periph) or \
                len(core_set) + len(periph) != int(spec.n_units):
            raise ValueError(
                f'ConstantBudgetApportioner: feasibility condition 6 failed -- '
                f'A_core {sorted(core_set)} and A_periph are not a disjoint '
                f'partition of the {int(spec.n_units)} spatial tubes.')
        self.core_coords = coords
        self.core_indices = indices
        self.periph_indices = periph

    def _validate_single_visual(self, t: int, m: int) -> None:
        """ADR-007 §2.6 feasibility for the ONE-visual policy.

        Evaluated in TABLE ORDER (1 -> 2 -> 3 -> 4 -> 5 -> 5b -> 6) so a tuple
        that violates several conditions always reports the first one. Rule 6
        is enforced while the partition is resolved (:meth:`_resolve_core_grid`);
        rules 5b and 6 apply only under ``anatomy_aware``.
        """
        spec = self.visual[0]
        units_v = int(spec.n_units)
        c_p = int(sum(self.physio_cap_tokens.values()))

        # 1. exact budget divisibility
        if not t or self.n_budget % t:
            raise ValueError(
                f'ConstantBudgetApportioner: feasibility condition 1 failed -- '
                f'n_budget {self.n_budget} is not a multiple of tokens_per_unit '
                f'T={t} (remainder {self.n_budget % t if t else "n/a"}), so '
                f'Mode 1 cannot map the whole budget onto whole visual tubes '
                f'(ADR-007 §2.6 rule 1).')
        s_max = self.n_budget // t

        # 2. non-trivial inpainting cap
        if not t <= c_p < self.n_budget:
            raise ValueError(
                f'ConstantBudgetApportioner: feasibility condition 2 failed -- '
                f'the physiological cap C_p={c_p} must satisfy '
                f'T={t} <= C_p < n_budget={self.n_budget} (ADR-007 §2.6 '
                f'rule 2); adjust physio_cap_ratio '
                f'({self.physio_cap_ratio!r}) or n_budget.')
        if self.physio_draw_min >= 1 and c_p < t:
            raise ValueError(
                f'ConstantBudgetApportioner: feasibility condition 2 failed -- '
                f'physio_draw_min={self.physio_draw_min} needs C_p={c_p} >= '
                f'T={t} so a whole-tube inpainting draw exists.')

        # 3. visual grid capacity
        if s_max > units_v:
            raise ValueError(
                f'ConstantBudgetApportioner: feasibility condition 3 failed -- '
                f'Mode 1 needs S_max = n_budget/T = {self.n_budget}/{t} = '
                f'{s_max} tube(s) but the visual stream {spec.name!r} has only '
                f'{units_v} (ADR-007 §2.6 rule 3).')

        # 4. high-masking ceiling
        kept_fraction = s_max / units_v
        if kept_fraction > self.video_cap_ratio:
            raise ValueError(
                f'ConstantBudgetApportioner: feasibility condition 4 failed -- '
                f'S_max/units_v = {s_max}/{units_v} = '
                f'{kept_fraction:.4f} exceeds the high-masking ceiling '
                f'video_cap_ratio={self.video_cap_ratio!r} (ADR-007 §2.6 '
                f'rule 4). A 1-visual arm must mask at least '
                f'{(1.0 - self.video_cap_ratio) * 100:.1f} % of its tubes.')

        # 5. stratified feasibility at the minimum visual allocation
        s_min = (self.n_budget - c_p) // t
        if s_min < 1:
            raise ValueError(
                f'ConstantBudgetApportioner: feasibility condition 5 failed -- '
                f'physiological saturation leaves S_min = (n_budget - C_p)/T = '
                f'({self.n_budget} - {c_p})//{t} = {s_min} visual tube(s), so '
                f'the encoder would receive no visual evidence (ADR-007 §2.6 '
                f'rule 5). Lower physio_cap_ratio or raise n_budget.')
        if m != s_min:
            raise ValueError(
                f'ConstantBudgetApportioner: feasibility condition 5 failed -- '
                f'video_min_tubes={m} must equal the derived floor '
                f'S_tir_min = (n_budget - C_p)//T = ({self.n_budget} - '
                f'{c_p})//{t} = {s_min} (ADR-007 §2.6 rule 5; it is an '
                f'assertion, not a knob).')

        # 5b. stratified core/periphery feasibility (anatomy-aware only).
        # The core subgrid is resolved HERE so that rules 1-5 report before a
        # malformed ``core_*_bounds`` does (plan §1.2: the order is normative).
        if self.anatomy_aware:
            self._resolve_core_grid(spec)
            n_core = len(self.core_indices)
            n_periph = len(self.periph_indices)
            if not 1 <= self.core_tubes_min <= self.core_tubes_max <= n_core:
                raise ValueError(
                    f'ConstantBudgetApportioner: feasibility condition 5b '
                    f'failed -- need 1 <= core_tubes_min '
                    f'({self.core_tubes_min}) <= core_tubes_max '
                    f'({self.core_tubes_max}) <= |A_core| = {n_core} '
                    f'(ADR-007 §2.6 rule 5).')
            if self.periph_min_tubes < 1:
                raise ValueError(
                    f'ConstantBudgetApportioner: feasibility condition 5b '
                    f'failed -- periph_min_tubes={self.periph_min_tubes} must '
                    f'be >= 1 (the peripheral context floor, ADR-007 §2.6 '
                    f'rule 5).')
            if s_min < self.core_tubes_max + self.periph_min_tubes:
                raise ValueError(
                    f'ConstantBudgetApportioner: feasibility condition 5b '
                    f'failed -- S_tir_min={s_min} must be at least '
                    f'core_tubes_max + periph_min_tubes = '
                    f'{self.core_tubes_max} + {self.periph_min_tubes} = '
                    f'{self.core_tubes_max + self.periph_min_tubes} '
                    f'(ADR-007 §2.6 rule 5).')
            if n_periph < s_max - self.core_tubes_min:
                raise ValueError(
                    f'ConstantBudgetApportioner: feasibility condition 5b '
                    f'failed -- |A_periph|={n_periph} must be at least '
                    f'S_tir_max - core_tubes_min = {s_max} - '
                    f'{self.core_tubes_min} (ADR-007 §2.6 rule 5).')

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _randint(self, lo: int, hi: int, generator) -> int:
        """Uniform integer on ``[lo, hi]`` inclusive, through ``generator``."""
        if hi < lo:
            raise RuntimeError(
                f'ConstantBudgetApportioner: empty draw interval [{lo}, {hi}] '
                f'-- the construction-time feasibility check should have '
                f'rejected this tuple.')
        if hi == lo:
            return int(lo)
        return int(torch.randint(lo, hi + 1, (1,), dtype=torch.int64,
                                 generator=generator).item())

    def _split_tubes(self, total: int, generator) -> Tuple[int, int]:
        """Split ``total`` tubes over the two visual streams, clamped-uniform."""
        m = self.video_min_tubes
        first, second = self.visual[0].name, self.visual[1].name
        lo = max(m, total - self.video_cap_units[second])
        hi = min(self.video_cap_units[first], total - m)
        s_first = self._randint(lo, hi, generator)
        return s_first, total - s_first

    def _plan(self, mode: int, units: Dict[str, int],
              tokens: Dict[str, int], regime: str,
              carrier: Optional[str] = None,
              core: Optional[Dict[str, int]] = None,
              periph: Optional[Dict[str, int]] = None) -> BudgetPlan:
        # ADR-005 contract 4: the RESOLVED mode is a function of the counts,
        # not of the draw. Computed here because this is the single
        # construction site of a plan and only the policy knows the signal set.
        by_counts = (MODE_VIRTUAL_SENSING
                     if all(int(tokens[s.name]) == 0 for s in self.signal)
                     else MODE_INPAINTING)
        plan = BudgetPlan(mode=int(mode), kept_tokens=dict(tokens),
                          kept_units=dict(units), mode_by_counts=by_counts,
                          regime=regime, carrier=carrier,
                          core_units=(None if core is None else dict(core)),
                          periph_units=(None if periph is None else dict(periph)))
        # exact sum, caps, floors and the tube multiple, re-checked
        if plan.total_tokens != self.n_budget:
            raise RuntimeError(
                f'ConstantBudgetApportioner: invariant violated — visible '
                f'tokens {plan.total_tokens} != n_budget {self.n_budget} '
                f'({plan.kept_tokens}, mode {mode}).')
        if regime == REGIME_SINGLE:
            # A single-carrier step IS a sensor-absent step, so the check is on
            # the RESOLVED label: a zero-count Mode-2 fall-through keeps
            # ``mode = MODE_INPAINTING`` (the sampled draw) while its counts
            # resolve to Mode 1 and its regime to Regime 3 (ADR-006 §2.3(4)).
            if int(by_counts) != MODE_VIRTUAL_SENSING:
                raise RuntimeError(
                    f'ConstantBudgetApportioner: invariant violated — a '
                    f'{REGIME_SINGLE!r} plan must resolve to '
                    f'MODE_VIRTUAL_SENSING; got {by_counts} '
                    f'(sampled mode {mode}, tokens {plan.kept_tokens}).')
            names = [s.name for s in self.visual]
            if carrier not in names:
                raise RuntimeError(
                    f'ConstantBudgetApportioner: invariant violated — carrier '
                    f'{carrier!r} is not a declared visual stream {names}.')
        for s in self.visual:
            kept = plan.kept_units[s.name]
            if regime == REGIME_SINGLE:
                # ADR-006 §2.3 contract 3: the floor is satisfied BY the
                # carrier, so it and the multi-carrier cap both skip here. The
                # carrier holds S_v and every other visual holds exactly 0.
                want = self.single_carrier_tubes if s.name == carrier else 0
                if kept != want:
                    raise RuntimeError(
                        f'ConstantBudgetApportioner: invariant violated — '
                        f'visual stream {s.name!r} keeps {kept} tube(s) on a '
                        f'single-carrier plan whose carrier is {carrier!r} '
                        f'(expected {want}).')
            elif not (self.video_min_tubes <= kept
                      <= self.video_cap_units[s.name]):
                raise RuntimeError(
                    f'ConstantBudgetApportioner: invariant violated — visual '
                    f'stream {s.name!r} keeps {kept} tube(s), outside '
                    f'[{self.video_min_tubes}, {self.video_cap_units[s.name]}].')
            if plan.kept_tokens[s.name] != kept * int(s.tokens_per_unit):
                raise RuntimeError(
                    f'ConstantBudgetApportioner: invariant violated — visual '
                    f'stream {s.name!r} keeps {plan.kept_tokens[s.name]} '
                    f'token(s) for {kept} tube(s) of {s.tokens_per_unit}.')
        # ADR-007 §2.4(5): a stratifying policy must carry its split, and the
        # split must stay inside the sampled bounds.
        if self.anatomy_aware:
            if plan.core_units is None:
                raise RuntimeError(
                    'ConstantBudgetApportioner: invariant violated — an '
                    'anatomy_aware plan must carry the stratified split; got '
                    f'core={plan.core_units!r}.')
            for s in self.visual:
                core = int(plan.core_units.get(s.name, 0))
                periph = int(plan.periph_units.get(s.name, 0))
                if not self.core_tubes_min <= core <= self.core_tubes_max:
                    raise RuntimeError(
                        f'ConstantBudgetApportioner: invariant violated — '
                        f'visual stream {s.name!r} keeps {core} core tube(s), '
                        f'outside [{self.core_tubes_min}, '
                        f'{self.core_tubes_max}].')
                if periph < self.periph_min_tubes:
                    raise RuntimeError(
                        f'ConstantBudgetApportioner: invariant violated — '
                        f'visual stream {s.name!r} keeps {periph} peripheral '
                        f'tube(s), below periph_min_tubes '
                        f'{self.periph_min_tubes}.')
        for s in self.signal:
            kept = plan.kept_tokens[s.name]
            if not 0 <= kept <= self.physio_cap_tokens[s.name]:
                raise RuntimeError(
                    f'ConstantBudgetApportioner: invariant violated — '
                    f'physiological stream {s.name!r} keeps {kept} token(s), '
                    f'outside [0, {self.physio_cap_tokens[s.name]}].')
        if mode == MODE_VIRTUAL_SENSING:
            if any(plan.kept_tokens[s.name] for s in self.signal):
                raise RuntimeError(
                    'ConstantBudgetApportioner: invariant violated — Mode 1 '
                    f'must keep no physiological token; got '
                    f'{plan.kept_tokens}.')
        return plan

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def p_at(self, epoch: Optional[int]) -> float:
        """Mode-1 probability at ``epoch`` (ADR-005 §2.7, 0-based index).

        ``p_warm`` strictly below the boundary, ``p_target`` from ``E_warm``
        onwards. ``epoch=None`` means "no epoch context" (pinned evaluation,
        a single-step probe) and resolves to ``p_target`` — the schedule's
        steady state, never its warm-up value.

        :param epoch: 0-based training-epoch index, or ``None``.
        """
        if epoch is None:
            return self.p_virtual_sensing
        if int(epoch) >= self.virtual_sensing_warmup_epochs:
            return self.p_virtual_sensing
        return self.virtual_sensing_warmup_p

    def p_s_at(self, epoch: Optional[int]) -> float:
        """Single-carrier probability at ``epoch`` (ADR-006 §2.7, 0-based).

        ``p_{s,warm}`` strictly below the boundary, ``p_{s,target}`` from
        ``E_warm`` onwards. ``epoch=None`` resolves to ``p_{s,target}``, the
        same convention as :meth:`p_at` (and therefore 0.0 for every
        configuration that does not mention the new keys).

        :param epoch: 0-based training-epoch index, or ``None``.
        """
        if epoch is None:
            return self.p_single_modality
        if int(epoch) >= self.virtual_sensing_warmup_epochs:
            return self.p_single_modality
        return self.single_modality_warmup_p

    def sample_step(self, generator: Optional[torch.Generator] = None,
                    epoch: Optional[int] = None) -> BudgetPlan:
        """Draw the step's mode and per-stream counts.

        :param generator: RNG to use; defaults to the one given at
            construction. Pass one for a replayable sequence.
        :param epoch: 0-based epoch index selecting ``p(e)`` (ADR-005 §2.7).
            ``None`` uses ``p_target``, so a caller that has no epoch context
            reproduces the pre-curriculum behaviour exactly.
        """
        g = generator if generator is not None else self.generator
        u = float(torch.rand(1, dtype=_F64, generator=g).item())
        if u < self.p_at(epoch):
            return self._mode1(g, epoch)
        return self._mode2(g, epoch)

    def _mode1(self, g, epoch: Optional[int] = None) -> BudgetPlan:
        """Sampled sensor-absent step (Mode 1): no physiology, budget visual.

        Delegates to :meth:`_sensor_absent`, which is also the path a Mode-2
        draw takes when its aligned respiration count lands on zero — both are
        the same step class (ADR-006 §2.3 contract 4).

        :param g: RNG for this step's draws.
        :param epoch: 0-based epoch index; ``None`` uses the steady state.
            Keep the default: ``tests/test_curriculum_schedule.py`` calls this
            helper with one positional argument.
        """
        return self._sensor_absent(g, epoch)

    def _sensor_absent(self, g, epoch: Optional[int],
                       mode: int = MODE_VIRTUAL_SENSING) -> BudgetPlan:
        """The ADR-006 sensor-absent class: Regime 3 with p_s(e), else Regime 2.

        Reached from BOTH draws, exactly as ADR-006 §2.5 specifies: a Mode-1
        draw AND a Mode-2 draw whose aligned respiration count is zero. The
        ``p_s > 0`` guard is evaluated BEFORE any draw, so with
        ``p_single_modality = 0`` (the default) this method consumes exactly the
        RNG the pre-ADR-006 engine did and the mask sequence is bit-identical.

        :param mode: the SAMPLED mode, kept for ADR-005 contract 4 — a Mode-2
            draw that falls through here keeps ``MODE_INPAINTING`` while its
            counts resolve to Mode 1.
        """
        if self.n_visual == 1:
            # ADR-007 §2.4(4): with a sole visual the sensor-absent class is a
            # single step shape (S_tir = S_max, K_resp = 0). The ADR-006
            # single-carrier sub-decision does not apply and cannot: a second
            # carrier stream would be needed to zero, and ``p_single_modality
            # > 0`` is rejected at construction. NO p_s draw is taken, so this
            # branch consumes exactly one random number per step (the mode
            # draw in ``sample_step``).
            return self._sole_visual(g, mode, k_resp=0)
        p_s = self.p_s_at(epoch)
        if p_s > 0.0:
            if float(torch.rand(1, dtype=_F64, generator=g).item()) < p_s:
                first = float(torch.rand(1, dtype=_F64,
                                         generator=g).item())
                carrier = self.visual[0] if first < 0.5 else self.visual[1]
                return self._single_carrier(carrier.name, mode)
        total = self.n_budget // self.tokens_per_unit
        s1, s2 = self._split_tubes(total, g)
        first, second = self.visual[0], self.visual[1]
        units = {first.name: s1, second.name: s2}
        tokens = {first.name: s1 * int(first.tokens_per_unit),
                  second.name: s2 * int(second.tokens_per_unit)}
        for s in self.signal:
            units[s.name] = 0
            tokens[s.name] = 0
        return self._plan(mode, units, tokens, REGIME_BI_VISUAL)

    def _single_carrier(self, carrier: str,
                        mode: int = MODE_VIRTUAL_SENSING) -> BudgetPlan:
        """ADR-006 Regime 3: one visual stream takes the whole budget.

        Built directly: ``_split_tubes`` clamps to ``[m, M]`` and would return
        9/5 rather than ``S_v``/0, so it must not be used here.
        """
        spec = next((s for s in self.visual if s.name == carrier), None)
        if spec is None:
            raise RuntimeError(
                f'ConstantBudgetApportioner: carrier {carrier!r} is not a '
                f'declared visual stream {[s.name for s in self.visual]}.')
        units = {s.name: 0 for s in self.streams}
        tokens = {s.name: 0 for s in self.streams}
        units[carrier] = self.single_carrier_tubes
        tokens[carrier] = self.single_carrier_tubes * int(spec.tokens_per_unit)
        return self._plan(mode, units, tokens, REGIME_SINGLE, carrier)

    def _sole_visual(self, g, mode: int, k_resp: int) -> BudgetPlan:
        """ADR-007 §2.4/§2.5: the allocation of the ONE-visual policy.

        Mode 1 (``k_resp = 0``) keeps the whole budget on the sole visual
        (``S_tir = N/T``); Mode 2 keeps ``(N - k_resp)/T`` tubes. ADR-007 D3
        labels the step :data:`REGIME_SINGLE_VISUAL` when no physiological
        token survives and :data:`REGIME_INPAINTING` when some do, so
        "regime == inpainting iff a signal stream keeps > 0 tokens" holds in
        this topology too.

        ``S_core`` is drawn HERE, once per step and AFTER the count/alignment
        draws (ADR-007 §2.4(6) + §2.5 order), and shared by the whole batch;
        only the tube INDICES vary per sample.
        """
        spec = self.visual[0]
        s_tir = (self.n_budget - int(k_resp)) // self.tokens_per_unit
        units = {s.name: 0 for s in self.streams}
        tokens = {s.name: 0 for s in self.streams}
        units[spec.name] = s_tir
        tokens[spec.name] = s_tir * int(spec.tokens_per_unit)
        for s in self.signal:
            units[s.name] = int(k_resp)
            tokens[s.name] = int(k_resp)
        core, periph = self._draw_stratification(g, spec.name, s_tir)
        regime = (REGIME_SINGLE_VISUAL if int(k_resp) == 0
                  else REGIME_INPAINTING)
        return self._plan(mode, units, tokens, regime,
                          core=core, periph=periph)

    def _draw_stratification(self, g, name: str, s_tir: int):
        """``(core_units, periph_units)`` for this step (ADR-007 §2.3).

        One draw per STEP, shared by the whole mini-batch (§2.4(6)); WHICH tubes
        are kept is decided per sample by the placement. Under ``uniform`` the
        literal is tested BEFORE any draw, so the pre-ADR-007 RNG stream is not
        disturbed -- a draw-then-ignore would shift every later count.
        """
        if not self.anatomy_aware:
            return None, None
        s_core = self._randint(self.core_tubes_min, self.core_tubes_max, g)
        return {name: int(s_core)}, {name: int(s_tir) - int(s_core)}

    def _mode2(self, g, epoch: Optional[int] = None) -> BudgetPlan:
        """Waveform inpainting: physiological draws, tube-aligned residual.

        A draw whose ALIGNED physiological total is zero is a sensor-absent step
        (contract 4), so it falls through to :meth:`_sensor_absent` rather than
        returning an ``inpainting`` plan with no respiration token — ADR-006
        §2.5. The sampled ``mode`` stays ``MODE_INPAINTING``; only the resolved
        regime changes.
        """
        t = self.tokens_per_unit
        raw: Dict[str, int] = {
            s.name: self._randint(self.physio_draw_min,
                                  self.physio_cap_tokens[s.name], g)
            for s in self.signal}

        # alignment: charge the modulo remainder to the first declared
        # physiological stream that can absorb it (ADR-002 §2, G14)
        remainder = (self.n_budget - sum(raw.values())) % t
        if remainder:
            for s in self.signal:
                if raw[s.name] + remainder <= self.physio_cap_tokens[s.name]:
                    raw[s.name] += remainder
                    remainder = 0
                    break
        if remainder:
            # ADR-007 §2.5: the downward branch ``K_tilde - (T - r)``. Its count
            # is NEVER clamped here: mutating it would break the modulo
            # alignment the next block asserts. With the ADR-007 draw floor
            # (``physio_draw_min >= 1``) a result below T is impossible for a
            # single signal — the floor is >= 1, ``C_p >= T`` is validated, and
            # the upward branch is preferred whenever it fits — so the check
            # below is an assertion, not a correction.
            first = self.signal[0].name
            raw[first] -= t - remainder
            if raw[first] < 0:
                raise RuntimeError(
                    f'ConstantBudgetApportioner: alignment remainder '
                    f'{remainder} cannot be absorbed — {first!r} would fall to '
                    f'{raw[first]}. Feasibility condition 3 should have '
                    f'rejected this tuple.')
            if (self.physio_draw_min >= 1 and len(self.signal) == 1
                    and raw[first] < t):
                raise RuntimeError(
                    f'ConstantBudgetApportioner: the ADR-007 draw floor cannot '
                    f'land below T={t} on a sole physiological stream, yet '
                    f'{first!r} aligned to {raw[first]}. ``_validate`` '
                    f'enforces C_p >= T for physio_draw_min > 0, so this is a '
                    f'bug in the alignment, not a configuration error.')

        # ADR-006 §2.3 contract 4 / §2.5: an aligned total of zero is a
        # SENSOR-ABSENT step, not an inpainting one — it falls through to the
        # same Regime-3/Regime-2 decision as a sampled Mode-1 step. (With
        # physio_draw_min >= 1 this is unreachable: the draw itself is >= 1 and
        # the alignment only rounds UP or down to a hole tube count >= T, so no
        # Mode-2 step can leak into Mode 1 — ADR-007 §2.5.)
        if sum(raw.values()) == 0:
            return self._sensor_absent(g, epoch, MODE_INPAINTING)

        remaining = self.n_budget - sum(raw.values())
        if remaining % t:
            raise RuntimeError(
                f'ConstantBudgetApportioner: aligned physiological total '
                f'{sum(raw.values())} leaves {remaining} visual token(s), not a '
                f'multiple of T={t}.')
        if self.n_visual == 1:
            # ADR-007 §2.5: S_tir = (N - K_resp)/T, no split.
            return self._sole_visual(g, MODE_INPAINTING,
                                     k_resp=sum(raw.values()))
        s1, s2 = self._split_tubes(remaining // t, g)
        first, second = self.visual[0], self.visual[1]
        units = {first.name: s1, second.name: s2}
        tokens = {first.name: s1 * int(first.tokens_per_unit),
                  second.name: s2 * int(second.tokens_per_unit)}
        for s in self.signal:
            units[s.name] = raw[s.name]
            tokens[s.name] = raw[s.name]
        return self._plan(MODE_INPAINTING, units, tokens, REGIME_INPAINTING)

    def plan_from_counts(self, kept_tokens: Mapping[str, int],
                         kept_core: Optional[int] = None) -> BudgetPlan:
        """Validate an EXPLICIT allocation (evaluation pins, tests).

        The masking policy owns the draws; this entry point exists so a caller
        that must reproduce one specific masking (a reconstruction figure, a
        pinned evaluation) can state the counts and have the same invariants
        enforced — including the exact budget — instead of fabricating masks.
        The mode is inferred: Mode 1 when no physiological token survives.

        :param kept_tokens: kept encoder tokens per DECLARED stream.
        :raises ValueError: missing/unknown stream, or any invariant broken.
        """
        declared = {s.name for s in self.streams}
        unknown = sorted(set(kept_tokens) - declared)
        if unknown:
            raise ValueError(
                f'ConstantBudgetApportioner.plan_from_counts: undeclared '
                f'stream(s) {unknown}; declared {sorted(declared)}.')
        missing = sorted(declared - set(kept_tokens))
        if missing:
            raise ValueError(
                f'ConstantBudgetApportioner.plan_from_counts: missing '
                f'stream(s) {missing}; every declared stream needs a count.')

        tokens: Dict[str, int] = {}
        units: Dict[str, int] = {}
        for s in self.streams:
            kept = int(kept_tokens[s.name])
            if kept < 0:
                raise ValueError(
                    f'ConstantBudgetApportioner.plan_from_counts: stream '
                    f'{s.name!r} keeps {kept} token(s); counts must be >= 0.')
            if s.kind == VISUAL:
                if kept % int(s.tokens_per_unit):
                    raise ValueError(
                        f'ConstantBudgetApportioner.plan_from_counts: visual '
                        f'stream {s.name!r} keeps {kept} token(s), not a '
                        f'multiple of T={s.tokens_per_unit} (a tube keeps its '
                        f'whole temporal column).')
                units[s.name] = kept // int(s.tokens_per_unit)
            else:
                units[s.name] = kept
            tokens[s.name] = kept

        mode = (MODE_VIRTUAL_SENSING
                if all(tokens[s.name] == 0 for s in self.signal)
                else MODE_INPAINTING)
        regime, carrier = self._infer_regime(tokens)
        core, periph = self._pin_stratification(units, kept_core)
        try:
            return self._plan(mode, units, tokens, regime, carrier,
                              core=core, periph=periph)
        except RuntimeError as exc:      # normalise to the caller-facing type
            raise ValueError(
                f'ConstantBudgetApportioner.plan_from_counts: {exc}') from exc

    def _pin_stratification(self, units: Mapping[str, int],
                            kept_core: Optional[int]
                            ) -> Tuple[Optional[Dict[str, int]],
                                       Optional[Dict[str, int]]]:
        """Turn an explicit ``kept_core`` into a stratified split.

        Only a STRATIFIED policy can consume one. Defaulting on a ``uniform``
        checkpoint (an ADR-006 arm, or the ADR-007 ``uniform`` comparator) would
        report a fabricated ``S_core`` in a figure caption, so an explicit
        ``kept_core`` there is an error rather than a no-op.
        """
        if not self.anatomy_aware:
            if kept_core is not None:
                raise ValueError(
                    f'ConstantBudgetApportioner.plan_from_counts: kept_core='
                    f'{kept_core} was given for a {SPATIAL_UNIFORM!r} policy. '
                    f'A uniform checkpoint has no anatomical core, so its '
                    f'core count would be fabricated; drop kept_core or run '
                    f'with spatial_sampling={SPATIAL_ANATOMY_AWARE!r}.')
            return None, None
        spec = self.visual[0]
        s_tir = int(units[spec.name])
        core_n = self.core_tubes_min if kept_core is None else int(kept_core)
        return {spec.name: core_n}, {spec.name: s_tir - core_n}

    def _infer_regime(self, tokens: Mapping[str, int]
                      ) -> Tuple[str, Optional[str]]:
        """Resolve the step class from an EXPLICIT allocation.

        Only :meth:`plan_from_counts` needs this: the sampler knows the class it
        drew, while a pinned allocation must be labelled from its counts. The
        rule is the sampler's (ADR-007 D3): ``inpainting`` iff a signal stream
        keeps > 0 tokens; otherwise ``single_visual`` for a one-visual set and
        ``single``/``bi_visual`` (by carrier count) for a two-visual set.
        """
        if any(int(tokens[s.name]) for s in self.signal):
            return REGIME_INPAINTING, None
        carriers = [s.name for s in self.visual if int(tokens[s.name]) > 0]
        if len(self.visual) == 1:
            return REGIME_SINGLE_VISUAL, None
        if len(carriers) == 1:
            return REGIME_SINGLE, carriers[0]
        return REGIME_BI_VISUAL, None
