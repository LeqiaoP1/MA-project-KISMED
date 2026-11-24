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
REGIME_INPAINTING = 'inpainting'      # K_resp > 0; both visuals carry tokens
REGIME_BI_VISUAL = 'bi_visual'        # K_resp = 0; both visuals carry tokens
REGIME_SINGLE = 'single'              # K_resp = 0; ONE visual carries all of N
REGIMES: Tuple[str, str, str] = (REGIME_INPAINTING, REGIME_BI_VISUAL,
                                 REGIME_SINGLE)

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
    """

    name: str
    kind: str
    n_units: int
    tokens_per_unit: int = 1

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

    @property
    def n_tokens(self) -> int:
        """Encoder tokens this stream can contribute when fully visible."""
        return int(self.n_units) * int(self.tokens_per_unit)


@dataclass(frozen=True)
class BudgetPlan:
    """Counts for one training step (identical for every sample of a batch).

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
    """

    mode: int
    kept_tokens: Dict[str, int]
    kept_units: Dict[str, int]
    mode_by_counts: int
    regime: str = REGIME_BI_VISUAL
    carrier: Optional[str] = None

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

    @property
    def total_tokens(self) -> int:
        """Visible encoder tokens over all streams (the constant budget)."""
        return int(sum(self.kept_tokens.values()))

    def kept(self, name: str) -> int:
        """Kept tokens of one stream (0 when the stream was not declared)."""
        return int(self.kept_tokens.get(name, 0))


class ConstantBudgetApportioner:
    """Draw one kept-token count per declared stream per step (ADR-002 §2).

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
        self.generator = generator

        self.visual: Tuple[StreamSpec, ...] = tuple(
            s for s in self.streams if s.kind == VISUAL)
        self.signal: Tuple[StreamSpec, ...] = tuple(
            s for s in self.streams if s.kind == SIGNAL)

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
        self.video_cap_units: Dict[str, int] = {
            s.name: int(math.floor(self.video_cap_ratio * int(s.n_units)))
            for s in self.visual}
        #: per-stream cap in TOKENS (``C_p``) for physiological streams.
        self.physio_cap_tokens: Dict[str, int] = {
            s.name: int(math.floor(self.physio_cap_ratio * int(s.n_units)))
            for s in self.signal}

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
        if n_visual != 2:
            raise ValueError(
                f'ConstantBudgetApportioner: the declared set needs exactly '
                f'TWO visual streams (ADR-002 contract 5); got {n_visual} '
                f'{[s.name for s in self.visual]} of {names}. A single visual '
                f'stream cannot supply the Mode-1 budget and three cannot '
                f'survive physiological saturation.')
        if not 1 <= n_signal <= 2:
            raise ValueError(
                f'ConstantBudgetApportioner: the declared set needs one or two '
                f'physiological streams (ADR-002 contract 5); got {n_signal} '
                f'{[s.name for s in self.signal]} of {names}.')

        tpu = {s.tokens_per_unit for s in self.visual}
        if len(tpu) != 1:
            per_stream = ', '.join(f'{s.name!r}: {s.tokens_per_unit}'
                                   for s in self.visual)
            raise ValueError(
                f'ConstantBudgetApportioner: both visual streams must share '
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
        if (self.virtual_sensing_warmup_epochs > 0
                and self.single_modality_warmup_p > self.p_single_modality):
            raise ValueError(
                f'ConstantBudgetApportioner: single_modality_warmup_p '
                f'{self.single_modality_warmup_p!r} exceeds p_single_modality '
                f'{self.p_single_modality!r}; the ADR-006 §2.7 schedule would '
                f'be decreasing.')
        if self.n_budget < 1:
            raise ValueError(
                f'ConstantBudgetApportioner: n_budget must be >= 1; got '
                f'{self.n_budget!r}.')

        t = self.tokens_per_unit
        m = self.video_min_tubes
        caps = self.video_cap_units
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
              carrier: Optional[str] = None) -> BudgetPlan:
        # ADR-005 contract 4: the RESOLVED mode is a function of the counts,
        # not of the draw. Computed here because this is the single
        # construction site of a plan and only the policy knows the signal set.
        by_counts = (MODE_VIRTUAL_SENSING
                     if all(int(tokens[s.name]) == 0 for s in self.signal)
                     else MODE_INPAINTING)
        plan = BudgetPlan(mode=int(mode), kept_tokens=dict(tokens),
                          kept_units=dict(units), mode_by_counts=by_counts,
                          regime=regime, carrier=carrier)
        # exact sum, caps, floors and the tube multiple, re-checked
        if plan.total_tokens != self.n_budget:
            raise RuntimeError(
                f'ConstantBudgetApportioner: invariant violated — visible '
                f'tokens {plan.total_tokens} != n_budget {self.n_budget} '
                f'({plan.kept_tokens}, mode {mode}).')
        if regime == REGIME_SINGLE:
            # A single-carrier step IS a sensor-absent step; anything else is a
            # mislabelled plan.
            if int(mode) != MODE_VIRTUAL_SENSING:
                raise RuntimeError(
                    f'ConstantBudgetApportioner: invariant violated — a '
                    f'{REGIME_SINGLE!r} plan must be '
                    f'MODE_VIRTUAL_SENSING; got mode {mode}.')
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
        return self._mode2(g)

    def _mode1(self, g, epoch: Optional[int] = None) -> BudgetPlan:
        """Pure virtual sensing: no physiological token, whole budget visual.

        ADR-006: a share ``p_s(e)`` of these steps is SINGLE-modality, i.e. the
        whole budget goes to exactly one visual stream. The ``p_s > 0`` guard is
        evaluated BEFORE any draw, so with ``p_single_modality = 0`` (the
        default) this method consumes exactly the RNG it always did and the
        pre-ADR-006 mask sequence is bit-identical.

        :param g: RNG for this step's draws.
        :param epoch: 0-based epoch index; ``None`` uses the steady state.
            Keep the default: ``tests/test_curriculum_schedule.py`` calls this
            helper with one positional argument.
        """
        p_s = self.p_s_at(epoch)
        if p_s > 0.0:
            if float(torch.rand(1, dtype=_F64, generator=g).item()) < p_s:
                first = float(torch.rand(1, dtype=_F64,
                                         generator=g).item())
                carrier = self.visual[0] if first < 0.5 else self.visual[1]
                return self._single_carrier(carrier.name)
        total = self.n_budget // self.tokens_per_unit
        s1, s2 = self._split_tubes(total, g)
        first, second = self.visual[0], self.visual[1]
        units = {first.name: s1, second.name: s2}
        tokens = {first.name: s1 * int(first.tokens_per_unit),
                  second.name: s2 * int(second.tokens_per_unit)}
        for s in self.signal:
            units[s.name] = 0
            tokens[s.name] = 0
        return self._plan(MODE_VIRTUAL_SENSING, units, tokens, REGIME_BI_VISUAL)

    def _single_carrier(self, carrier: str) -> BudgetPlan:
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
        return self._plan(MODE_VIRTUAL_SENSING, units, tokens, REGIME_SINGLE,
                          carrier)

    def _mode2(self, g) -> BudgetPlan:
        """Waveform inpainting: physiological draws, tube-aligned residual."""
        t = self.tokens_per_unit
        raw: Dict[str, int] = {
            s.name: self._randint(0, self.physio_cap_tokens[s.name], g)
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
            first = self.signal[0].name
            raw[first] -= t - remainder
            if raw[first] < 0:
                raise RuntimeError(
                    f'ConstantBudgetApportioner: alignment remainder '
                    f'{remainder} cannot be absorbed — {first!r} would fall to '
                    f'{raw[first]}. Feasibility condition 3 should have '
                    f'rejected this tuple.')

        remaining = self.n_budget - sum(raw.values())
        if remaining % t:
            raise RuntimeError(
                f'ConstantBudgetApportioner: aligned physiological total '
                f'{sum(raw.values())} leaves {remaining} visual token(s), not a '
                f'multiple of T={t}.')
        s1, s2 = self._split_tubes(remaining // t, g)
        first, second = self.visual[0], self.visual[1]
        units = {first.name: s1, second.name: s2}
        tokens = {first.name: s1 * int(first.tokens_per_unit),
                  second.name: s2 * int(second.tokens_per_unit)}
        for s in self.signal:
            units[s.name] = raw[s.name]
            tokens[s.name] = raw[s.name]
        return self._plan(MODE_INPAINTING, units, tokens, REGIME_INPAINTING)

    def plan_from_counts(self, kept_tokens: Mapping[str, int]) -> BudgetPlan:
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
        try:
            return self._plan(mode, units, tokens, regime, carrier)
        except RuntimeError as exc:      # normalise to the caller-facing type
            raise ValueError(
                f'ConstantBudgetApportioner.plan_from_counts: {exc}') from exc

    def _infer_regime(self, tokens: Mapping[str, int]
                      ) -> Tuple[str, Optional[str]]:
        """Resolve the ADR-006 regime from an EXPLICIT allocation.

        Only :meth:`plan_from_counts` needs this: the sampler knows the regime
        it drew, while a pinned allocation must be labelled from its counts.
        """
        if any(int(tokens[s.name]) for s in self.signal):
            return REGIME_INPAINTING, None
        carriers = [s.name for s in self.visual if int(tokens[s.name]) > 0]
        if len(carriers) == 1:
            return REGIME_SINGLE, carriers[0]
        return REGIME_BI_VISUAL, None
