"""``FoldPlan`` — the only object that knows which subjects hold which role.

ADR-001 I1-I4 live here: the plan is derived once from the subject universe,
validated against the specification, and then carried around as a value. The
record it serialises to (``partition.json``) is **write-once** and is the single
source of truth shared by Stage 2, Stage 3 and the evaluator, so no consumer can
re-derive a different partition.
"""
import hashlib
import json
import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .constants import (K, ROLES, ROLE_SHARE, SCHEMA_PARTITION, SHARE_TOLERANCE,
                        SPLIT_STRATEGIES, DEFAULT_VAL_SEED)
from .partition import allocate_strata, split_universe, subject_stratum

__all__ = ['ProtocolError', 'FoldPlan', 'build_plan', 'validate_plan',
           'plan_from_record', 'to_record', 'save_plan', 'load_plan',
           'partition_digest', 'canonical_json']


class ProtocolError(RuntimeError):
    """A violation of ADR-001. Never downgraded to a warning."""


def canonical_json(payload: Any) -> bytes:
    """Byte-stable JSON: sorted keys, no whitespace, ASCII only.

    Two constructions of the same partition must serialise identically
    (Criterion 1), so nothing here may depend on dict or set iteration order.
    """
    return json.dumps(payload, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True).encode('utf-8')


def partition_digest(n_folds: int, split_seed: Optional[int], split_stratify: str,
                     val_seed: Optional[int],
                     roles: Mapping[str, Sequence[str]]) -> str:
    """SHA-256 over the partition itself (roles + the seeds that produced them).

    This is the pairing key: a Stage-3 run may only use a Stage-2 checkpoint
    whose record carries the same digest, and a resume may only continue a
    checkpoint of the same partition.
    """
    payload = {
        'n_folds': int(n_folds),
        'split_seed': None if split_seed is None else int(split_seed),
        'split_stratify': str(split_stratify),
        'val_seed': None if val_seed is None else int(val_seed),
        'roles': {role: sorted(str(s) for s in roles[role]) for role in ROLES},
    }
    return hashlib.sha256(canonical_json(payload)).hexdigest()


@dataclass(frozen=True)
class FoldPlan:
    """One fold's immutable role assignment (ADR-001 I1, I3)."""

    fold: int
    n_folds: int
    split_seed: Optional[int]
    split_stratify: str
    val_seed: int
    universe: Tuple[str, ...]
    roles: Mapping[str, Tuple[str, ...]]
    dev_smoke: bool = False

    def role(self, name: str) -> Tuple[str, ...]:
        """Subject ids of ``name`` (``train`` / ``val`` / ``test``)."""
        if name not in self.roles:
            raise ProtocolError(
                f'unknown role {name!r}; the protocol defines {ROLES}.')
        return self.roles[name]

    @property
    def counts(self) -> Dict[str, int]:
        return {role: len(self.roles[role]) for role in ROLES}

    @property
    def share(self) -> Dict[str, float]:
        size = len(self.universe) or 1
        return {role: self.counts[role] / size for role in ROLES}

    @property
    def digest(self) -> str:
        return partition_digest(self.n_folds, self.split_seed,
                                self.split_stratify, self.val_seed, self.roles)

    @property
    def universe_digest(self) -> str:
        return hashlib.sha256(canonical_json(list(self.universe))).hexdigest()


def _draw_val(pool: Sequence[str], n_val: int, stratify: str,
              rng: np.random.Generator) -> List[str]:
    """Draw the validation subjects from the non-test pool (ADR-001 I3).

    ``stratify='none'`` draws uniformly; ``'sex'`` allocates ``n_val`` over the
    pool's strata in proportion (largest remainder) and draws within each, so
    ``train = pool - val`` does not inherit a sex skew.
    """
    if stratify == 'none':
        picked = rng.choice(list(pool), size=n_val, replace=False).tolist()
    else:
        strata: Dict[str, List[str]] = {}
        for subject in pool:
            strata.setdefault(subject_stratum(subject), []).append(subject)
        allocation = allocate_strata(n_val, {name: len(v)
                                            for name, v in strata.items()})
        picked: List[str] = []
        # Strata consumed in sorted order with a single rng: deterministic
        # across processes and independent of dict iteration order.
        for name in sorted(strata):
            take = min(allocation[name], len(strata[name]))
            if take:
                picked.extend(rng.choice(strata[name], size=take,
                                         replace=False).tolist())
        if len(picked) != n_val:                 # defensive: never silently short
            remaining = [s for s in pool if s not in set(picked)]
            extra = rng.choice(remaining, size=n_val - len(picked), replace=False)
            picked.extend(extra.tolist())
    return sorted(picked)


def _blocks(universe: Sequence[str], split_seed: Optional[int],
            split_stratify: str) -> List[List[str]]:
    """`split_universe` with its argument errors promoted to ``ProtocolError``.

    One exception type leaves: a caller should never have to catch both
    ``ValueError`` and ``ProtocolError`` to know that the protocol refused.
    """
    try:
        return split_universe(universe, K, split_seed=split_seed,
                              split_stratify=split_stratify)
    except ValueError as exc:
        raise ProtocolError(str(exc)) from None


def build_plan(universe: Sequence[str], fold: int, *,
               split_seed: Optional[int] = None,
               split_stratify: str = 'none',
               val_seed: Optional[int] = None,
               dev_smoke: bool = False) -> FoldPlan:
    """Derive the three roles of ``fold`` from the subject universe (I2, I3).

    :param universe: subject ids from `data.universe.corpus_subjects` — the
        corpus, never a filtered/capped view. Order and duplicates are
        irrelevant: the list is deduplicated and sorted first.
    :param fold: block index in ``[0, K)``; that block becomes the test role.
    :param split_seed: partition seed. ``None`` keeps sorted contiguous blocks.
    :param split_stratify: ``'none'`` or ``'sex'`` (see `partition`).
    :param val_seed: seed of the validation draw. Defaults to ``split_seed``,
        then to :data:`protocol.constants.DEFAULT_VAL_SEED` — the draw adds no
        free parameter of its own.
    :param dev_smoke: report the Criterion-1 share tolerance instead of raising
        it. The resulting plan is written to ``partition.dev.json`` and can never
        be pooled.
    :raises ProtocolError: the universe cannot support the protocol, or the
        resulting partition violates an invariant.
    """
    unique = tuple(sorted({str(s).strip() for s in universe if str(s).strip()}))
    if len(unique) < K:
        raise ProtocolError(
            f'the universe has {len(unique)} subject(s); ADR-001 requires at '
            f'least K={K} to form {K} non-empty blocks.')
    if not 0 <= int(fold) < K:
        raise ProtocolError(f'fold must be in [0, {K}); got {fold}')
    stratify = str(split_stratify or 'none').strip().lower()
    if stratify not in SPLIT_STRATEGIES:
        raise ProtocolError(
            f'split_stratify must be one of {SPLIT_STRATEGIES}; got '
            f'{split_stratify!r}')

    resolved_val_seed = (val_seed if val_seed is not None
                         else split_seed if split_seed is not None
                         else DEFAULT_VAL_SEED)

    blocks = _blocks(unique, split_seed, stratify)
    test = tuple(blocks[int(fold)])
    test_set = set(test)
    pool = [subject for subject in unique if subject not in test_set]

    n_val = max(1, int(round(ROLE_SHARE['val'] * len(unique))))
    n_val = min(n_val, len(pool) - 1) if pool else 0
    if n_val <= 0:
        raise ProtocolError(
            f'the universe is too small for a validation role under fold '
            f'{fold} ({len(pool)} non-test subject(s)).')
    rng = np.random.default_rng(int(resolved_val_seed) + int(fold))
    val = tuple(_draw_val(pool, n_val, stratify, rng))
    val_set = set(val)
    train = tuple(sorted(s for s in pool if s not in val_set))

    plan = FoldPlan(
        fold=int(fold), n_folds=K, split_seed=split_seed,
        split_stratify=stratify, val_seed=int(resolved_val_seed),
        universe=unique,
        roles=MappingProxyType({'train': train, 'val': val, 'test': test}),
        dev_smoke=bool(dev_smoke),
    )
    validate_plan(plan)
    return plan


def validate_plan(plan: FoldPlan) -> None:
    """Assert every structural invariant of ``plan`` (I1, I3, Criterion 1).

    The test role is **recomputed** from the seeds and compared, so a record
    that was edited by hand cannot pass its own self-check.
    """
    if plan.n_folds != K:
        raise ProtocolError(
            f'n_folds is {plan.n_folds}; ADR-001 fixes it at K={K}. Changing '
            f'the fold count requires a new ADR.')
    if not 0 <= plan.fold < K:
        raise ProtocolError(f'fold must be in [0, {K}); got {plan.fold}')
    if set(plan.roles) != set(ROLES):
        raise ProtocolError(
            f'roles must be exactly {ROLES}; got {sorted(plan.roles)}')

    for role in ROLES:
        members = plan.roles[role]
        if list(members) != sorted(set(members)):
            raise ProtocolError(
                f'role {role!r} must be sorted and duplicate-free; got '
                f'{len(members)} entries.')
        if not members:
            raise ProtocolError(
                f'role {role!r} is empty (fold {plan.fold}): ADR-001 requires '
                f'a non-empty train, validation and test role.')

    seen: Dict[str, str] = {}
    for role in ROLES:
        for subject in plan.roles[role]:
            if subject in seen:
                raise ProtocolError(
                    f'{subject!r} appears in both {seen[subject]!r} and '
                    f'{role!r}; roles must be disjoint (I1).')
            seen[subject] = role

    if set(seen) != set(plan.universe):
        missing = sorted(set(plan.universe) - set(seen))
        extra = sorted(set(seen) - set(plan.universe))
        raise ProtocolError(
            f'roles do not partition the universe: {len(missing)} missing '
            f'(e.g. {missing[:3]}), {len(extra)} unknown (e.g. {extra[:3]}).')

    expected_test = _blocks(plan.universe, plan.split_seed,
                            plan.split_stratify)[plan.fold]
    if list(plan.roles['test']) != list(expected_test):
        raise ProtocolError(
            f'the test role of fold {plan.fold} is not block {plan.fold} of the '
            f'partition implied by split_seed={plan.split_seed!r}, '
            f'split_stratify={plan.split_stratify!r}; the record does not match '
            f'its seeds.')

    if not plan.dev_smoke:
        for role in ROLES:
            drift = abs(plan.share[role] - ROLE_SHARE[role])
            if drift > SHARE_TOLERANCE + 1e-12:
                raise ProtocolError(
                    f'realised {role!r} share is {plan.share[role]:.3f} '
                    f'({plan.counts[role]}/{len(plan.universe)}), '
                    f'{drift * 100:.1f} pp away from the required '
                    f'{ROLE_SHARE[role]:.2f} (Criterion 1 allows '
                    f'+/-{SHARE_TOLERANCE * 100:.0f} pp). Use --dev_smoke only '
                    f'for plumbing tests: a dev plan is never pooled.')


def to_record(plan: FoldPlan, writer: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Serialise ``plan`` to the ``partition.json`` payload (ADR-001 section 2.3).

    The universe is stored in full, not only as a digest: the record must be
    verifiable on its own, without re-reading the raw corpus, so that an audit
    can tell a drifted universe from a hand-edited role list.
    """
    return {
        'schema': SCHEMA_PARTITION,
        'adr': 'ADR-001',
        'fold': plan.fold,
        'n_folds': plan.n_folds,
        'split_seed': plan.split_seed,
        'split_stratify': plan.split_stratify,
        'val_seed': plan.val_seed,
        'universe': {'size': len(plan.universe),
                     'digest': plan.universe_digest,
                     'subjects': list(plan.universe)},
        'roles': {role: list(plan.roles[role]) for role in ROLES},
        'counts': plan.counts,
        'share': {role: round(plan.share[role], 4) for role in ROLES},
        'digest': plan.digest,
        'dev_smoke': plan.dev_smoke,
        'writer': dict(writer or {}),
    }


def plan_from_record(record: Mapping[str, Any]) -> FoldPlan:
    """Rebuild and verify a plan from a ``partition.json`` payload."""
    if not isinstance(record, Mapping):
        raise ProtocolError(
            f'partition record must be a mapping; got {type(record).__name__}')
    schema = record.get('schema')
    if schema != SCHEMA_PARTITION:
        raise ProtocolError(
            f'unknown partition schema {schema!r}; this code writes and reads '
            f'{SCHEMA_PARTITION!r} only.')
    try:
        roles = {role: tuple(str(s) for s in record['roles'][role])
                 for role in ROLES}
        universe = tuple(str(s) for s in record['universe']['subjects'])
    except (KeyError, TypeError) as exc:
        raise ProtocolError(f'partition record is missing {exc}') from None

    plan = FoldPlan(
        fold=int(record['fold']), n_folds=int(record['n_folds']),
        split_seed=(None if record.get('split_seed') is None
                    else int(record['split_seed'])),
        split_stratify=str(record['split_stratify']),
        val_seed=int(record['val_seed']),
        universe=universe,
        roles=MappingProxyType(roles),
        dev_smoke=bool(record.get('dev_smoke', False)),
    )
    stored = record.get('digest')
    if stored is not None and stored != plan.digest:
        raise ProtocolError(
            f'partition record digest {stored!r} does not match its own roles '
            f'({plan.digest}); the file was edited after it was written.')
    validate_plan(plan)
    return plan


def save_plan(path: str, plan: FoldPlan,
              writer: Optional[Mapping[str, Any]] = None) -> FoldPlan:
    """Write the record **once**; an existing record must agree (D4).

    An existing record with a different digest is a hard error: two stages, two
    arms or two resumes must never disagree about who is held out.
    """
    payload = json.dumps(to_record(plan, writer=writer), indent=2,
                         sort_keys=True) + '\n'
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    def _reuse() -> FoldPlan:
        existing = load_plan(path)
        if existing.digest != plan.digest:
            raise ProtocolError(
                f'{path} already records a different partition '
                f'(digest {existing.digest[:12]} vs {plan.digest[:12]}). '
                f'Two writers disagree about the held-out subjects; keep one '
                f'--exp_root per sweep or use a fresh one.')
        return existing

    if os.path.exists(path):
        return _reuse()

    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'w') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(tmp, path)          # atomic, and fails if the record appeared
    except FileExistsError:
        return _reuse()
    except OSError:
        try:                        # filesystem without hard links
            with open(path, 'x') as handle:
                handle.write(payload)
        except FileExistsError:
            return _reuse()
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return plan


def load_plan(path: str) -> FoldPlan:
    """Read and verify a ``partition.json``."""
    try:
        with open(path) as handle:
            record = json.load(handle)
    except FileNotFoundError:
        raise ProtocolError(
            f'no partition record at {path}. Create it first with '
            f'`runners/run_partition.py` or Stage 2.') from None
    return plan_from_record(record)
