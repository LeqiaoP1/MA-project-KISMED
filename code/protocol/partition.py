"""Subject-block partitioning (ADR-001 I2, Criterion 1).

The whole module is a pure function of its arguments: no global RNG, no
dataset access, no torch, no sklearn. ``sklearn.model_selection.GroupKFold`` is
deliberately **not** used — it balances SAMPLES per group, so a subject gaining
or losing a clip would move a fold boundary. ADR-001 partitions SUBJECTS.
"""
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from .constants import K, SPLIT_STRATEGIES

__all__ = ['subject_stratum', 'allocate_strata', 'split_universe']


def subject_stratum(subject: str) -> str:
    """Stratum label of one subject id: its leading letter, uppercased.

    BP4D+ ids are ``F001..`` / ``M001..``, so the stratum is sex. An id that does
    not start with a letter becomes ``'?'``: a foreign naming scheme then
    degrades to "no stratification" instead of raising.
    """
    text = str(subject).strip()
    return text[0].upper() if text and text[0].isalpha() else '?'


def allocate_strata(n_total: int, sizes: Dict[str, int]) -> Dict[str, int]:
    """Split ``n_total`` slots over strata in proportion to ``sizes``.

    Largest-remainder method, ties broken by stratum name, so the allocation is
    deterministic and sums to ``n_total`` exactly. ADR-001 I2 requires the
    validation role to be a function of the stratifier; an unstratified draw of
    14 of ~111 subjects can land on one sex and skew ``train = pool - val``.
    """
    if n_total < 0:
        raise ValueError(f'n_total must be >= 0; got {n_total}')
    total = sum(sizes.values())
    if total <= 0 or n_total == 0:
        return {name: 0 for name in sizes}

    exact = {name: n_total * size / total for name, size in sizes.items()}
    base = {name: int(math.floor(value)) for name, value in exact.items()}
    remainder = n_total - sum(base.values())
    # Descending fractional part, then name: a total order, never dict order.
    order = sorted(exact, key=lambda name: (-(exact[name] - base[name]), name))
    for name in order[:remainder]:
        base[name] += 1
    return base


def _interleave_strata(strata: Dict[str, List[str]]) -> List[str]:
    """Merge strata into one sequence with every stratum spread uniformly.

    Member ``j`` of an ``n``-member stratum sits at global rank ``(j + 0.5) / n``,
    so ANY contiguous chunk of the merged sequence holds each stratum in (almost
    exactly) its corpus proportion. That is what lets ``np.array_split`` produce
    folds that are equal in SIZE and in COMPOSITION; concatenating the strata
    would rebuild the very blocks the stratifier exists to remove.
    """
    slots = []
    for name in sorted(strata):
        members = strata[name]
        n_members = len(members)
        for j, subject in enumerate(members):
            slots.append(((j + 0.5) / n_members, name, j, subject))
    slots.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in slots]


def split_universe(universe: Sequence[str], k: int = K,
                   split_seed: Optional[int] = None,
                   split_stratify: str = 'none') -> List[List[str]]:
    """Partition the subject universe into ``k`` disjoint, size-matched blocks.

    :param universe: subject ids (duplicates and order are irrelevant).
    :param k: number of blocks; ``K`` for any protocol run.
    :param split_seed: ``None`` keeps the sorted order (contiguous blocks); an
        int makes the assignment a reproducible draw. Required when
        ``split_stratify != 'none'``, because an unseeded random partition
        cannot be reproduced by the next stage of the sweep.
    :param split_stratify: one of ``SPLIT_STRATEGIES``. ``'sex'`` shuffles within
        each stratum, then interleaves proportionally.
    :returns: ``k`` lists of subject ids, each sorted; block sizes differ by at
        most one. The result is a pure function of the four arguments (I2).
    :raises ValueError: an invalid ``k``/strategy, or a stratified draw without
        a seed.
    """
    if k <= 0:
        raise ValueError(f'k must be > 0; got {k}')
    stratify = str(split_stratify or 'none').strip().lower()
    if stratify not in SPLIT_STRATEGIES:
        raise ValueError(
            f'split_stratify must be one of {SPLIT_STRATEGIES}; got '
            f'{split_stratify!r}')
    if stratify != 'none' and split_seed is None:
        raise ValueError(
            f'split_stratify={stratify!r} requires an explicit split_seed: an '
            f'unseeded partition cannot be reproduced by the next stage or by '
            f'another arm.')

    unique = sorted(set(str(subject) for subject in universe))
    if not unique:
        return [[] for _ in range(k)]

    if split_seed is None and stratify == 'none':
        return [block.tolist() for block in np.array_split(unique, k)]

    rng = np.random.default_rng(int(split_seed))
    if stratify == 'none':
        order = rng.permutation(len(unique))
        sequence = [unique[i] for i in order]
    else:
        strata: Dict[str, List[str]] = {}
        for subject in unique:
            strata.setdefault(subject_stratum(subject), []).append(subject)
        # One rng, strata consumed in sorted order -> deterministic across
        # processes (no dependence on dict/set iteration).
        for name in sorted(strata):
            order = rng.permutation(len(strata[name]))
            strata[name] = [strata[name][i] for i in order]
        sequence = _interleave_strata(strata)

    return [sorted(block.tolist()) for block in np.array_split(sequence, k)]
