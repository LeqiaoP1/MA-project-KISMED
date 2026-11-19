"""ADR-001 constants — the protocol's fixed numbers.

Nothing in this module may be tuned at run time. ADR-001 section 2.1: the fold
count ``K`` and the role ratio are thesis-level properties; changing either
requires a new ADR. They are therefore module constants with **no CLI flag and
no config key** behind them.
"""
from typing import Dict, Tuple

__all__ = [
    'K', 'ROLES', 'ROLE_SHARE', 'SHARE_TOLERANCE', 'MIN_EVAL_FREQ',
    'SPLIT_STRATEGIES', 'CRITERIA', 'SELECTION_RULES', 'DEFAULT_VAL_SEED',
    'SCHEMA_PARTITION', 'SCHEMA_EVAL', 'SCHEMA_SELECTION',
]

#: ADR-001 section 2.1: the subject universe is divided into ``K`` blocks and
#: block ``k`` is the test role of fold ``k``.
K: int = 5

#: The three roles, in the canonical order every record uses.
ROLES: Tuple[str, ...] = ('train', 'val', 'test')

#: ADR-001 section 2.1: 70 / 10 / 20 by SUBJECT, not by sample.
ROLE_SHARE: Dict[str, float] = {'train': 0.70, 'val': 0.10, 'test': 0.20}

#: ADR-001 section 3, Criterion 1: realised shares must be within +/-2
#: percentage points of ``ROLE_SHARE``.
SHARE_TOLERANCE: float = 0.02

#: Plan decision D10. A validation pass over the 10% role costs about
#: ``0.10 / 0.70`` of a training epoch, so an evaluation every ``MIN_EVAL_FREQ``
#: epochs stays under the 5% budget of ADR-001 Criterion 4. Lower values are
#: refused rather than silently overspending.
MIN_EVAL_FREQ: int = 3

#: Partition strategies. ``'sex'`` shuffles within each stratum and interleaves
#: proportionally; ``'none'`` keeps sorted contiguous blocks (or a global
#: shuffle when a seed is supplied).
SPLIT_STRATEGIES: Tuple[str, ...] = ('none', 'sex')

#: Validation criterion -> optimisation direction. ``best_val`` must never
#: maximise a loss or minimise a correlation.
CRITERIA: Dict[str, str] = {'joint_loss': 'min', 'pearson_band': 'max'}

#: Fallback seed for the validation draw when neither ``val_seed`` nor
#: ``split_seed`` is given. The draw adds no free parameter of its own: it
#: defaults to ``split_seed`` and only falls back to this value when the
#: partition itself is seedless (sorted contiguous blocks).
DEFAULT_VAL_SEED: int = 42

#: ADR-001 section 2.3: ``fixed_epoch`` (default) or ``best_val``.
SELECTION_RULES: Tuple[str, ...] = ('fixed_epoch', 'best_val')

#: One schema string per artefact family; a reader refuses anything else.
SCHEMA_PARTITION: str = 'adr-001/partition/1'
SCHEMA_EVAL: str = 'adr-001/eval/1'
SCHEMA_SELECTION: str = 'adr-001/selection/1'
