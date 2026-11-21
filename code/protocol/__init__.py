"""ADR-001 implementation: the only place a train/val/test role is computed.

Public API (see `code/notes/plan/features/ADR-001-subject-kfold-plan.md`)::

    from protocol import build_plan, load_plan, save_plan, FoldPlan

Consumers receive a `FoldPlan` and a role name — never a seed, a fold index or a
ratio — so two stages, two arms or a resume cannot disagree about who is held
out. Everything here is pure Python + NumPy: no torch, no dataset import.
"""
from .constants import (CRITERIA, DEFAULT_VAL_SEED, K, MIN_EVAL_FREQ, ROLES,
                        ROLE_SHARE, SCHEMA_EVAL, SCHEMA_PARTITION,
                        SCHEMA_SELECTION, SELECTION_RULES, SHARE_TOLERANCE,
                        SPLIT_STRATEGIES)
from .guards import LeakGuard, assert_roles_disjoint
from .partition import allocate_strata, split_universe, subject_stratum
from .plan import (FoldPlan, ProtocolError, build_plan, canonical_json,
                   load_plan, partition_digest, plan_from_record, save_plan,
                   to_record, validate_plan)
from .records import (append_jsonl, make_eval_record, read_json, read_jsonl,
                      read_selection, write_json, write_selection)
from .selection import SelectionTracker, val_mask_seed

__all__ = [
    # constants
    'K', 'ROLES', 'ROLE_SHARE', 'SHARE_TOLERANCE', 'MIN_EVAL_FREQ',
    'SPLIT_STRATEGIES', 'CRITERIA', 'SELECTION_RULES', 'DEFAULT_VAL_SEED',
    'SCHEMA_PARTITION', 'SCHEMA_EVAL', 'SCHEMA_SELECTION',
    # plan
    'ProtocolError', 'FoldPlan', 'build_plan', 'validate_plan', 'save_plan',
    'load_plan', 'to_record', 'plan_from_record', 'partition_digest',
    'canonical_json',
    # partition
    'split_universe', 'subject_stratum', 'allocate_strata',
    # selection
    'SelectionTracker', 'val_mask_seed',
    # records
    'write_json', 'read_json', 'append_jsonl', 'read_jsonl',
    'make_eval_record', 'write_selection', 'read_selection',
    # guards
    'assert_roles_disjoint', 'LeakGuard',
]
