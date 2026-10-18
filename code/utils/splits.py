import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = ['subject_folds', 'get_stage_splits', 'write_fold_manifest',
           'resolve_split_keys']


def subject_folds(subjects: Sequence[str], n_folds: int) -> List[List[str]]:
    """Partition sorted unique subject ids into n_folds disjoint folds.

    Unlike sklearn.model_selection.GroupKFold, this partitions the SUBJECTS
    evenly, not the samples. This guarantees that a subject gaining or losing
    clips does not shift the fold boundaries.

    Args:
        subjects: Sequence of subject IDs (e.g., ['F001', 'F002', ...]).
        n_folds: Number of folds to create.

    Returns:
        A list of n_folds lists, where each inner list contains the subject IDs
        for that fold.
    """
    if n_folds <= 0:
        raise ValueError(f"n_folds must be > 0; got {n_folds}")
    
    unique_subjects = sorted(set(subjects))
    if not unique_subjects:
        return [[] for _ in range(n_folds)]
        
    # np.array_split handles cases where len(unique_subjects) is not perfectly
    # divisible by n_folds.
    folds = np.array_split(unique_subjects, n_folds)
    return [fold.tolist() for fold in folds]


def get_stage_splits(subjects: Sequence[str], fold: int, n_folds: int = 5,
                     stage: int = 3, use_small_val: bool = False,
                     val_seed: int = 42) -> Dict[str, List[str]]:
    """Generates subject-exclusive splits for Stage 2 (SSL) and Stage 3.

    Implements the Cyclic Rotation strategy:
    - test = Fold k
    - train = All folds != k (if val is unused)
    - val = ~10% of non-test pool (if Option B is used)

    Args:
        subjects: Sequence of subject IDs.
        fold: Current fold index (0 to n_folds - 1).
        n_folds: Total number of folds. If <= 1, returns all subjects in train
            and test, with an empty val split.
        stage: Pipeline stage (2 for Domain SSL, 3 for Task Fine-Tuning).
        use_small_val: If True in Stage 3, carves ~10% of the non-test pool for val.
        val_seed: Seed for the Option B validation draw.

    Returns:
        Dict with keys 'train', 'val', 'test', each containing a list of subject IDs.
    """
    unique_subjects = sorted(set(subjects))
    
    if n_folds <= 1:
        return {
            'train': list(unique_subjects),
            'val': [],
            'test': list(unique_subjects)
        }

    if not (0 <= fold < n_folds):
        raise ValueError(f"fold must be in [0, {n_folds - 1}]; got {fold}")
    if stage not in (2, 3):
        raise ValueError(f"stage must be 2 or 3; got {stage}")
    if stage == 2 and use_small_val:
        use_small_val = False

    folds = subject_folds(unique_subjects, n_folds)
    
    test_subjects = folds[fold]
    
    # Non-test pool
    train_pool = []
    for i in range(n_folds):
        if i != fold:
            train_pool.extend(folds[i])
            
    if not use_small_val:
        val_subjects = []
        train_subjects = train_pool
    else:
        if not train_pool:
            raise ValueError("Corpus is too small to support a validation role.")
            
        n_val_subs = max(1, int(round(0.10 * len(unique_subjects))))
        n_val_subs = min(n_val_subs, len(train_pool) - 1)
        
        if n_val_subs <= 0:
            raise ValueError("Corpus is too small to support a validation role.")
            
        rng = np.random.default_rng(val_seed + fold)
        val_subjects = rng.choice(train_pool, size=n_val_subs, replace=False).tolist()
        train_subjects = [s for s in train_pool if s not in val_subjects]

    return {
        'train': sorted(train_subjects),
        'val': sorted(val_subjects),
        'test': sorted(test_subjects)
    }


def write_fold_manifest(path: str, fold: int, n_folds: int, stage: int,
                        roles: Dict[str, List[str]]) -> None:
    """Writes the fold partition to a JSON manifest for auditability."""
    manifest = {
        'fold': fold,
        'n_folds': n_folds,
        'stage': stage,
        'roles': roles
    }
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    # ATOMIC: several writers can target this path (the DDP ranks of one run, a
    # resumed run, an --eval_only re-run), and a truncated JSON would silently
    # destroy the only audit record of the partition. Same pattern as the
    # Stage-1 weight cache: process-unique temp, then one atomic replace.
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'w') as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, path)


def resolve_split_keys(usable_sessions: List[Dict], is_train: bool,
                       train_ratio: float, split_by: str,
                       val_subject: str, split_role: str,
                       n_folds: int, fold: int, use_small_val: bool,
                       dataset_name: str = '') -> Tuple[List[str], str]:
    """Resolve the split keys (subjects or sessions) for ONE dataset view.

    Two policies, selected by ``n_folds``:

    * ``n_folds > 1`` -- the k-fold cyclic rotation. ``split_role`` names the
      view (``train`` / ``val`` / ``test``) and the keys are SUBJECT ids.
    * ``n_folds <= 1`` -- the legacy ratio cut (``train_ratio`` / ``split_by``
      with the ``val_subject`` LOSO override), byte-identical to the previous
      inline code in both ROI dataset classes.

    An EMPTY ``train`` role is an error (there is nothing to learn from). An
    empty ``val`` or ``test`` role is a LEGITIMATE outcome and is returned as
    ``[]``: a held-out subject can lose every clip to the rail/sentinel rules
    (locally F004 is rail-saturated), and Stage 3 Option A has no val at all.
    Callers must represent such a role as an EMPTY dataset, never by forwarding
    ``subjects=[]`` to the base dataset -- a falsy filter there means "every
    subject" and would silently load the whole corpus.

    :returns: ``(split_keys, split_key_kind)`` where ``split_key_kind`` is
        ``'subject'`` or ``'session'`` (the legacy ``split_by`` value).
    """
    who = dataset_name or 'dataset'
    if n_folds > 1:
        if split_by == 'session':
            raise ValueError(
                f"{who}: split_by='session' is not supported when n_folds > 1; "
                f"a session split can put two sessions of one subject on both "
                f"sides (k-fold scenario 5). Use split_by='subject'.")

        subs = sorted({s['subject'] for s in usable_sessions})
        splits = get_stage_splits(subs, fold=fold, n_folds=n_folds, stage=3,
                                  use_small_val=use_small_val)

        role = split_role or ('train' if is_train else 'val')
        if role not in splits:
            raise ValueError(
                f'{who}: unknown split_role {role!r}; must be train, val or '
                f'test.')

        keep = splits[role]
        if not keep and role == 'train':
            raise ValueError(
                f'{who}: n_folds={n_folds}, fold={fold} leaves the train split '
                f'empty ({len(subs)} usable subject(s)); nothing to train on.')

        return sorted(keep), 'subject'

    # Legacy path (n_folds <= 1)
    if val_subject:
        subs = sorted({s['subject'] for s in usable_sessions})
        if val_subject not in subs:
            raise ValueError(
                f'val_subject={val_subject!r} is not among the usable '
                f'subjects {subs}.')
        keep = ({val_subject} if not is_train
                else set(subs) - {val_subject})
        if not keep:
            raise ValueError(
                f'val_subject={val_subject!r} leaves the train split '
                f'empty (it is the only usable subject).')
        return sorted(keep), 'subject'
    else:
        keys = sorted({s['subject'] if split_by == 'subject'
                       else s['session'] for s in usable_sessions})
        n_keep = max(1, int(round(len(keys) * train_ratio)))
        keep = set(keys[:n_keep]) if is_train else set(keys[n_keep:])
        if not keep:
            raise ValueError(
                f"split_by={split_by!r} with train_ratio {train_ratio} "
                f'leaves the {"train" if is_train else "val"} split '
                f'empty: {len(keys)} {split_by}(s) -> {n_keep} train '
                f'side(s). Lower train_ratio, or pass val_subject.')
        return sorted(keep), split_by
