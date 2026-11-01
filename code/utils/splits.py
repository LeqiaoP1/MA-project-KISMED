import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ['subject_folds', 'get_stage_splits', 'write_fold_manifest',
           'resolve_split_keys', 'subject_stratum', 'SPLIT_STRATEGIES',
           'as_split_seed', 'as_split_strategy']

#: Partition strategies accepted by :func:`subject_folds`.
#:
#: ``'none'`` = the legacy rule: sorted ids cut into CONTIGUOUS blocks with
#: ``np.array_split``. On BP4D+ that makes fold = sex block, because ``F001..``
#: sorts before ``M001..``, so a per-fold spread measures cohorts rather than
#: model stability.
#:
#: ``'sex'``  = a seeded shuffle WITHIN each stratum, then a proportional
#: interleave, so every fold is a near-representative sample of the corpus and
#: the fold sizes are still the ``array_split`` ones. Requires ``split_seed``.
SPLIT_STRATEGIES = ('none', 'sex')


def subject_stratum(subject: str) -> str:
    """Stratum label of one subject id -> its leading letter, uppercased.

    BP4D+ ids are ``F001..F082`` / ``M001..M058``, so this is SEX. An id that
    does not start with a letter becomes its own ``'?'`` stratum: a foreign
    naming scheme then degrades to "no stratification" instead of raising.
    """
    s = str(subject).strip()
    return s[0].upper() if s and s[0].isalpha() else '?'


def as_split_seed(value) -> Optional[int]:
    """Coerce a config/CLI value to a partition seed (``None`` = legacy)."""
    if value is None or value == '' or value == 'None':
        return None
    return int(value)


def as_split_strategy(value) -> str:
    """Coerce a config/CLI value to a validated :data:`SPLIT_STRATEGIES`."""
    strategy = str(value if value is not None else 'none').strip().lower()
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(
            f'split_stratify must be one of {SPLIT_STRATEGIES}; '
            f'got {value!r}')
    return strategy


def _interleave_strata(strata: Dict[str, List[str]]) -> List[str]:
    """Merge strata into ONE sequence with every stratum spread uniformly.

    Member ``j`` of a stratum with ``n`` members sits at global rank
    ``(j + 0.5) / n``, so ANY contiguous chunk of the merged sequence holds
    each stratum in (almost exactly) its corpus proportion. That is what lets
    ``np.array_split`` produce folds that are equal in SIZE and equal in
    COMPOSITION; concatenating the strata instead would rebuild the very sex
    blocks this exists to remove.

    Deterministic: ties fall back to the stratum name, then the member index.
    """
    slots = []
    for name in sorted(strata):
        members = strata[name]
        n_members = len(members)
        for j, subj in enumerate(members):
            slots.append(((j + 0.5) / n_members, name, j, subj))
    slots.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in slots]


def subject_folds(subjects: Sequence[str], n_folds: int,
                  split_seed: Optional[int] = None,
                  stratify: str = 'none') -> List[List[str]]:
    """Partition sorted unique subject ids into n_folds disjoint folds.

    Unlike sklearn.model_selection.GroupKFold, this partitions the SUBJECTS
    evenly, not the samples. This guarantees that a subject gaining or losing
    clips does not shift the fold boundaries.

    Args:
        subjects: Sequence of subject IDs (e.g., ['F001', 'F002', ...]).
        n_folds: Number of folds to create.
        split_seed: Seed of the partition. ``None`` (with ``stratify='none'``)
            reproduces the LEGACY contiguous partition byte-for-byte; any int
            makes the assignment a reproducible random draw.
        stratify: One of :data:`SPLIT_STRATEGIES`. ``'sex'`` shuffles within
            each stratum and interleaves proportionally, which removes the
            "fold = sex block" confound of the contiguous rule.

    Returns:
        A list of n_folds lists, where each inner list contains the subject IDs
        for that fold (sorted, so a manifest is canonical).
    """
    if n_folds <= 0:
        raise ValueError(f"n_folds must be > 0; got {n_folds}")
    stratify = str(stratify or 'none').strip().lower()
    if stratify not in SPLIT_STRATEGIES:
        raise ValueError(
            f"split_stratify must be one of {SPLIT_STRATEGIES}; got {stratify!r}")
    if stratify != 'none' and split_seed is None:
        raise ValueError(
            f"stratify={stratify!r} requires an explicit split_seed: an "
            f"unseeded random partition cannot be reproduced by the next "
            f"stage of the sweep, which would silently break the pairing "
            f"between the Stage-2 and Stage-3 folds.")

    unique_subjects = sorted(set(subjects))
    if not unique_subjects:
        return [[] for _ in range(n_folds)]

    if stratify == 'none' and split_seed is None:
        # LEGACY path, kept bit-identical so every existing config/script and
        # the recorded 5-fold sweep stay reproducible.
        folds = np.array_split(unique_subjects, n_folds)
        return [fold.tolist() for fold in folds]

    rng = np.random.default_rng(int(split_seed))
    if stratify == 'none':
        order = rng.permutation(len(unique_subjects))
        sequence = [unique_subjects[i] for i in order]
    else:
        strata: Dict[str, List[str]] = {}
        for subj in unique_subjects:
            strata.setdefault(subject_stratum(subj), []).append(subj)
        # One rng, strata consumed in sorted order -> deterministic across
        # processes (no dependence on dict/set iteration).
        for name in sorted(strata):
            order = rng.permutation(len(strata[name]))
            strata[name] = [strata[name][i] for i in order]
        sequence = _interleave_strata(strata)

    folds = np.array_split(sequence, n_folds)
    return [sorted(fold.tolist()) for fold in folds]


def get_stage_splits(subjects: Sequence[str], fold: int, n_folds: int = 5,
                     stage: int = 3, use_small_val: bool = False,
                     val_seed: int = 42, split_seed: Optional[int] = None,
                     stratify: str = 'none') -> Dict[str, List[str]]:
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
        split_seed: Seed of the PARTITION itself (:func:`subject_folds`).
            ``None`` + ``stratify='none'`` = the legacy contiguous blocks.
        stratify: Partition strategy, see :data:`SPLIT_STRATEGIES`. Stage 2 and
            Stage 3 of one sweep MUST use the same pair of values, otherwise
            fold k's encoder was pre-trained on fold k's test subjects.

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

    folds = subject_folds(unique_subjects, n_folds,
                          split_seed=split_seed, stratify=stratify)
    
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
                        roles: Dict[str, List[str]],
                        split_seed: Optional[int] = None,
                        stratify: str = 'none') -> None:
    """Writes the fold partition to a JSON manifest for auditability.

    ``split_seed`` / ``split_stratify`` record HOW the partition was DRAWN: the
    legacy contiguous rule and a seeded stratified draw produce different folds
    from the same corpus, so a manifest that omits them cannot tell the two
    apart when a Stage-2 / Stage-3 pair is audited.
    """
    manifest = {
        'fold': fold,
        'n_folds': n_folds,
        'stage': stage,
        'split_seed': split_seed,
        'split_stratify': stratify,
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
                       dataset_name: str = '',
                       split_seed: Optional[int] = None,
                       stratify: str = 'none') -> Tuple[List[str], str]:
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
                                  use_small_val=use_small_val,
                                  split_seed=split_seed, stratify=stratify)

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
