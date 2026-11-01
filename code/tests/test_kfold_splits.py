import pytest
import numpy as np

from utils.splits import (get_stage_splits, resolve_split_keys,
                          subject_folds, subject_stratum)


def test_subject_folds_disjoint():
    subjects = [f"F{i:03d}" for i in range(140)]
    folds = subject_folds(subjects, 5)
    
    assert len(folds) == 5
    all_subjects = []
    for fold in folds:
        assert len(fold) == 28
        all_subjects.extend(fold)
        
    assert len(set(all_subjects)) == 140
    assert sorted(all_subjects) == sorted(subjects)


def test_get_stage_splits_stage2():
    subjects = [f"F{i:03d}" for i in range(140)]
    splits = get_stage_splits(subjects, fold=0, n_folds=5, stage=2)
    
    assert len(splits['test']) == 28
    assert len(splits['val']) == 0
    assert len(splits['train']) == 112
    
    assert set(splits['test']).isdisjoint(set(splits['train']))
    assert sorted(splits['test'] + splits['train']) == sorted(subjects)


def test_get_stage_splits_stage3_option_a():
    subjects = [f"F{i:03d}" for i in range(140)]
    splits = get_stage_splits(subjects, fold=2, n_folds=5, stage=3, use_small_val=False)
    
    assert len(splits['test']) == 28
    assert len(splits['val']) == 0
    assert len(splits['train']) == 112
    
    assert set(splits['test']).isdisjoint(set(splits['train']))


def test_get_stage_splits_stage3_option_b():
    subjects = [f"F{i:03d}" for i in range(140)]
    splits = get_stage_splits(subjects, fold=4, n_folds=5, stage=3, use_small_val=True)
    
    assert len(splits['test']) == 28
    assert len(splits['val']) == 14
    assert len(splits['train']) == 98
    
    assert set(splits['test']).isdisjoint(set(splits['val']))
    assert set(splits['test']).isdisjoint(set(splits['train']))
    assert set(splits['val']).isdisjoint(set(splits['train']))
    
    assert sorted(splits['test'] + splits['val'] + splits['train']) == sorted(subjects)


def test_cyclic_rotation_determinism():
    subjects = [f"F{i:03d}" for i in range(140)]
    
    splits1 = get_stage_splits(subjects, fold=1, n_folds=5, stage=3, use_small_val=True)
    splits2 = get_stage_splits(subjects, fold=1, n_folds=5, stage=3, use_small_val=True)
    
    assert splits1['test'] == splits2['test']
    assert splits1['val'] == splits2['val']
    assert splits1['train'] == splits2['train']


def test_dynamic_fold_count_compatibility():
    subjects = ["F001", "F002", "F003", "F004"]
    splits = get_stage_splits(subjects, fold=0, n_folds=4, stage=3, use_small_val=False)
    
    assert len(splits['test']) == 1
    assert len(splits['val']) == 0
    assert len(splits['train']) == 3
    
    assert "F001" in splits['test']
    assert "F002" in splits['train']
    assert "F003" in splits['train']
    assert "F004" in splits['train']


def test_legacy_path_reproduces_keys():
    subjects = ["F001", "F002", "F003", "F004"]
    splits = get_stage_splits(subjects, fold=0, n_folds=0, stage=3)
    
    assert len(splits['test']) == 4
    assert len(splits['val']) == 0
    assert len(splits['train']) == 4
    assert splits['train'] == sorted(subjects)


# --------------------------------------------------------------------------- #
# Empty folds are a legitimate outcome, not an error.
# A held-out subject can lose every clip to the rail/sentinel rules (locally
# F004 is rail-saturated), so `n_folds > n_subjects` leaves folds with no
# subject at all. The split layer must represent that as an empty role.
# --------------------------------------------------------------------------- #
def _sessions(*subjects):
    """Minimal session records: `resolve_split_keys` only reads 'subject'."""
    return [{'subject': s} for s in subjects]


def test_subject_folds_tolerates_more_folds_than_subjects():
    folds = subject_folds(["F001", "F002"], 4)

    assert len(folds) == 4
    assert folds[0] == ["F001"]
    assert folds[1] == ["F002"]
    assert folds[2] == []
    assert folds[3] == []


def test_empty_test_role_is_returned_not_raised():
    keys, kind = resolve_split_keys(
        _sessions("F001", "F002"), is_train=False, train_ratio=0.8,
        split_by="subject", val_subject=None, split_role="test",
        n_folds=4, fold=2, use_small_val=False,
        dataset_name="dummy")

    assert keys == []
    assert kind == "subject"


def test_empty_train_role_still_raises():
    """Training on nothing is a real error, unlike an unscorable fold."""
    with pytest.raises(ValueError, match="train split empty"):
        resolve_split_keys(
            _sessions("F001"), is_train=True, train_ratio=0.8,
            split_by="subject", val_subject=None, split_role="train",
            n_folds=4, fold=0, use_small_val=False,
            dataset_name="dummy")


def test_session_split_rejected_with_folds():
    with pytest.raises(ValueError, match="split_by='session'"):
        resolve_split_keys(
            _sessions("F001", "F002", "F003"), is_train=True,
            train_ratio=0.8, split_by="session", val_subject=None,
            split_role="train", n_folds=3, fold=0, use_small_val=False,
            dataset_name="dummy")


# --------------------------------------------------------------------------- #
# Stage 1/2 isolation: a fold may only ever resume its OWN fold.
# --------------------------------------------------------------------------- #
def _ns(**kw):
    from types import SimpleNamespace
    return SimpleNamespace(**kw)


def test_resume_accepts_the_same_fold():
    from utils.checkpoint import check_resume_fold

    verdict = check_resume_fold(_ns(n_folds=4, fold=2),
                               _ns(n_folds=4, fold=2))
    assert verdict['match'] is True


def test_resume_refuses_another_fold():
    from utils.checkpoint import check_resume_fold

    with pytest.raises(SystemExit, match="FOLD MISMATCH"):
        check_resume_fold(_ns(n_folds=4, fold=1), _ns(n_folds=4, fold=0))


def test_resume_warns_on_a_non_fold_checkpoint():
    """A whole-corpus pretrain resumed into a fold run is allowed but flagged."""
    from utils.checkpoint import check_resume_fold

    verdict = check_resume_fold(_ns(n_folds=4, fold=1), _ns(n_folds=0, fold=0))
    assert verdict['match'] is False


def test_resume_noop_when_not_a_fold_run():
    from utils.checkpoint import check_resume_fold

    assert check_resume_fold(_ns(n_folds=0, fold=0), _ns(n_folds=4, fold=3)) == {}


# --------------------------------------------------------------------------- #
# Stratified partition (2026-11-01).
#
# The legacy rule cuts SORTED ids into contiguous blocks, so on BP4D+ (where
# 'F001' sorts before 'M001') a fold is a sex block: fold 0-1 all female,
# 3-4 all male. Per-fold spread then measures COHORTS, not model stability.
# 'sex' shuffles within each stratum and interleaves proportionally, keeping
# the array_split fold SIZES while making every fold representative.
# --------------------------------------------------------------------------- #
def _corpus(n_f: int = 82, n_m: int = 57):
    """The real universe size: 139 usable BP4D+ subjects."""
    return ([f'F{i:03d}' for i in range(1, n_f + 1)]
            + [f'M{i:03d}' for i in range(1, n_m + 1)])


def test_subject_stratum_reads_the_id_prefix():
    assert subject_stratum('F001') == 'F'
    assert subject_stratum('m058') == 'M'
    assert subject_stratum('') == '?'
    assert subject_stratum('001') == '?'


def test_legacy_partition_is_unchanged_and_is_a_sex_block():
    """The default MUST stay byte-identical (the recorded sweep depends on it)."""
    subjects = _corpus()
    folds = subject_folds(subjects, 5)

    assert [len(f) for f in folds] == [28, 28, 28, 28, 27]
    assert folds[0] == [f'F{i:03d}' for i in range(1, 29)]
    # the confound this whole change exists to remove
    assert {s[0] for s in folds[0]} == {'F'}
    assert {s[0] for s in folds[4]} == {'M'}


def test_stratified_partition_is_deterministic():
    subjects = _corpus()
    first = subject_folds(subjects, 5, split_seed=42, stratify='sex')
    again = subject_folds(subjects, 5, split_seed=42, stratify='sex')
    other = subject_folds(subjects, 5, split_seed=7, stratify='sex')

    assert first == again
    assert first != other


def test_stratified_partition_keeps_sizes_and_balances_sex():
    subjects = _corpus()
    folds = subject_folds(subjects, 5, split_seed=42, stratify='sex')

    # Fold sizes are still the array_split ones -> the compute budget per fold
    # does not change when the confound is removed.
    assert [len(f) for f in folds] == [28, 28, 28, 28, 27]
    flat = [s for f in folds for s in f]
    assert sorted(flat) == sorted(subjects)      # complete
    assert len(set(flat)) == len(flat)           # disjoint

    # Every fold holds both sexes, and the female count is within 1 subject of
    # the corpus proportion.
    total_f = sum(1 for s in subjects if s[0] == 'F')
    for fold in folds:
        n_f = sum(1 for s in fold if s[0] == 'F')
        assert 0 < n_f < len(fold)
        assert abs(n_f - total_f * len(fold) / len(subjects)) <= 1.0


def test_stratified_partition_requires_an_explicit_seed():
    """An unseeded draw cannot be reproduced by the next stage of the sweep."""
    with pytest.raises(ValueError, match='split_seed'):
        subject_folds(_corpus(), 5, split_seed=None, stratify='sex')


def test_unknown_partition_strategy_is_rejected():
    with pytest.raises(ValueError, match='split_stratify'):
        subject_folds(_corpus(), 5, split_seed=1, stratify='age')


def test_get_stage_splits_threads_the_partition_rule():
    """Stage 2's held-out set MUST be Stage 3's test set for the same fold."""
    subjects = _corpus()
    kwargs = dict(split_seed=42, stratify='sex')
    s2 = get_stage_splits(subjects, fold=0, n_folds=5, stage=2, **kwargs)
    s3 = get_stage_splits(subjects, fold=0, n_folds=5, stage=3, **kwargs)

    assert s2['test'] == s3['test']
    assert set(s2['test']).isdisjoint(s2['train'])
    assert len(s2['train']) + len(s2['test']) == len(subjects)


def test_resolve_split_keys_threads_the_partition_rule():
    kwargs = dict(split_seed=42, stratify='sex')
    keys, kind = resolve_split_keys(
        _sessions(*_corpus()), is_train=False, train_ratio=0.8,
        split_by='subject', val_subject=None, split_role='test',
        n_folds=5, fold=3, use_small_val=False, dataset_name='dummy', **kwargs)

    assert kind == 'subject'
    assert len(keys) == 28
    assert {s[0] for s in keys} == {'F', 'M'}
