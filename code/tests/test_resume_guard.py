"""Resuming may only continue the SAME partition (ADR-001 I2/I4).

A checkpoint from another fold loads perfectly: shapes match, the loss falls, and
the run looks healthy while it continues weights that saw this fold's held-out
subjects. The guard therefore compares the partition DIGEST, not the fold index.
"""
import argparse

import pytest

from utils.checkpoint import check_resume_fold


def _args(**kw):
    return argparse.Namespace(**kw)


def test_same_partition_resumes():
    verdict = check_resume_fold(_args(partition_digest='abc123'),
                               _args(partition_digest='abc123'))
    assert verdict['match'] is True


def test_another_partition_is_refused():
    with pytest.raises(SystemExit, match='PARTITION MISMATCH'):
        check_resume_fold(_args(partition_digest='abc123'),
                          _args(partition_digest='def456'))


def test_same_fold_index_is_not_enough():
    """Two partitions can share ``fold == 2`` and still hold out other people."""
    with pytest.raises(SystemExit, match='PARTITION MISMATCH'):
        check_resume_fold(_args(fold=2, partition_digest='abc123'),
                          _args(fold=2, partition_digest='def456'))


def test_legacy_checkpoint_warns_instead_of_passing():
    verdict = check_resume_fold(_args(partition_digest='abc123'),
                                _args(fold=2))
    assert verdict['match'] is False
    assert verdict['checkpoint_digest'] is None


def test_no_plan_means_nothing_to_check():
    assert check_resume_fold(_args(), _args(fold=2)) == {}
