"""Universe tests (ADR-001 I2) — the availability predicate, on a synthetic tree.

The real corpus is 139 subjects / 1384 usable sessions, but a test must not need
95 GB of pixels: the predicate is exercised on a hand-built raw tree instead.
"""
import inspect
import os

import pytest

from data.universe import (available_data_sets, corpus_sessions,
                           corpus_subjects, is_usable_session, resolve_corpus)


def _touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w'):
        pass


def _session(root: str, subject: str, task: str = 'T1', *,
             video: bool = True, ir: bool = True, resp: bool = True) -> None:
    """Create one ``(subject, task)`` session of the BP4D+ raw layout."""
    if video:
        _touch(os.path.join(root, 'Thermal', subject, f'{subject}_{task}.wmv'))
    if ir:
        _touch(os.path.join(root, 'IRFeatures', f'{subject}_{task}.txt'))
    if resp:
        _touch(os.path.join(root, 'Physiology', subject, task, 'Resp_Volts.txt'))


def test_universe_uses_the_availability_predicate(tmp_path):
    """A subject without IRFeatures or Resp_Volts is not in the universe."""
    root = str(tmp_path)
    _session(root, 'F001')                                   # complete
    _session(root, 'F002', ir=False)                         # no landmark track
    _session(root, 'F003', resp=False)                       # no respiration
    _session(root, 'F004', video=False)                      # nothing to discover

    assert corpus_subjects(raw_root=root, data_set='tir_roi') == ['F001']
    assert len(corpus_sessions(root, 'tir_roi')) == 1


def test_universe_ignores_task_coverage(tmp_path):
    """A subject is in the universe whatever tasks they have.

    Task selection narrows which clips a role reads; letting it move a subject
    between roles would make the partition depend on the task set (I2).
    """
    root = str(tmp_path)
    _session(root, 'F001', 'T1')
    _session(root, 'F002', 'T9')            # only one, unusual task
    subjects = corpus_subjects(raw_root=root, data_set='tir_roi')
    assert subjects == ['F001', 'F002']


def test_universe_takes_no_narrowing_arguments():
    """The enumerator must not grow a tasks/subjects/caps parameter (I2)."""
    params = set(inspect.signature(corpus_subjects).parameters)
    assert params == {'args', 'raw_root', 'data_set'}
    assert 'tasks' not in params and 'subjects' not in params


def test_universe_is_sorted_and_unique(tmp_path):
    root = str(tmp_path)
    for subject in ('M002', 'F010', 'F002'):
        _session(root, subject, 'T1')
        _session(root, subject, 'T2')
    subjects = corpus_subjects(raw_root=root, data_set='tir_roi')
    assert subjects == sorted(set(subjects)) == ['F002', 'F010', 'M002']


def test_usable_predicate_reads_status_for_the_rgb_family():
    assert is_usable_session({'status': 'usable'})
    assert not is_usable_session({'status': 'unusable'})
    assert not is_usable_session({'ir_file': None, 'resp_file': 'x'})
    assert is_usable_session({'ir_file': 'x', 'resp_file': 'y'})


def test_family_without_an_enumerator_is_refused():
    with pytest.raises(NotImplementedError, match='no subject enumerator'):
        corpus_subjects(raw_root='/tmp', data_set='bp4d+')


def test_all_protocol_families_are_advertised():
    assert 'tir_roi' in available_data_sets()
    assert 'rgb_roi' in available_data_sets()
    assert 'tir_roi_resp_rgb2d_bp' in available_data_sets()


def test_resolve_corpus_prefers_the_explicit_root(monkeypatch, tmp_path):
    monkeypatch.setenv('RAW_DATA_PATH', '/from/env')
    root, name = resolve_corpus(None, raw_root=str(tmp_path), data_set='tir_roi')
    assert root == str(tmp_path) and name == 'tir_roi'
    assert resolve_corpus(None, data_set='tir_roi')[0] == '/from/env'
    with pytest.raises(ValueError, match='data_set'):
        resolve_corpus(None, raw_root=str(tmp_path))
