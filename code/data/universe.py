"""The corpus subject universe — the single definition ADR-001 partitions.

Invariant I2 requires the partition to be a pure function of the *usable subject
universe*, the seeds, the stratifier and ``K`` — and explicitly NOT of caps,
task selection or filter thresholds. So the universe is computed here, once, from
the raw tree only, and every other module consumes it.

Two rules make that true:

* this module takes **no** ``subjects``/``tasks``/``max_sessions``/cleaning
  arguments: those narrow what a loader READS, never who is held out;
* availability is one named predicate (:func:`is_usable_session`), not an
  expression repeated at three call sites.

Consequence worth knowing: a subject that owns no usable clip after the
rail/flatline rules is still in the universe and can still be a fold's test
subject. The dataset then returns an EMPTY view for it and the run records
``skipped: true``; it does not silently disappear from the partition.
"""
from typing import Any, Dict, List, Optional, Sequence

import os

__all__ = ['is_usable_session', 'corpus_sessions', 'corpus_subjects',
           'resolve_corpus', 'available_data_sets']

# Dataset families are imported lazily: ``data.tir_resp_dataset`` pulls in torch,
# and the partition utility must stay importable on a CPU-only login node.
# BP availability is a CLIP-level filter in the four-stream dataset, not a
# universe predicate: a subject without a usable BP segment still belongs to
# the partition and is scored as skipped inside a role (I2).
_TIR_FAMILIES = ('tir_roi', 'tir_roi_resp', 'tir-roi',
                 'tir_roi_resp_rgb2d', 'trimodal',
                 'tir_roi_resp_rgb2d_bp')
_RGB_FAMILIES = ('rgb_roi', 'rgb_roi_resp', 'rgb-roi')


def available_data_sets() -> Sequence[str]:
    """``data_set`` values this module can enumerate a universe for."""
    return tuple(_TIR_FAMILIES) + tuple(_RGB_FAMILIES)


def is_usable_session(session: Dict[str, Any]) -> bool:
    """Does this session contribute to the universe? (ADR-001 availability)

    * TIR-ROI families: the thermal video (discovery's own key),
      ``IRFeatures/<S>_<T>.txt`` and ``Physiology/<S>/<T>/Resp_Volts.txt``;
    * RGB-ROI family: discovery's ``status == 'usable'`` (which already covers
      the jpeg/2D-feature frame-count mismatch).

    Deliberately NOT part of the predicate: rail/flatline cleaning, clip length,
    task selection, any cap. Those filter CLIPS inside a role; letting them move
    a subject between roles would make the partition depend on them (I2).
    """
    if 'status' in session:
        return session['status'] == 'usable'
    return bool(session.get('ir_file')) and bool(session.get('resp_file'))


def _discover(raw_root: str, data_set: str) -> List[Dict[str, Any]]:
    name = str(data_set or '').strip().lower()
    if name in _TIR_FAMILIES:
        from .tir_resp_dataset import discover_sessions
        return discover_sessions(raw_root)          # no tasks, no subjects
    if name in _RGB_FAMILIES:
        from .rgb_roi_dataset import discover_rgb_sessions
        return discover_rgb_sessions(raw_root, tasks=None, attach_landmarks=False)
    raise NotImplementedError(
        f'data_set={data_set!r} has no subject enumerator, so ADR-001 cannot '
        f'partition it. Supported: {available_data_sets()}. The canonical '
        f'"paired" / bp / au lineages are off-protocol until a follow-up adds '
        f'them; they must not produce a thesis number.')


def corpus_sessions(raw_root: str, data_set: str) -> List[Dict[str, Any]]:
    """Every usable ``(subject, task)`` session of the corpus, in a stable order."""
    sessions = [s for s in _discover(raw_root, data_set) if is_usable_session(s)]
    return sorted(sessions, key=lambda s: (s['subject'], s['session']))


def resolve_corpus(args: Optional[Any] = None, *, raw_root: Optional[str] = None,
                   data_set: Optional[str] = None):
    """Resolve ``(raw_root, data_set)`` from an argparse namespace and/or keywords."""
    if args is not None:
        data_set = data_set or getattr(args, 'data_set', '')
        raw_root = (raw_root or getattr(args, 'raw_root', None)
                    or getattr(args, 'data_path', None))
    if not data_set:
        raise ValueError(
            'a data_set is required (--data_set or args.data_set).')
    if not raw_root:
        raw_root = (os.environ.get('RAW_DATA_PATH')
                    or os.path.join(os.path.dirname(os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__)))),
                        'data', 'raw', 'BP4D'))
    return str(raw_root), str(data_set)


def corpus_subjects(args: Optional[Any] = None, *, raw_root: Optional[str] = None,
                    data_set: Optional[str] = None) -> List[str]:
    """Sorted subject ids of the corpus — the input to ``protocol.build_plan``.

    :param args: an argparse namespace carrying ``data_set`` and
        ``raw_root``/``data_path``. Pass explicit keywords instead when there is
        no namespace (tests, the partition utility).
    :returns: sorted unique subject ids. Ordering and deduplication are what make
        the plan a pure function of the corpus (I2).
    """
    root, name = resolve_corpus(args, raw_root=raw_root, data_set=data_set)
    sessions = corpus_sessions(root, name)
    subjects = sorted({str(s['subject']) for s in sessions})
    if not subjects:
        raise RuntimeError(
            f'no usable subject under {root!r} for data_set={name!r}; '
            f'nothing to partition.')
    return subjects
