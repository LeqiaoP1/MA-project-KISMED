"""Dataset builders.

Dispatch on ``args.data_set`` / the domain config and return a
``torch.utils.data.Dataset``. This is the thesis-specific part — port your
datasets here following these references:

- **BP4D+ (target of this thesis)**: multimodal RGB + Thermal-IR video with
  synchronised 1D physiological streams (BP / RESP). Register it, e.g.
  ``@register_dataset('bp4d+')``, and implement the paired (visual, signal,
  mask) collation described in code/notes/plan/experiments/RGB_BP_plan.md §3.2.
- VideoMAE   ``tmp/videomae/datasets.py`` + ``kinetics.py`` + ``ssv2.py``
            (video; yields ``(samples, labels, index, bool_masked_pos)``)
- MultiMAE  ``tmp/MultiMAE/utils/datasets.py``,
            ``tmp/MultiMAE/utils/dataset_folder.py``,
            ``tmp/MultiMAE/utils/dataset_regression.py``,
            ``tmp/MultiMAE/utils/datasets_semseg.py`` (multi-modal 2D)

Register a new dataset with ``@register_dataset`` and reference it from the
YAML config via ``data_set``.
"""
from typing import Callable, Dict, Optional

import copy
import os

__all__ = ['build_dataset', 'build_pretraining_dataset',
           'build_pretraining_corpus_dataset', 'register_dataset',
           'list_datasets', 'TIR_ROI_DATA_SETS', 'RGB_ROI_DATA_SETS',
           'TIR_ROI_RGB2D_DATA_SETS']

#: ``data_set`` values that select the ADD-ON thermal-ROI + respiration Stage-2
#: dataset (RAW tree) instead of the canonical paired layout.
TIR_ROI_DATA_SETS = ('tir_roi', 'tir_roi_resp', 'tir-roi')

#: ``data_set`` values that select the TRI-MODAL Stage-2 path (TIR-ROI + 2D
#: texture + respiration, all three from offline caches). Stage 2 ONLY: there is
#: no fine-tune view of the 2D stream yet, so ``build_dataset`` rejects it.
TIR_ROI_RGB2D_DATA_SETS = ('tir_roi_resp_rgb2d', 'trimodal')

#: ``data_set`` values that select the ADD-ON RGB face-ROI path (RAW tree): the
#: visual stream is the landmark-defined face crop, so Stage 2 and Stage 3 share
#: one front-end. A ``(subject, task)`` whose 2DFeatures frame count differs from
#: the jpeg count is EXCLUDED as ``unusable`` (its 1-D physiology is dropped too).
RGB_ROI_DATA_SETS = ('rgb_roi', 'rgb_roi_resp', 'rgb-roi')

#: name -> builder(is_train, test_mode, args) -> Dataset
_DATASET_BUILDERS: Dict[str, Callable] = {}


def _as_subject_list(value) -> list:
    """Normalise ``--subjects``: a comma string, a list/tuple, or empty."""
    if not value:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [s.strip() for s in str(value).split(',') if s.strip()]


def register_dataset(name: str):
    """Decorator registering a dataset builder under ``name``."""

    def _decorator(fn: Callable) -> Callable:
        if name in _DATASET_BUILDERS:
            raise ValueError(f'Dataset "{name}" already registered.')
        _DATASET_BUILDERS[name] = fn
        return fn

    return _decorator


def list_datasets() -> list:
    return sorted(_DATASET_BUILDERS)


def build_dataset(args, plan, role: str = 'train'):
    """Build a supervised (fine-tuning / evaluation) dataset for ONE role.

    The role comes from the fold's `protocol.FoldPlan`; nothing here derives a
    split (ADR-001 I2, D3). Three roles exist and all are legal: ``train``,
    ``val`` and ``test`` — Stage 3 never builds ``test`` while training, but the
    dedicated scoring process does (I5).

    ``data_set in TIR_ROI_DATA_SETS`` / ``RGB_ROI_DATA_SETS`` use the ADD-ON
    ROI paths, whose universes `data.universe` can enumerate. The canonical
    ``paired`` layout has no subject enumerator, so it is refused: those
    lineages are off-protocol and must not produce a thesis number.
    """
    from protocol import ProtocolError

    if role not in ('train', 'val', 'test'):
        raise ProtocolError(f'unknown role {role!r}; expected train/val/test.')

    from .paired_dataset import PAIRED_DATA_SETS

    name = str(getattr(args, 'data_set', '') or '').strip().lower()
    if name in PAIRED_DATA_SETS:
        raise NotImplementedError(
            f'data_set={name!r} has no ADR-001 universe (no subject-discovery '
            f'hook), so it cannot provide a role. The canonical paired / bp / '
            f'au lineages are off-protocol until a follow-up adds an '
            f'enumerator; they must not produce a thesis number.')
    if name in TIR_ROI_DATA_SETS:
        from .tir_resp_dataset import build_tir_roi_finetune_dataset
        return build_tir_roi_finetune_dataset(args, plan=plan, role=role)
    if name in RGB_ROI_DATA_SETS:
        from .rgb_roi_dataset import build_rgb_roi_finetune_dataset
        return build_rgb_roi_finetune_dataset(args, plan=plan, role=role)
    raise NotImplementedError(
        f'Unknown data_set "{name}". Registered datasets: {list_datasets()}. '
        f'Each registered builder must accept (args, plan, role) to take part '
        f'in the protocol.')


def build_pretraining_dataset(args, plan, role: str = 'train'):
    """Build the masked pre-training dataset for ONE role of an ADR-001 fold.

    Stage 2 builds ``train`` and ``val`` only. The test block is not "scored
    later" — it is never loaded (I5). Role membership comes from the fold's
    `protocol.FoldPlan`; ``--tasks`` and ``--subjects`` narrow which SESSIONS of
    that role are read and can never move a subject between roles (I2).

    Sources, selected by ``args.data_set``:

    * ``tir_roi`` / ``tir_roi_resp`` -> :class:`TirRoiRespPretrainDataset`, the
      ADD-ON thermal-ROI + respiration path over the RAW BP4D tree (the ROI
      crop needs native-resolution frames and the ``IRFeatures`` landmark
      track, neither of which exists in the canonical session layout).
    * ``tir_roi_resp_rgb2d`` / ``trimodal`` -> the three-stream Stage-2 path.
    * ``rgb_roi`` -> :class:`RGBRoiPretrainDataset`.

    Stage 2 CONTRACT: at least TWO modalities — >=1 video (``rgb``/``tir``) and
    >=1 physiological 1-D signal (``bp``/``resp``, the Stage-3 target).
    Per-stream masks are produced inside the multimodal MAE forward pass.
    """
    from protocol import ProtocolError

    if role not in ('train', 'val'):
        raise ProtocolError(
            f'Stage 2 may only build the train/val roles; got {role!r}. The '
            f'test role is never loaded during training (ADR-001 I5).')

    subjects = list(plan.roles[role])
    requested = _as_subject_list(getattr(args, 'subjects', ''))
    if requested:
        subjects = [s for s in subjects if s in set(requested)]
    if not subjects:
        raise RuntimeError(
            f'Stage 2: the {role!r} role of fold {plan.fold} is empty'
            + (f' after --subjects {sorted(requested)}' if requested else '')
            + ' (partition: '
            + ', '.join(f'{name}={plan.counts[name]}'
                        for name in ('train', 'val', 'test')) + ').')
    print(f'[data] Stage 2 role={role}: {len(subjects)} subject(s) '
          f'(fold {plan.fold}/{plan.n_folds}, digest {plan.digest[:12]})')

    name = str(getattr(args, 'data_set', '') or '').strip().lower()
    if role == 'val' and name not in (TIR_ROI_DATA_SETS + TIR_ROI_RGB2D_DATA_SETS
                                      + RGB_ROI_DATA_SETS):
        raise ProtocolError(
            f'data_set={name!r} has no ADR-001 universe, so it cannot provide '
            f'a validation role. Off-protocol lineages (paired / bp / au) must '
            f'not produce a thesis number until they are migrated.')

    # A SHALLOW COPY, not `args.subjects = ...`: one process builds several
    # roles, and mutating the shared namespace made the second build intersect
    # its role against the first role's subject list (measured: the val role
    # came out empty right after a populated train role).
    view_args = copy.copy(args)
    view_args.subjects = subjects
    return _build_pretrain_source(view_args)


def build_pretraining_corpus_dataset(args):
    """WHOLE Stage-2 corpus, no role — DIAGNOSTICS ONLY.

    Two inspection tools (a reconstruction figure and a zero-shot probe) need
    every clip, including the held-out subjects they then select by name. That
    is not a protocol path: a number or figure produced from this dataset may
    never be pooled or reported, because it can see the test block. Training
    and evaluation go through :func:`build_pretraining_dataset` with a role.
    """
    view_args = copy.copy(args)          # never mutate the shared namespace
    view_args.subjects = ''              # every discoverable subject
    return _build_pretrain_source(view_args)


def _build_pretrain_source(args):
    """Dispatch to the Stage-2 dataset family selected by ``args.data_set``."""
    name = str(getattr(args, 'data_set', '') or '').strip().lower()
    if name in TIR_ROI_DATA_SETS:
        from .tir_resp_dataset import build_tir_roi_pretrain_dataset
        return build_tir_roi_pretrain_dataset(args)
    if name in TIR_ROI_RGB2D_DATA_SETS:
        from .trimodal_dataset import build_tir_roi_rgb2d_pretrain_dataset
        return build_tir_roi_rgb2d_pretrain_dataset(args)
    if name in RGB_ROI_DATA_SETS:
        from .rgb_roi_dataset import build_rgb_roi_pretrain_dataset
        return build_rgb_roi_pretrain_dataset(args)

    from .paired_dataset import PairedPretrainDataset

    streams = tuple(s.strip() for s in
                    str(getattr(args, 'streams', 'rgb,tir,bp')).split(',')
                    if s.strip())
    if not streams:
        raise ValueError(
            'build_pretraining_dataset: --streams must name at least two '
            'modalities (>=1 video rgb/tir and >=1 physiological 1-D '
            'bp/resp); got an empty list.')

    return PairedPretrainDataset(
        data_path=getattr(args, 'data_path', ''),
        fs=getattr(args, 'fs', 100.0),
        fps=getattr(args, 'fps', 25.0),
        clip_duration=getattr(args, 'clip_duration', 4.0),
        clip_stride=getattr(args, 'clip_stride', None) or None,
        temporal_stride=int(getattr(args, 'temporal_stride', 1) or 1),
        seq_len=getattr(args, 'seq_len', None) or None,
        input_size=getattr(args, 'input_size', 64),
        streams=streams,
        max_sessions=getattr(args, 'max_sessions', None),
        max_clips=getattr(args, 'max_clips', None),
        max_entries=getattr(args, 'max_entries', None))
