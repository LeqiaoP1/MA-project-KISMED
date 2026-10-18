"""Dataset builders.

Dispatch on ``args.data_set`` / the domain config and return a
``torch.utils.data.Dataset``. This is the thesis-specific part — port your
datasets here following these references:

- **BP4D+ (target of this thesis)**: multimodal RGB + Thermal-IR video with
  synchronised 1D physiological streams (BP / RESP / EDA). Register it, e.g.
  ``@register_dataset('bp4d+')``, and implement the paired (visual, signal,
  mask) collation described in code/notes/plan/ImplementationPlan.md.
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

import os

__all__ = ['build_dataset', 'build_pretraining_dataset', 'register_dataset',
           'list_datasets', 'TIR_ROI_DATA_SETS', 'RGB_ROI_DATA_SETS']

#: ``data_set`` values that select the ADD-ON thermal-ROI + respiration Stage-2
#: dataset (RAW tree) instead of the canonical paired layout.
TIR_ROI_DATA_SETS = ('tir_roi', 'tir_roi_resp', 'tir-roi')

#: ``data_set`` values that select the ADD-ON RGB face-ROI path (RAW tree): the
#: visual stream is the landmark-defined face crop, so Stage 2 and Stage 3 share
#: one front-end. A ``(subject, task)`` whose 2DFeatures frame count differs from
#: the jpeg count is EXCLUDED as ``unusable`` (its 1-D physiology is dropped too).
RGB_ROI_DATA_SETS = ('rgb_roi', 'rgb_roi_resp', 'rgb-roi')

#: name -> builder(is_train, test_mode, args) -> Dataset
_DATASET_BUILDERS: Dict[str, Callable] = {}


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


def build_dataset(is_train: bool, test_mode: bool, args, split_role: Optional[str] = None):
    """Build a supervised (fine-tuning / eval) dataset from ``args.data_set``.

    ``data_set in ('bp4d+', 'paired')`` uses the recorded RGB(jpg-seq)+TIR(wmv)
    layout via :func:`paired_dataset.build_paired_dataset`;
    ``data_set in TIR_ROI_DATA_SETS`` (``tir_roi`` / ``tir_roi_resp``) uses the
    ADD-ON RAW-tree thermal-ROI + respiration path via
    :func:`tir_resp_dataset.build_tir_roi_finetune_dataset` -- the Stage-3
    counterpart of the Stage-2 pretraining view, so a TIR-ROI checkpoint gets
    the SAME ROI crop it was pre-trained on (train/val split by subject);
    any other value must be registered with ``@register_dataset``.
    """
    from .paired_dataset import PAIRED_DATA_SETS, build_paired_dataset

    # An explicit --split_role overrides the role the runner asked for. That is
    # the single-view use case (e.g. `--split_role test` to score a fold), and
    # it is the ONLY meaning of the flag: one process cannot be three roles at
    # once, so the runner still builds its train/val/test views itself.
    explicit_role = str(getattr(args, 'split_role', '') or '').strip().lower()
    if explicit_role:
        split_role = explicit_role

    name = getattr(args, 'data_set', None)
    if name in PAIRED_DATA_SETS:
        if split_role and str(split_role) != ('train' if is_train else 'val'):
            raise NotImplementedError(
                f'data_set={name!r} cannot honour split_role={split_role!r}: '
                f'the canonical paired layout has no subject-discovery hook for '
                f'the k-fold rotation. Use a ROI data_set, or drop --split_role.')
        return build_paired_dataset(is_train, test_mode, args)
    if str(name or '').strip().lower() in TIR_ROI_DATA_SETS:
        from .tir_resp_dataset import build_tir_roi_finetune_dataset
        return build_tir_roi_finetune_dataset(is_train, test_mode, args, split_role=split_role)
    if str(name or '').strip().lower() in RGB_ROI_DATA_SETS:
        from .rgb_roi_dataset import build_rgb_roi_finetune_dataset
        return build_rgb_roi_finetune_dataset(is_train, test_mode, args, split_role=split_role)
    if name not in _DATASET_BUILDERS:
        raise NotImplementedError(
            f'Unknown data_set "{name}". Registered datasets: {list_datasets()}. '
            f'Implement and @register_dataset your dataset builder in '
            f'code/data/datasets.py (see tmp/videomae/datasets.py).')
    return _DATASET_BUILDERS[name](is_train, test_mode, args)


def build_pretraining_dataset(args):
    """Build the masked pre-training dataset (Stage 2, multimodal MAE).

    Two sources, selected by ``args.data_set``:

    * ``tir_roi`` / ``tir_roi_resp`` -> :class:`TirRoiRespPretrainDataset`, the
      ADD-ON thermal-ROI + respiration path over the RAW BP4D tree (the ROI
      crop needs native-resolution frames and the ``IRFeatures`` landmark
      track, neither of which exists in the canonical session layout).
    * anything else (default) -> :class:`PairedPretrainDataset` over the
      canonical ``<data_path>/<session>/`` layout, exactly as before.

    For the paired path: the dataset serves exactly the modalities listed in
    ``args.streams``. Stage-2 CONTRACT: at least TWO modalities -- >=1 video
    (``rgb``/``tir``) AND >=1 physiological 1-D signal (``bp``/``resp``/``eda``,
    the Stage-3 reconstruction target). PRETRAIN-ON-ALL: no split, every session
    is used. Per-stream masks are produced inside the multimodal MAE forward
    pass.
    """
    split = str(getattr(args, 'pretrain_split', 'all') or '').strip().lower()
    if split not in ('all', 'fold'):
        raise ValueError(
            f'Stage-2 pre-training requires pretrain_split="all" or "fold"; '
            f'got {split!r}.')

    name = str(getattr(args, 'data_set', '') or '').strip().lower()

    if split == 'fold':
        # Stage 2 must NOT see the test fold of the fold it is paired with, or
        # the Stage-3 number is contaminated by pretraining. The partition is
        # computed over the FULL usable subject list (never the capped/filtered
        # one): --max_sessions / --max_clips would otherwise shift the folds
        # between the Stage-2 and Stage-3 runs of the same fold.
        from utils.splits import get_stage_splits
        from . import task_groups as tgrp

        n_folds = int(getattr(args, 'n_folds', 0) or 0)
        fold = int(getattr(args, 'fold', 0) or 0)

        if n_folds <= 1:
            print('[data] pretrain_split=fold but n_folds <= 1: the whole '
                  'corpus is used (no fold to exclude).')
        elif name in TIR_ROI_DATA_SETS:
            from .tir_resp_dataset import default_raw_root, discover_sessions
            root = (getattr(args, 'raw_root', None)
                    or getattr(args, 'data_path', None) or default_raw_root())
            sel = tgrp.resolve_task_selection(
                tasks=getattr(args, 'tasks', None),
                task_set=getattr(args, 'task_set', None),
                task_groups=getattr(args, 'task_groups', None))
            usable = [s for s in discover_sessions(root, tasks=sel.tasks)
                      if s['ir_file'] and s['resp_file']]
            subs = sorted({s['subject'] for s in usable})
        elif name in RGB_ROI_DATA_SETS:
            from .rgb_roi_dataset import default_raw_root, discover_rgb_sessions
            root = (getattr(args, 'raw_root', None)
                    or getattr(args, 'data_path', None) or default_raw_root())
            usable = [s for s in discover_rgb_sessions(root, tasks=None,
                                                       attach_landmarks=False)
                      if s['status'] == 'usable']
            subs = sorted({s['subject'] for s in usable})
        else:
            raise NotImplementedError(
                "pretrain_split='fold' is implemented for the ROI paths "
                '(tir_roi / rgb_roi) only; the canonical paired layout has no '
                'subject discovery hook yet.')

        if n_folds > 1:
            if not subs:
                raise RuntimeError(
                    f"pretrain_split='fold': no usable subject under {root}; "
                    f'nothing to partition.')
            splits = get_stage_splits(subs, fold=fold, n_folds=n_folds,
                                      stage=2)
            train_subjects = splits['train']
            # honour an explicit --subjects filter, but never silently widen
            requested = [s.strip() for s in
                         str(getattr(args, 'subjects', '') or '').split(',')
                         if s.strip()]
            if requested:
                train_subjects = [s for s in train_subjects
                                  if s in set(requested)]
                if not train_subjects:
                    raise RuntimeError(
                        f"pretrain_split='fold': --subjects "
                        f'{sorted(requested)} is disjoint from the fold '
                        f'{fold} train pool.')
            args.subjects = train_subjects
            print(f'[data] Stage 2 pretrain_split=fold: fold {fold}/{n_folds} '
                  f'-> {len(train_subjects)} train subject(s), '
                  f'{len(splits["test"])} held out '
                  f'(test fold): {splits["test"]}')
            # Audit artefact (trap: the partition can drift between the Stage-2
            # and Stage-3 runs of the same fold, e.g. when --tasks or
            # --max_sessions change the subject universe). Written next to the
            # checkpoint so a mismatched pair is visible after the fact.
            out_dir = str(getattr(args, 'output_dir', '') or '').strip()
            if out_dir:
                from utils.dist import is_main_process
                from utils.splits import write_fold_manifest
                # RANK-0 ONLY: every rank builds this dataset, but the manifest
                # is an audit artefact of the RUN, not per-rank state. Four
                # ranks truncating+writing one path could interleave into a
                # corrupt JSON -- and it is the only record of the partition
                # (write_fold_manifest itself is atomic, so a reader never sees
                # a half-written file either way).
                if is_main_process():
                    write_fold_manifest(
                        os.path.join(out_dir, 'fold_manifest.json'),
                        fold=fold, n_folds=n_folds, stage=2, roles=splits)
                    print(f'[data] wrote '
                          f'{os.path.join(out_dir, "fold_manifest.json")}')

    if name in TIR_ROI_DATA_SETS:
        from .tir_resp_dataset import build_tir_roi_pretrain_dataset
        return build_tir_roi_pretrain_dataset(args)
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
            'bp/resp/eda); got an empty list.')

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
