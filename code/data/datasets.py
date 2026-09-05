"""Dataset builders.

Dispatch on ``args.data_set`` / the domain config and return a
``torch.utils.data.Dataset``. This is the thesis-specific part — port your
datasets here following these references:

- **BP4D+ (target of this thesis)**: multimodal RGB + Thermal-IR video with
  synchronised 1D physiological streams (BVP / RESP / EDA). Register it, e.g.
  ``@register_dataset('bp4d+')``, and implement the paired (visual, signal,
  mask) collation described in docs/ImplementationPlan.md.
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

__all__ = ['build_dataset', 'build_pretraining_dataset', 'register_dataset',
           'list_datasets']

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


def build_dataset(is_train: bool, test_mode: bool, args):
    """Build a supervised (fine-tuning / eval) dataset from ``args.data_set``.

    ``data_set in ('bp4d+', 'paired')`` uses the recorded RGB(jpg-seq)+TIR(wmv)
    layout via :func:`paired_dataset.build_paired_dataset`; any other value
    must be registered with ``@register_dataset``.
    """
    from .paired_dataset import PAIRED_DATA_SETS, build_paired_dataset

    name = getattr(args, 'data_set', None)
    if name in PAIRED_DATA_SETS:
        return build_paired_dataset(is_train, test_mode, args)
    if name not in _DATASET_BUILDERS:
        raise NotImplementedError(
            f'Unknown data_set "{name}". Registered datasets: {list_datasets()}. '
            f'Implement and @register_dataset your dataset builder in '
            f'code/data/datasets.py (see tmp/videomae/datasets.py).')
    return _DATASET_BUILDERS[name](is_train, test_mode, args)


def build_pretraining_dataset(args):
    """Build the masked pre-training dataset (Stage 2, first local milestone).

    Streams returned per sample: ``{'rgb','tir','bvp'}`` (dict of aligned
    per-stream tensors). PRETRAIN-ON-ALL: no split, every session is used.
    Per-stream masks are produced inside the multimodal MAE forward pass.
    """
    from .paired_dataset import PairedPretrainDataset

    return PairedPretrainDataset(
        data_path=getattr(args, 'data_path', ''),
        fs=getattr(args, 'fs', 100.0),
        fps=getattr(args, 'fps', 25.0),
        clip_duration=getattr(args, 'clip_duration', 4.0),
        seq_len=getattr(args, 'seq_len', None) or None,
        input_size=getattr(args, 'input_size', 64),
        max_sessions=getattr(args, 'max_sessions', None),
        max_entries=getattr(args, 'max_entries', None))
