"""TIR-ROI -> AU-occurrence probe dataset -- ADD-ON / diagnostic.

**Protocol of record:** ``analysis/tir_resp/AU_OccurenceProbe.md`` (highest
priority for AU-probing). This module implements the *input* half of that
protocol; the model half is ``core/au_probe.py`` (reused verbatim) and the
train/eval half is ``engines/au_probe.py``.

It reuses :class:`data.tir_resp_dataset.TirRoiRespPretrainDataset` **unchanged**
for everything that defines *which pixels reach the encoder* -- the ROI cache,
the ``IRFeatures`` sentinel gate, the static per-clip :func:`roi_box_from_landmarks`
box, and the ``temporal_stride`` decimation -- and only changes the LABEL: each
clip is labelled with the **AU-occurrence vector of its centre frame** instead of
the respiration window.

Contract (see the protocol §4):

* Input = the **thermal ROI crop** only (``streams: ('tir',)``). The respiration
  stream is present only to reuse the geometry; it is **never returned**, so no
  physio leaks into the AU prediction.
* Window = ``clip_duration`` seconds centred on the AU-coded frame ``f_c``;
  ``f_c`` = ``frame_start + clip_frames // 2`` (``clip_frames`` source frames,
  e.g. 200 = 8 s @ 25 fps). The label is the ``AU_OCC`` row at ``f_c`` *after*
  the ``temporal_stride`` decimation, i.e. at the centre of the 8 s span.
* A clip is DROPPED when the centre frame is not AU-coded, its target-AU cells
  contain a ``9`` (unknown), a source frame is a landmark sentinel
  (``missing_frame_mask``, applied by the base class), or the window leaves the
  video range.

The respiraton-quality cleaning knobs of the base class (``min_signal_spread``,
``rail_touch_v``) are deliberately turned OFF: they are label-quality filters for
respiration and irrelevant to an AU target.
"""
import csv
import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .au_dataset import default_au_root, resolve_au_list
from .tir_resp_dataset import (TARGET_LANDMARKS, TirRoiRespPretrainDataset,
                               default_raw_root)

__all__ = [
    'TIER_A_AU_LIST', 'ROI_FORBIDDEN_AUS', 'ROI_BORDERLINE_AUS',
    'classify_au_set', 'check_au_gate', 'au_probe_num_frames',
    'centre_frame_index', 'resolve_labels', 'discover_au_sessions',
    'load_au_table', 'AuTirRoiDataset', 'build_au_tir_roi_datasets',
    'TIR_ROI_DATA_SETS',
]

_MISSING = '9'

#: ``data_set`` values that select this probe path.
TIR_ROI_DATA_SETS = ('tir_roi', 'tir_roi_resp', 'tir-roi')

#: Headline AU set (protocol §3.1): the ROI-valid members of the BP4D-12 subset.
TIER_A_AU_LIST = (10, 12, 14, 15, 17, 23, 24)

#: AU ids whose primary evidence is OUTSIDE the nose+mouth ROI (protocol §3.3):
#: brows (1, 2, 4) and eyelids (5, 7). Measuring them on this encoder is a
#: chance-level artefact.
ROI_FORBIDDEN_AUS = (1, 2, 4, 5, 7)

#: AU ids whose evidence straddles the padded ROI edge (protocol §3.2). Report
#: them, but never in the headline.
ROI_BORDERLINE_AUS = (6, 13)


# --------------------------------------------------------------------------- #
# AU-set gating (protocol §3.1-3.3)
# --------------------------------------------------------------------------- #
def classify_au_set(au_list: Sequence[int]) -> Dict[str, List[int]]:
    """Split an AU list into ``{'forbidden', 'borderline', 'core'}`` (protocol
    §3). ``core`` is the ROI-valid headline-eligible subset."""
    aus = [int(a) for a in au_list]
    return {
        'forbidden': [a for a in aus if a in ROI_FORBIDDEN_AUS],
        'borderline': [a for a in aus if a in ROI_BORDERLINE_AUS],
        'core': [a for a in aus
                 if a not in ROI_FORBIDDEN_AUS and a not in ROI_BORDERLINE_AUS],
    }


def check_au_gate(au_list: Sequence[int], strict: bool = False) -> str:
    """Human-readable warning for a non-ROI-valid AU set (empty string = clean).

    With ``strict=True`` a forbidden AU is a HARD ERROR (the runner exposes this
    as ``--strict_au_gate``); otherwise the message is returned for the caller to
    print as a warning.
    """
    rep = classify_au_set(au_list)
    msgs: List[str] = []
    if rep['forbidden']:
        msgs.append(
            f"AU(s) {rep['forbidden']} are OUTSIDE the nose+mouth ROI (brow / "
            f"eyelid): the encoder has no pixels for them, so their F1 reads as "
            f"chance. Use the Tier-A set {list(TIER_A_AU_LIST)} (protocol §3.3).")
    if rep['borderline']:
        msgs.append(
            f"AU(s) {rep['borderline']} straddle the ROI edge (cheek); report "
            f"them, but keep them out of the headline (protocol §3.2).")
    msg = ' '.join(msgs)
    if msg and strict and rep['forbidden']:
        raise ValueError('[au_probe] ' + msg)
    return msg


# --------------------------------------------------------------------------- #
# geometry helpers (protocol §4)
# --------------------------------------------------------------------------- #
def au_probe_num_frames(clip_duration: float, fps: float, temporal_stride: int,
                        tubelet_t: int) -> int:
    """The encoder's ``num_frames`` for the probe -- identical to
    ``TirRoiRespPretrainDataset`` and ``build_pretraining_model``:
    ``n = round(clip_duration * fps / temporal_stride)`` rounded DOWN to a
    multiple of ``tubelet_t`` (>= ``tubelet_t``)."""
    n = max(1, int(round(float(clip_duration) * float(fps)
                         / max(1, int(temporal_stride)))))
    t = max(1, int(tubelet_t))
    if n % t:
        n -= n % t
    return max(t, n)


def centre_frame_index(clip_frames: int) -> int:
    """Offset (source frames) of the clip's centre within the window."""
    return int(clip_frames) // 2


# --------------------------------------------------------------------------- #
# AU table IO (protocol §4)
# --------------------------------------------------------------------------- #
def discover_au_sessions(au_root: str) -> List[dict]:
    """Every ``<subject>_<task>.csv`` under ``au_root`` (sorted)."""
    if not os.path.isdir(au_root):
        raise FileNotFoundError(f'au_root does not exist: {au_root}')
    out: List[dict] = []
    for path in sorted(glob.glob(os.path.join(au_root, '*.csv'))):
        session = os.path.splitext(os.path.basename(path))[0]
        subject, _, task = session.rpartition('_')
        if not subject:
            subject, task = session, ''
        out.append({'session': session, 'subject': subject, 'task': task,
                    'path': path})
    return out


def load_au_table(path: str, au_list: Sequence[int]) -> dict:
    """Parse one ``AU_OCC`` csv -> ``{'frame_to_label', 'max_frame_1based'}``.

    ``frame_to_label`` maps a **0-based** frame index (``csv_frame - 1``) to the
    ``[len(au_list)]`` float32 occurrence vector. A row whose target cells hold a
    ``9`` (unknown) is skipped -- it is never treated as absent. A target AU that
    is not in the header is a hard error (a silent column miss would corrupt the
    label).
    """
    aus = [int(a) for a in au_list]
    frame_to_label: Dict[int, np.ndarray] = {}
    max_frame_1based = 0
    with open(path, newline='') as fh:
        reader = csv.reader(fh)
        header = next(reader)
        ids = [int(x) for x in header[1:]]
        col = {a: 1 + ids.index(a) for a in aus if a in ids}
        missing = [a for a in aus if a not in col]
        if missing:
            raise ValueError(
                f'{path}: target AU(s) {missing} not in header; header AUs = '
                f'{ids}')
        ncol = max(col.values())
        for row in reader:
            if not row or len(row) <= ncol:
                continue
            try:
                frame = int(row[0])
            except ValueError:
                continue
            if frame < 1:
                continue
            max_frame_1based = max(max_frame_1based, frame)
            vals = [row[col[a]].strip() for a in aus]
            if any(v == _MISSING for v in vals):
                continue
            frame_to_label[frame - 1] = np.asarray(
                [int(v) for v in vals], dtype=np.float32)
    return {'frame_to_label': frame_to_label,
            'max_frame_1based': max_frame_1based, 'au_ids': aus}


def resolve_labels(entries: Sequence[dict], clip_frames: int,
                   au_tables: Dict[str, Optional[dict]]
                   ) -> Tuple[List[dict], List[np.ndarray]]:
    """Drop entries without a usable centre-frame AU label.

    ``au_tables`` maps ``session -> load_au_table(...)`` (or ``None`` for an
    unusable table). Returns the kept entries and their labels, in order.
    """
    kept: List[dict] = []
    labels: List[np.ndarray] = []
    off = centre_frame_index(clip_frames)
    for entry in entries:
        tab = au_tables.get(entry['session'])
        if not tab:
            continue
        lab = tab['frame_to_label'].get(entry['frame_start'] + off)
        if lab is None:
            continue
        kept.append(entry)
        labels.append(lab)
    return kept, labels


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class AuTirRoiDataset(TirRoiRespPretrainDataset):
    """Thermal-ROI clip -> centre-frame AU-occurrence vector.

    Yields ``(tir [3, num_frames, S, S] float32, au_label [len(au_list)] float32)``.
    Only sessions that are BOTH AU-coded (have an ``AU_OCC`` csv) and usable by
    the base TIR-ROI loader (thermal video + ``IRFeatures`` + ``Resp_Volts``) are
    kept; unusable ones are reported in ``.warnings``.
    """

    def __init__(self, au_root: str, au_list: Sequence[int],
                 subjects: Optional[Sequence[str]] = None,
                 tasks: Optional[Sequence[str]] = None,
                 raw_root: Optional[str] = None,
                 clip_duration: float = 8.0, clip_stride: Optional[float] = None,
                 temporal_stride: int = 2, tubelet_t: int = 2,
                 input_size: int = 64, roi_padding: float = 0.2,
                 fps: float = 25.0,
                 target_landmarks: Sequence[int] = TARGET_LANDMARKS,
                 max_clips_per_session: Optional[int] = None,
                 max_entries: Optional[int] = None,
                 roi_cache: Optional[str] = None, roi_cache_lru: int = 2,
                 verbose: bool = False):
        self.au_root = str(au_root)
        self.au_list = [int(a) for a in au_list]
        if not self.au_list:
            raise ValueError('AuTirRoiDataset: au_list is empty')

        au_sessions = discover_au_sessions(self.au_root)
        if not au_sessions:
            raise FileNotFoundError(
                f'AuTirRoiDataset: no AU csv under {self.au_root}')
        if subjects is None:
            subjects = sorted({s['subject'] for s in au_sessions})
        if tasks is None:
            tasks = sorted({s['task'] for s in au_sessions})

        super().__init__(
            raw_root=raw_root, streams=('tir', 'resp'),
            fs=100.0, fps=fps, clip_duration=clip_duration,
            clip_stride=clip_stride, temporal_stride=temporal_stride,
            tubelet_t=tubelet_t, input_size=input_size,
            roi_padding=roi_padding, target_landmarks=target_landmarks,
            min_signal_spread=0.0, rail_touch_v=0.0, physio_norm='none',
            subjects=subjects, tasks=tasks,
            max_clips_per_session=max_clips_per_session,
            max_entries=max_entries, roi_cache=roi_cache,
            roi_cache_lru=roi_cache_lru, verbose=verbose)

        self.n_au = len(self.au_list)
        self.frame_mismatch: List[dict] = []
        self._relabel()

    # ------------------------------------------------------------------ relabel
    def _relabel(self) -> None:
        au_paths = {s['session']: s['path']
                    for s in discover_au_sessions(self.au_root)}
        tables: Dict[str, Optional[dict]] = {}
        self._au_loaded: List[str] = []
        for session, path in au_paths.items():
            if session not in self.base.sessions:
                continue                      # base never built clips for it
            try:
                tables[session] = load_au_table(path, self.au_list)
                self._au_loaded.append(session)
                n_thermal = int(self.base.sessions[session]['n_frames'])
                if int(tables[session]['max_frame_1based']) > n_thermal:
                    self.frame_mismatch.append({
                        'session': session,
                        'au_max_frame': int(tables[session]['max_frame_1based']),
                        'n_thermal': n_thermal})
            except Exception as exc:          # unusable table -> drop the session
                tables[session] = None
                self.base.warnings.append(f'{session}: AU table unusable ({exc})')

        kept, labels = resolve_labels(self.entries, self.base.clip_frames, tables)
        if not kept:
            raise RuntimeError(
                f'AuTirRoiDataset: no AU-labelled clip survived for au_list '
                f'{self.au_list} over {len(self.base.sessions)} usable TIR '
                f'session(s). Check --au_root / --au_list and the centre-frame '
                f'coverage.')
        # ``self.entries`` and ``self.base.entries`` are the SAME list object
        # (see TirRoiRespPretrainDataset.__init__); reassign BOTH so the base
        # index used by ``super().__getitem__`` stays aligned with ``self.labels``.
        self.entries = kept
        self.base.entries = kept
        self.labels = np.stack(labels).astype(np.float32)
        self.subjects = sorted({e['subject'] for e in kept})
        if self.frame_mismatch:
            self.base.warnings.append(
                f'{len(self.frame_mismatch)} session(s) have AU coding beyond '
                f'the thermal frame count (possible frame-index drift); see '
                f'.frame_mismatch (protocol §4 frame-alignment prerequisite)')
        self.stats['au_probe'] = {
            'au_root': self.au_root,
            'au_list': list(self.au_list),
            'sessions_with_au': len(au_paths),
            'sessions_au_usable': len(self._au_loaded),
            'clips_au_labelled': int(self.labels.shape[0]),
            'subjects': self.subjects,
            'frame_mismatch': list(self.frame_mismatch),
            'tier_report': classify_au_set(self.au_list),
        }

    # ------------------------------------------------------------------ item
    def __getitem__(self, index: int):
        if index < 0:
            index += len(self.entries)
        item = super().__getitem__(index)              # {'tir':..., 'resp':...}
        tir = item['tir']
        if not torch.is_tensor(tir):
            tir = torch.from_numpy(np.ascontiguousarray(tir))
        return tir, torch.from_numpy(self.labels[index])

    def describe(self) -> str:
        rep = classify_au_set(self.au_list)
        return (f'AuTirRoiDataset au_list={self.au_list} '
                f'n_au={self.n_au} subjects={self.subjects}\n'
                f'  AUs core={rep["core"]} borderline={rep["borderline"]} '
                f'forbidden={rep["forbidden"]}\n'
                f'  clips {len(self)} (centre-frame labelled)\n'
                + super().describe())


# --------------------------------------------------------------------------- #
# builder (runner-facing)
# --------------------------------------------------------------------------- #
def _parse_int_csv(value) -> Tuple[int, ...]:
    return tuple(int(x) for x in str(value).split(',') if str(x).strip())


def _split_subjects(args, all_subjects: List[str]
                    ) -> Tuple[List[str], List[str], List[str]]:
    """Subject-disjoint ``(train, val, test)`` split (test must be set explicitly
    via ``--test_subjects``; empty otherwise)."""
    def _parse(s: str) -> List[str]:
        return [x.strip() for x in str(s).split(',') if x.strip()]

    train_csv = getattr(args, 'train_subjects', '') or ''
    val_csv = getattr(args, 'val_subjects', '') or ''
    test = _parse(getattr(args, 'test_subjects', '') or '')

    if train_csv.strip() and val_csv.strip():
        train, val = _parse(train_csv), _parse(val_csv)
    else:
        ratio = float(getattr(args, 'train_ratio', 0.8))
        n_train = max(1, min(len(all_subjects) - 1,
                             int(round(len(all_subjects) * ratio))))
        train, rest = all_subjects[:n_train], all_subjects[n_train:]
        val = [s for s in rest if s not in set(test)]

    unknown = [s for s in (train + val + test) if s not in set(all_subjects)]
    if unknown:
        raise ValueError(
            f'--train/val/test_subjects reference unknown subject(s) {unknown}; '
            f'AU-coded subjects = {all_subjects}')
    return train, val, test


def build_au_tir_roi_datasets(args, with_test: bool = False):
    """Build the ``(train, val[, test])`` TIR-ROI AU datasets from ``args``.

    The AU set comes from :func:`data.au_dataset.resolve_au_list` (``--au_list``
    or ``--au_freq_topk``); the split is subject-disjoint, mirroring
    :func:`data.au_dataset.build_au_datasets`.
    """
    au_root = getattr(args, 'au_root', '') or default_au_root()
    au_list = resolve_au_list(args)

    all_subjects = sorted({s['subject'] for s in discover_au_sessions(au_root)})
    if not all_subjects:
        raise FileNotFoundError(f'no AU-coded subject under {au_root}')
    max_sessions = getattr(args, 'max_sessions', None)
    if max_sessions:
        all_subjects = all_subjects[:int(max_sessions)]

    train_subs, val_subs, test_subs = _split_subjects(args, all_subjects)
    if not val_subs:
        raise ValueError(
            f'subject split left the val side empty (subjects={all_subjects}, '
            f'train_ratio={getattr(args, "train_ratio", 0.8)}); lower '
            f'train_ratio or pass --val_subjects.')

    tubelet_t = _parse_int_csv(getattr(args, 'tubelet', '2,16,16'))[0]
    common = dict(
        au_root=au_root, au_list=au_list,
        raw_root=(getattr(args, 'raw_root', '') or None)
        or default_raw_root(),
        clip_duration=float(getattr(args, 'clip_duration', 8.0)),
        fps=float(getattr(args, 'fps', 25.0)),
        temporal_stride=int(getattr(args, 'temporal_stride', 1) or 1),
        tubelet_t=tubelet_t,
        input_size=int(getattr(args, 'input_size', 64)),
        roi_padding=float(getattr(args, 'roi_padding', 0.2)),
        max_clips_per_session=getattr(args, 'max_clips', None),
        max_entries=getattr(args, 'max_entries', None),
        roi_cache=(getattr(args, 'roi_cache', '') or None),
        roi_cache_lru=int(getattr(args, 'roi_cache_lru', 2) or 2),
    )
    train = AuTirRoiDataset(subjects=train_subs, **common)
    val = AuTirRoiDataset(subjects=val_subs, **common)
    if not with_test:
        return train, val
    test = (AuTirRoiDataset(subjects=test_subs, **common)
            if test_subs else None)
    return train, val, test
