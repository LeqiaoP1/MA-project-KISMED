"""Four-stream Stage-2 dataset: ``tir`` + ``rgb2d`` + ``bp`` + ``resp``.

Contract
--------
``TirRoiResp2DPretrainDataset`` already guarantees that ``tir``, ``rgb2d`` and
``resp`` share ONE ``frame_start/frame_end`` window (it post-filters the parent's
entries and reads the 2D cache on the same frames). This module adds the fourth
stream, ``bp``, from the 1000 Hz BP artefact on the SAME clip start:

    s = round(t_start * fs_out)
    n = round(kept_seconds * fs_out)

where ``t_start`` is the parent entry's own ``t_start`` and ``kept_seconds`` is
the model's effective window (``seq_len / fs``, after the tubelet-tail trim).
A clip survives only when ``bp_windows.usable(mask, s, n)`` is true, i.e. the
whole window lies in one valid BP segment. The surviving ``bp`` window is
z-scored per clip with the SAME population z-score the parent uses for ``resp``
(``ddof=0``, ``ZSCORE_EPS``), so both physiological targets live in one label
space.

The output of one sample is::

    {'tir':   float32 [3, T, S, S]   thermal ROI, [0, 1]
     'rgb2d': float32 [3, T, S, S]   2D-texture ROI, [0, 1]
     'resp':  float32 [1, L]         per-clip z-scored respiration
     'bp':    float32 [1, L]         per-clip z-scored blood-pressure waveform}

with one shared ``T`` and one shared ``L``. The four tensors are temporally
identical, which is exactly what
``MultiModalMAE._check_space_time_alignment`` requires.

Cache identity
--------------
``bp_artefact`` is the OUT_ROOT (the parent of the ``<CACHE_KEY>`` directory),
never a keyed directory. ``find_cache`` matches the full frozen parameter set
(``fs_in == fs_out``, gate thresholds, ``min_segment_s``, ``store``) and raises
a field-by-field diff on a mismatch. A missing shard WITHOUT a ``skips/`` marker
is an incomplete build and raises; a ``skips/`` marker is a legitimate
exclusion and drops the clip with a counter.
"""
from __future__ import annotations

import collections
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import bp_artefact as bpa                        # noqa: E402
from data import bp_windows as bpw                         # noqa: E402
from data import rgb2d_cache as c2d                        # noqa: E402
from data import rgb_features as rf                        # noqa: E402
from data import tir_resp_dataset as trd                   # noqa: E402
from data import trimodal_dataset as td                    # noqa: E402

__all__ = ['FOURSTREAM_DATA_SETS', 'FOURSTREAM_STREAMS',
           'TirRoiResp2DBPPretrainDataset',
           'build_tir_roi_rgb2d_bp_pretrain_dataset']

#: ``data_set`` aliases this module serves (registration happens in
#: ``data/datasets.py``; the constant lives here so the builder and the
#: dispatcher cannot drift).
FOURSTREAM_DATA_SETS = ('tir_roi_resp_rgb2d_bp',)

#: The exact stream set this class serves.
FOURSTREAM_STREAMS = ('tir', 'rgb2d', 'bp', 'resp')


def _normalise_streams(streams) -> Tuple[str, ...]:
    if isinstance(streams, str):
        return tuple(s.strip() for s in streams.split(',') if s.strip())
    return tuple(str(s).strip() for s in streams if str(s).strip())


def _resolve_bp_params(raw_root: str, bp_fs: float,
                       bp_params: Optional[Dict]) -> Dict:
    """The frozen BP identity this run must match.

    ``bp_params`` is an escape hatch for synthetic tests / explicit artefacts.
    The production path discovers the corpus and builds the SAME parameter set
    the builder writes: no clip length, hop, split or normalisation enters it.
    """
    if bp_params is not None:
        return dict(bp_params)
    sessions = bpa.discover_sessions(raw_root)
    return bpa.build_params(sessions, fs_in=bp_fs, fs_out=bp_fs)


class TirRoiResp2DBPPretrainDataset(td.TirRoiResp2DPretrainDataset):
    """Stage-2 masked pre-training on four streams on one shared window.

    :param bp_artefact: OUT_ROOT of a BP artefact built by
        ``runners/run_build_bp_artefact.py`` (parent of the keyed directories).
    :param bp_fs: sample rate of the artefact. The artefact applies no
        resampling, so it MUST equal the model's ``fs``; ``None`` uses ``fs``.
    :param bp_params: optional frozen BP parameter dict. ``None`` builds it from
        the raw corpus with the frozen defaults.
    :param bp_cache_lru: number of open BP shard payloads kept per worker.
    """

    def __init__(self, raw_root: Optional[str] = None,
                 streams: Sequence[str] = FOURSTREAM_STREAMS,
                 *,
                 bp_artefact: Optional[str] = None,
                 bp_fs: Optional[float] = None,
                 bp_params: Optional[Dict] = None,
                 bp_cache_lru: int = 8,
                 allow_empty: bool = False,
                 **parent_kwargs):
        declared = _normalise_streams(streams)
        unknown = [s for s in declared if s not in FOURSTREAM_STREAMS]
        if unknown:
            raise ValueError(
                f'TirRoiResp2DBPPretrainDataset: unknown stream(s) {unknown}; '
                f'this path serves exactly {FOURSTREAM_STREAMS}.')
        dupes = sorted({s for s in declared if declared.count(s) > 1})
        if dupes:
            raise ValueError(
                f'TirRoiResp2DBPPretrainDataset: duplicate stream(s) '
                f'{dupes}; got {declared}.')
        missing = [s for s in FOURSTREAM_STREAMS if s not in declared]
        if missing:
            raise ValueError(
                f'TirRoiResp2DBPPretrainDataset: Stage 2 needs all four '
                f'streams, missing {missing}; got {declared}.')
        if not str(bp_artefact or '').strip():
            raise ValueError(
                'TirRoiResp2DBPPretrainDataset: bp_artefact is required (the '
                'OUT_ROOT of a BP artefact; build it with '
                '`runners/run_build_bp_artefact.py --init --build`).')

        # The parent has its own three-stream validator; it must never see 'bp'.
        parent_kwargs.pop('streams', None)
        parent_kwargs.pop('allow_empty', None)
        super().__init__(raw_root=raw_root, streams=('tir', 'rgb2d', 'resp'),
                         allow_empty=allow_empty, **parent_kwargs)
        # The parent sets self.streams to its three-stream declaration; restore
        # the four-stream contract for consumers and for __getitem__'s checks.
        self.streams = declared

        self.bp_fs = float(self.fs if bp_fs is None else bp_fs)
        if self.bp_fs <= 0:
            raise ValueError(
                f'bp_fs must be > 0; got {self.bp_fs!r}.')
        if abs(self.bp_fs - float(self.fs)) > 1e-9:
            raise ValueError(
                f'bp_artefact stores the source grid unchanged (no resampling); '
                f'this dataset reads it at fs={self.fs:g} Hz but bp_fs='
                f'{self.bp_fs:g} Hz. Set fs to the artefact rate and rebuild '
                f'the model geometry, or add an explicit resampler.')
        self.bp_clip_seconds = float(self.seq_len) / self.bp_fs
        self.bp_artefact_out_root = str(bp_artefact)
        self._bp_shard_lru = max(1, int(bp_cache_lru))
        self._bp_shards: 'collections.OrderedDict[str, Dict]' = \
            collections.OrderedDict()

        self.bp_params = _resolve_bp_params(self.base.raw_root, self.bp_fs,
                                            bp_params)
        self.bp_cache_root = str(
            bpa.find_cache(self.bp_artefact_out_root, self.bp_params))
        bpa.check_params_compatible(self.bp_cache_root, self.bp_params)
        self.bp_cache_key = os.path.basename(self.bp_cache_root.rstrip(os.sep))
        print(f'[data] bp_artefact: {self.bp_cache_root}')
        print(f'[data]   grid {self.bp_params["fs_in"]} -> '
              f'{self.bp_params["fs_out"]} Hz, range '
              f'{self.bp_params["gate1_min_mmhg"]:g}..'
              f'{self.bp_params["gate1_max_mmhg"]:g} mmHg, '
              f'min segment {self.bp_params["min_segment_s"]:g} s, '
              f'format v{self.bp_params["format_version"]}')

        self._filter_bp_entries(allow_empty)

    # ------------------------------------------------------------------ cache
    def _bp_shard(self, session: str) -> Dict:
        """One cached BP shard, LRU-cached.

        A MISSING shard is an ERROR (an incomplete build); a ``skips/`` entry
        is a legitimate exclusion handled in :meth:`_filter_bp_entries`.
        """
        sh = self._bp_shards.get(session)
        if sh is not None:
            self._bp_shards.move_to_end(session)
            return sh
        path = bpa.shard_path(self.bp_cache_root, session)
        if not path.is_file():
            raise FileNotFoundError(
                f'{session}: no BP shard in {self.bp_cache_root} -- the '
                f'artefact is INCOMPLETE. Run '
                f'`runners/run_build_bp_artefact.py --status` then `--build`.')
        sh = bpa.read_shard(path)
        self._bp_shards[session] = sh
        while len(self._bp_shards) > self._bp_shard_lru:
            self._bp_shards.popitem(last=False)
        return sh

    @staticmethod
    def _bp_skip_reason(root: str, session: str) -> Optional[str]:
        """Reason recorded in ``skips/<session>.json``, or ``None``.

        Mirrors the RESP cache's contract: a skip marker is a legitimate
        exclusion, while a missing shard without one is an incomplete build.
        """
        path = os.path.join(str(root), bpa.SUBDIR_SKIPS, f'{session}.json')
        if not os.path.isfile(path):
            return None
        try:
            with open(path) as fh:
                return str(json.load(fh).get('reason') or 'unknown')
        except Exception:
            return 'unknown'

    # --------------------------------------------------------------- window
    def _bp_window(self, entry: Dict) -> Tuple[int, int]:
        """The BP window ``(s, n)`` of one parent entry.

        The parent entry's ``t_start`` is the SAME time origin used by the
        TIR/RESP window, so the BP window is temporally identical. ``n`` is the
        model's effective length (``seq_len``), not necessarily the nominal
        clip length when the tubelet tail is trimmed.
        """
        s, n = bpw.window_indices(float(entry['t_start']),
                                  self.bp_clip_seconds, self.bp_fs)
        if n != self.seq_len:
            raise RuntimeError(
                f'{entry["session"]}: BP window length {n} != model seq_len '
                f'{self.seq_len} (t_start={entry["t_start"]}, '
                f'clip={self.bp_clip_seconds}, fs={self.bp_fs:g}).')
        return int(s), int(n)

    def bp_clip(self, index: int, normalized: bool = True) -> np.ndarray:
        """The clip's BP window ``[L]`` -- the LABEL (z-scored by default)."""
        if index < 0:
            index += len(self.entries)
        entry = self.entries[index]
        shard = self._bp_shard(str(entry['session']))
        s, n = self._bp_window(entry)
        y = np.asarray(bpw.read_window(shard, s, n), dtype=np.float32)
        return self.base._normalize(y) if normalized else y

    # ----------------------------------------------------------------- filter
    def _filter_bp_entries(self, allow_empty: bool) -> None:
        """Subset the parent entries by BP coverage and the BP validity mask."""
        parent_entries = list(self.entries)
        parent_base_idx = list(self._base_idx)
        n_skipped = n_unusable = 0
        skipped_sessions = set()
        keep_idx: List[int] = []

        for i, entry in enumerate(parent_entries):
            session = str(entry['session'])
            reason = self._bp_skip_reason(self.bp_cache_root, session)
            if reason is not None:
                n_skipped += 1
                skipped_sessions.add(session)
                continue
            shard = self._bp_shard(session)      # missing -> fail loudly
            s, n = self._bp_window(entry)
            if not bpw.usable(shard['mask'], s, n):
                n_unusable += 1
                continue
            keep_idx.append(i)

        # Re-apply the dev caps AFTER the filter, exactly as the rgb2d parent
        # does. A BP drop can still shrink a --max_entries smoke below the
        # requested count; the smoke must assert non-emptiness.
        if self._max_clips:
            per: Dict[str, int] = {}
            capped: List[int] = []
            for i in keep_idx:
                sess = str(parent_entries[i]['session'])
                if per.get(sess, 0) >= int(self._max_clips):
                    continue
                per[sess] = per.get(sess, 0) + 1
                capped.append(i)
            keep_idx = capped
        if self._max_entries:
            keep_idx = keep_idx[:int(self._max_entries)]

        self.entries = [parent_entries[i] for i in keep_idx]
        self._base_idx = [parent_base_idx[i] for i in keep_idx]
        self.stats = dict(self.stats)
        self.stats.update({
            'clips_before_bp_filter': len(parent_entries),
            'clips_after_bp_filter': len(self.entries),
            'clips_dropped_bp_skipped': n_skipped,
            'clips_dropped_bp_unusable': n_unusable,
            'sessions_bp_skipped': len(skipped_sessions),
            'bp_cache': self.bp_cache_root,
            'bp_cache_key': self.bp_cache_key,
            'bp_fs': self.bp_fs,
            'bp_clip_seconds': self.bp_clip_seconds,
        })
        if not self.entries and not allow_empty:
            raise RuntimeError(
                f'TirRoiResp2DBPPretrainDataset: the BP gates removed EVERY '
                f'clip of the {len(parent_entries)} the tri-modal path kept '
                f'(skipped={n_skipped}, unusable={n_unusable}, '
                f'sessions_skipped={len(skipped_sessions)}). Check the BP '
                f'artefact coverage / min_segment_s or the clip geometry.')

    # ------------------------------------------------------------------ item
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index < 0:
            index += len(self.entries)
        out = super().__getitem__(index)          # tir / rgb2d / resp
        bp = self.bp_clip(index)
        if bp.size != self.seq_len:
            raise RuntimeError(
                f'{self.entries[index]["session"]}: BP clip has {bp.size} '
                f'sample(s), geometry needs {self.seq_len}')
        out['bp'] = torch.from_numpy(bp).unsqueeze(0)      # [1, L]
        return out

    def describe(self) -> str:
        s = self.stats
        return (f'TirRoiResp2DBPPretrainDataset streams={list(self.streams)}\n'
                f'  bp artefact {self.bp_cache_root} '
                f'(fs {self.bp_fs:g} Hz, clip {self.bp_clip_seconds:g} s, '
                f'lru {self._bp_shard_lru})\n'
                f'  bp clips {s.get("clips_after_bp_filter")} of '
                f'{s.get("clips_before_bp_filter")} '
                f'(skipped {s.get("clips_dropped_bp_skipped")}, '
                f'unusable {s.get("clips_dropped_bp_unusable")}, '
                f'sessions_skipped {s.get("sessions_bp_skipped")})\n'
                + super().describe())


def build_tir_roi_rgb2d_bp_pretrain_dataset(
        args) -> TirRoiResp2DBPPretrainDataset:
    """Stage-2 builder for ``data_set: tir_roi_resp_rgb2d_bp``.

    Mirrors ``build_tir_roi_rgb2d_pretrain_dataset`` and adds only the BP
    artefact arguments. ``args.bp_artefact`` is required; the runner flag is
    wired in the registration task.
    """
    tubelet = str(getattr(args, 'tubelet', '2,16,16')).split(',')
    resp_cache = (str(getattr(args, 'resp_cache', '') or '').strip() or None)
    resp_fs = (float(getattr(args, 'fs', 100.0)) if resp_cache
               else float(getattr(args, 'resp_fs', trd.DEFAULT_RESP_FS)))
    fs = float(getattr(args, 'fs', 100.0))
    if resp_cache and abs(resp_fs - fs) > 1e-9:
        raise ValueError(
            f'resp_cache: resp_fs IS the artefact grid and must equal fs; '
            f'got resp_fs={resp_fs:g} vs fs={fs:g}')

    rgb2d_cache = (str(getattr(args, 'rgb2d_cache', '') or '').strip() or None)
    rgb2d_landmarks = getattr(args, 'rgb2d_landmarks', 'face')
    landmarks = (td.FACE_LANDMARKS
                 if str(rgb2d_landmarks).strip() in ('', 'face')
                 else rf.resolve_roi_landmarks_2d(rgb2d_landmarks))

    bp_artefact = (str(getattr(args, 'bp_artefact', '') or '').strip() or None)
    if not bp_artefact:
        raise ValueError(
            'data_set=tir_roi_resp_rgb2d_bp requires --bp_artefact (the '
            'OUT_ROOT of a BP artefact built by '
            'runners/run_build_bp_artefact.py).')

    streams = tuple(s.strip() for s in
                    str(getattr(args, 'streams', ','.join(FOURSTREAM_STREAMS)))
                    .split(',') if s.strip())

    ds = TirRoiResp2DBPPretrainDataset(
        raw_root=getattr(args, 'raw_root', None) or trd.default_raw_root(),
        streams=streams,
        fs=fs,
        fps=getattr(args, 'fps', trd.DEFAULT_FPS),
        clip_duration=getattr(args, 'clip_duration', 4.0),
        clip_stride=getattr(args, 'clip_stride', None) or None,
        temporal_stride=int(getattr(args, 'temporal_stride', 1) or 1),
        tubelet_t=int(tubelet[0]),
        input_size=getattr(args, 'input_size', trd.DEFAULT_INPUT_SIZE),
        roi_padding=float(getattr(args, 'roi_padding', trd.DEFAULT_ROI_PADDING)),
        crop_rule=str(getattr(args, 'crop_rule', trd.DEFAULT_CROP_RULE)),
        max_sway_px=float(getattr(args, 'max_sway_px',
                                  trd.DEFAULT_MAX_SWAY_PX)),
        target_landmarks=trd.resolve_roi_landmarks(
            getattr(args, 'roi_landmarks', None)),
        min_signal_spread=float(getattr(args, 'min_signal_spread', 0.0) or 0.0),
        rail_touch_v=float(getattr(args, 'rail_touch_v', 0.0) or 0.0),
        physio_norm=str(getattr(args, 'physio_norm', trd.TARGET_NORM)
                        or trd.TARGET_NORM),
        resp_fs=resp_fs,
        subjects=getattr(args, 'subjects', None),
        tasks=getattr(args, 'tasks', None),
        task_groups=getattr(args, 'task_groups', None),
        task_set=getattr(args, 'task_set', None),
        max_clips_per_session=getattr(args, 'max_clips', None),
        max_entries=getattr(args, 'max_entries', None),
        roi_cache=(str(getattr(args, 'roi_cache', '') or '').strip() or None),
        resp_cache=resp_cache,
        rgb2d_cache=rgb2d_cache,
        rgb2d_cache_lru=int(getattr(args, 'rgb2d_cache_lru', 8) or 8),
        rgb2d_store=str(getattr(args, 'rgb2d_store', None)
                        or c2d.STORE_RESOLUTION),
        rgb2d_landmarks=landmarks,
        rgb2d_min_valid_ratio=float(
            getattr(args, 'rgb2d_min_valid_ratio', 1.0)),
        rgb2d_max_sway_px=float(
            getattr(args, 'rgb2d_max_sway_px', td.DEFAULT_RGB2D_MAX_SWAY_PX)),
        rgb2d_area_min_frac=float(getattr(args, 'rgb2d_area_min_frac', 0.5)),
        bp_artefact=bp_artefact,
        bp_fs=getattr(args, 'bp_fs', None),
        bp_params=getattr(args, 'bp_params', None),
        bp_cache_lru=int(getattr(args, 'bp_cache_lru', 8) or 8),
        allow_empty=bool(getattr(args, 'allow_empty', False)),
    )
    trd._log_corpus_cleaning(ds.stats, 'stage2')
    return ds
