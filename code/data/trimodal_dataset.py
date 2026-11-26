"""Tri-modal Stage-2 dataset: ``tir`` + ``rgb2d`` + ``resp`` on ONE window.

Contract: ``notes/design/ADR-012-rgb2d-native-memmap-roi-cache.md`` (v2).
Build order: ``notes/plan/features/ADR-012-rgb2d-native-memmap-roi-cache-plan.md`` (Phase 3).

Why a SUBCLASS: ``BP4DPlusTIRRespDataset._build_entries`` is the authority for
clip enumeration, the sentinel/RESP gates and the motion statistic, and must stay
untouched for the existing lineage. This class passes ``streams=('tir','resp')``
to :class:`TirRoiRespPretrainDataset` (so nothing in that path changes), attaches
the ``rgb2d`` cache itself, and then **post-filters** the parent's entries.

Filtering a SUPERSET is a subset operation: every surviving entry keeps the
parent's exact ``frame_start``/``frame_end``, which is what makes the three
streams temporally identical and therefore lets
``MultiModalMAE._check_space_time_alignment`` pass unmodified.

Every ``rgb2d`` gate is a LOAD-time selection rule with its own counter --
``clips_dropped_2d_short`` / ``_invalid`` / ``_envelope`` / ``_sway`` -- and none
of them is a cache-key field, so changing one costs no rebuild.
"""
import collections
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import rgb2d_cache as c2d                            # noqa: E402
from data import rgb_features as rf                            # noqa: E402
from data import rgb_roi_dataset as rrd                        # noqa: E402
from data import tir_resp_dataset as trd                       # noqa: E402

#: Aliases accepted by ``data/datasets.py`` for this Stage-2-only path.
TIR_ROI_RGB2D_DATA_SETS = ('tir_roi_resp_rgb2d', 'trimodal')

#: The full 49-point face set: the box spans brow -> chin, i.e. the anatomy the
#: design doc's square crop is defined on.
FACE_LANDMARKS = tuple(range(1, rrd.NUM_LANDMARKS_2D + 1))

#: Max head excursion INSIDE a clip window for the 2D stream, in source pixels
#: (``--rgb2d_max_sway_px``); ``0`` disables the gate. Same statistic as the TIR
#: gate (``tir_resp_dataset.landmark_sway``), converted to this camera's pixels:
#: RGB reports 3.2x the TIR excursion for the same head motion, so the TIR
#: module's 40 px corresponds to ~110 px here (retention-matched; pure scale
#: matching gives 128 px). Measured on 1375 sessions / 7005 paired windows --
#: ``code/notes/eval/rgb2d/tir_rgb2d_sway_ratio.py``. Do NOT copy one number to
#: the other modality: 40 px here would reject a third of the 2D clips.
DEFAULT_RGB2D_MAX_SWAY_PX = 110.0


class TirRoiResp2DPretrainDataset(trd.TirRoiRespPretrainDataset):
    """Stage-2 masked pretraining on ``tir`` + ``rgb2d`` + ``resp``.

    ``__getitem__`` returns::

        {'tir':   float32 [3, T, s, s]   thermal ROI crop, [0, 1]
         'rgb2d': float32 [3, T, s, s]   2D-texture ROI crop, [0, 1]
         'resp':  float32 [1, L]         per-clip z-scored respiration}

    with ``s = input_size`` (96 in the shipped config) shared by both visual
    streams, because the model has ONE patch grid.
    """

    def __init__(self, raw_root: Optional[str] = None,
                 streams: Sequence[str] = ('tir', 'rgb2d', 'resp'),
                 rgb2d_cache: Optional[str] = None,
                 rgb2d_cache_lru: int = 8,
                 rgb2d_store: str = c2d.STORE_RESOLUTION,
                 rgb2d_landmarks: Sequence[int] = FACE_LANDMARKS,
                 rgb2d_min_valid_ratio: float = 1.0,
                 rgb2d_max_sway_px: float = DEFAULT_RGB2D_MAX_SWAY_PX,
                 rgb2d_area_min_frac: float = 0.5,
                 allow_empty: bool = False,
                 **parent_kwargs):
        self._declared_streams = tuple(
            str(s).strip() for s in streams if str(s).strip())
        unknown = [s for s in self._declared_streams
                   if s not in ('tir', 'rgb2d', 'resp')]
        if unknown:
            raise ValueError(
                f'TirRoiResp2DPretrainDataset: unknown stream(s) {unknown}; '
                f'this path serves exactly (tir, rgb2d, resp).')
        missing = [s for s in ('tir', 'rgb2d', 'resp')
                   if s not in self._declared_streams]
        if missing:
            raise ValueError(
                f'TirRoiResp2DPretrainDataset: Stage 2 needs all three streams, '
                f'missing {missing}; got {self._declared_streams}.')
        if not rgb2d_cache:
            raise ValueError(
                'TirRoiResp2DPretrainDataset: rgb2d_cache is required (build it '
                'with `runners/run_build_2d_roi_cache.py --init --build`).')

        # ---- the parent builds the SHARED window (tir + resp) --------------
        self._max_entries = parent_kwargs.get('max_entries')
        self._max_clips = parent_kwargs.get('max_clips_per_session')
        super().__init__(raw_root=raw_root, streams=('tir', 'resp'),
                         allow_empty=allow_empty, **parent_kwargs)
        # Declared streams describe the OUTPUT; the parent's validator only ever
        # sees ('tir','resp'), which is exactly the point of the split.
        self.streams = self._declared_streams

        self.rgb2d_landmarks = tuple(int(v) for v in rgb2d_landmarks)
        self.rgb2d_target_idx = np.asarray(
            c2d.landmark_indices_2d(self.rgb2d_landmarks), dtype=np.int64)
        self.rgb2d_min_valid_ratio = float(rgb2d_min_valid_ratio)
        if not 0.0 < self.rgb2d_min_valid_ratio <= 1.0:
            raise ValueError(
                f'rgb2d_min_valid_ratio must be in (0, 1]; got '
                f'{rgb2d_min_valid_ratio!r}')
        self.rgb2d_max_sway_px = float(rgb2d_max_sway_px)
        if self.rgb2d_max_sway_px < 0.0:
            raise ValueError('rgb2d_max_sway_px must be >= 0 (0 = gate off)')
        self.rgb2d_area_min_frac = float(rgb2d_area_min_frac)
        if not 0.0 <= self.rgb2d_area_min_frac <= 1.0:
            raise ValueError('rgb2d_area_min_frac must be in [0, 1]')

        # ---- attach the 2D cache ------------------------------------------
        # Like --roi_cache/--resp_cache, the flag takes the OUT ROOT (parent of
        # the <CACHE_KEY> directory); find_cache resolves the directory.
        self.rgb2d_out_root = str(rgb2d_cache)
        # Which STORE to attach. 'native' keeps the historical artefact;
        # 112/'px112' selects a cache built resized at BUILD time (42x less I/O
        # per clip, at the price of input_size <= N). It is a key field, so a
        # wrong value fails loudly in find_cache instead of silently reading a
        # different resolution.
        self.rgb2d_store = c2d.parse_store_resolution(rgb2d_store)
        self._rgb2d_expected = c2d.loader_params(
            self.base.raw_root, c2d.discover_sessions(self.base.raw_root),
            self.rgb2d_landmarks, self.base.roi_padding,
            crop_rule=self.base.crop_rule,
            clip_seconds=self.base.clip_seconds, fps=self.base.fps,
            store_resolution=self.rgb2d_store)
        _root = c2d.find_cache(self.rgb2d_out_root, self._rgb2d_expected)
        c2d.check_params_compatible(_root, c2d.read_params(_root))
        self.rgb2d_cache_root = str(_root)
        self._rgb2d_params: Dict = c2d.read_params(_root)
        self._rgb2d_lru = max(1, int(rgb2d_cache_lru))
        self._rgb2d_handles: 'collections.OrderedDict[str, c2d.ShardHandle]' = \
            collections.OrderedDict()
        self._rgb2d_tracks: Dict[str, np.ndarray] = {}

        # ---- post-filter the parent's entries -----------------------------
        self._filter_entries(allow_empty)

    # ------------------------------------------------------------------ cache
    def _handle(self, session: str) -> 'c2d.ShardHandle':
        h = self._rgb2d_handles.get(session)
        if h is not None:
            self._rgb2d_handles.move_to_end(session)
            return h
        h = c2d.open_shard(self.rgb2d_cache_root, session)
        self._rgb2d_handles[session] = h
        while len(self._rgb2d_handles) > self._rgb2d_lru:
            _old, evicted = self._rgb2d_handles.popitem(last=False)
            try:
                evicted.close()
            finally:
                pass
        return h

    def _track(self, session: str) -> np.ndarray:
        """The stored 49-point track as a plain array (``[T, 49, 2]`` float32).

        Copied out of the mapping on first use (~0.6 MB per 1500-frame session)
        because every gate below needs it, none of them needs a pixel, and a
        copy cannot outlive a closed handle.
        """
        t = self._rgb2d_tracks.get(session)
        if t is None:
            t = np.array(self._handle(session).landmarks, dtype=np.float32)
            self._rgb2d_tracks[session] = t
        return t

    # ----------------------------------------------------------------- filter
    def _filter_entries(self, allow_empty: bool) -> None:
        """Subset the parent's entries by the 2D cache's coverage + three gates."""
        pad = float(self.base.roi_padding)
        rule = str(self.base.crop_rule)
        n_short = n_invalid = n_envelope = n_sway = 0
        n_excluded = 0
        n_missing = 0
        missing: List[str] = []
        keep_idx: List[int] = []
        for i, entry in enumerate(self.base.entries):
            session = str(entry['session'])
            # --- cache coverage: a `skipped/` marker is a legitimate exclusion
            #     (a data fact the builder decided). A session with NO shard and
            #     no marker means the cache does not cover it YET -- typically a
            #     subject whose archive was still being copied when the cache
            #     was built. Excluded and counted, never fatal: one missing
            #     subject must not abort a 1400-session run, and the next build
            #     picks it up once its archive is complete. Corruption
            #     (truncated shard, params mismatch) still raises.
            skip_reason = self.base._cache_skip_reason(
                self.rgb2d_cache_root, session)
            if skip_reason is not None:
                n_excluded += 1
                continue
            try:
                h = self._handle(session)
            except FileNotFoundError:
                n_missing += 1
                if len(missing) < 10:
                    missing.append(session)
                continue
            f0, f1 = int(entry['frame_start']), int(entry['frame_end'])
            n_2d = int(h.n_frames)
            if f1 > n_2d:
                n_short += 1
                continue
            track = self._track(session)
            # --- 1. tracking validity ---------------------------------------
            bad = c2d.valid_mask_2d(track[f0:f1], self.rgb2d_landmarks)
            if 1.0 - float(bad.mean()) < self.rgb2d_min_valid_ratio:
                n_invalid += 1
                continue
            # --- 2. clip-level envelope (design §7) -------------------------
            try:
                box = c2d.square_span_2d(track[f0:f1], pad, rule)
            except ValueError:
                n_invalid += 1
                continue
            src_h, src_w = h.src_hw
            if c2d.in_frame_share(box, src_w, src_h) < self.rgb2d_area_min_frac:
                n_envelope += 1
                continue
            # --- 3. motion ---------------------------------------------------
            if self.rgb2d_max_sway_px > 0.0:
                sway = trd.landmark_sway(track[f0:f1], self.rgb2d_target_idx)
                if sway > self.rgb2d_max_sway_px:
                    n_sway += 1
                    continue
            keep_idx.append(i)

        self._base_idx = keep_idx
        parent_entries = self.base.entries
        # Dev caps are re-applied AFTER the filter: the parent applied them to a
        # superset, so a --max_entries smoke would otherwise shrink silently.
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
            self._base_idx = keep_idx
        if self._max_entries:
            keep_idx = keep_idx[:int(self._max_entries)]
            self._base_idx = keep_idx
        self.entries = [parent_entries[i] for i in keep_idx]
        self.stats = dict(self.base.stats)
        self.stats.update({
            'clips_dropped_2d_short': n_short,
            'clips_dropped_2d_invalid': n_invalid,
            'clips_dropped_2d_envelope': n_envelope,
            'clips_dropped_2d_sway': n_sway,
            'sessions_2d_excluded': n_excluded,
            'sessions_2d_missing': n_missing,
            'clips_before_2d_filter': len(parent_entries),
            'clips_after_2d_filter': len(self.entries),
        })
        if n_missing:
            more = '' if n_missing <= len(missing) else \
                f' (+{n_missing - len(missing)} more)'
            print(f'[data] rgb2d: {n_missing} session(s) not covered by the '
                  f'cache, excluded: {", ".join(missing)}{more}')
        if not self.entries and not allow_empty:
            raise RuntimeError(
                f'TirRoiResp2DPretrainDataset: the rgb2d gates removed EVERY '
                f'clip of the {len(parent_entries)} the TIR/RESP path kept '
                f'(short={n_short} invalid={n_invalid} envelope={n_envelope} '
                f'sway={n_sway}, sessions_excluded={n_excluded}, '
                f'missing={n_missing}). Relax the gates, rebuild the cache or '
                f'check its coverage.')

    # ------------------------------------------------------------------ item
    def _rgb2d_clip(self, base_index: int) -> torch.Tensor:
        """``[3, T', s, s]`` float32 in [0, 1] for one base entry."""
        entry = self.base.entries[base_index]
        session = str(entry['session'])
        h = self._handle(session)
        track = self._track(session)
        f0, f1 = int(entry['frame_start']), int(entry['frame_end'])

        hx0, hx1, hy0, hy1 = h.box
        bx0, bx1, by0, by1 = c2d.square_span_2d(
            track[f0:f1], float(self.base.roi_padding), str(self.base.crop_rule))
        # Coordinate frame (binding): the stored index IS
        # ``(source - hull origin) * store_scale``, and the black bars are
        # already part of the array, so `pad` is NOT added. Containment (I2) is
        # checked in SOURCE px -- that is the invariant the build guarantees --
        # and the crop itself is taken in STORED px with ONE side length, so a
        # resized store still applies exactly one isotropic scale (I6).
        sx0, sx1 = bx0 - hx0, bx1 - hx0
        sy0, sy1 = by0 - hy0, by1 - hy0
        if min(sx0, sy0) < 0 or sx1 > hx1 - hx0 or sy1 > hy1 - hy0:
            raise IndexError(
                f'{session}: clip box {(bx0, bx1, by0, by1)} is not inside the '
                f'stored hull {(hx0, hx1, hy0, hy1)} (I2) -- the cache was built '
                f'for a different clip length or crop rule. Refusing to clamp.')
        if (bx1 - bx0) != (by1 - by0):
            raise ValueError(
                f'{session}: crop_rule={self.base.crop_rule!r} but the box is '
                f'{bx1 - bx0}x{by1 - by0}; a non-square box would apply a '
                f'per-axis scale (I6).')

        scale = float(h.store_scale)
        stored_h, stored_w = int(h.frames.shape[1]), int(h.frames.shape[2])
        side = int(math.ceil((bx1 - bx0) * scale))
        side = max(1, min(side, stored_h, stored_w))
        cx = (sx0 + sx1) / 2.0 * scale
        cy = (sy0 + sy1) / 2.0 * scale
        x0 = min(max(0, int(math.floor(cx - side / 2.0))), stored_w - side)
        y0 = min(max(0, int(math.floor(cy - side / 2.0))), stored_h - side)
        x1, y1 = x0 + side, y0 + side

        cv2 = rrd._cv2()
        s = int(self.input_size)
        interp = cv2.INTER_AREA if s <= min(x1 - x0, y1 - y0) else cv2.INTER_LINEAR
        out = np.empty((f1 - f0, s, s, 3), np.uint8)
        for i in range(f1 - f0):
            out[i] = cv2.resize(
                np.ascontiguousarray(h.frames[f0 + i, y0:y1, x0:x1]),
                (s, s), interpolation=interp)
        t = torch.from_numpy(out.transpose(3, 0, 1, 2)).to(torch.float32)
        t = t.div_(255.0)
        if self.temporal_stride > 1:
            t = t[:, ::self.temporal_stride]
        t = t[:, :self.num_frames]
        if t.shape[1] != self.num_frames:
            raise RuntimeError(
                f'{session}: rgb2d clip has {t.shape[1]} frames, geometry needs '
                f'{self.num_frames}')
        return t.contiguous()

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index < 0:
            index += len(self.entries)
        base_index = self._base_idx[index]
        out = super().__getitem__(base_index)      # {'tir', 'resp'}
        out['rgb2d'] = self._rgb2d_clip(base_index)
        return out

    def __len__(self) -> int:
        return len(self.entries)

    # --------------------------------------------------------------- describe
    def describe(self) -> str:
        s = self.stats
        cached = sum(1 for _ in self._rgb2d_handles)
        return (f'TirRoiResp2DPretrainDataset streams={list(self.streams)}\n'
                f'  rgb2d cache {self.rgb2d_cache_root} '
                f'(handles open {cached}, lru {self._rgb2d_lru})\n'
                f'  rgb2d gates: min_valid_ratio={self.rgb2d_min_valid_ratio:g}, '
                f'area_min_frac={self.rgb2d_area_min_frac:g}, '
                f'max_sway_px={self.rgb2d_max_sway_px:g}\n'
                f'  clips {s.get("clips_after_2d_filter")} of '
                f'{s.get("clips_before_2d_filter")} '
                f'(short {s.get("clips_dropped_2d_short")}, '
                f'invalid {s.get("clips_dropped_2d_invalid")}, '
                f'envelope {s.get("clips_dropped_2d_envelope")}, '
                f'sway {s.get("clips_dropped_2d_sway")}, '
                f'sessions_excluded {s.get("sessions_2d_excluded")})\n'
                + super().describe())


def build_tir_roi_rgb2d_pretrain_dataset(args) -> TirRoiResp2DPretrainDataset:
    """Stage-2 builder for ``data_set: tir_roi_resp_rgb2d`` (see schemas).

    A SEPARATE builder is required: ``build_tir_roi_pretrain_dataset`` forwards
    ``args.streams`` verbatim into ``TirRoiRespPretrainDataset``, which raises on
    anything outside ``('tir','resp')``. Here the declared streams are the
    three-stream set and the parent only ever sees the thermal pair.
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
    landmarks = (FACE_LANDMARKS if str(rgb2d_landmarks).strip() in ('', 'face')
                 else rf.resolve_roi_landmarks_2d(rgb2d_landmarks))
    ds = TirRoiResp2DPretrainDataset(
        raw_root=getattr(args, 'raw_root', None) or trd.default_raw_root(),
        streams=tuple(s.strip() for s in
                      str(getattr(args, 'streams', 'tir,rgb2d,resp')).split(',')
                      if s.strip()),
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
            getattr(args, 'rgb2d_max_sway_px', DEFAULT_RGB2D_MAX_SWAY_PX)),
        rgb2d_area_min_frac=float(getattr(args, 'rgb2d_area_min_frac', 0.5)),
    )
    trd._log_corpus_cleaning(ds.stats, 'stage2')
    return ds
