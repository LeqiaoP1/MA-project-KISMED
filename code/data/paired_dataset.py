"""Paired-session dataset for the recorded RGB(jpg-seq) + TIR(.wmv) layout.

Expected per-session layout under ``data_path``::

    <data_path>/<session>/
        rgb/            # ordered jpg frames  (25 fps nominal)
        tir.wmv         # single video ~60 s  (25 fps nominal)
        signals.csv     # header: [time,] bp, resp, eda  at fs Hz

``PairedSessionDataset`` returns ``(samples, target)`` per clip:
  * ``samples`` : float tensor [3+C, T, H, W]  (RGB 3ch + TIR ``C``ch, temporal
    stack; ``C = tir_channels``, default 3 -> 6 channels, because the TIR
    stream is a false-colour rendering -- see ``video_io`` and ``tir_channels``)
  * ``target``  : float tensor [seq_len]       (chosen waveform: BP, RESP or EDA)

Both videos are read on a single common time grid (see ``data/alignment.py``),
handling any RGB/TIR fps mismatch; 1D signals are sliced on the same axis and
resampled to ``seq_len``. Splits are per *session* (never per frame/clip) to
avoid subject leakage.

Dev/quick-run caps are applied in order: ``max_sessions`` bounds the decode
budget up front, ``max_clips`` bounds the windows taken from *each* session,
and ``max_entries`` bounds the global total (``self.entries``).

NOTE: the spatio-temporal video *model* front-end is the remaining port (see
code/README.md Stage 1) -- this module only produces the synchronised inputs.
"""
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from . import alignment as align
from . import video_io as vio

__all__ = ['PairedSessionDataset', 'build_paired_dataset', 'scan_sessions']

PAIRED_DATA_SETS = ('bp4d+', 'paired')

_CSV_DELIM = ','
_SIGNAL_ALIASES = {'bp': ('bp', 'ppg', 'pulse'),
                   'resp': ('resp', 'respiration'),
                   'eda': ('eda', 'gsr', 'scr', 'electrodermal')}

#: visual stream names the Stage-2 dataset can serve (RGB 3ch, TIR 1ch)
_PRETRAIN_VISUAL_STREAMS = ('rgb', 'tir')


def _subject_of(session: str) -> str:
    """``'F001_T1' -> 'F001'`` (the BP4D session prefix is the subject id).

    Deliberately duplicated from ``data.au_dataset``: importing that module
    drags pandas and the AU-coding tables into the Stage-3 path for one
    string operation.
    """
    return str(session).rsplit('_', 1)[0]


def _first_col(header: List[str], aliases: Tuple[str, ...]) -> Optional[str]:
    low = [h.strip().lower() for h in header]
    for alias in aliases:
        if alias in low:
            return header[low.index(alias)]
    return None


def _canonical_signals(cols: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Map raw csv columns -> canonical stream names ``{'bp','resp','eda'}``.

    Only streams whose csv column resolves via :data:`_SIGNAL_ALIASES` are
    included; missing streams are simply absent from the returned dict (callers
    decide whether that absence is an error).
    """
    out = {}
    for canon, aliases in _SIGNAL_ALIASES.items():
        col = _first_col(list(cols.keys()), aliases)
        if col is not None:
            out[canon] = np.nan_to_num(cols[col]).astype(np.float32)
    return out


def _load_signals_csv(path: str, fs: float) -> Tuple[Dict[str, np.ndarray], float]:
    """Parse the signals csv -> {name: 1D array} and the true sample rate."""
    with open(path) as f:
        header = next(f).strip().split(_CSV_DELIM)
    header = [h.strip() for h in header]
    data = np.genfromtxt(path, delimiter=_CSV_DELIM, skip_header=1)

    if data.ndim == 1:                       # single sample / degenerate file
        data = data.reshape(1, -1)
    n_cols = data.shape[1]
    if len(header) != n_cols:
        raise ValueError(
            f'{path}: header has {len(header)} cols but data has {n_cols}')

    cols = {}
    for i, name in enumerate(header):
        cols[name] = data[:, i]

    time_col = _first_col(header, ('time', 't', 'timestamp', 'sec'))
    if time_col is not None and cols[time_col].size > 1:
        t = cols[time_col]
        t = t[~np.isnan(t)]
        dt = np.diff(t)
        dt = dt[dt > 0]
        if dt.size:
            fs = 1.0 / float(np.median(dt))     # time column is in seconds
    return cols, fs


def scan_sessions(data_path: str, rgb_dir: str = 'rgb', tir_file: str = 'tir.wmv',
                  signals_file: str = 'signals.csv') -> List[dict]:
    """Discover session directories and their raw files."""
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f'data_path does not exist: {data_path}')
    sessions = []
    for name in sorted(os.listdir(data_path)):
        root = os.path.join(data_path, name)
        if not os.path.isdir(root):
            continue
        rgb_path = os.path.join(root, rgb_dir)
        tir_path = os.path.join(root, tir_file)
        if not os.path.isdir(rgb_path):
            continue
        if not os.path.isfile(tir_path):
            # fall back to the first *.wmv found in the session root
            wmvs = [os.path.join(root, f) for f in os.listdir(root)
                    if f.lower().endswith('.wmv')]
            if not wmvs:
                continue
            tir_path = sorted(wmvs)[0]
        sig_path = os.path.join(root, signals_file)
        if not os.path.isfile(sig_path):
            continue
        sessions.append({'session': name, 'root': root, 'rgb_dir': rgb_path,
                         'tir_file': tir_path, 'signals_file': sig_path})
    if not sessions:
        raise FileNotFoundError(
            f'No sessions of the expected layout found under {data_path}. '
            f'Expected <session>/{rgb_dir}/, <session>/{tir_file}, '
            f'<session>/{signals_file}.')
    return sessions


class PairedSessionDataset(Dataset):
    """Fixed-length synchronised clips over sessions (train/val per session)."""

    def __init__(self, data_path: str, target: str = 'bp',
                 is_train: bool = True, test_mode: bool = False,
                 fs: float = 100.0, fps: float = 25.0,
                 clip_duration: float = 10.0,
                 clip_stride: Optional[float] = None,
                 temporal_stride: int = 1,
                 seq_len: Optional[int] = None,
                 input_size: int = 224, train_ratio: float = 0.8,
                 split_by: str = 'session',
                 rgb_dir: str = 'rgb', tir_file: str = 'tir.wmv',
                 signals_file: str = 'signals.csv',
                 max_sessions: Optional[int] = None,
                 max_clips: Optional[int] = None,
                 max_entries: Optional[int] = None,
                 tir_channels: int = 3,
                 use_tir: bool = True,
                 signal_norm: str = 'none'):
        assert target in ('bp', 'resp', 'eda'), target
        assert int(tir_channels) in (1, 3), \
            f'tir_channels must be 1 or 3, got {tir_channels}'
        if signal_norm not in ('none', 'ac', 'zscore'):
            raise ValueError(
                f"signal_norm must be 'none', 'ac' or 'zscore'; got "
                f'{signal_norm!r}')
        if split_by not in ('session', 'subject'):
            raise ValueError(
                f"split_by must be 'session' or 'subject'; got {split_by!r}")
        self.split_by = split_by
        self.target = target
        #: waveform normalisation per clip. The recorded BP4D streams are NOT
        #: zero-mean: the BP (blood pulse) column is raw blood pressure in mmHg (measured
        #: mean ~101, std ~11 on the local sessions), so a regression head would
        #: have to reproduce a ~100 offset and the MR-STFT term would be
        #: dominated by that DC component (its 0 Hz bin) instead of the
        #: pulsatile band. Stage 3 therefore uses 'zscore' (per-window
        #: zero-mean/unit-std, the standard rPPG convention; Pearson/PSD/MAE are
        #: all reported on the normalised waveform).
        #: 'none' keeps the raw units (default -- the inspection tooling plots
        #: absolute waveforms), 'ac' removes the mean only.
        self.signal_norm = signal_norm
        #: False => RGB-only clips (no TIR decode at all). Stage 3 uses this
        #: when the Stage-2 encoder was pre-trained on ``rgb`` (+ physio) only,
        #: so the TIR adapter has no trained weights and TIR would be a domain
        #: shift; it also halves the decode cost and the RAM cache.
        self.use_tir = bool(use_tir)
        self.fs = float(fs)
        self.max_sessions = max_sessions
        self.max_clips = max_clips
        self.max_entries = max_entries
        self.fps_rgb = float(fps)
        self.input_size = input_size
        self.clip_duration = float(clip_duration)
        # stride between window starts (seconds); None/<=0 => non-overlapping
        # windows (stride == clip_duration). clip_stride < clip_duration yields
        # overlapping windows and therefore more samples per session.
        self.clip_stride = (
            self.clip_duration if not clip_stride or float(clip_stride) <= 0
            else float(clip_stride))
        # temporal decimation INSIDE a window: keep one frame every
        # ``temporal_stride`` frames of the common video grid (1 = every frame;
        # 2/4/8 -> 12.5/6.25/3.125 fps at the nominal 25 fps). This is NOT the
        # window hop -- that is ``clip_stride`` (in seconds, above).
        self.temporal_stride = max(1, int(temporal_stride))
        self.seq_len = seq_len or int(round(self.clip_duration * self.fs))
        # TIR channels: 3 keeps the false-colour thermal rendering as written
        # (verified wmv3/yuv420p with real chroma), 1 = legacy luma-only path
        self.tir_channels = int(tir_channels)

        sessions = scan_sessions(data_path, rgb_dir=rgb_dir,
                                 tir_file=tir_file, signals_file=signals_file)

        # quick/dev mode: cap the number of sessions BEFORE any heavy decode
        if max_sessions is not None and max_sessions > 0:
            sessions = sessions[:max_sessions]

        # deterministic split with no frame-level leakage.
        # 'session' = the historical behaviour (one subject's tasks may straddle
        # the split). 'subject' = SUBJECT-DISJOINT, which the Stage-3 protocol
        # requires: every session of a subject lands on the same side, so the
        # model can never see the evaluated subject during training.
        if split_by == 'subject':
            subjects = sorted({_subject_of(s['session']) for s in sessions})
            n_sub = max(1, int(round(len(subjects) * train_ratio)))
            keep = set(subjects[:n_sub] if is_train else subjects[n_sub:])
            sessions = [s for s in sessions
                        if _subject_of(s['session']) in keep]
            if not sessions:
                raise ValueError(
                    f"split_by='subject' left the "
                    f"{'train' if is_train else 'val'} split empty: "
                    f'{len(subjects)} subject(s) with train_ratio '
                    f'{train_ratio} gives {n_sub} train subject(s). Lower '
                    f'train_ratio or add subjects.')
            print(f"[data] split_by=subject: {'train' if is_train else 'val'} "
                  f'= {sorted(keep)} ({len(sessions)} sessions)')
        else:
            n_train = max(1, int(round(len(sessions) * train_ratio)))
            sessions = sessions[:n_train] if is_train else sessions[n_train:]

        self.entries = []          # (session_meta, t_start_s)
        self._tir_cache = {}
        self._sig_cache = {}
        self._sig_all = {}          # session -> {canonical signal name: array}
        self._files_cache = {}

        for s in sessions:
            sig_path = s['signals_file']
            cols, fs_real = _load_signals_csv(sig_path, self.fs)
            s['fs'] = fs_real                      # per-session true sample rate
            # cache EVERY canonical signal stream present (bp/resp/eda) from
            # a single csv parse, so subclasses (Stage-2 multimodal pretraining)
            # can read extra streams without re-parsing the file.
            canon = _canonical_signals(cols)
            self._sig_all[s['session']] = canon
            if target not in canon:
                raise ValueError(
                    f'{sig_path}: no column for {target}; header={list(cols)}')
            self._sig_cache[s['session']] = canon[target]

            rgb_files = vio.list_image_files(s['rgb_dir'])
            self._files_cache[s['session']] = rgb_files

            tir_fps = self.fps_rgb
            if self.use_tir:
                with vio.open_video(s['tir_file']) as reader:
                    tir_fps = reader.fps
                    s['tir_fps'] = tir_fps
                    self._tir_cache[s['session']] = reader.read_all(
                        gray=self.tir_channels == 1, target_size=self.input_size)
            else:
                s['tir_fps'] = tir_fps

            durations = [len(rgb_files) / self.fps_rgb,
                         len(canon[target]) / s['fs']]
            if self.use_tir:
                durations.insert(1, len(self._tir_cache[s['session']]) / tir_fps)
            dur = align.available_duration(durations)
            s['dur'] = dur                       # session's available duration
            stride = self.clip_stride
            # windows start at 0, stride, ... while the window still fits
            if dur >= self.clip_duration:
                n_windows = 1 + int((dur - self.clip_duration) / stride)
            else:
                n_windows = 1
            n_clips = max(1, n_windows)
            s['n_clips_raw'] = n_clips          # before the per-session cap
            # quick/dev mode: cap the windows taken from each session
            if max_clips is not None and max_clips > 0:
                n_clips = min(n_clips, max_clips)
            s['n_clips'] = n_clips              # after the per-session cap
            for k in range(n_clips):
                self.entries.append((s, k * stride))

        # optional hard cap on the total number of clips (quick/dev runs)
        if max_entries is not None and max_entries > 0:
            self.entries = self.entries[:max_entries]

    def __len__(self):
        return len(self.entries)

    def _get_rgb(self, rgb_files, indices):
        frames = []
        n = len(rgb_files)
        for i in indices:
            if 0 <= i < n:
                frames.append(vio.read_image(rgb_files[i],
                                             target_size=self.input_size))
        if not frames:
            raise IndexError('RGB read range out of bounds')
        return np.stack(frames, axis=0).astype(np.float32) / 255.0   # [T,H,W,3]

    def _signal_at(self, s, t_start: float, name: str) -> np.ndarray:
        """Canonical signal ``name`` over this clip, resampled to ``seq_len``."""
        sig = self._sig_all[s['session']][name]
        fs_s = s['fs']
        plan = align.plan_clip(t_start, self.clip_duration,
                               self.fps_rgb, s['tir_fps'], fs_s,
                               temporal_stride=self.temporal_stride)
        sig_slice = align.slice_1d(
            sig, fs_s, start=plan['signal']['start'], n=plan['signal']['n'])
        w = align.resample_1d(
            sig_slice, fs_s, float(self.seq_len / self.clip_duration),
            length=self.seq_len)
        # per-clip waveform normalisation (see signal_norm in __init__)
        if self.signal_norm != 'none':
            w = w - float(w.mean())
            if self.signal_norm == 'zscore':
                w = w / (float(w.std()) + 1e-6)
        return w

    def _load_rgb(self, s, t_start: float) -> torch.Tensor:
        """RGB clip ``[3, T, H, W]`` on the common time grid."""
        session = s['session']
        plan = align.plan_clip(t_start, self.clip_duration,
                               self.fps_rgb, s['tir_fps'], s['fs'],
                               temporal_stride=self.temporal_stride)
        rgb = self._get_rgb(self._files_cache[session],
                            plan['rgb']['indices'])
        tgt_t = plan['rgb']['n']
        rgb = _pad_time(rgb, tgt_t)
        return torch.from_numpy(rgb).permute(3, 0, 1, 2)       # [3,T,H,W]

    def _load_tir(self, s, t_start: float) -> torch.Tensor:
        """TIR clip ``[C, T, H, W]`` on the common time grid (C = 3 by default)."""
        session = s['session']
        tir_frames = self._tir_cache[session]          # [T, H, W] uint8
        plan = align.plan_clip(t_start, self.clip_duration,
                               self.fps_rgb, s['tir_fps'], s['fs'],
                               temporal_stride=self.temporal_stride)
        tir_idx = plan['tir']['indices']
        tir_idx = tir_idx[(tir_idx >= 0) & (tir_idx < len(tir_frames))]
        tir = tir_frames[tir_idx].astype(np.float32) / 255.0  # [T,H,W] or [T,H,W,C]
        if tir.ndim == 3:                                     # legacy 1-channel
            tir = tir[..., None]                              # [T,H,W,1]
        # pad/trim to the RGB grid so both visuals share the same T
        tir = _pad_time(tir, plan['rgb']['n'])
        return torch.from_numpy(tir).permute(3, 0, 1, 2)      # [C,T,H,W]

    def _load_visual(self, s, t_start: float) -> torch.Tensor:
        """Aligned RGB(+TIR) clip stack ``[3(+C), T, H, W]`` on the common grid.

        ``use_tir=False`` returns the RGB branch alone (``[3, T, H, W]``): the
        Stage-3 configuration used when the Stage-2 encoder was pre-trained on
        ``rgb`` (+ 1-D physio) and therefore has no trained TIR adapter.
        """
        rgb = self._load_rgb(s, t_start)
        if not self.use_tir:
            return rgb
        tir = self._load_tir(s, t_start)
        return torch.cat([rgb, tir], dim=0)                    # [3+C,T,H,W]

    def __getitem__(self, idx):
        s, t_start = self.entries[idx]
        samples = self._load_visual(s, t_start)
        # target waveform on the same window, resampled to seq_len
        target = self._signal_at(s, t_start, self.target)
        return samples, torch.from_numpy(target)


def _pad_time(arr: np.ndarray, t: int) -> np.ndarray:
    """Pad/trim the temporal axis of [T, H, W(, C)] to exactly ``t`` frames."""
    if arr.shape[0] == t:
        return arr
    if arr.shape[0] < t:
        pad = [(0, t - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1)
        return np.pad(arr, pad, mode='edge')
    return arr[:t]


class PairedPretrainDataset(PairedSessionDataset):
    """Stage-2 masked-pretraining dataset with FLEXIBLE modalities.

    Reuses ``PairedSessionDataset`` (aligned RGB/TIR frames + 1-D signals on a
    common time grid) but returns a *dict of raw per-stream tensors* consumed
    by the multimodal MAE. Exactly the streams listed in ``streams`` are
    returned, e.g.::

        streams = ('rgb', 'bp')       -> video + physio (Stage-2 minimum)
        streams = ('rgb','tir','bp')  -> two video + one physio
        streams = ('rgb','tir','bp','resp','eda') -> all five

    Stage-2 CONTRACT: ``streams`` must contain at least TWO modalities --
    >=1 video (``rgb``/``tir``) AND >=1 physiological 1-D signal
    (``bp``/``resp``/``eda``, the waveform later regressed in Stage 3);
    video-only or signal-only lists are rejected. Split policy =
    PRETRAIN-ON-ALL: every session is used (``train_ratio=1.0``). Masking is
    applied inside the model forward, not here.
    """

    def __init__(self, data_path: str, fs: float = 100.0, fps: float = 25.0,
                 clip_duration: float = 4.0,
                 clip_stride: Optional[float] = None,
                 temporal_stride: int = 1,
                 seq_len: Optional[int] = None,
                 input_size: int = 64, rgb_dir: str = 'rgb',
                 tir_file: str = 'tir.wmv', signals_file: str = 'signals.csv',
                 streams: Sequence[str] = ('rgb', 'tir', 'bp', 'resp', 'eda'),
                 max_sessions: Optional[int] = None,
                 max_clips: Optional[int] = None,
                 max_entries: Optional[int] = None,
                 tir_channels: int = 3):
        streams = tuple(streams)
        if not streams:
            raise ValueError(
                'PairedPretrainDataset: Stage-2 needs at least TWO streams '
                '(>=1 video and >=1 physiological 1-D signal); got an empty '
                'list.')
        allowed = _PRETRAIN_VISUAL_STREAMS + tuple(_SIGNAL_ALIASES)
        unknown = [s for s in streams if s not in allowed]
        if unknown:
            raise ValueError(
                f'PairedPretrainDataset: unknown stream(s) {unknown}; '
                f'allowed streams: {allowed}')
        self.streams = streams
        self.visual_streams = tuple(
            s for s in streams if s in _PRETRAIN_VISUAL_STREAMS)
        self.signal_streams = tuple(
            s for s in streams if s not in _PRETRAIN_VISUAL_STREAMS)
        # Stage-2 contract: >=1 video (rgb/tir) AND >=1 1-D physiological
        # (bp/resp/eda); the physio stream(s) are the Stage-3 targets.
        if not self.visual_streams or not self.signal_streams:
            raise ValueError(
                'PairedPretrainDataset: Stage-2 requires >=1 video stream '
                f'({", ".join(_PRETRAIN_VISUAL_STREAMS)}) AND >=1 '
                f'physiological 1-D stream ({", ".join(_SIGNAL_ALIASES)}); '
                f'got visual={self.visual_streams}, '
                f'signal={self.signal_streams}.')

        # the base loader only needs ONE signal to size/fs the window.
        target = self.signal_streams[0]
        super().__init__(
            data_path=data_path, target=target,
            is_train=True, test_mode=True,
            fs=fs, fps=fps, clip_duration=clip_duration,
            clip_stride=clip_stride, temporal_stride=temporal_stride,
            seq_len=seq_len,
            input_size=input_size, train_ratio=1.0,
            rgb_dir=rgb_dir, tir_file=tir_file, signals_file=signals_file,
            max_sessions=max_sessions, max_clips=max_clips,
            max_entries=max_entries, tir_channels=tir_channels,
            # decode/cache the TIR video ONLY when 'tir' is a requested stream:
            # an rgb+bp run would otherwise pay the warm .wmv decode plus up to
            # ~1.6 GB (224 px) of RAM per dataset instance for nothing.
            use_tir=('tir' in self.visual_streams))

        # every requested 1-D stream must exist in each cached session's csv
        checked = set()
        for meta, _ in self.entries:
            if meta['session'] in checked:
                continue
            checked.add(meta['session'])
            avail = set(self._sig_all[meta['session']])
            missing = [s for s in self.signal_streams if s not in avail]
            if missing:
                raise ValueError(
                    f'{meta["signals_file"]}: requested stream(s) {missing} '
                    f'missing; available: {sorted(avail)}')

    def __getitem__(self, idx):
        s, t_start = self.entries[idx]
        out = {}
        if 'rgb' in self.visual_streams:
            out['rgb'] = self._load_rgb(s, t_start)       # [3, T, H, W]
        if 'tir' in self.visual_streams:
            out['tir'] = self._load_tir(s, t_start)       # [1, T, H, W]
        for name in self.signal_streams:
            w = self._signal_at(s, t_start, name)         # np [seq_len]
            out[name] = torch.from_numpy(w).unsqueeze(0)  # [1, seq_len]
        return out


def build_paired_dataset(is_train: bool, test_mode: bool, args):
    """Build a PairedSessionDataset from runner/YAML ``args``."""
    return PairedSessionDataset(
        data_path=getattr(args, 'data_path', ''),
        target=getattr(args, 'target', 'bp'),
        is_train=is_train, test_mode=test_mode,
        fs=getattr(args, 'fs', 100.0),
        fps=getattr(args, 'fps', 25.0),
        clip_duration=getattr(args, 'clip_duration', 10.0),
        clip_stride=getattr(args, 'clip_stride', None),
        temporal_stride=int(getattr(args, 'temporal_stride', 1) or 1),
        seq_len=getattr(args, 'seq_len', None),
        input_size=getattr(args, 'input_size', 224),
        train_ratio=getattr(args, 'train_ratio', 0.8),
        split_by=str(getattr(args, 'split_by', 'session')),
        rgb_dir=getattr(args, 'rgb_dir', 'rgb'),
        tir_file=getattr(args, 'tir_file', 'tir.wmv'),
        signals_file=getattr(args, 'signals_file', 'signals.csv'),
        max_sessions=getattr(args, 'max_sessions', None),
        max_clips=getattr(args, 'max_clips', None),
        max_entries=getattr(args, 'max_entries', None),
        tir_channels=int(getattr(args, 'tir_channels', 3)),
        use_tir=bool(getattr(args, 'use_tir', True)),
        signal_norm=str(getattr(args, 'signal_norm', 'none')))
