"""Paired-session dataset for the recorded RGB(jpg-seq) + TIR(.wmv) layout.

Expected per-session layout under ``data_path``::

    <data_path>/<session>/
        rgb/            # ordered jpg frames  (25 fps nominal)
        tir.wmv         # single video ~60 s  (25 fps nominal)
        signals.csv     # header: [time,] bvp, resp, eda  at fs Hz

``PairedSessionDataset`` returns ``(samples, target)`` per clip:
  * ``samples`` : float tensor [4, T, H, W]  (RGB 3ch + TIR 1ch, temporal stack)
  * ``target``  : float tensor [seq_len]     (chosen waveform: BVP, RESP or EDA)

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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from . import alignment as align
from . import video_io as vio

__all__ = ['PairedSessionDataset', 'build_paired_dataset', 'scan_sessions']

PAIRED_DATA_SETS = ('bp4d+', 'paired')

_CSV_DELIM = ','
_SIGNAL_ALIASES = {'bvp': ('bvp', 'ppg', 'pulse'),
                   'resp': ('resp', 'respiration'),
                   'eda': ('eda', 'gsr', 'scr', 'electrodermal')}


def _first_col(header: List[str], aliases: Tuple[str, ...]) -> Optional[str]:
    low = [h.strip().lower() for h in header]
    for alias in aliases:
        if alias in low:
            return header[low.index(alias)]
    return None


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

    def __init__(self, data_path: str, target: str = 'bvp',
                 is_train: bool = True, test_mode: bool = False,
                 fs: float = 100.0, fps: float = 25.0,
                 clip_duration: float = 10.0,
                 clip_stride: Optional[float] = None,
                 seq_len: Optional[int] = None,
                 input_size: int = 224, train_ratio: float = 0.8,
                 rgb_dir: str = 'rgb', tir_file: str = 'tir.wmv',
                 signals_file: str = 'signals.csv',
                 max_sessions: Optional[int] = None,
                 max_clips: Optional[int] = None,
                 max_entries: Optional[int] = None):
        assert target in ('bvp', 'resp', 'eda'), target
        self.target = target
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
        self.seq_len = seq_len or int(round(self.clip_duration * self.fs))

        sessions = scan_sessions(data_path, rgb_dir=rgb_dir,
                                 tir_file=tir_file, signals_file=signals_file)

        # quick/dev mode: cap the number of sessions BEFORE any heavy decode
        if max_sessions is not None and max_sessions > 0:
            sessions = sessions[:max_sessions]

        # deterministic per-session split (no frame-level leakage)
        n_train = max(1, int(round(len(sessions) * train_ratio)))
        if is_train:
            sessions = sessions[:n_train]
        elif test_mode:
            sessions = sessions[n_train:]
        else:
            sessions = sessions[n_train:]

        self.entries = []          # (session_meta, t_start_s)
        self._tir_cache = {}
        self._sig_cache = {}
        self._files_cache = {}

        for s in sessions:
            sig_path = s['signals_file']
            cols, fs_real = _load_signals_csv(sig_path, self.fs)
            s['fs'] = fs_real                      # per-session true sample rate
            for alias in _SIGNAL_ALIASES[target]:
                if alias in cols:
                    sig = cols[alias]
                    break
            else:
                raise ValueError(
                    f'{sig_path}: no column for {target}; header={list(cols)}')
            self._sig_cache[s['session']] = np.nan_to_num(sig).astype(np.float32)

            rgb_files = vio.list_image_files(s['rgb_dir'])
            self._files_cache[s['session']] = rgb_files

            tir_fps = self.fps_rgb
            with vio.open_video(s['tir_file']) as reader:
                tir_fps = reader.fps
                s['tir_fps'] = tir_fps
                self._tir_cache[s['session']] = reader.read_all(
                    gray=True, target_size=self.input_size)

            dur = align.available_duration([
                len(rgb_files) / self.fps_rgb,
                len(self._tir_cache[s['session']]) / s['tir_fps'],
                len(sig) / s['fs']])
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

    def __getitem__(self, idx):
        s, t_start = self.entries[idx]
        session = s['session']

        rgb_files = self._files_cache[session]
        tir_frames = self._tir_cache[session]          # [T, H, W] uint8
        sig = self._sig_cache[session]
        fs_s = s['fs']

        # common time grid between both videos
        plan = align.plan_clip(t_start, self.clip_duration,
                               self.fps_rgb, s['tir_fps'], fs_s)

        rgb = self._get_rgb(rgb_files, plan['rgb']['indices'])
        tir_idx = plan['tir']['indices']
        tir_idx = tir_idx[(tir_idx >= 0) & (tir_idx < len(tir_frames))]
        tir = tir_frames[tir_idx].astype(np.float32) / 255.0   # [T,H,W]
        tir = tir[..., None]                                   # [T,H,W,1]

        # pad/trim the two streams to the same T on the common grid
        tgt_t = plan['rgb']['n']
        rgb = _pad_time(rgb, tgt_t)
        tir = _pad_time(tir, tgt_t)

        # target waveform on the same window, resampled to seq_len
        sig_slice = align.slice_1d(
            sig, fs_s, start=plan['signal']['start'], n=plan['signal']['n'])
        target = align.resample_1d(
            sig_slice, fs_s, float(self.seq_len / self.clip_duration),
            length=self.seq_len)

        # [C, T, H, W] = 3xRGB + 1xTIR
        rgb = torch.from_numpy(rgb).permute(3, 0, 1, 2)        # [3,T,H,W]
        tir = torch.from_numpy(tir).permute(3, 0, 1, 2)        # [1,T,H,W]
        samples = torch.cat([rgb, tir], dim=0)                 # [4,T,H,W]
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
    """Stage-2 multimodal masked-pretraining dataset (first local milestone).

    Reuses ``PairedSessionDataset`` (aligned RGB/TIR frames + 1-D signals on a
    common time grid) but returns a *dict of raw per-stream tensors* that the
    multimodal MAE consumes::

        {'rgb': torch [3, T, H, W],   # float in ~[0, 1]
         'tir': torch [1, T, H, W],
         'bvp': torch [1, seq_len]}   # raw waveform over the same window

    Split policy = PRETRAIN-ON-ALL: every session is used for training
    (``train_ratio=1.0``, no subject-disjoint pretrain split). Masking is
    applied inside the model forward, not here.

    NOTE: signal streams beyond BVP (resp/eda) are a later extension; caching
    more columns in ``_sig_cache`` mirrors the target loop below.
    """

    def __init__(self, data_path: str, fs: float = 100.0, fps: float = 25.0,
                 clip_duration: float = 4.0,
                 clip_stride: Optional[float] = None,
                 seq_len: Optional[int] = None,
                 input_size: int = 64, rgb_dir: str = 'rgb',
                 tir_file: str = 'tir.wmv', signals_file: str = 'signals.csv',
                 max_sessions: Optional[int] = None,
                 max_clips: Optional[int] = None,
                 max_entries: Optional[int] = None):
        super().__init__(
            data_path=data_path, target='bvp',
            is_train=True, test_mode=True,
            fs=fs, fps=fps, clip_duration=clip_duration,
            clip_stride=clip_stride, seq_len=seq_len,
            input_size=input_size, train_ratio=1.0,
            rgb_dir=rgb_dir, tir_file=tir_file, signals_file=signals_file,
            max_sessions=max_sessions, max_clips=max_clips,
            max_entries=max_entries)

    def __getitem__(self, idx):
        # super(): samples [4,T,H,W] = RGB(3) + TIR(1); bvp [seq_len]
        samples, bvp = super().__getitem__(idx)
        return {
            'rgb': samples[:3],     # [3, T, H, W]
            'tir': samples[3:4],    # [1, T, H, W]
            'bvp': bvp.unsqueeze(0)  # [1, seq_len]
        }


def build_paired_dataset(is_train: bool, test_mode: bool, args):
    """Build a PairedSessionDataset from runner/YAML ``args``."""
    return PairedSessionDataset(
        data_path=getattr(args, 'data_path', ''),
        target=getattr(args, 'target', 'bvp'),
        is_train=is_train, test_mode=test_mode,
        fs=getattr(args, 'fs', 100.0),
        fps=getattr(args, 'fps', 25.0),
        clip_duration=getattr(args, 'clip_duration', 10.0),
        clip_stride=getattr(args, 'clip_stride', None),
        seq_len=getattr(args, 'seq_len', None),
        input_size=getattr(args, 'input_size', 224),
        train_ratio=getattr(args, 'train_ratio', 0.8),
        rgb_dir=getattr(args, 'rgb_dir', 'rgb'),
        tir_file=getattr(args, 'tir_file', 'tir.wmv'),
        signals_file=getattr(args, 'signals_file', 'signals.csv'),
        max_sessions=getattr(args, 'max_sessions', None),
        max_clips=getattr(args, 'max_clips', None),
        max_entries=getattr(args, 'max_entries', None))
