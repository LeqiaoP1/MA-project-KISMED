"""Convert the fixed raw BP4D layout into the canonical per-session layout.

The canonical layout is what ``data/paired_dataset.PairedSessionDataset``
consumes (see its module docstring):

    <data_path>/<session>/
        rgb/            # ordered jpg frames      (fps_rgb)
        tir.wmv         # single thermal video    (fps_tir)
        signals.csv     # header: time,bvp,resp,eda   at fs Hz
        meta.json       # provenance + probed rates/counts

Raw BP4D layout handled here::

    <raw_root>/2D+3D/<S>/<T>/*.jpg
    <raw_root>/Thermal/<S>/<T>.wmv
    <raw_root>/Physiology/<S>/<T>/<Channel>_*.txt     (one value per line)

Channel mapping (canonical column <- raw file):
    bvp  <- BP_mmHg.txt        (continuous pulse-pressure surrogate)
    resp <- Resp_Volts.txt     (raw respiration belt)
    eda  <- EDA_microsiemens.txt

The raw physiology .txt files carry no time column. The script assumes the
dataset's nominal physiology sample rate (``--phys_fs``, default 1000 Hz, see
the BP4D+ user guide) and, when ``--phys_fs 0``, auto-estimates it per session
from the RGB duration. Channels are low-pass filtered + resampled to ``--fs``
and written as one synchronised CSV with a synthetic time column.

Usage (from ``code/``)::

    # all sessions
    python data/prepare_bp4d.py --raw_root /path/to/BP4D --out_root /path/out
    # quick smoke on the first two sessions
    python data/prepare_bp4d.py --limit_sessions 2
"""
import argparse
import csv
import json
import os
import shutil
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# --- make top-level packages (data.*, utils.*) importable from code/ -------- #
_CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

import numpy as np

from data import video_io as vio

# Canonical column name -> raw physiology file basename (contains match).
CHANNEL_FILES = {
    'bvp': 'BP_mmHg.txt',
    'resp': 'Resp_Volts.txt',
    'eda': 'EDA_microsiemens.txt',
}


def _repo_root() -> str:
    return os.path.dirname(_CODE_ROOT)


def default_raw_root() -> str:
    return os.path.join(_repo_root(), 'data', 'raw', 'BP4D')


def default_out_root() -> str:
    return os.path.join(_repo_root(), 'data', 'processed', 'bp4d_canonical')


def discover_sessions(raw_root: str) -> List[Tuple[str, str]]:
    """Return sorted (subject, task) keys present in all three modality trees."""
    phys = os.path.join(raw_root, 'Physiology')
    c2d = os.path.join(raw_root, '2D+3D')
    th = os.path.join(raw_root, 'Thermal')
    sessions: List[Tuple[str, str]] = []
    if not os.path.isdir(phys):
        raise FileNotFoundError(f'No Physiology/ tree under {raw_root}')
    for subj in sorted(os.listdir(phys)):
        if not os.path.isdir(os.path.join(phys, subj)):
            continue
        for task in sorted(os.listdir(os.path.join(phys, subj))):
            if not os.path.isdir(os.path.join(phys, subj, task)):
                continue
            if not os.path.isdir(os.path.join(c2d, subj, task)):
                continue
            if not os.path.isfile(os.path.join(th, subj, task + '.wmv')):
                continue
            sessions.append((subj, task))
    if not sessions:
        raise FileNotFoundError(
            f'No complete BP4D sessions (RGB+Thermal+Physiology) under {raw_root}')
    return sessions


def load_channel(path: str) -> np.ndarray:
    """Load a single-column physiology .txt as float64 1-D."""
    return np.loadtxt(path, dtype=np.float64).reshape(-1)


def resample_antialiased(x: np.ndarray, fs_in: float, fs_out: float,
                         n_out: int) -> np.ndarray:
    """Low-pass (anti-alias) then linear-interp to ``n_out`` samples."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.zeros(n_out, dtype=np.float64)
    if fs_in > fs_out:
        try:
            from scipy.signal import butter, sosfiltfilt
            nyq = fs_in / 2.0
            cutoff = fs_out * 0.5
            if cutoff < nyq * 0.99:
                sos = butter(4, cutoff / nyq, btype='lowpass', output='sos')
                pad = min(x.size - 1, max(1, int(round(fs_in / fs_out)) * 4))
                if x.size > 2 and pad > 0:
                    xp = np.pad(x, (pad, pad), mode='edge')
                    xp = sosfiltfilt(sos, xp)
                    x = xp[pad:pad + x.size]
        except Exception as exc:  # pragma: no cover - graceful degradation
            print(f'  [warn] anti-alias filter unavailable ({exc}); '
                  f'falling back to plain interpolation')
    old = np.linspace(0, x.size - 1, num=x.size)
    new = np.linspace(0, x.size - 1, num=n_out)
    return np.interp(new, old, x)


def probe_tir_fps(tir_path: str, fallback: float) -> Tuple[float, int, str]:
    """Return (fps, num_frames, status) for a .wmv; never raises."""
    try:
        with vio.open_video(tir_path) as reader:
            return float(reader.fps), int(reader.num_frames), 'ok'
    except Exception as exc:
        return float(fallback), -1, f'decode-failed: {exc}'


def _copy_images(src_dir: str, dst_dir: str, mode: str) -> int:
    os.makedirs(dst_dir, exist_ok=True)
    files = vio.list_image_files(src_dir)
    for i, p in enumerate(files):
        dst = os.path.join(dst_dir, os.path.basename(p))
        if os.path.exists(dst):
            continue
        if mode == 'link':
            try:
                os.link(p, dst)
                continue
            except OSError:
                pass
            try:
                os.symlink(p, dst)
                continue
            except OSError:
                pass
            # fall through to a copy if linking is not permitted
        shutil.copy2(p, dst)
    return len(files)


def build_session(raw_root: str, out_root: str, subject: str, task: str,
                  fs: float, fps_rgb: float, fps_tir_cfg: float,
                  phys_fs_cfg: float, images_mode: str) -> Dict:
    """Convert one (subject, task) session; returns a summary dict."""
    session = f'{subject}_{task}'
    dst = os.path.join(out_root, session)
    os.makedirs(dst, exist_ok=True)

    rgb_src = os.path.join(raw_root, '2D+3D', subject, task)
    tir_src = os.path.join(raw_root, 'Thermal', subject, task + '.wmv')
    phys_src = os.path.join(raw_root, 'Physiology', subject, task)

    rgb_files = vio.list_image_files(rgb_src)
    n_rgb = len(rgb_files)

    # ---- thermal: probe fps (or fall back) + copy ------------------------ #
    fps_tir, n_tir, tir_status = probe_tir_fps(
        tir_src, fps_rgb if fps_tir_cfg <= 0 else fps_tir_cfg)
    tir_dst = os.path.join(dst, 'tir.wmv')
    if not os.path.exists(tir_dst):
        shutil.copy2(tir_src, tir_dst)

    # ---- physiology channels --------------------------------------------- #
    def _find_channel(needle: str) -> Optional[str]:
        """Pick the raw file for a canonical channel.

        Prefers an exact basename match; otherwise matches by substring but
        excludes derived channels (Dia/Mean/Systolic/Rate) so e.g. 'BP_mmHg'
        does not accidentally select 'LA Mean BP_mmHg'.
        """
        names = sorted(os.listdir(phys_src))
        needle_l = needle.lower()
        exact = [f for f in names if f.lower() == needle_l]
        if exact:
            return os.path.join(phys_src, exact[0])
        excluded = ('dia', 'mean', 'systolic', 'diastolic', 'rate')
        cand = [f for f in names
                if needle_l in f.lower()
                and not any(tok in f.lower() for tok in excluded)]
        return os.path.join(phys_src, cand[0]) if cand else None

    found: Dict[str, str] = {}
    for col, needle in CHANNEL_FILES.items():
        p = _find_channel(needle)
        if p is not None:
            found[col] = p
    missing = [c for c in CHANNEL_FILES if c not in found]
    if missing:
        return {'session': session, 'ok': False,
                'error': f'missing physiology channels {missing}'}

    chans = {c: load_channel(p) for c, p in found.items()}
    len_ref = int(np.median([len(v) for v in chans.values()]))

    # auto-estimate the raw physiology sample rate when requested
    phys_fs = phys_fs_cfg
    if phys_fs <= 0:
        dur_rgb = n_rgb / fps_rgb
        phys_fs = (len_ref / dur_rgb) if dur_rgb > 0 else 0.0
        print(f'  [info] auto phys_fs ~= {phys_fs:.1f} Hz (from {len_ref} samples '
              f'over {dur_rgb:.1f}s RGB)')

    dur_sig = len_ref / phys_fs
    n_out = max(1, int(round(dur_sig * fs)))
    sigs = {c: resample_antialiased(v, phys_fs, fs, n_out)
            for c, v in chans.items()}
    t_axis = np.arange(n_out, dtype=np.float64) / fs

    # sanity: RGB vs signal duration consistency (warn only)
    dur_rgb = n_rgb / fps_rgb
    warnings = []
    if dur_rgb > 0 and abs(dur_sig - dur_rgb) / dur_rgb > 0.1:
        warnings.append(
            f'signal duration {dur_sig:.1f}s vs RGB {dur_rgb:.1f}s '
            f'(check --phys_fs / --fps_rgb)')

    # ---- write signals.csv ----------------------------------------------- #
    csv_path = os.path.join(dst, 'signals.csv')
    cols = ['time'] + list(CHANNEL_FILES)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        rows = np.column_stack([t_axis] + [sigs[c] for c in CHANNEL_FILES])
        writer.writerows(rows)

    # ---- RGB frames ------------------------------------------------------ #
    n_copied = 0
    if images_mode != 'none':
        n_copied = _copy_images(rgb_src, os.path.join(dst, 'rgb'), images_mode)

    meta = {
        'session': session, 'subject': subject, 'task': task,
        'fs': fs, 'fps_rgb': fps_rgb, 'fps_tir': fps_tir,
        'phys_fs': phys_fs, 'n_rgb': n_rgb, 'n_tir': n_tir,
        'n_sig': n_out, 'n_rgb_copied': n_copied,
        'durations': {'rgb': dur_rgb, 'tir': (n_tir / fps_tir) if n_tir > 0 else -1,
                      'signal': dur_sig},
        'tir_status': tir_status, 'warnings': warnings, 'ok': True,
    }
    with open(os.path.join(dst, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw_root', default=os.environ.get('RAW_DATA_PATH', ''),
                    help=f'raw BP4D root (default: {default_raw_root()})')
    ap.add_argument('--out_root', default=os.environ.get('DATA_PATH', ''),
                    help=f'canonical output root (default: {default_out_root()})')
    ap.add_argument('--fs', type=float, default=100.0, help='canonical fs (Hz)')
    ap.add_argument('--phys_fs', type=float, default=1000.0,
                    help='raw physiology sample rate (Hz); 0 = auto-estimate')
    ap.add_argument('--fps_rgb', type=float, default=25.0,
                    help='RGB video frame rate (Hz)')
    ap.add_argument('--fps_tir', type=float, default=0.0,
                    help='TIR video frame rate (Hz); 0 = probe the .wmv')
    ap.add_argument('--limit_sessions', type=int, default=0,
                    help='only convert the first N sessions (0 = all)')
    ap.add_argument('--sessions', type=str, default='',
                    help='comma list of "Subject/Task", e.g. F001/T1,F002/T3')
    ap.add_argument('--images', default='copy', choices=['copy', 'link', 'none'],
                    help='how to materialise rgb/ frames')
    ap.add_argument('--force', action='store_true',
                    help='re-convert sessions even if output exists')
    args = ap.parse_args()

    raw_root = args.raw_root or default_raw_root()
    out_root = args.out_root or default_out_root()
    os.makedirs(out_root, exist_ok=True)

    sessions = discover_sessions(raw_root)
    if args.sessions:
        wanted = {tuple(s.strip().split('/')) for s in args.sessions.split(',')
                  if s.strip()}
        sessions = [s for s in sessions if (s[0], s[1]) in wanted]
    elif args.limit_sessions > 0:
        sessions = sessions[:args.limit_sessions]

    if not sessions:
        raise SystemExit('No sessions selected to convert.')
    print(f'Converting {len(sessions)} session(s) from {raw_root} -> {out_root}')

    manifest = os.path.join(out_root, 'manifest.csv')
    write_header = not os.path.exists(manifest) or args.force
    with open(manifest, 'a', newline='') as mf:
        writer = csv.writer(mf)
        if write_header:
            writer.writerow(['session', 'subject', 'task', 'fs', 'fps_rgb',
                             'fps_tir', 'n_rgb', 'n_tir', 'n_sig', 'duration_s'])
        for subject, task in sessions:
            dst = os.path.join(out_root, f'{subject}_{task}')
            if os.path.isdir(dst) and not args.force:
                print(f'  [skip] {subject}_{task} (exists; use --force to redo)')
                continue
            print(f'  [convert] {subject}/{task}')
            try:
                meta = build_session(
                    raw_root=raw_root, out_root=out_root, subject=subject,
                    task=task, fs=args.fs, fps_rgb=args.fps_rgb,
                    fps_tir_cfg=args.fps_tir, phys_fs_cfg=args.phys_fs,
                    images_mode=args.images)
            except Exception as exc:
                print(f'  [FAIL] {subject}/{task}: {exc}')
                continue
            if not meta.get('ok', False):
                print(f"  [FAIL] {subject}/{task}: {meta.get('error')}")
                continue
            writer.writerow([meta['session'], meta['subject'], meta['task'],
                             meta['fs'], meta['fps_rgb'], meta['fps_tir'],
                             meta['n_rgb'], meta['n_tir'], meta['n_sig'],
                             round(meta['durations']['signal'], 3)])
            for w in meta.get('warnings', []):
                print(f'  [warn] {subject}_{task}: {w}')
    print(f'Done. Manifest: {manifest}')


if __name__ == '__main__':
    main()
