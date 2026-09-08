"""AU-occurrence probe dataset -- ADD-ON / diagnostic (does not touch the
Stage-1/2/3 datasets or their builders).

Builds supervised samples for the *semantic-representation-quality* probe
(FACS AU *occurrence* detection): each sample is a fixed-length RGB clip
centred on an AU-coded frame and labelled with that centre frame's AU
occurrence vector (0/1 per target AU). Prediction is clip-level but
frame-anchored: the model outputs ONE multi-label vector per clip, supervised
by the CENTRE frame only (temporal context helps disambiguate AU onsets).

Layout compatibility
--------------------
* RGB frames: the SAME canonical per-session layout the rest of the project
  consumes (``data/prepare_bp4d.py`` output):
  ``<data_path>/<session>/rgb/NNNN.jpg``  (0-based jpg names, 25 fps nominal).
* AU occurrence: raw BP4D+ FACS annotation (see ``BP4D+UserGuide_v0.2.pdf``),
  one file per session ``<au_root>/<session>.csv``. The csv has a header row
  of AU indices and a header *column* of 1-based GLOBAL frame indices; values
  are 0 (absent), 1 (present), 9 (missing/unknown). Only tasks T1/T6/T7/T8 are
  FACS-coded (560 files = 140 subjects x 4 tasks).

Frame mapping (verified): AU frame ``f`` (1-based) <-> RGB image index/jpg
number ``f - 1``. Coded rows whose centre window falls partly before the first
frame or whose labels contain a ``9`` are skipped.

Splits are by SUBJECT (session prefix before ``_``), never by frame/session
(mandatory protocol -- matches the Stage-3 subject-disjoint decision).
"""
import csv
import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from . import video_io as vio

__all__ = ['DEFAULT_AU_LIST', 'AuOccurrenceDataset', 'build_au_datasets',
           'resolve_au_list', 'default_au_root', 'default_data_path']

#: standard BP4D 12-AU evaluation subset (everything else is too rare)
DEFAULT_AU_LIST = (1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24)

_RGB_DIR = 'rgb'
_MISSING = '9'


def _repo_root() -> str:
    # .../MA-project-KISMED/code/data/au_dataset.py -> repo root
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def default_data_path() -> str:
    return os.environ.get('DATA_PATH') or os.path.join(
        _repo_root(), 'data', 'processed', 'bp4d_canonical')


def default_au_root() -> str:
    return os.environ.get('AU_ROOT') or os.path.join(
        _repo_root(), 'data', 'raw', 'BP4D', 'AUCoding', 'AU_OCC')


def _subject_of(session: str) -> str:
    """'F001_T1' -> 'F001' (session prefix is the subject id)."""
    return session.rsplit('_', 1)[0]


def _discover(data_path: str, au_root: str) -> List[dict]:
    """Canonical sessions that have BOTH an rgb/ tree AND an AU csv."""
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f'data_path does not exist: {data_path}')
    if not os.path.isdir(au_root):
        raise FileNotFoundError(f'au_root does not exist: {au_root}')
    out = []
    for name in sorted(os.listdir(data_path)):
        root = os.path.join(data_path, name)
        if not os.path.isdir(root):
            continue
        rgb_dir = os.path.join(root, _RGB_DIR)
        if not os.path.isdir(rgb_dir):
            continue
        au_file = os.path.join(au_root, name + '.csv')
        if not os.path.isfile(au_file):
            continue
        out.append({'session': name, 'subject': _subject_of(name),
                    'rgb_dir': rgb_dir, 'au_file': au_file})
    if not out:
        raise FileNotFoundError(
            f'No AU-usable sessions (rgb/ + AU csv) under {data_path} with '
            f'AU files in {au_root}.')
    return out


class AuOccurrenceDataset(Dataset):
    """RGB clip (centre-frame AU label) dataset over AU-coded sessions.

    :param data_path: canonical session root (default ``bp4d_canonical``)
    :param au_root:   raw BP4D+ ``AUCoding/AU_OCC`` directory
    :param au_list:   target AUs (default the 12-AU BP4D subset)
    :param input_size: square frame resize/crop (must match probe geometry)
    :param num_frames: clip length in frames == the Stage-2 encoder's
        ``num_frames = clip_duration * fps`` (geometry contract; even)
    :param subjects:  restrict to these subject ids (used for subject splits)
    :param max_subjects: cap the number of subjects (dev smoke)
    :param max_entries:  cap the total number of samples (dev smoke)
    """

    def __init__(self, data_path: Optional[str] = None,
                 au_root: Optional[str] = None,
                 au_list: Sequence[int] = DEFAULT_AU_LIST,
                 input_size: int = 64, num_frames: int = 100,
                 subjects: Optional[Sequence[str]] = None,
                 max_subjects: Optional[int] = None,
                 max_entries: Optional[int] = None):
        self.data_path = data_path or default_data_path()
        self.au_root = au_root or default_au_root()
        self.au_list = list(au_list)
        self.input_size = int(input_size)
        self.num_frames = int(num_frames)
        if self.num_frames <= 0 or self.num_frames % 2 != 0:
            raise ValueError(
                f'au_dataset: num_frames must be >0 and even (symmetric '
                f'centre window), got {self.num_frames}')
        self.half = self.num_frames // 2

        sessions = _discover(self.data_path, self.au_root)
        if subjects is not None:
            keep = set(subjects)
            sessions = [s for s in sessions if s['subject'] in keep]
        subject_ids = sorted({s['subject'] for s in sessions})
        if max_subjects is not None and max_subjects > 0:
            subject_ids = subject_ids[:int(max_subjects)]
            keep = set(subject_ids)
            sessions = [s for s in sessions if s['subject'] in keep]
        if not sessions:
            raise ValueError('au_dataset: no sessions for the selected subjects.')

        # frame-number -> position maps (AU frames are jpg *numbers*, 1-based
        # in the csv -> number = frame - 1). Files are read by position, so we
        # additionally require the window's numbers to be position-contiguous.
        self._files: Dict[str, List[str]] = {}
        self._nums: Dict[str, np.ndarray] = {}
        self._entries: List[Tuple[str, int, np.ndarray]] = []   # (rgb_dir,pos,label)

        for meta in sessions:
            self._add_session(meta)
        if max_entries is not None and max_entries > 0:
            self._entries = self._entries[:int(max_entries)]
        if not self._entries:
            raise ValueError(
                'au_dataset: no usable centre-frame samples (all rows with a '
                f'"9" label or window out of range were skipped). '
                f'sessions={len(sessions)}, au_list={self.au_list}')
        self.subjects = sorted({_subject_of(s['session'])
                                for s in sessions})

    # ------------------------------------------------------------------ #
    def _add_session(self, meta: dict) -> None:
        session, rgb_dir, au_file = (meta['session'], meta['rgb_dir'],
                                     meta['au_file'])
        try:
            files = vio.list_image_files(rgb_dir)
        except FileNotFoundError:
            return
        n = len(files)
        self._files[rgb_dir] = files
        nums = []
        for p in files:
            stem = os.path.splitext(os.path.basename(p))[0]
            try:
                nums.append(int(stem))
            except ValueError:
                nums.append(len(nums))          # non-numeric -> fall back to pos
        nums = np.asarray(nums)
        self._nums[rgb_dir] = nums
        num2pos = {int(v): i for i, v in enumerate(nums)}

        with open(au_file, newline='') as fh:
            reader = csv.reader(fh)
            header = next(reader)
            aus = [int(x) for x in header[1:]]
            col = {a: 1 + aus.index(a) for a in self.au_list if a in aus}
            missing_aus = [a for a in self.au_list if a not in col]
            if missing_aus:
                raise ValueError(
                    f'{au_file}: target AU(s) {missing_aus} not in header; '
                    f'header AUs = {aus}')
            for row in reader:
                if not row or len(row) <= max(col.values()):
                    continue
                try:
                    frame = int(row[0])
                except ValueError:
                    continue
                center = frame - 1                       # jpg number (0-based)
                if center < 0:
                    continue
                label = []
                bad = False
                for a in self.au_list:
                    v = row[col[a]].strip()
                    if v == _MISSING:
                        bad = True
                        break
                    label.append(int(v))
                if bad:
                    continue
                start_num = center - self.half
                start = num2pos.get(start_num, -1)
                if start < 0 or start + self.num_frames > n:
                    continue
                # require the window's jpg numbers to be contiguous positions
                if not np.array_equal(
                        nums[start:start + self.num_frames],
                        np.arange(start_num, start_num + self.num_frames)):
                    continue
                self._entries.append(
                    (rgb_dir, start, np.asarray(label, dtype=np.int8)))

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int):
        rgb_dir, start, label = self._entries[idx]
        files = self._files[rgb_dir]
        sub = files[start:start + self.num_frames]
        frames = [vio.read_image(p, target_size=self.input_size) for p in sub]
        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0   # [T,H,W,3]
        samples = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()  # [3,T,H,W]
        target = torch.from_numpy(label.astype(np.float32))               # [n_aus]
        return samples, target


# --------------------------------------------------------------------------- #
# builder (runner-facing; resolves args like the other data builders)
# --------------------------------------------------------------------------- #
def frequent_aus(au_files: Sequence[str], topk: int) -> List[int]:
    """Return the ``topk`` AUs with the highest PRESENCE rate over ``au_files``.

    ``au_files``: AU occurrence csv paths (BP4D+ ``AUCoding/AU_OCC`` layout).
    Ranking is over the whole ANNOTATION CORPUS (independent of which media
    sessions are downloaded) so the selected set is stable across machine/subset
    and reproducible (e.g. ``topk=5`` -> the most frequent expressive AUs).
    Presence rate = present / (present + absent); rows marked '9' (unknown)
    are excluded, exactly as in training. AU codes break ties ascending.
    """
    if topk <= 0:
        return []
    total_present = {}
    total_known = {}
    for au_file in au_files:
        with open(au_file, newline='') as fh:
            reader = csv.reader(fh)
            header = next(reader)
            aus = [int(x) for x in header[1:]]
        present = {a: 0 for a in aus}
        known = {a: 0 for a in aus}
        with open(au_file, newline='') as fh:
            reader = csv.reader(fh)
            next(reader)
            for row in reader:
                if not row or len(row) <= len(aus):
                    continue
                for j, a in enumerate(aus):
                    v = row[1 + j].strip()
                    if v == _MISSING:
                        continue
                    known[a] += 1
                    if v == '1':
                        present[a] += 1
        for a in aus:
            total_present[a] = total_present.get(a, 0) + present[a]
            total_known[a] = total_known.get(a, 0) + known[a]

    scored = [(a, total_present[a] / max(1, total_known[a]))
              for a in total_known if total_known[a] > 0]
    scored.sort(key=lambda t: (-t[1], t[0]))
    out = [a for a, _ in scored[:max(0, topk)]]
    if not out:
        raise ValueError(
            'frequent_aus: no AU with any known (0/1) label found in the AU '
            'files.')
    return out


def resolve_au_list(args) -> List[int]:
    """Resolve the target AU set from args (precedence):

    1. ``--au_list``   : explicit AU set -- a comma string (``"6,7,10,12,14"``)
       or a list/tuple of ints (e.g. YAML ``au_list: [6,7,10,12,14]``).
    2. ``--au_freq_topk N`` : auto-select the N most frequent AUs by presence
       rate over the available AU-usable sessions (e.g. ``5`` -> AU6/7/10/12/14).
    3. otherwise        : :data:`DEFAULT_AU_LIST` (BP4D 12-AU subset).
    """
    raw = getattr(args, 'au_list', '') or ''
    if isinstance(raw, str):
        aus = ([int(x) for x in raw.split(',') if x.strip()]
               if raw.strip() else [])
    else:                                   # already a list/tuple of AU ids
        aus = [int(x) for x in raw]
    if aus:
        return aus
    topk = int(getattr(args, 'au_freq_topk', 0) or 0)
    if topk > 0:
        au_root = getattr(args, 'au_root', '') or default_au_root()
        au_files = sorted(glob.glob(os.path.join(au_root, '*.csv')))
        return frequent_aus(au_files, topk)
    return list(DEFAULT_AU_LIST)


def build_au_datasets(args):
    """Return ``(train, val)`` AU datasets split by SUBJECT.

    Explicit subject lists (``--train_subjects`` / ``--val_subjects``) take
    precedence; otherwise a deterministic ``--train_ratio`` split of the
    (sorted) subjects is used. The AU set comes from :func:`resolve_au_list`.
    """
    data_path = getattr(args, 'data_path', '') or default_data_path()
    au_root = getattr(args, 'au_root', '') or default_au_root()
    sessions_all = _discover(data_path, au_root)
    au_list = resolve_au_list(args)
    fps = float(getattr(args, 'fps', 25.0))
    clip_duration = float(getattr(args, 'clip_duration', 4.0))
    num_frames = int(getattr(args, 'num_frames', 0)) or max(
        1, int(round(clip_duration * fps)))
    input_size = int(getattr(args, 'input_size', 64))
    max_entries = getattr(args, 'max_entries', None)

    common = dict(data_path=data_path, au_root=au_root, au_list=au_list,
                  input_size=input_size, num_frames=num_frames,
                  max_entries=max_entries)

    train_csv = getattr(args, 'train_subjects', '') or ''
    val_csv = getattr(args, 'val_subjects', '') or ''
    if train_csv and val_csv:
        train_subs = [s.strip() for s in train_csv.split(',') if s.strip()]
        val_subs = [s.strip() for s in val_csv.split(',') if s.strip()]
    else:
        all_subs = sorted({s['subject'] for s in sessions_all})
        ratio = float(getattr(args, 'train_ratio', 0.8))
        n_train = max(1, min(len(all_subs) - 1,
                             int(round(len(all_subs) * ratio))))
        train_subs, val_subs = all_subs[:n_train], all_subs[n_train:]

    train_ds = AuOccurrenceDataset(subjects=train_subs, **common)
    val_ds = AuOccurrenceDataset(subjects=val_subs, **common)
    return train_ds, val_ds
