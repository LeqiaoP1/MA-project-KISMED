"""Archived (``2D+3D/<S>.zip``) RGB sequences -- the zip-backed frame source.

BP4D+ ships the visible-light frames as **140 per-subject ZIP archives** and
unpacks nothing, while ``data.rgb_roi_dataset`` used to require an extracted
``2D+3D/<S>/<T>/`` directory (``discover_rgb_sessions`` raised *"No 2D+3D
sequences under ..."*). This file pins the fix:

* an archived task is discovered exactly like a directory one, and is exposed as
  a ``zip+file://<zip>!/<member prefix>`` source URI, so ``os.path.join(uri,
  '0000.jpg')`` still names the frame (that is what every access site does);
* ``_read_rgb`` / ``_source_size`` decode a member through ``cv2.imdecode`` and
  return the SAME pixels as ``cv2.imread`` of the same file on disk;
* a full ``RGBRoiDataset`` built from an archive passes ``_check_clip`` -- i.e.
  the ROI box, "the tensor IS the resize of that crop", the ``jpg == mat`` gate
  and the 1-D window slice are all verified on zip-decoded frames, not only on
  directory-decoded ones;
* an extracted directory still WINS over an archive of the same subject, so a
  tree that is being unpacked keeps its old (faster) behaviour;
* the member prefix is found whether the archive root is the subject
  (``F900/T1/0000.jpg``, how BP4D+ ships) or wraps it (``2D+3D/F900/T1/...``).

The synthetic tree mirrors ``tests/test_tir_resp_sentinel.py``: one session, one
49-point ``2DFeatures`` track, one ``BP_mmHg.txt`` and tiny 20x16 JPEG frames
(written with OpenCV), packed into a real archive with ``zipfile``.
"""
import io
import json
import os
import sys
import zipfile

import numpy as np
import pytest

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from data import rgb_roi_dataset as rrd

FPS = 25.0
PHYS_FS = 1000.0
CLIP_S = 0.4                                  # 10 frames @ 25 fps
N_FRAMES = 60                                 # 2.4 s -> 6 non-overlapping clips
CLIP_FRAMES = int(round(CLIP_S * FPS))
N_SIG = int(round(N_FRAMES / FPS * PHYS_FS))  # 2400 samples
SUBJECT = 'F900'
TASK = 'T1'
SRC_HW = (16, 20)                             # tiny frames: (h, w)


def _frame(i: int) -> np.ndarray:
    """A deterministic BGR frame (the gradient makes a mis-decode obvious)."""
    h, w = SRC_HW
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    img[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    img[..., 2] = (i * 4) % 256
    return img


def _jpeg_bytes(i: int) -> bytes:
    import cv2
    ok, buf = cv2.imencode('.jpg', _frame(i), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    return buf.tobytes()


def _landmarks(n: int = N_FRAMES) -> np.ndarray:
    """``[n, 49, 2]`` (x, y) SOURCE pixels, a face-ish patch inside the frame."""
    h, w = SRC_HW
    base = np.zeros((49, 2), np.float64)
    # a 7x7 grid of points spanning x 4..16, y 4..12 -- inside 20x16, and a
    # non-degenerate box (a collinear set would make the ROI padding ambiguous)
    for k in range(49):
        base[k, 0] = 4.0 + 2.0 * (k % 7)
        base[k, 1] = 4.0 + 1.33 * (k // 7)
    out = np.repeat(base[None], n, axis=0)
    # a slow drift so the landmark span is not bit-identical across frames
    out[..., 0] += 0.02 * np.arange(n)[:, None]
    return out


def _write_mat(path: str, pts: np.ndarray) -> None:
    from scipy.io import savemat
    n = pts.shape[0]
    fit = np.zeros((1, n), dtype=[('frame', 'O'), ('pts_2d', 'O'),
                                  ('headPose', 'O')])
    for i in range(n):
        fit[0, i]['frame'] = np.array([[i + 1]], np.uint16)
        fit[0, i]['pts_2d'] = pts[i]
        fit[0, i]['headPose'] = np.zeros((3, 1), np.float32)
    savemat(path, {'fit': fit})


def _write_bp(path: str, n: int = N_SIG) -> None:
    t = np.arange(n) / PHYS_FS
    y = 80.0 + 30.0 * np.sin(2.0 * np.pi * 1.2 * t)      # ~72 bpm
    with open(path, 'w') as fh:
        fh.write('\n'.join(f'{v:.6f}' for v in y) + '\n')


def _build_tree(root: str, prefix: str = '', archived: bool = True,
                extracted: bool = False) -> str:
    """Write a raw tree into ``root``; returns the archive path ('' if none)."""
    os.makedirs(os.path.join(root, '2D+3D'), exist_ok=True)
    os.makedirs(os.path.join(root, '2DFeatures'), exist_ok=True)
    phys = os.path.join(root, 'Physiology', SUBJECT, TASK)
    os.makedirs(phys, exist_ok=True)
    _write_mat(os.path.join(root, '2DFeatures', f'{SUBJECT}_{TASK}.mat'),
               _landmarks())
    _write_bp(os.path.join(phys, 'BP_mmHg.txt'))

    zip_path = ''
    if archived:
        zip_path = os.path.join(root, '2D+3D', f'{SUBJECT}.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
            for i in range(N_FRAMES):
                zf.writestr(f'{prefix}{SUBJECT}/{TASK}/{i:04d}.jpg',
                            _jpeg_bytes(i))
            # a NON-image member in the same prefix: it must never be counted
            zf.writestr(f'{prefix}{SUBJECT}/{TASK}/0000.obj', b'v 0 0 0\n')
    if extracted:
        seq = os.path.join(root, '2D+3D', SUBJECT, TASK)
        os.makedirs(seq, exist_ok=True)
        import cv2
        for i in range(N_FRAMES):
            cv2.imwrite(os.path.join(seq, f'{i:04d}.jpg'), _frame(i))
        with open(os.path.join(seq, '0000.obj'), 'w') as fh:
            fh.write('v 0 0 0\n')
    return zip_path


def _dataset(root: str, **kw) -> rrd.RGBRoiDataset:
    args = dict(raw_root=root, subjects=[SUBJECT], tasks=[TASK],
                signals=('bp',), clip_seconds=CLIP_S, input_size=16,
                landmarks=rrd.FACE_LANDMARKS)
    args.update(kw)
    return rrd.RGBRoiDataset(**args)


# --------------------------------------------------------------------------- #
# source URI helpers
# --------------------------------------------------------------------------- #
def test_source_uri_roundtrip(tmp_path):
    """``zip_source`` / ``split_zip_source`` / ``is_zip_source`` agree."""
    uri = rrd.zip_source(str(tmp_path / 'a.zip'), '/F900/T1/')
    assert rrd.is_zip_source(uri)
    zip_path, member = rrd.split_zip_source(uri)
    assert zip_path == os.path.abspath(str(tmp_path / 'a.zip'))
    assert member == 'F900/T1'                      # prefix, no trailing slash
    # a directory path is NOT a zip source
    assert rrd.split_zip_source('/tmp/plain') is None
    assert not rrd.is_zip_source('/tmp/plain')
    # ... and os.path.join still builds a member URI out of it
    assert os.path.join(uri, '0000.jpg') == (
        f'{rrd.ZIP_SCHEME}{os.path.abspath(str(tmp_path / "a.zip"))}'
        f'!/F900/T1/0000.jpg')


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def test_archived_session_is_discovered(tmp_path):
    """A zip-only subject yields the same spec shape as an extracted one."""
    root = str(tmp_path)
    zip_path = _build_tree(root)
    specs = rrd.discover_rgb_sessions(root, subjects=[SUBJECT], tasks=[TASK])
    assert len(specs) == 1
    spec = specs[0]
    assert spec['session'] == f'{SUBJECT}_{TASK}'
    assert spec['source_kind'] == 'zip'
    assert spec['n_jpg'] == N_FRAMES                    # .obj NOT counted
    assert spec['n_mat'] == N_FRAMES
    assert spec['status'] == 'usable', spec['reason']
    assert rrd.is_zip_source(spec['rgb_dir'])
    assert spec['rgb_dir'] == rrd.zip_source(zip_path, f'{SUBJECT}/{TASK}/')
    assert spec['feat_file'].endswith(f'{SUBJECT}_{TASK}.mat')


def test_nested_archive_prefix_is_found(tmp_path):
    """An archive wrapping ``2D+3D/`` resolves to the same task prefix."""
    root = str(tmp_path)
    zip_path = _build_tree(root, prefix='2D+3D/')
    spec = rrd.discover_rgb_sessions(root, subjects=[SUBJECT], tasks=[TASK])[0]
    assert spec['source_kind'] == 'zip'
    assert spec['rgb_dir'] == rrd.zip_source(zip_path,
                                             f'2D+3D/{SUBJECT}/{TASK}/')
    assert spec['n_jpg'] == N_FRAMES


def test_extracted_dir_wins_over_archive(tmp_path):
    """Both present -> the directory is used (old, faster behaviour)."""
    root = str(tmp_path)
    _build_tree(root, archived=True, extracted=True)
    spec = rrd.discover_rgb_sessions(root, subjects=[SUBJECT], tasks=[TASK])[0]
    assert spec['source_kind'] == 'dir'
    assert spec['rgb_dir'] == os.path.join(root, '2D+3D', SUBJECT, TASK)


def test_find_rgb_dir_returns_uri_or_dir(tmp_path):
    root = str(tmp_path)
    zip_path = _build_tree(root)
    got = rrd.find_rgb_dir(root, SUBJECT, TASK)
    assert got == rrd.zip_source(zip_path, f'{SUBJECT}/{TASK}/')
    assert rrd.find_rgb_dir(root, SUBJECT, 'T9') is None      # absent task

    root2 = str(tmp_path / 'extracted')
    _build_tree(root2, archived=False, extracted=True)
    assert rrd.find_rgb_dir(root2, SUBJECT, TASK) == os.path.join(
        root2, '2D+3D', SUBJECT, TASK)


# --------------------------------------------------------------------------- #
# frame access
# --------------------------------------------------------------------------- #
def test_list_rgb_files_ignores_non_images(tmp_path):
    root = str(tmp_path)
    zip_path = _build_tree(root)
    uri = rrd.zip_source(zip_path, f'{SUBJECT}/{TASK}/')
    names = rrd._list_rgb_files(uri)
    assert names == [f'{i:04d}.jpg' for i in range(N_FRAMES)]
    assert not any(n.endswith('.obj') for n in names)


def test_zip_decode_matches_disk_decode(tmp_path):
    """``_read_rgb`` from a member == ``cv2.imread`` of the same JPEG."""
    root = str(tmp_path)
    zip_path = _build_tree(root, archived=True, extracted=True)
    ds_dir = _dataset(root)
    uri = rrd.zip_source(zip_path, f'{SUBJECT}/{TASK}/')
    from_zip = ds_dir._read_rgb(os.path.join(uri, '0007.jpg'))
    from_disk = ds_dir._read_rgb(os.path.join(root, '2D+3D', SUBJECT, TASK,
                                              '0007.jpg'))
    assert from_zip.shape == from_disk.shape == (*SRC_HW, 3)
    assert np.array_equal(from_zip, from_disk)
    # and the channel order is the provider's (BGR reversed): in `_frame` BGR
    # channel 2 is the per-frame constant (28 here) and channel 0 the
    # horizontal gradient, so they must come back as RGB channel 0 and 2. JPEG
    # chroma subsampling blurs a constant by a few levels, so compare the shape
    # of the two channels rather than exact values.
    assert np.all(np.diff(from_zip[0, :, 2].astype(int)) >= 0)     # gradient
    assert abs(float(from_zip[..., 0].mean()) - 28.0) < 12.0       # constant
    assert int(np.ptp(from_zip[..., 0].astype(int))) < int(
        np.ptp(from_zip[..., 2].astype(int)))


def test_decode_scale_factors_the_frame(tmp_path):
    """``decode_scale`` halves the decoded size for a zip member too."""
    root = str(tmp_path)
    zip_path = _build_tree(root)
    uri = rrd.zip_source(zip_path, f'{SUBJECT}/{TASK}/')
    ds = _dataset(root, decode_scale=2)
    img = ds._read_rgb(os.path.join(uri, '0000.jpg'))
    assert img.shape[:2] == (SRC_HW[0] // 2, SRC_HW[1] // 2)


# --------------------------------------------------------------------------- #
# the dataset on top of an archive
# --------------------------------------------------------------------------- #
def test_dataset_from_archive_passes_its_own_checks(tmp_path):
    """The whole clip path (ROI, tensor, 1-D window) works on zip frames."""
    root = str(tmp_path)
    _build_tree(root)
    ds = _dataset(root)
    assert ds.stats['sessions_from_zip'] == 1
    assert ds.sessions[f'{SUBJECT}_{TASK}']['source_kind'] == 'zip'
    assert len(ds) == N_FRAMES // CLIP_FRAMES            # 6 clips

    item = ds[0]
    assert item['rgb_video'].shape == (3, CLIP_FRAMES, 16, 16)
    assert item['rgb_video'].dtype.is_floating_point
    assert float(item['rgb_video'].min()) >= 0.0
    assert float(item['rgb_video'].max()) <= 1.0
    assert item['bp_signal'].shape == (int(round(CLIP_S * PHYS_FS)),)

    for index in range(len(ds)):
        assert rrd._check_clip(ds, index) == [], index


def test_dataset_from_extracted_dir_is_unchanged(tmp_path):
    """The directory path still passes the same checks (no regression)."""
    root = str(tmp_path)
    _build_tree(root, archived=False, extracted=True)
    ds = _dataset(root)
    assert ds.stats['sessions_from_zip'] == 0
    assert ds.sessions[f'{SUBJECT}_{TASK}']['source_kind'] == 'dir'
    for index in range(len(ds)):
        assert rrd._check_clip(ds, index) == [], index


def test_unusable_rule_still_fires_on_an_archive(tmp_path):
    """A frame-count mismatch excludes the archived pair (its BP is not read)."""
    root = str(tmp_path)
    zip_path = os.path.join(root, '2D+3D', f'{SUBJECT}.zip')
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    # 59 frames in the archive, 60 rows in the .mat -> unusable
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
        for i in range(N_FRAMES - 1):
            zf.writestr(f'{SUBJECT}/{TASK}/{i:04d}.jpg', _jpeg_bytes(i))
    os.makedirs(os.path.join(root, '2DFeatures'), exist_ok=True)
    _write_mat(os.path.join(root, '2DFeatures', f'{SUBJECT}_{TASK}.mat'),
               _landmarks())
    spec = rrd.discover_rgb_sessions(root, subjects=[SUBJECT], tasks=[TASK])[0]
    assert spec['status'] == 'unusable'
    assert spec['n_jpg'] == N_FRAMES - 1 and spec['n_mat'] == N_FRAMES
    with pytest.raises(RuntimeError):
        _dataset(root)


def test_zip_cache_is_reopened_after_a_fork(tmp_path, monkeypatch):
    """The handle cache is keyed on the pid -- a forked child must reopen it."""
    root = str(tmp_path)
    zip_path = _build_tree(root)
    zf_a, names = rrd._zip_handle(zip_path)
    assert len(names) == N_FRAMES + 1                    # frames + the .obj
    real_pid = os.getpid()                               # BEFORE the patch
    monkeypatch.setattr(rrd.os, 'getpid', lambda: real_pid + 1)
    zf_b, _names = rrd._zip_handle(zip_path)
    assert zf_b is not zf_a                              # reopened, not inherited
