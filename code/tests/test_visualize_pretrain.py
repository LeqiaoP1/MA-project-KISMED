"""``runners/run_visualize_pretrain.py``: the token -> pixel inverse.

The qualitative figure is only trustworthy if the prediction tensor is folded
back to pixels EXACTLY as ``MultiModalMAE._targets`` unfolded it. A silent
misalignment there would still produce a plausible-looking image, so the
inverse is pinned by a test against the forward view itself.
"""
from types import SimpleNamespace

import torch

from runners.run_visualize_pretrain import (_blank_masked_patches,
                                            _tokens_to_video)


def _model(B=2, C=3, T=8, H=32, W=32):
    return SimpleNamespace(tubelet=(2, 16, 16), grid_t=T // 2,
                           grid_h=H // 16, grid_w=W // 16)


def _tokens(x, m):
    """The forward view from ``MultiModalMAE._targets`` (verbatim)."""
    t, ph, pw = m.tubelet
    B, C, T, H, W = x.shape
    Gt, Gh, Gw = T // t, H // ph, W // pw
    v = x.view(B, C, Gt, t, Gh, ph, Gw, pw)
    v = v.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return v.view(B, Gt * Gh * Gw, -1)


def test_tokens_to_video_inverts_the_patch_view():
    m = _model()
    x = torch.arange(2 * 3 * 8 * 32 * 32, dtype=torch.float32).reshape(2, 3, 8, 32, 32)
    assert torch.equal(_tokens_to_video(_tokens(x, m), m), x)


def test_blank_masked_patches_blanks_exactly_the_masked_cells():
    m = _model()                                   # 2 x 2 spatial patches
    img = torch.ones(3, 32, 32)
    mask_spatial = torch.tensor([0, 1, 1, 0])      # (h0,w0), (h0,w1), ...
    out = _blank_masked_patches(img, mask_spatial, m, level=0.0)
    # (h0,w1) and (h1,w0) must be 0, the diagonal patches must stay 1
    assert out[:, 0:16, 16:32].max() == 0
    assert out[:, 16:32, 0:16].max() == 0
    assert out[:, 0:16, 0:16].min() == 1
    assert out[:, 16:32, 16:32].min() == 1


# --------------------------------------------------------------------------- #
# held-out subject selection (the "--split test" path)
# --------------------------------------------------------------------------- #
def test_held_out_subjects_reads_the_run_manifest(tmp_path):
    """The run's own fold_manifest.json is the authority (NOT a re-derivation).

    Re-deriving from the dataset entries partitions the CLIP-SURVIVING subjects
    (133) instead of the discovered usable universe (139), which re-splits every
    fold -- measured once as 15 of 27 "unseen" subjects sitting in the
    checkpoint's train list.
    """
    import json

    from runners.run_visualize_pretrain import _held_out_subjects

    man = tmp_path / 'fold_manifest.json'
    man.write_text(json.dumps({'fold': 0, 'n_folds': 5, 'stage': 2,
                               'roles': {'train': ['F002', 'M001'],
                                         'val': [],
                                         'test': ['F001', 'M003']}}))
    test, train, src = _held_out_subjects(SimpleNamespace(), str(man))
    assert test == {'F001', 'M003'}
    assert train == {'F002', 'M001'}
    assert src == 'fold_manifest.json'


def test_recorded_subjects_parses_every_saved_form():
    """``args.subjects`` is sometimes a real list, sometimes its repr string."""
    from runners.run_visualize_pretrain import _recorded_subjects

    assert _recorded_subjects(['F001', 'M002']) == {'F001', 'M002'}
    assert _recorded_subjects("['F001', 'M002']") == {'F001', 'M002'}
    assert _recorded_subjects('F001, M002') == {'F001', 'M002'}
    assert _recorded_subjects('') == set()
