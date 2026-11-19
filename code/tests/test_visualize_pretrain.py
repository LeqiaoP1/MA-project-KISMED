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
def test_held_out_subjects_reads_the_partition_record(tmp_path):
    """The fold's write-once ``partition.json`` is the authority (ADR-001 I2).

    Re-deriving from the dataset entries partitions the CLIP-SURVIVING subjects
    (133) instead of the discovered usable universe (139), which re-splits every
    fold -- measured once as 15 of 27 "unseen" subjects sitting in the
    checkpoint's train list. A legacy ``fold_manifest.json`` carries no digest,
    so it cannot be verified and is refused rather than trusted.
    """
    import json

    import pytest

    from protocol import build_plan, save_plan
    from runners.run_visualize_pretrain import _held_out_subjects

    universe = [f'F{i:03d}' for i in range(1, 7)] + [f'M{i:03d}' for i in range(1, 5)]
    plan = build_plan(universe, fold=0, split_seed=42, split_stratify='none')
    path = tmp_path / 'partition.json'
    save_plan(str(path), plan, writer={'stage': 'test'})

    test, train, src = _held_out_subjects(SimpleNamespace(), str(path))
    assert test == set(plan.roles['test'])
    assert train == set(plan.roles['train'])
    assert src == 'partition.json'

    legacy = tmp_path / 'fold_manifest.json'
    legacy.write_text(json.dumps({'fold': 0, 'n_folds': 5, 'stage': 2,
                                  'roles': {'train': ['F002', 'M001'],
                                            'val': [],
                                            'test': ['F001', 'M003']}}))
    with pytest.raises(SystemExit) as exc:
        _held_out_subjects(SimpleNamespace(), str(legacy))
    # refused at schema validation: the legacy record carries no schema tag,
    # so it cannot be verified and is never trusted
    assert 'partition schema' in str(exc.value)

    with pytest.raises(SystemExit) as exc:
        _held_out_subjects(SimpleNamespace(), '')
    assert '--partition' in str(exc.value)


def test_recorded_subjects_parses_every_saved_form():
    """``args.subjects`` is sometimes a real list, sometimes its repr string."""
    from runners.run_visualize_pretrain import _recorded_subjects

    assert _recorded_subjects(['F001', 'M002']) == {'F001', 'M002'}
    assert _recorded_subjects("['F001', 'M002']") == {'F001', 'M002'}
    assert _recorded_subjects('F001, M002') == {'F001', 'M002'}
    assert _recorded_subjects('') == set()


# --------------------------------------------------------------------------- #
# "--sessions": re-show the SAME subject+task under a different masking
# --------------------------------------------------------------------------- #
def _session_pool():
    """5 clips of F053_T3, 3 of M012_T1, 2 of M058_T3 (indices are the pool)."""
    tags = (['F053_T3'] * 5) + (['M012_T1'] * 3) + (['M058_T3'] * 2)
    return list(range(len(tags))), dict(enumerate(tags))


def test_pin_session_indices_one_clip_per_tag_in_the_given_order():
    from runners.run_visualize_pretrain import _pin_session_indices

    pool, sessions = _session_pool()
    want = ['M058_T3', 'F053_T3', 'M012_T1']
    idx = _pin_session_indices(pool, sessions, want, seed=0)
    assert len(idx) == 3
    assert [sessions[i] for i in idx] == want          # ORDER is the ask
    # same seed -> same clips (the figure stays reproducible)
    assert _pin_session_indices(pool, sessions, want, seed=0) == idx


def test_pin_session_indices_never_leaves_the_session():
    """Every pick must stay INSIDE its session (the split filter is upstream)."""
    from runners.run_visualize_pretrain import _pin_session_indices

    pool, sessions = _session_pool()
    for seed in range(8):
        idx = _pin_session_indices(pool, sessions, ['F053_T3', 'M012_T1'], seed)
        assert sessions[idx[0]] == 'F053_T3' and sessions[idx[1]] == 'M012_T1'


def test_pin_session_indices_rejects_a_tag_outside_the_pool():
    """A tag absent from the (split-filtered) pool must ABORT, not silently leak.

    This is what keeps ``--split test --sessions X`` honest: X is only reachable
    when its subject is in the fold's held-out list.
    """
    import pytest

    from runners.run_visualize_pretrain import _pin_session_indices

    pool, sessions = _session_pool()
    with pytest.raises(SystemExit):
        _pin_session_indices(pool, sessions, ['F053_T3', 'M001_T2'], seed=0)
    with pytest.raises(SystemExit):
        _pin_session_indices(pool, sessions, [], seed=0)


def test_pin_session_indices_pins_an_exact_clip_start():
    """``TAG@t`` must select THAT window (the point of re-using an earlier clip)."""
    from runners.run_visualize_pretrain import _pin_session_indices

    pool, sessions = _session_pool()
    # F053_T3 clips at 0/2/4..8 s, M012_T1 at 0/2/4 s, M058_T3 at 0/2 s
    starts = {i: float(2 * (i if sessions[i] == 'F053_T3' else
                            i - 5 if sessions[i] == 'M012_T1' else i - 8))
              for i in pool}
    idx = _pin_session_indices(pool, sessions, ['F053_T3@4', 'M058_T3@2'],
                               seed=0, starts=starts)
    assert [sessions[i] for i in idx] == ['F053_T3', 'M058_T3']
    assert [starts[i] for i in idx] == [4.0, 2.0]
    # an unknown start must abort (and name the spec), never fall back to a pick
    import pytest
    with pytest.raises(SystemExit):
        _pin_session_indices(pool, sessions, ['F053_T3@99'], seed=0, starts=starts)
