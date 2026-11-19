"""The ONE visual input preprocessing -- clip-level z-score before patch division.

Contract: ``code/notes/design/Video_input_Preprocessing.md``.
Implementation: ``code/notes/plan/video_input_preprocessing_implementation_plan.md``.

Two rules, both model-side (the datasets keep returning ``[0, 1]``):

1. **Input.** ``core.input_adapters.zscore_clip`` z-scores the WHOLE clip per
   channel, per sample, over ``(T, H, W)``, AFTER ``_fit_time`` and BEFORE
   patch division. Both Stage 2 (``MultiModalMAE``) and Stage 3
   (``MultiModalWaveformRegressor``) call it, so there is one input space.
2. **Targets.** Only MASKED cubes get the local per-cube mean/var; unmasked
   cubes carry the clip-normalised values (no local map).

The gates below (G1-G5 of the plan) are the properties that would fail
SILENTLY: a wrong input space still trains and still logs a plausible loss.
``_tiny_model`` keeps everything CPU-only and fast.
"""
import os
import sys

import pytest
import torch

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import core.multimae as mm                                     # noqa: E402
import core.waveform_model as wm                               # noqa: E402
from core.input_adapters import EPS, zscore_clip               # noqa: E402
from core.multimae import _tiny_model                          # noqa: E402
from core.waveform_model import MultiModalWaveformRegressor    # noqa: E402

#: the eps floor makes two affine-equivalent clips differ by ~1e-7 relative
#: (``sqrt(var+eps)`` vs ``sqrt(k^2*var+eps)``), so the invariance gates are
#: float32-tight, not bit-exact.
AFFINE_ATOL = 1e-4


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _fixed_masks(model, B=2, seed=1234):
    """One deterministic mask set, so a forward pass carries no RNG.

    ADR-002 gives the batch ONE count per stream, so the plan is stated
    explicitly (half the constant budget per visual stream, no physiological
    token) and only the PLACEMENT is drawn from a seeded generator.
    """
    per_visual = model.n_budget // len(model.visual)
    plan = model.plan_from_counts(
        {s: (per_visual if s in model.visual else 0) for s in model.streams})
    generator = torch.Generator().manual_seed(seed)
    return model.make_masks(B, torch.device('cpu'), plan=plan,
                            placement_generator=generator)


def _clip_cubes(x, tubelet):
    """The [B, N, C*t*ph*pw] flattening ``_targets`` starts from (tubelet order)."""
    B, C, T, H, W = x.shape
    t, ph, pw = tubelet
    Gt, Gh, Gw = T // t, H // ph, W // pw
    x = x.view(B, C, Gt, t, Gh, ph, Gw, pw)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return x.view(B, Gt * Gh * Gw, -1)


def _adapter_tokens(model, x, stream='tir'):
    """The tokens the VISUAL ADAPTER produces -- i.e. the network's input space.

    Measured at ``adapters.<stream>`` and not at ``enc_norm``: the encoder's
    first LayerNorm is (nearly) blind to a global affine rescaling of its
    input, so an encoder-output probe cannot tell a normalised clip from a raw
    one. The adapter output can.
    """
    seen = {}

    def hook(_mod, _inp, out):
        seen['z'] = out.detach().clone()

    h = model.adapters[stream].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(x)
    finally:
        h.remove()
    return seen['z']


def _unmasked_tiny(monkeypatch, **over):
    """A frozen ``_tiny_model`` with a FIXED mask set (no RNG in forward)."""
    torch.manual_seed(0)
    m = _tiny_model(**over)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    masks, plan, idx = _fixed_masks(m)
    monkeypatch.setattr(m, 'make_masks',
                        lambda B, device, **kwargs: (masks, plan, idx))
    return m


def _batch(m, tir=None, B=2):
    """A three-stream batch at the tiny model's geometry (both visual streams
    share one grid, which is what the model's alignment check enforces)."""
    v = (B, 3, m.num_frames, m.input_size, m.input_size)
    return {'tir': torch.rand(*v) if tir is None else tir,
            'rgb2d': torch.rand(*v),
            'resp': torch.randn(B, 1, m.seq_len)}


def _stage3_model(**over):
    """Smallest Stage-3 regressor: 10 tubelet steps x 2x2 spatial grid."""
    kw = dict(streams=('tir',), embed_dim=32, enc_depth=1, enc_num_heads=2,
              tubelet=(2, 8, 8), input_size=16, num_frames=20,
              output_len=80, sig_kernel=8, fps=25.0, fs=100.0,
              pos_init='sincos3d', head_init='none')
    kw.update(over)
    return MultiModalWaveformRegressor(**kw)


# --------------------------------------------------------------------------- #
# G1 -- the helper itself
# --------------------------------------------------------------------------- #
def test_zscore_clip_per_channel():
    """Every channel is centred and scaled on its own, whatever its raw scale."""
    torch.manual_seed(0)
    x = torch.rand(2, 3, 4, 8, 8) * 5.0 + 3.0
    x[:, 0] = x[:, 0] * 40.0                 # channel 0 has a wildly different scale
    x[:, 2] = x[:, 2] * 0.2

    y = zscore_clip(x)
    assert y.shape == x.shape

    per_ch = y.mean(dim=(2, 3, 4))           # [B, C]
    per_ch_std = y.std(dim=(2, 3, 4), unbiased=False)
    assert torch.allclose(per_ch, torch.zeros_like(per_ch), atol=1e-5)
    assert torch.allclose(per_ch_std, torch.ones_like(per_ch_std), atol=1e-4)

    # per SAMPLE, not one global scalar: the two batch entries are independent
    assert not torch.allclose(x[0], x[1])


def test_zscore_clip_constant_channel_is_finite_zero():
    """The eps guard: a constant clip becomes exactly 0, never NaN/Inf."""
    for fill in (0.0, 7.0, -3.25):
        y = zscore_clip(torch.full((1, 3, 4, 8, 8), fill))
        assert torch.isfinite(y).all()
        assert int(torch.count_nonzero(y)) == 0

    # ... and one constant channel does not poison the varying ones
    x = torch.rand(1, 3, 4, 8, 8) * 2.0 + 1.0
    x[:, 1] = 4.0
    y = zscore_clip(x)
    assert torch.isfinite(y).all()
    assert int(torch.count_nonzero(y[:, 1])) == 0
    assert abs(float(y[:, 0].mean())) < 1e-5


def test_zscore_clip_dtype_and_batch_independence():
    """float32 stays float32, and slot 1 never shifts slot 0."""
    torch.manual_seed(0)
    x = torch.rand(2, 3, 4, 8, 8)
    x[1] = 3.5                               # constant clip in slot 1
    y = zscore_clip(x)

    assert y.dtype == torch.float32 and x.dtype == torch.float32
    assert torch.allclose(y[0], zscore_clip(x[0:1])[0])
    assert int(torch.count_nonzero(y[1])) == 0

    # a wider dtype is preserved too (no silent cast to the default)
    y64 = zscore_clip(x.double())
    assert y64.dtype == torch.float64


# --------------------------------------------------------------------------- #
# G4 -- the encoder sees an affine-invariant input space
# --------------------------------------------------------------------------- #
def test_encoder_input_is_affine_invariant(monkeypatch):
    """tokens(2x + 5) == tokens(x), and the tokens are NOT the pre-change ones."""
    m = _unmasked_tiny(monkeypatch)
    B = 2
    x = _batch(m, tir=torch.rand(B, 3, m.num_frames, m.input_size,
                                 m.input_size) * 0.7 + 0.2)

    z_plain = _adapter_tokens(m, x)
    z_affine = _adapter_tokens(m, dict(x, tir=2.0 * x['tir'] + 5.0))
    assert z_plain.shape == z_affine.shape
    assert torch.allclose(z_plain, z_affine, atol=AFFINE_ATOL)

    # the change is REAL: the pre-change path (no normalisation) sees a
    # completely different token stream for the same [0, 1]-ish clip
    monkeypatch.setattr(mm, 'zscore_clip', lambda t, eps=EPS: t)
    z_legacy = _adapter_tokens(m, x)
    assert float((z_plain - z_legacy).abs().max()) > 1e-2


# --------------------------------------------------------------------------- #
# G3 -- only the INPUT moved: the masked MSE of a frozen model is unchanged
# --------------------------------------------------------------------------- #
def test_masked_mse_is_unchanged_by_input_normalisation(monkeypatch):
    """A clip already in the input space (z-scored) yields the SAME loss as the
    pre-change code path, where ``zscore_clip`` did not exist.

    The pre-change behaviour is emulated exactly -- patch ``zscore_clip`` to
    the identity and hand the model the already-normalised clip -- so a
    regression in ``_targets``, the masking or the decoder shows up here.
    """
    m = _unmasked_tiny(monkeypatch)
    B = 2
    x_raw = torch.randn(B, 3, m.num_frames, m.input_size, m.input_size)
    x = _batch(m, tir=zscore_clip(x_raw))
    # BOTH visual streams must already be in the input space, else patching
    # ``zscore_clip`` away would also strip the rgb2d normalisation.
    x['rgb2d'] = zscore_clip(x['rgb2d'])

    # the clip really is in the input space (otherwise the comparison is vacuous)
    assert torch.allclose(zscore_clip(x['tir']), x['tir'], atol=1e-5)

    with torch.no_grad():
        now = m(x)
    monkeypatch.setattr(mm, 'zscore_clip', lambda t, eps=EPS: t)
    with torch.no_grad():
        ref = m(x)

    assert now['losses_mse']['tir'].item() == pytest.approx(
        ref['losses_mse']['tir'].item(), rel=1e-5, abs=1e-7)
    assert now['losses_mse']['resp'].item() == pytest.approx(
        ref['losses_mse']['resp'].item(), rel=1e-5, abs=1e-7)
    assert now['loss'].item() == pytest.approx(
        ref['loss'].item(), rel=1e-5, abs=1e-7)

    # and the masks/prediction shapes are untouched by the change
    assert {s: tuple(v.shape) for s, v in now['masks'].items()} == \
        {s: tuple(v.shape) for s, v in ref['masks'].items()}


# --------------------------------------------------------------------------- #
# G2 -- masked cubes are per-cube z-scores, unmasked cubes stay clip-space
# --------------------------------------------------------------------------- #
def test_unmasked_cube_target_is_clip_space():
    torch.manual_seed(0)
    m = _tiny_model()
    B = 1
    x_clip = zscore_clip(
        torch.rand(B, 3, m.num_frames, m.input_size, m.input_size) * 0.6 + 0.3)

    mask = torch.zeros(B, m.n_visual, dtype=torch.long)
    mask[0, :3] = 1                                     # 3 masked, rest unmasked
    tgt = m._targets(x_clip, 'tir', mask)
    assert tuple(tgt.shape) == (B, m.n_visual, m._flat['tir'])

    mean = tgt.mean(dim=-1)
    std = tgt.std(dim=-1, unbiased=False)
    masked, unmasked = mask.bool(), ~mask.bool()

    # masked cubes: local per-cube standardisation
    assert torch.allclose(mean[masked], torch.zeros_like(mean[masked]),
                          atol=1e-5)
    assert torch.allclose(std[masked], torch.ones_like(std[masked]), atol=1e-4)

    # unmasked cubes: the clip-normalised values, verbatim, NO local map
    flat = _clip_cubes(x_clip, m.tubelet)
    assert torch.allclose(tgt[unmasked], flat[unmasked])
    assert bool((mean[unmasked].abs() > 1e-3).any())

    # D7: mask=None means "every cube masked" (the pre-change behaviour)
    assert torch.allclose(m._targets(x_clip, 'tir'),
                          m._targets(x_clip, 'tir', torch.ones_like(mask)))


def test_targets_ignore_the_mask_for_one_dimensional_streams():
    """The 1-D branch has one label space; the mask argument is accepted and
    ignored, so Stage-2 can pass the loss mask for every stream."""
    m = _tiny_model(target_norm='none')
    B = 2
    x = torch.randn(B, 1, m.n_signal * m.sig_kernel)
    want = x.reshape(B, m.n_signal, m.sig_kernel)
    assert torch.allclose(m._targets(x, 'resp'), want)
    mask = torch.zeros(B, m.n_signal, dtype=torch.long)
    mask[:, :2] = 1
    assert torch.allclose(m._targets(x, 'resp', mask), want)


# --------------------------------------------------------------------------- #
# G5 -- Stage 3 shares the same input space as Stage 2
# --------------------------------------------------------------------------- #
def test_stage3_is_affine_invariant(monkeypatch):
    torch.manual_seed(0)
    m = _stage3_model()
    m.eval()
    x = torch.rand(2, 3, m.num_frames, m.input_size, m.input_size) * 0.7 + 0.2

    with torch.no_grad():
        plain = m(x)
        affine = m(2.0 * x + 5.0)
        as_dict = m({'tir': x})

    assert tuple(plain.shape) == (2, m.output_len)
    assert torch.allclose(plain, affine, atol=AFFINE_ATOL)
    assert torch.allclose(plain, as_dict)          # tensor input == dict input

    # the change is real here too -- measured at the adapter, because the
    # encoder's LayerNorm hides an input rescaling
    t_plain = _adapter_tokens(m, {'tir': x})
    monkeypatch.setattr(wm, 'zscore_clip', lambda t, eps=EPS: t)
    t_legacy = _adapter_tokens(m, {'tir': x})
    assert float((t_plain - t_legacy).abs().max()) > 1e-3
