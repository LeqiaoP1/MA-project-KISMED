"""Unit tests for the Stage-1 VideoMAE weight-spec grammar.

Stage 1 is **ViT-Base only** and is **always** initialised from a VideoMAE
checkpoint (Kinetics-400 or Something-Something-v2), so ``models/pretrained.py``
serves exactly two corpora and rejects every blank / non-VideoMAE source. These
tests pin that contract (no network: ``plan_download`` never touches the Hub).

The loaders' behaviour (the ``n_loaded == 0`` hard error, the 3-D-only patch
embed) is covered at the bottom and needs torch.
"""
import pytest

from models import pretrained as P


# --------------------------------------------------------------------------- #
# the source table
# --------------------------------------------------------------------------- #
def test_available_specs_are_videomae_base_only():
    assert P.available_specs() == ['videomae:k400', 'videomae:ssv2']
    assert set(P.VIT_VARIANTS) == {'base'}


@pytest.mark.parametrize('spec,dataset', [
    ('videomae:k400', 'k400'),
    ('videomae:ssv2', 'ssv2'),
    ('videomae:base', 'k400'),        # legacy alias
    ('base', 'k400'),                 # bare alias
    ('videomae', 'k400'),             # source-only -> default corpus
])
def test_spec_resolves_to_a_corpus(spec, dataset):
    plan = P.plan_download(spec)
    assert plan['kind'] == 'remote'
    assert plan['source'] == 'videomae'
    assert plan['dataset'] == dataset
    assert plan['variant'] == 'base'


def test_source_only_spec_honours_the_dataset_knob():
    plan = P.plan_download('videomae', dataset='ssv2')
    assert plan['dataset'] == 'ssv2'
    assert plan['dest'].name == 'videomae_base_ssv2_patch16_224.pth'


def test_the_two_corpora_have_distinct_cache_files():
    k400 = P.plan_download('videomae:k400')['dest'].name
    ssv2 = P.plan_download('videomae:ssv2')['dest'].name
    assert k400 == 'videomae_base_patch16_224.pth'
    assert ssv2 == 'videomae_base_ssv2_patch16_224.pth'
    assert k400 != ssv2


# --------------------------------------------------------------------------- #
# eliminated sources
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('spec', [
    'mae:base', 'mae:large', 'large', 'videomae:large', 'huge',
    'timm:vit_base_patch16_224.mae',
])
def test_removed_sources_are_rejected(spec):
    with pytest.raises((KeyError, ValueError)):
        P.plan_download(spec)


def test_path_like_typo_raises_filenotfound():
    with pytest.raises(FileNotFoundError):
        P.plan_download('does/not/exist.pth')


# --------------------------------------------------------------------------- #
# the blank-backbone path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('spec', ['', 'none', 'random', 'scratch', 'c0'])
def test_blank_spec_is_empty_without_require(spec):
    """Stage-3 keeps treating an empty --finetune as "no checkpoint"."""
    assert P.resolve_encoder_weights(spec) == ''


@pytest.mark.parametrize('spec', ['', 'none', 'random', 'scratch', 'c0'])
def test_blank_spec_is_a_hard_error_when_required(spec):
    """Stage 1 (``require=True``) has no blank-backbone path."""
    with pytest.raises(ValueError, match='blank/random'):
        P.resolve_encoder_weights(spec, require=True)


# --------------------------------------------------------------------------- #
# loader guards (need torch)
# --------------------------------------------------------------------------- #
def test_stage1_loader_rejects_a_zero_tensor_checkpoint(tmp_path):
    """A Stage-2 ``MultiModalMAE`` file (``enc_blocks.*``) cannot init Stage 1:
    the mapping reads zero tensors, so the loader must raise instead of leaving
    the encoder silently random."""
    torch = pytest.importorskip('torch')
    from core import multimae as M

    model = M._tiny_model()
    fake = {
        'enc_blocks.0.attn.qkv.weight': torch.zeros(3, 3, 3),
        'enc_norm.weight': torch.zeros(3),
    }
    path = tmp_path / 'fake_stage2.pth'
    torch.save(fake, str(path))
    with pytest.raises(ValueError, match='no encoder tensors'):
        M.load_pretrained_encoder(model, str(path))


def test_fit_visual_patch_embed_rejects_a_2d_source():
    """The legacy 2-D boxcar inflation was pruned: only 5-D is accepted."""
    torch = pytest.importorskip('torch')
    from core import multimae as M

    model = M._tiny_model()
    conv2d = torch.zeros(3, 3, 16, 16)              # [out, in, ph, pw]
    assert M.fit_visual_patch_embed(model, conv2d,
                                    'adapters.tir.patch_embed.weight') is None
    conv3d = torch.zeros(3, 3, *model.tubelet)      # [out, in, t, ph, pw]
    got = M.fit_visual_patch_embed(model, conv3d,
                                   'adapters.tir.patch_embed.weight')
    assert got is conv3d
