"""Random-init contract for Stage 2.

``run_pretrain.py`` used to require a VideoMAE checkpoint unconditionally. The
explicit blank aliases (``''`` / ``none`` / ``random`` / ``scratch`` / ``c0``)
now select random initialisation, which is the only valid path for
``project_multimae_small`` (no small Stage-1 checkpoint is published). Any other
unknown spec must still fail.
"""
import argparse

import pytest

torch = pytest.importorskip('torch')

import core.multimae as M
import models.pretrained as P
from runners import run_pretrain as rp


def _args(spec: str) -> argparse.Namespace:
    return argparse.Namespace(pretrained_encoder=spec,
                              videomae_dataset='k400')


@pytest.mark.parametrize('spec', ['', 'none', 'random', 'scratch', 'c0'])
def test_blank_aliases_are_random_init(spec, capsys):
    out = rp.init_stage1_encoder(object(), _args(spec))
    assert out == {'path': '', 'loaded': 0, 'random': True}
    assert 'random init' in capsys.readouterr().out


def test_loaded_checkpoint_branch(monkeypatch):
    monkeypatch.setattr(P, 'resolve_encoder_weights',
                        lambda *a, **k: '/fake/ckpt.pth')
    monkeypatch.setattr(M, 'load_pretrained_encoder',
                        lambda model, path: {'loaded': 7})
    out = rp.init_stage1_encoder(object(), _args('videomae:k400'))
    assert out == {'path': '/fake/ckpt.pth', 'loaded': 7, 'random': False}


def test_unknown_spec_still_fails():
    with pytest.raises((KeyError, ValueError, FileNotFoundError)):
        rp.init_stage1_encoder(object(), _args('not-a-real-spec'))


def test_small_variant_is_registered():
    assert M.multimae_variant('project_multimae_small') == {
        'embed_dim': 384, 'enc_depth': 12, 'enc_num_heads': 6}
