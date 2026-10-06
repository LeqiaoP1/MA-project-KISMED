"""Unit tests for YAML-config resolution and the unknown-key guard.

``runners/_common.py::parse_args_with_config`` implements the contract
documented in ``code/spec/04_configs_and_execution.md``:

    explicit CLI flag  >  YAML value  >  argparse hardcoded default

A YAML file supplies argparse *defaults* only, so a key it does not mention
keeps the value hardcoded in the runner's ``get_args()``. Because
``parser.set_defaults(**cfg)`` accepts arbitrary keywords, a typo
(``roi_paddingg``) or a stale key used to be a silent no-op: the run proceeded
with the default and looked perfectly healthy. The worst case is a key that
misspells into a *different valid default* -- losing ``mask_ratio_resp: 0.50``
reverts to 0.90, and losing ``pretrained_encoder: videomae:base`` silently
trains from scratch. These tests pin the guard that turns that into a hard
error.

The tests use a SYNTHETIC parser (no torch, no dataset) so they stay fast and
isolate the resolution logic. ``test_config_loading.py`` covers the real
runners and the shipped configs.
"""
import argparse

import pytest

from runners._common import _check_config_keys, parse_args_with_config


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_parser() -> argparse.ArgumentParser:
    """Stand-in with the same SHAPE of keys as a real runner parser.

    Deliberately includes the three cases that make the guard non-trivial:
    a geometry key, a key whose default differs from the shipped configs
    (``mask_ratio_resp`` 0.90 vs 0.50), and an option that has an ALIAS so the
    dest and the flag name disagree.
    """
    p = argparse.ArgumentParser('Test Runner')
    p.add_argument('--input_size', default=64, type=int)
    p.add_argument('--roi_padding', default=0.2, type=float)
    p.add_argument('--mask_ratio_resp', default=0.90, type=float)
    p.add_argument('--streams', default='rgb,tir,bp', type=str)
    p.add_argument('--task_groups', default='', type=str)
    p.add_argument('--lr', default=1e-4, type=float)
    p.add_argument('--spectral_fft_sizes', '--fft_sizes',
                   dest='spectral_fft_sizes', default='', type=str)
    return p


def parse(argv):
    """``parse_args_with_config`` with a fresh synthetic parser."""
    return parse_args_with_config(make_parser(), list(argv))


def reject_message(cfg: str) -> str:
    """Parse ``cfg`` expecting a rejection; return the SystemExit message."""
    with pytest.raises(SystemExit) as exc:
        parse(['-c', cfg])
    return str(exc.value)


# --------------------------------------------------------------------------- #
# precedence: CLI > YAML > hardcoded default
# --------------------------------------------------------------------------- #
def test_no_config_keeps_hardcoded_defaults():
    args = parse([])
    assert (args.input_size, args.roi_padding) == (64, 0.2)
    assert args.mask_ratio_resp == pytest.approx(0.90)


def test_yaml_overrides_hardcoded_default(write_yaml):
    args = parse(['-c', write_yaml('input_size: 96\n')])
    assert args.input_size == 96


def test_yaml_sets_only_the_keys_it_contains(write_yaml):
    """A key ABSENT from the YAML keeps the runner's hardcoded default.

    This is the property that makes a lost key dangerous -- it is exactly why
    the guard below exists.
    """
    args = parse(['-c', write_yaml('input_size: 96\n')])
    assert args.input_size == 96                 # from the YAML
    assert args.roi_padding == 0.2               # not in the YAML -> default
    assert args.mask_ratio_resp == pytest.approx(0.90)


def test_cli_flag_overrides_yaml(write_yaml):
    cfg = write_yaml('input_size: 96\nroi_padding: 0.35\n')
    args = parse(['-c', cfg, '--input_size', '128'])
    assert args.input_size == 128                # CLI wins
    assert args.roi_padding == pytest.approx(0.35)   # untouched YAML value


def test_cli_flag_overrides_hardcoded_default():
    assert parse(['--input_size', '7']).input_size == 7


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse(['-c', str(tmp_path / 'does_not_exist.yaml')])


# --------------------------------------------------------------------------- #
# the unknown-key guard
# --------------------------------------------------------------------------- #
def test_typo_key_is_rejected(write_yaml):
    cfg = write_yaml('input_size: 96\nroi_paddingg: 0.99\n')
    msg = reject_message(cfg)
    assert cfg in msg                       # names the offending file
    assert 'roi_paddingg' in msg            # names the offending key
    assert 'SILENTLY IGNORED' in msg        # explains the failure mode
    assert '1 key(s)' in msg


def test_typo_key_gets_a_spelling_suggestion(write_yaml):
    msg = reject_message(write_yaml('roi_paddingg: 0.99\n'))
    assert "did you mean 'roi_padding'" in msg


def test_unrelated_key_gets_no_suggestion(write_yaml):
    msg = reject_message(write_yaml('zzzzzzzz_qqqq: 1\n'))
    assert 'zzzzzzzz_qqqq' in msg
    assert 'did you mean' not in msg


def test_every_unknown_key_is_listed(write_yaml):
    msg = reject_message(write_yaml('aaa_bogus: 1\nbbb_bogus: 2\nccc_bogus: 3\n'))
    for key in ('aaa_bogus', 'bbb_bogus', 'ccc_bogus'):
        assert key in msg
    assert '3 key(s)' in msg


def test_rejection_names_the_runner(write_yaml):
    assert 'Test Runner' in reject_message(write_yaml('nope_key: 1\n'))


def test_valid_config_is_not_rejected(write_yaml):
    cfg = write_yaml('input_size: 96\nroi_padding: 0.35\nstreams: tir,resp\n')
    args = parse(['-c', cfg])
    assert (args.input_size, args.roi_padding) == (96, pytest.approx(0.35))
    assert args.streams == 'tir,resp'


def test_check_config_keys_accepts_a_plain_dict():
    _check_config_keys(make_parser(), {'input_size': 112}, 'inline.yaml')


def test_check_config_keys_rejects_an_unknown_key_inline():
    with pytest.raises(SystemExit) as exc:
        _check_config_keys(make_parser(), {'input_sizeg': 112}, 'inline.yaml')
    assert 'input_sizeg' in str(exc.value)


# --------------------------------------------------------------------------- #
# edge cases the guard must NOT break
# --------------------------------------------------------------------------- #
def test_empty_yaml_keeps_all_defaults(write_yaml):
    """An empty / comment-only file is legal: it means "all defaults"."""
    for body in ('', '\n', '# nothing but a comment\n'):
        args = parse(['-c', write_yaml(body)])
        assert (args.input_size, args.roi_padding) == (64, 0.2)


def test_non_mapping_yaml_is_rejected(write_yaml):
    for body, kind in (('- a\n- b\n', 'list'), ('just a string\n', 'str')):
        with pytest.raises(SystemExit) as exc:
            parse(['-c', write_yaml(body)])
        msg = str(exc.value)
        assert 'expected a YAML mapping' in msg
        assert kind in msg


def test_null_yaml_value_arrives_as_none(write_yaml):
    """``lr:`` (empty) must reach the runner as None.

    ``run_pretrain`` treats ``lr is None`` as "derive the LR from blr", so the
    guard must not coerce or reject a null value.
    """
    assert parse(['-c', write_yaml('lr:\n')]).lr is None
    assert parse([]).lr == pytest.approx(1e-4)


def test_yaml_key_must_be_the_dest_not_an_alias(write_yaml):
    """``--fft_sizes`` is an ALIAS; the YAML must use the dest name."""
    ok = parse(['-c', write_yaml('spectral_fft_sizes: "128,256,512"\n')])
    assert ok.spectral_fft_sizes == '128,256,512'

    msg = reject_message(write_yaml('fft_sizes: "128,256,512"\n'))
    assert 'fft_sizes' in msg and 'SILENTLY IGNORED' in msg


def test_yaml_mapping_flows_into_a_str_arg_unconverted(write_yaml):
    """``set_defaults`` bypasses argparse's ``type=`` conversion by design.

    ``task_groups`` / ``mask_span_s`` are declared ``type=str`` yet the shipped
    configs use YAML MAPPINGS; the readers accept either form, so the guard must
    accept a dict value for a str-typed option.
    """
    args = parse(['-c', write_yaml('task_groups:\n  low: [T2, T3]\n')])
    assert args.task_groups == {'low': ['T2', 'T3']}


def test_guard_does_not_swallow_other_cli_errors(write_yaml, capsys):
    """An unknown CLI FLAG must still be argparse's error, not ours.

    argparse writes its message to stderr and exits with code 2, so the text is
    read from the captured stderr rather than from SystemExit.code.
    """
    with pytest.raises(SystemExit):
        parse(['-c', write_yaml('input_size: 96\n'), '--not_a_flag', '1'])
    err = capsys.readouterr().err
    assert 'unrecognized arguments' in err
    assert 'SILENTLY IGNORED' not in err
