"""``MultiModalMAE`` with the ``rgb2d`` visual stream (3 streams, one grid).

The tri-modal Stage-2 config declares ``streams: tir,rgb2d,resp``. Both visual
streams must share ``input_size`` / ``tubelet`` / ``num_frames`` -- the model has
ONE patch grid, and ``_check_space_time_alignment`` is what enforces it. These
cases pin:

* the stream name is accepted and gets a 3-channel adapter;
* a 3-stream forward emits ``losses_mse`` / ``masked_<s>`` for EXACTLY those
  three streams (the DDP key-set rule: a key set that depends on the step or the
  rank can deadlock ``MetricLogger.synchronize_between_processes``);
* at ``input_size=64`` / 8 frames / ``tubelet (2,16,16)`` each visual stream is
  4x4 = 16 spatial tubes x 4 tubelet steps = 64 tokens, and the 1-D stream is
  64 samples / 16 = 4 tokens -- i.e. one signal token per video tubelet;
* a mismatched second visual geometry is REJECTED rather than silently cropped.
"""
import os
import sys

import pytest
import torch

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from core import multimae as mm

STREAMS = ('tir', 'rgb2d', 'resp')


def _model(**over):
    kw = dict(streams=STREAMS, embed_dim=32, enc_depth=1, enc_num_heads=2,
              dec_depth=1, tubelet=(2, 16, 16), input_size=64, num_frames=8,
              sig_kernel=8, seq_len=32, fps=25.0, fs=100.0,
              temporal_stride=1, n_budget=20)
    kw.update(over)
    return mm.MultiModalMAE(**kw)


def _batch(model, B=2):
    v = (B, 3, model.num_frames, model.input_size, model.input_size)
    return {'tir': torch.rand(*v), 'rgb2d': torch.rand(*v),
            'resp': torch.randn(B, 1, model.seq_len)}


def test_rgb2d_is_a_registered_visual_stream():
    assert 'rgb2d' in mm._VISUAL_STREAMS
    assert mm.STREAM_CHANNELS['rgb2d'] == 3
    m = _model()
    assert m.visual == ['tir', 'rgb2d']
    assert m.signal == ['resp']
    assert m.stream_channels['rgb2d'] == 3


def test_three_stream_forward_emits_exactly_three_key_sets():
    torch.manual_seed(0)
    m = _model()
    out = m(_batch(m))
    assert set(out['losses_mse']) == set(STREAMS)
    assert set(out['losses']) == set(STREAMS)
    for s in STREAMS:
        assert torch.isfinite(out['losses_mse'][s])
    # the two visual streams are tokenised identically
    assert m.n_visual == m.grid_t * m.grid_h * m.grid_w
    assert m.n_visual == 4 * 4 * 4          # 8 frames / tubelet 2, 64/16 = 4x4


def test_96px_geometry_gives_36_patches_and_25_tubelets():
    """The ADR-002 arm geometry: 100 frames / tubelet_t 4 = 25 visual steps and
    8000 signal samples / sig_kernel 32 = 250 tokens, i.e. R = 10."""
    m = _model(input_size=96, num_frames=100, tubelet=(4, 16, 16),
               fs=1000.0, sig_kernel=32, seq_len=8000, temporal_stride=2,
               n_budget=350, spatial_patches=36, physio_tokens=250)
    assert (m.grid_h, m.grid_w, m.grid_t) == (6, 6, 25)
    assert m.n_visual == 36 * 25            # 900 tokens per visual stream
    assert m.n_signal == 250                # R = 10 signal tokens per tube
    assert m.signal_tokens_per_visual == 10
    out = m(_batch(m))
    assert set(out['losses_mse']) == set(STREAMS)
    assert out['n_batch'] == 350


def test_a_non_integer_signal_token_ratio_is_rejected():
    """ADR-002 §5.1: R = n_signal/Gt must be an integer and span the same time.

    ``seq_len=40`` / ``sig_kernel=8`` gives 5 signal tokens over the 4 video
    steps (R = 1.25), which no tube tiling can represent.
    """
    with pytest.raises(ValueError, match='not an integer'):
        _model(seq_len=40)

    # An integer ratio whose spans still disagree (R = 2 but 2*160 ms != 80 ms)
    # must be rejected by the seconds check, not silently accepted.
    with pytest.raises(ValueError, match='cover'):
        _model(sig_kernel=16, seq_len=64)


def test_a_second_visual_stream_with_another_grid_is_rejected():
    """One grid, so a per-stream geometry cannot silently differ.

    The failure is a shape error at the position add (``RuntimeError``) or an
    explicit check -- either way it is LOUD, which is the property that matters;
    a silently cropped/padded second stream would be the real defect.
    """
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        m = _model(stream_channels={'tir': 3, 'rgb2d': 3})
        m({'tir': torch.rand(1, 3, 8, 64, 64),
           'rgb2d': torch.rand(1, 3, 8, 96, 96),          # wrong size
           'resp': torch.randn(1, 1, 32)})


def test_unknown_stream_names_still_fail():
    with pytest.raises(ValueError, match='unknown stream'):
        mm.MultiModalMAE(streams=('tir', 'tex2d', 'resp'), embed_dim=32,
                         enc_depth=1, enc_num_heads=2, dec_depth=1,
                         num_frames=4, input_size=32, sig_kernel=8,
                         seq_len=16)
