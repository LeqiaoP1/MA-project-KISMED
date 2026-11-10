"""``MultiModalMAE`` with the ``rgb2d`` visual stream (3 streams, one grid).

The tri-modal Stage-2 config declares ``streams: tir,rgb2d,resp``. Both visual
streams must share ``input_size`` / ``tubelet`` / ``num_frames`` -- the model has
ONE patch grid, and ``_check_space_time_alignment`` is what enforces it. These
cases pin:

* the stream name is accepted and gets a 3-channel adapter;
* a 3-stream forward emits ``losses_mse`` / ``masked_<s>`` for EXACTLY those
  three streams (the DDP key-set rule: a key set that depends on the step or the
  rank can deadlock ``MetricLogger.synchronize_between_processes``);
* at ``input_size=96`` / 100 frames / ``tubelet (2,16,16)`` each visual stream is
  36 spatial patches x 50 tubelet steps = 1800 tokens, and the 1-D stream is
  800 samples / 16 = 50 tokens -- i.e. one signal token per video tubelet;
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
              dec_depth=1, tubelet=(2, 16, 16), input_size=32, num_frames=4,
              sig_kernel=16, seq_len=32, fps=25.0, fs=100.0,
              temporal_stride=2)
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
    assert m.n_visual == 2 * 2 * 2          # 4 frames / tubelet 2, 32/16 = 2x2


def test_96px_geometry_gives_36_patches_and_50_tubelets():
    m = _model(input_size=96, num_frames=100, sig_kernel=16, seq_len=800)
    assert (m.grid_h, m.grid_w, m.grid_t) == (6, 6, 50)
    assert m.n_visual == 36 * 50            # 1800 tokens per visual stream
    assert m.n_signal == 50                 # ONE signal token per tubelet
    out = m(_batch(m))
    assert set(out['losses_mse']) == set(STREAMS)


def test_a_second_visual_stream_with_another_grid_is_rejected():
    """One grid, so a per-stream geometry cannot silently differ.

    The failure is a shape error at the position add (``RuntimeError``) or an
    explicit check -- either way it is LOUD, which is the property that matters;
    a silently cropped/padded second stream would be the real defect.
    """
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        _model(stream_channels={'tir': 3, 'rgb2d': 3}).__call__(
            {'tir': torch.rand(1, 3, 4, 32, 32),
             'rgb2d': torch.rand(1, 3, 4, 64, 64),          # wrong size
             'resp': torch.randn(1, 1, 32)})


def test_unknown_stream_names_still_fail():
    with pytest.raises(ValueError, match='unknown stream'):
        mm.MultiModalMAE(streams=('tir', 'tex2d', 'resp'), embed_dim=32,
                         enc_depth=1, enc_num_heads=2, dec_depth=1,
                         num_frames=4, input_size=32, sig_kernel=16,
                         seq_len=32)
