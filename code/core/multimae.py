"""Multimodal masked autoencoder for Stage-2 pre-training.

Streams: ``rgb`` + ``tir`` (3-D tubelet spatio-temporal video) plus any
physiological 1-D signals (``bp`` / ``resp`` / ``eda``; milestone runs
``rgb,tir,bp``). STAGE-2 CONTRACT: at least TWO streams -- >=1 video
(rgb/tir) AND >=1 physiological 1-D signal (bp/resp/eda, the waveform
later regressed in Stage 3). All-visual or all-signal stream lists are
rejected. Pipeline inside ``MultiModalMAE.forward``:

  * per-stream adapters -> tokens (+ learned positional embedding)
  * per-stream ASYMMETRIC masks: tube masks for the videos, random-window for
    the signal (visual 50-75 %, signals 90 %+); masks == 1 => "masked / to
    reconstruct" (matches ``MaskedMSELoss``)
  * visible tokens of ALL streams are concatenated into ONE shared transformer
  * a shared lightweight decoder re-inserts a learned [MASK] token at every
    masked position and each stream reconstructs its NORMALIZED tubelet/window
    patches (VideoMAE-style ``norm_pix``) through a per-stream linear head;
    the loss is a masked MSE computed INDEPENDENTLY per modality on its masked
    positions only, then combined as a WEIGHTED sum: visual streams (rgb/tir)
    keep lambda = 1.0 while physio (1-D) streams get lambda = ``signal_weight``
    (default 0.5; ``loss_weights`` gives a full per-stream override).
  * OPTIONAL periodicity prior on the 1-D streams: a multi-resolution STFT
    MAGNITUDE loss (``core.waveform_losses.MultiResolutionSTFTLoss``) on the
    ASSEMBLED clip waveform, weighted by ``spectral_weight`` (``spectral_weights``
    overrides per stream). OFF by default (0.0), so the previous objective is
    reproduced exactly. The FFT windows are resolved PER STREAM
    (``spectral_fft_sizes``; see :func:`parse_fft_sizes` and
    :data:`SPECTRAL_FFT_DEFAULTS`), because one window set cannot police both
    the BP (1-2.5 Hz) and the RESP (0.16-0.4 Hz) band; the per-modality
    defaults mirror the Stage-3 finetune configs one for one, so a stream gets
    the same spectral objective in both stages. Rationale: a per-token masked
    MSE constrains amplitude
    only, so a low-frequency surrogate can lower it without modelling the
    cardiac/respiratory cycle; the STFT term gives the shared encoder a direct
    gradient on periodicity. It REQUIRES ``target_norm='clip'`` (one mean/std
    per sample+stream) so the token windows reassemble into a coherent
    waveform -- enforced at construction, not assumed.

SPACE-TIME ATTENTION AND ALIGNMENT (checked, not assumed)
---------------------------------------------------------
* The encoder is a plain ViT (``core.blocks.Block``) applied to the SINGLE
  concatenated token sequence of all streams, i.e. JOINT space-time attention
  over the tubelet tokens -- there is no divided/ factorised space-time
  attention and no per-frame attention anywhere in this path. The 3-D
  structure comes from the Conv3d tubelet embed: token index
  ``gt*Gh*Gw + gh*Gw + gw``.
* ``_check_space_time_alignment`` enforces at CONSTRUCTION time that one video
  tubelet and one 1-D signal token cover the same time span *and* that there
  are equally many of them (``Gt == n_signal``), so token ``i`` means the same
  instant in every stream. A geometry that violates this raises instead of
  silently letting the shared encoder learn a non-existent cross-modal
  relation. ``_fit_time`` then refuses to PAD at forward time (padding would
  re-warp the time axis); only a trailing trim (which keeps the time origin)
  is allowed.
* ``_init_pos_embeds`` initialises the (learned) positions with a 3-D
  ``(t, h, w)`` sincos grid for the videos and the SAME 1-D temporal sincos for
  the physio streams, so the space-time prior exists at init -- the Stage-1
  MAE/ViT checkpoint cannot provide it (its 2-D ``pos_embed`` has a different
  token count and is dropped by ``load_pretrained_encoder``). Disable with
  ``pos_init='random'``.

Tensor layouts::

    x = {'rgb': [B, 3, T, H, W],
         'tir': [B, 3, T, H, W],   # false-colour thermal rendering (3 ch)
         'bp': [B, 1, S]}         # any physio stream (resp/eda share this layout)

``tir`` is 3-channel by default because the BP4D thermal ``.wmv`` is a
false-colour (rainbow) thermal *rendering* with a burned-in degC legend -- not a
gray thermal image (verified with two independent decoders: ``wmv3``/``yuv420p``,
real chroma on every session). ``--tir_channels 1`` restores the legacy
luma-only path, which is what checkpoints trained before 2026-09 expect.

TODO(extend): MultiMAE-style prediction-task sampling, separate deeper
per-stream decoders, 3-D sincos pos-embed, visual-stream weight sharing.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import math

import torch
import torch.nn as nn

from .blocks import Block, trunc_normal_
from .criterion import MaskedMSELoss
from .registry import register_model
from .waveform_losses import MultiResolutionSTFTLoss

__all__ = ['TubeletEmbed', 'MultiModalMAE', 'build_pretraining_model',
           'load_pretrained_encoder', 'canonicalise_vit_state_dict',
           'fit_visual_patch_embed', 'MULTIMAE_VARIANTS', 'multimae_variant',
           'STREAM_CHANNELS']

_EPS = 1e-6
_VISUAL_STREAMS = ('rgb', 'tir')
_SIGNAL_STREAMS = ('bp', 'resp', 'eda')

#: Default input channels per stream. ``tir`` is 3, NOT 1: the BP4D thermal
#: stream is a false-colour (rainbow) rendering with a burned-in degC legend --
#: verified with two independent decoders (OpenCV + PyAV): ``wmv3``/``yuv420p``,
#: 3 planes, mean |U-128| ~ 23-27 on every session, chroma following the scene.
#: Feeding luma only would discard the palette the camera wrote. Override per
#: run with ``--tir_channels 1`` (legacy), but note that changes the TIR
#: adapter geometry: Stage-2 checkpoints are only compatible within one setting.
STREAM_CHANNELS = {'rgb': 3, 'tir': 3, 'bp': 1, 'resp': 1, 'eda': 1}

#: Named multimodal encoder geometries. The NAME implies the geometry (the same
#: timm-style convention as the ``project_vit_*`` family in :mod:`core.model`),
#: so a config only needs ``model: project_multimae_base`` and embed_dim /
#: depth / heads follow from it -- they can no longer contradict the
#: ``pretrained_encoder`` checkpoint. **Only ``base`` and ``large`` are
#: supported** (the project's two operating sizes, and the two sizes the Stage-1
#: weights are published for); ``small`` / ``huge`` remain registered model
#: entrypoints with no checkpoint source. Set ``enc_embed_dim`` /
#: ``enc_depth`` / ``enc_num_heads`` explicitly only to OVERRIDE.
MULTIMAE_VARIANTS: Dict[str, Dict[str, int]] = {
    'project_multimae_small': {'embed_dim': 384, 'enc_depth': 12,
                               'enc_num_heads': 6},
    'project_multimae_base': {'embed_dim': 768, 'enc_depth': 12,
                              'enc_num_heads': 12},
    'project_multimae_large': {'embed_dim': 1024, 'enc_depth': 24,
                               'enc_num_heads': 16},
    'project_multimae_huge': {'embed_dim': 1280, 'enc_depth': 32,
                              'enc_num_heads': 16},
}


def multimae_variant(model_name: str) -> Optional[Dict[str, int]]:
    """Geometry of a ``project_multimae_*`` variant (``None`` if unknown)."""
    return MULTIMAE_VARIANTS.get(str(model_name or ''))


# --------------------------------------------------------------------------- #
# adapters
# --------------------------------------------------------------------------- #
class TubeletEmbed(nn.Module):
    """3-D tubelet patch embed: ``[B, C, T, H, W] -> tokens [B, N, D]``.

    One Conv3d doubles as tubelet-partition + linear projection (VideoMAE).
    """

    def __init__(self, in_chans: int, dim_tokens: int,
                 tubelet: Tuple[int, int, int]):
        super().__init__()
        t, ph, pw = tubelet
        self.t, self.ph, self.pw = t, ph, pw
        self.patch_embed = nn.Conv3d(
            in_chans, dim_tokens, kernel_size=(t, ph, pw), stride=(t, ph, pw))

    def forward(self, x):
        # x: [B, C, T, H, W] -> [B, Nv, D]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SignalEmbed(nn.Module):
    """1-D window embed: ``[B, C, S] -> tokens [B, Ns, D]`` (kernel=stride)."""

    def __init__(self, num_channels: int, dim_tokens: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.embed = nn.Conv1d(num_channels, dim_tokens,
                               kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        x = self.embed(x)          # [B, D, Ns]
        return x.transpose(1, 2)   # [B, Ns, D]


class _PosMask(nn.Module):
    """Learned 1-D position embedding + single [MASK] token for one stream.

    ``num_tokens`` is ``Gt*Gh*Gw`` for a video stream (tubelet tokens, ordered
    ``(t, h, w)``) and ``seq_len // sig_kernel`` for a 1-D physio stream; in a
    correctly aligned Stage-2 geometry the two counts are EQUAL, see
    :meth:`MultiModalMAE._check_space_time_alignment`.
    """

    def __init__(self, num_tokens: int, dim_tokens: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, dim_tokens))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim_tokens))
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.mask_token, std=0.02)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class MultiModalMAE(nn.Module):
    """Single shared-encoder, per-stream asymmetric-masked autoencoder."""

    def __init__(self, streams: Sequence[str] = ('rgb', 'tir', 'bp'),
                 stream_channels: Optional[Dict[str, int]] = None,
                 embed_dim: int = 768, enc_depth: int = 12,
                 enc_num_heads: int = 12, mlp_ratio: float = 4.0,
                 dec_depth: int = 2, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0, drop_path_rate: float = 0.0,
                 tubelet: Tuple[int, int, int] = (2, 16, 16),
                 input_size: int = 64, num_frames: int = 100,
                 sig_kernel: int = 8, seq_len: int = 400,
                 fps: float = 25.0, fs: float = 100.0,
                 temporal_stride: int = 1,
                 mask_ratios: Optional[Dict[str, float]] = None,
                 signal_weight: float = 0.5,
                 loss_weights: Optional[Dict[str, float]] = None,
                 pos_init: str = 'sincos3d',
                 target_norm: str = 'token',
                 spectral_weight: float = 0.0,
                 spectral_weights: Optional[Dict[str, float]] = None,
                 spectral_fft_sizes=None,
                 spectral_hop_ratio: float = 0.25,
                 physio_mask: str = 'random',
                 mask_span_s=None):
        super().__init__()
        self.streams = list(streams)
        # per-stream input channels (tir = 3 by default; see STREAM_CHANNELS)
        self.stream_channels = dict(STREAM_CHANNELS)
        if stream_channels:
            self.stream_channels.update(
                {k: int(v) for k, v in stream_channels.items()})
        if not self.streams:
            raise ValueError(
                'MultiModalMAE: Stage-2 needs at least TWO streams (>=1 video '
                'and >=1 physiological 1-D signal); got an empty list.')
        unknown = [s for s in self.streams
                   if s not in _VISUAL_STREAMS and s not in _SIGNAL_STREAMS]
        if unknown:
            raise ValueError(
                f'MultiModalMAE: unknown stream(s) {unknown}; allowed: '
                f'{_VISUAL_STREAMS + _SIGNAL_STREAMS}')
        self.visual = [s for s in self.streams if s in _VISUAL_STREAMS]
        self.signal = [s for s in self.streams if s not in _VISUAL_STREAMS]
        # Stage-2 contract: >=1 video (rgb/tir) AND >=1 1-D physiological
        # (bp/resp/eda) -- the physio stream(s) are the Stage-3 regression
        # targets, so video-only or signal-only runs are not allowed.
        if not self.visual or not self.signal:
            raise ValueError(
                'MultiModalMAE: Stage-2 requires >=1 video stream '
                f'({", ".join(_VISUAL_STREAMS)}) AND >=1 physiological 1-D '
                f'stream ({", ".join(_SIGNAL_STREAMS)}); got '
                f'visual={self.visual}, signal={self.signal}.')

        t, ph, pw = tubelet
        assert input_size % ph == 0 and input_size % pw == 0, \
            f'patch {tubelet} must divide input_size {input_size}'
        assert num_frames % t == 0, \
            f'tubelet_t {t} must divide num_frames {num_frames}'
        self.tubelet = tubelet
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.seq_len = seq_len
        self.input_size = input_size
        self.sig_kernel = sig_kernel
        self.fps = float(fps)
        self.fs = float(fs)
        self.temporal_stride = max(1, int(temporal_stride or 1))

        Gh = Gw = input_size // ph
        Gt = num_frames // t
        self.grid_t, self.grid_h, self.grid_w = Gt, Gh, Gw
        self.n_visual = Gt * Gh * Gw            # tokens per visual stream
        self.n_signal = seq_len // sig_kernel   # tokens for a signal stream
        assert self.n_signal > 0, 'seq_len < sig_kernel'
        # HARD space-time alignment contract: one tubelet token and one signal
        # token must cover the SAME time span and there must be the same number
        # of them, so token index i means the same instant in every stream.
        self._check_space_time_alignment()

        # per-stream positional embedding init: 3-D (t, h, w) sincos for the
        # video streams + the matching 1-D temporal sincos for the physio
        # streams (see _init_pos_embeds). 'random' keeps the trunc_normal init.
        if pos_init not in ('sincos3d', 'random'):
            raise ValueError(
                f"MultiModalMAE: pos_init must be 'sincos3d' or 'random'; "
                f'got {pos_init!r}')
        self.pos_init = pos_init

        # ---- normalisation of the reconstruction target ------------------- #
        # 'token': per-token z-score -- every token window is rescaled by its
        #   OWN mean/std (the pre-2026-09 behaviour).
        # 'clip' : per-clip z-score -- ONE mean/std per (sample, stream), so
        #   the token windows concatenate back into a coherent waveform. This
        #   is what the spectral term below requires: per-token rescaling puts
        #   a step at every token boundary, i.e. broadband energy the STFT loss
        #   would chase instead of the physiological band.
        if target_norm not in ('token', 'clip'):
            raise ValueError(
                f"MultiModalMAE: target_norm must be 'token' or 'clip'; "
                f'got {target_norm!r}')
        self.target_norm = target_norm

        # ---- optional spectral (MR-STFT magnitude) loss on the 1-D streams - #
        # Default 0.0 => OFF, i.e. the masked-MSE-only objective is unchanged.
        self.spectral_weights = {
            s: (0.0 if s in self.visual else float(spectral_weight))
            for s in self.streams}
        if spectral_weights:
            self.spectral_weights.update(
                {s: float(w) for s, w in spectral_weights.items()
                 if s in self.streams})
        bad = [s for s in self.visual if self.spectral_weights.get(s, 0.0) > 0]
        if bad:
            raise ValueError(
                f'MultiModalMAE: the spectral (MR-STFT) term is defined on 1-D '
                f'waveforms only, but spectral_weights sets a positive weight '
                f'for video stream(s) {bad}. Use it on the physio streams '
                f'({self.signal}).')
        self.spectral_hop_ratio = float(spectral_hop_ratio)
        # streams that actually carry the term (empty => no spectral graph at all)
        self._spectral_streams = [s for s in self.signal
                                  if self.spectral_weights.get(s, 0.0) > 0]
        # FFT windows are resolved PER PHYSIO STREAM, not once for the whole
        # run: BP (1-2.5 Hz) and RESP (0.16-0.4 Hz) differ by ~10x in period,
        # so a single window set cannot police both bands -- and the Stage-3
        # finetune configs already carry one set PER BRANCH (configs/finetune/
        # {bp,resp,eda}.yaml --fft_sizes). Accepted forms (see parse_fft_sizes):
        #   None/''/'auto'                      -> SPECTRAL_FFT_DEFAULTS
        #   '64,128,256' / [64,128,256] / 256   -> the SAME set for EVERY physio
        #                                          stream (pre-2026-09 behaviour)
        #   'resp=128/256/512,bp=64/128/256'    -> per stream (CLI)
        #   {'resp': [128, 256, 512]}           -> per stream (YAML)
        # With an empty spec each stream falls back to the window set its own
        # Stage-3 branch uses, so Stage-2 pretraining imposes the same spectral
        # objective the downstream branch will fine-tune with.
        fft_spec = parse_fft_sizes(spectral_fft_sizes)
        #: resolved {stream: [windows]} after the <= clip filter below
        self.spectral_fft_sizes: Dict[str, List[int]] = {}
        #: resolved {stream: MultiResolutionSTFTLoss}; streams with identical
        #: windows share ONE instance (so the uniform case is byte-identical to
        #: the single-module version this replaced)
        self.spectral_fns: Dict[str, MultiResolutionSTFTLoss] = {}
        if self._spectral_streams and self.target_norm != 'clip':
            raise ValueError(
                'MultiModalMAE: a positive spectral_weight requires '
                "target_norm='clip'. The MR-STFT term is computed on the "
                'ASSEMBLED [B, n_signal*sig_kernel] waveform; under the '
                'per-token target normalisation the assembled target is '
                'independently rescaled inside every token, so its '
                'spectrum carries token-boundary artefacts instead of the '
                'physiological band the loss is meant to enforce.')
        if self._spectral_streams:
            n_samples = self.n_signal * self.sig_kernel
            print(f'[spectral] MR-STFT (hop_ratio '
                  f'{self.spectral_hop_ratio:g}, target_norm '
                  f'{self.target_norm}) on the assembled waveform '
                  f'({n_samples} samples = {n_samples / self.fs:.2f} s at '
                  f'fs {self.fs:g} Hz); windows are PER STREAM:')
            by_windows: Dict[Tuple[int, ...], MultiResolutionSTFTLoss] = {}
            for s in self._spectral_streams:
                raw = fft_spec.get(s, fft_spec.get('*'))
                if raw is None:
                    raw = SPECTRAL_FFT_DEFAULTS.get(s, SPECTRAL_FFT_FALLBACK)
                    src = 'per-modality default'
                else:
                    src = 'requested'
                sizes = [int(n) for n in raw]
                kept = [n for n in sizes if n <= n_samples]
                dropped = [n for n in sizes if n > n_samples]
                if not kept:
                    raise ValueError(
                        f'MultiModalMAE: none of the {src} spectral_fft_sizes '
                        f'for stream {s!r} ({sizes} samples) fits the '
                        f'assembled waveform ({n_samples} samples = seq_len = '
                        f'{n_samples / self.fs:.2f} s). Use a longer clip '
                        f'(-c clip_duration) or smaller FFT windows; the '
                        f'per-modality defaults are {SPECTRAL_FFT_DEFAULTS}.')
                self.spectral_fft_sizes[s] = kept
                fn = by_windows.get(tuple(kept))
                if fn is None:
                    fn = MultiResolutionSTFTLoss(
                        fft_sizes=kept, hop_ratio=self.spectral_hop_ratio)
                    by_windows[tuple(kept)] = fn
                self.spectral_fns[s] = fn
                drop = (f'  [DROPPED {"/".join(str(n) for n in dropped)} '
                        f'(> clip {n_samples} samples = '
                        f'{n_samples / self.fs:.2f} s)]' if dropped else '')
                print(f'[spectral]   {s}: weight '
                      f'{self.spectral_weights[s]:g}, {src}: '
                      f'{_window_summary(kept, self.fs)}{drop}')

        # default asymmetric ratios (visual 50-75 %, signals 90 %+)
        ratios = {'rgb': 0.75, 'tir': 0.50, 'bp': 0.90}
        if mask_ratios:
            ratios.update(mask_ratios)
        self.mask_ratios = {s: ratios.get(s, 0.90) for s in self.streams}

        # ---- 1-D masking pattern: scattered dropout or contiguous SPANS ---- #
        # 'random' (default) = the historical `_random_mask`, byte-identical
        # behaviour for every existing config. 'span' = contiguous blocks, which
        # removes the "interpolate the gap from its visible neighbours" shortcut
        # that makes a scattered mask locally solvable -- see
        # code/SpanMask_PhysioSignals.md.
        #
        # The span geometry is resolved ONCE here, per stream, from
        # ``mask_span_s`` (seconds, per stream) and the stream's mask ratio:
        #     span_tokens = round(span_s * fs / sig_kernel)
        #     n_spans     = round(ratio * n_signal / span_tokens)   (>= 1)
        # Both are CONSTANTS of the run (not drawn per sample), which keeps the
        # visible-token count identical for every sample in a batch. That is
        # required, not cosmetic: `forward` gathers visible tokens with a
        # BATCH-MAX k, so unequal counts would leak masked tokens into the
        # encoder. Only the span PLACEMENT is random per sample.
        self.physio_mask = str(physio_mask or 'random').lower()
        if self.physio_mask not in ('random', 'span'):
            raise ValueError(
                f"physio_mask must be 'random' or 'span', got "
                f"{self.physio_mask!r}")
        self.mask_span_s: Dict[str, float] = {}
        self.mask_span_tokens: Dict[str, int] = {}
        self.mask_n_spans: Dict[str, int] = {}
        if self.physio_mask == 'span':
            spec = parse_mask_span(mask_span_s)
            wildcard = spec.get('*')
            for s in self.signal:
                secs = spec.get(s, wildcard)
                if secs is None:
                    secs = MASK_SPAN_DEFAULTS.get(s, self.seq_len / self.fs)
                span = max(1, int(round(float(secs) * self.fs / self.sig_kernel)))
                span = min(span, max(1, self.n_signal - 1))
                n = max(1, int(round(self.mask_ratios[s] * self.n_signal / span)))
                n = max(1, min(n, max(1, (self.n_signal - 1) // span)))
                self.mask_span_s[s] = float(secs)
                self.mask_span_tokens[s] = span
                self.mask_n_spans[s] = n
                eff = n * span / self.n_signal
                note = ('' if abs(eff - self.mask_ratios[s]) < 0.02
                        else f'  [NOTE: configured ratio '
                             f'{self.mask_ratios[s]:.3f} -> effective '
                             f'{eff:.3f}; adjust mask_span_s/mask_ratio_*]')
                print(f'[mask] {s}: span {secs:g} s = {span} tokens '
                      f'({span * self.sig_kernel / self.fs:.2f} s) x {n} '
                      f'span(s) -> effective masked fraction {eff:.3f}'
                      f'{note}')

        # per-modality weights of the final weighted masked-MSE sum.
        # default policy: lambda = 1.0 for the visual streams (rgb/tir) and
        # lambda = ``signal_weight`` for every physio (1-D) stream, so the
        # low-amplitude signals neither dominate nor get ignored by the loss
        # gradient. ``loss_weights`` overrides this per stream when given.
        self.loss_weights = {
            s: (1.0 if s in self.visual else float(signal_weight))
            for s in self.streams
        }
        if loss_weights:
            self.loss_weights.update(
                {s: float(w) for s, w in loss_weights.items()
                 if s in self.streams})

        # masked-MSE criterion; mask == 1 => masked / to reconstruct
        self.loss_fn = MaskedMSELoss()

        # --- adapters ----------------------------------------------------- #
        self.adapters = nn.ModuleDict()
        for s in self.streams:
            if s in self.visual:
                in_ch = int(self.stream_channels.get(s, 3))
                self.adapters[s] = TubeletEmbed(in_ch, embed_dim, tubelet)
            else:
                self.adapters[s] = SignalEmbed(
                    int(self.stream_channels.get(s, 1)), embed_dim, sig_kernel)

        # --- positions / mask tokens -------------------------------------- #
        self.positions = nn.ModuleDict()
        for s in self.streams:
            n = self.n_visual if s in self.visual else self.n_signal
            self.positions[s] = _PosMask(n, embed_dim)

        # --- shared encoder ------------------------------------------------ #
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, enc_depth)]
        self.enc_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=enc_num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=nn.LayerNorm)
            for i in range(enc_depth)
        ])
        self.enc_norm = nn.LayerNorm(embed_dim)

        # --- shared decoder + per-stream heads ----------------------------- #
        self.dec_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=enc_num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=0.0, norm_layer=nn.LayerNorm)
            for _ in range(dec_depth)
        ])

        # flat reconstruction dim per stream (normalized patch of raw values).
        # MUST use the same channel counts as the adapters, otherwise the
        # per-stream prediction and target shapes disagree (tir = 3 by default).
        self.heads = nn.ModuleDict()
        self._flat = {}
        for s in self.streams:
            if s in self.visual:
                in_ch = int(self.stream_channels.get(s, 3))
                f = t * ph * pw * in_ch
            else:
                f = sig_kernel
            self._flat[s] = f
            self.heads[s] = nn.Linear(embed_dim, f)

        self.apply(self._init_weights)
        if self.pos_init == 'sincos3d':
            self._init_pos_embeds()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------------ #
    # space-time alignment contract
    # ------------------------------------------------------------------ #
    def _check_space_time_alignment(self):
        """Verify that the video tubelets and the 1-D signal tokens agree.

        The shared encoder sees ONE concatenated token sequence, so a video
        tubelet token and a physio token are only interchangeable if both
        cover the same time span and there are equally many of them. Three
        conditions are enforced (all hard errors -- a silent mismatch would
        let the model "learn" a cross-modal relation that does not exist):

        1. the tubelet grid tiles the clip exactly (``tubelet_t | num_frames``);
        2. the signal grid tiles the window exactly (``sig_kernel | seq_len``);
        3. one tubelet token and one signal token span the SAME number of
           seconds (``tubelet_t*temporal_stride/fps == sig_kernel/fs``) and
           there is the SAME number of them (``Gt == n_signal``).
        """
        t, _, _ = self.tubelet
        self.sec_per_visual_token = t * self.temporal_stride / self.fps
        self.sec_per_signal_token = self.sig_kernel / self.fs

        if self.num_frames % t:
            raise ValueError(
                f'Space-time alignment: tubelet_t {t} must divide num_frames '
                f'{self.num_frames} so the tubelet grid covers the clip '
                f'exactly.')

        if self.seq_len % self.sig_kernel:
            rest = self.seq_len % self.sig_kernel
            raise ValueError(
                f'Space-time alignment: seq_len {self.seq_len} is not a '
                f'multiple of sig_kernel {self.sig_kernel} -- the trailing '
                f'{rest} signal sample(s) would fall outside the last token '
                f'and the 1-D token grid would drift off the video tubelet '
                f'grid. Use seq_len = {self.seq_len - rest} (or a divisor of '
                f'the window).')

        if not math.isclose(self.sec_per_visual_token,
                            self.sec_per_signal_token,
                            rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(
                f'Space-time alignment: one video tubelet covers '
                f'tubelet_t*temporal_stride/fps = '
                f'{t}*{self.temporal_stride}/{self.fps} = '
                f'{self.sec_per_visual_token * 1000:.2f} ms but one signal '
                f'token covers sig_kernel/fs = {self.sig_kernel}/{self.fs} = '
                f'{self.sec_per_signal_token * 1000:.2f} ms, so token i means '
                f'a different instant in the two streams. Set '
                f'sig_kernel = {int(round(self.sec_per_visual_token * self.fs))} '
                f'(recommended) or temporal_stride = 1 / tune fps-fs.')

        if self.grid_t != self.n_signal:
            raise ValueError(
                f'Space-time alignment: the video covers '
                f'{self.grid_t} tubelet steps '
                f'({self.num_frames} frames / {t} = '
                f'{self.grid_t * self.sec_per_visual_token:.3f} s) but the '
                f'signal covers {self.n_signal} tokens '
                f'({self.seq_len} samples / {self.sig_kernel} = '
                f'{self.n_signal * self.sec_per_signal_token:.3f} s). Both '
                f'streams must span the same window: set seq_len = '
                f'{int(round(self.grid_t * self.sec_per_visual_token * self.fs))} '
                f'(= num_frames * temporal_stride / fps * fs).')

    def _init_pos_embeds(self):
        """Deterministic space-time prior for the learned positional embeds.

        A Stage-1 MAE/ViT-Base checkpoint carries a 2-D ``pos_embed`` over a
        14x14 spatial grid, which cannot be reused for ``Gt*Gh*Gw`` tubelet
        tokens (and ``load_pretrained_encoder`` drops it), so without this the
        temporal structure of the clip would be pure noise at init. Here the
        video positions start from the VideoMAE 3-D sincos grid, and each
        physio stream starts from the SAME 1-D temporal sincos, placed in the
        SAME embedding dims -- i.e. visual token ``(t, h, w)`` and signal
        token ``t`` share an identical time basis. Both stay trainable
        parameters, so this is only an initialisation.
        """
        from utils.pos_embed import (get_1d_sincos_pos_embed_from_grid,
                                     get_3d_sincos_pos_embed)
        import numpy as np

        vis = torch.from_numpy(get_3d_sincos_pos_embed(
            self.embed_dim, (self.grid_t, self.grid_h, self.grid_w))).float()
        d_t = 2 * (self.embed_dim // 6)        # size of the [t] block
        time = torch.from_numpy(get_1d_sincos_pos_embed_from_grid(
            d_t, np.arange(self.n_signal, dtype=np.float32))).float()
        sig = time.new_zeros(self.n_signal, self.embed_dim)
        sig[:, :d_t] = time

        for s in self.streams:
            pe = vis if s in self.visual else sig
            with torch.no_grad():
                self.positions[s].pos_embed.copy_(pe.unsqueeze(0))

    # ------------------------------------------------------------------ #
    # input geometry check (loud failure instead of a silent time warp)
    # ------------------------------------------------------------------ #
    def _fit_time(self, x, n, stream):
        """Trim the trailing frames/samples of ``x`` to ``n`` (or fail loudly).

        Trimming keeps the shared time origin, so it preserves the alignment of
        the remaining tokens. PADDING cannot: replicating frames/samples would
        silently re-warp the time axis against the other stream, which is what
        the old modulo-index helper did -- so a too-short input is an error.
        """
        T = x.shape[2]
        if T == n:
            return x
        if T > n:
            return x[:, :, :n]
        raise ValueError(
            f'Space-time alignment: stream "{stream}" provides T={T} but this '
            f'model needs T={n} per clip (num_frames for the videos, seq_len '
            f'for the 1-D signals). The dataset grid is '
            f'round(clip_duration*fps_common/temporal_stride) with '
            f'fps_common = min(fps_rgb, fps_tir), so check that --fps matches '
            f'the video fps the dataset actually decodes, and that '
            f'--clip_duration/--temporal_stride/--seq_len match the geometry '
            f'this checkpoint was trained with.')

    # ------------------------------------------------------------------ #
    # masking (masks == 1 => masked / to reconstruct)
    # ------------------------------------------------------------------ #
    def _tube_mask(self, B: int, device, mask_ratio: float):
        """Random subset of spatial patches, replicated across all frames."""
        n_spatial = self.grid_h * self.grid_w
        k = max(1, min(n_spatial - 1, int(mask_ratio * n_spatial)))
        perm = torch.rand(B, n_spatial, device=device).argsort(dim=1)
        hidden = perm[:, :k]                                   # [B, k]
        m = torch.zeros(B, n_spatial, device=device, dtype=torch.long)
        m.scatter_(1, hidden, 1)                               # [B, n_spatial]
        mask = m.unsqueeze(1).expand(B, self.grid_t, n_spatial)
        return mask.reshape(B, self.n_visual)

    def _random_mask(self, B: int, device, N: int, mask_ratio: float):
        k = max(1, min(N - 1, int(mask_ratio * N)))
        perm = torch.rand(B, N, device=device).argsort(dim=1)
        hidden = perm[:, :k]
        m = torch.zeros(B, N, device=device, dtype=torch.long)
        m.scatter_(1, hidden, 1)
        return m

    def _span_mask(self, B: int, device, N: int, n_spans: int, span: int):
        """Contiguous-block mask: ``n_spans`` spans of ``span`` tokens.

        The masked COUNT is ``min(n_spans * span, N - 1)`` for every sample in
        the batch (the ``N - 1`` clamp keeps >= 1 visible token, which is the
        physio stream's only gradient path to the encoder); only the PLACEMENT
        is random per sample. That uniformity is what makes the batch-max
        visible gather in ``forward`` exact -- see
        ``code/SpanMask_PhysioSignals.md`` §4.3.
        """
        span = max(1, int(span))
        n_spans = max(1, int(n_spans))
        while n_spans > 1 and n_spans * span > N - 1:
            n_spans -= 1
        span = min(span, max(1, N - 1))
        total = min(n_spans * span, N - 1)

        # distribute the slack (positions the spans may start at) over the
        # n_spans + 1 gaps, so the spans never overlap and never run past the end
        slack = max(0, N - total)
        cut = torch.rand(B, n_spans + 1, device=device)
        cut = cut / cut.sum(dim=1, keepdim=True)
        gap = (cut * slack).long()                        # [B, n_spans + 1]
        offs = torch.cumsum(gap[:, :-1], dim=1)           # start offset of span j
        starts = offs + torch.arange(n_spans, device=device) * span
        idx = (starts.unsqueeze(-1)
               + torch.arange(span, device=device))       # [B, n_spans, span]
        m = torch.zeros(B, N, device=device, dtype=torch.long)
        m.scatter_(1, idx.reshape(B, -1), 1)
        return m

    def make_masks(self, B: int, device):
        out = {}
        for s in self.streams:
            if s in self.visual:
                out[s] = self._tube_mask(B, device, self.mask_ratios[s])
            elif self.physio_mask == 'span' and self.mask_span_tokens.get(s):
                out[s] = self._span_mask(B, device, self.n_signal,
                                         self.mask_n_spans[s],
                                         self.mask_span_tokens[s])
            else:
                out[s] = self._random_mask(B, device, self.n_signal,
                                           self.mask_ratios[s])
        return out

    # ------------------------------------------------------------------ #
    # reconstruction targets (normalized flat patches)
    # ------------------------------------------------------------------ #
    def _targets(self, x, stream: str):
        if stream in self.visual:
            B, C, T, H, W = x.shape
            t, ph, pw = self.tubelet
            Gt = T // t
            Gh, Gw = H // ph, W // pw
            # tokens indexed by conv flatten order: (t, h, w)
            x = x.view(B, C, Gt, t, Gh, ph, Gw, pw)
            x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
            x = x.view(B, Gt * Gh * Gw, -1)          # [B, N, C*t*ph*pw]
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, unbiased=False, keepdim=True)
            return (x - mean) / torch.sqrt(var + _EPS)
        x = x.reshape(x.shape[0], self.n_signal, self.sig_kernel)
        if self.target_norm == 'clip':
            # ONE mean/std per (sample, stream) -- the flattening inverse of
            # this (tokens are consecutive non-overlapping windows) reproduces
            # exactly the z-scored clip waveform, which is what makes the
            # assembled prediction/target pair a coherent STFT input.
            mean = x.mean(dim=(1, 2), keepdim=True)
            var = x.var(dim=(1, 2), unbiased=False, keepdim=True)
            return (x - mean) / torch.sqrt(var + _EPS)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + _EPS)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resize_t(x, n: int):
        """Slice the time dim (dim 2) of a [B, C, T, ...] tensor to ``n``.

        Kept for backward compatibility; the analysis-critical behaviour lives
        in :meth:`_fit_time`, which REFUSES to pad (padding would re-warp the
        time axis against the other streams).
        """
        if x.shape[2] <= n:
            return x
        return x[:, :, :n]

    def forward(self, x):
        B = next(iter(x.values())).shape[0]
        device = next(iter(x.values())).device

        # Enforce the configured time/signal lengths. Both streams are trimmed
        # from the END, which keeps the shared time origin and therefore the
        # tubelet <-> signal token alignment; a too-short stream raises.
        xin = {}
        for s in self.streams:
            n = self.num_frames if s in self.visual else self.seq_len
            xin[s] = self._fit_time(x[s], n, s)

        # --- tokenize + add positional embedding -------------------------- #
        tokens = {}
        for s in self.streams:
            tok = self.adapters[s](xin[s])           # [B, N, D]
            tok = tok + self.positions[s].pos_embed
            tokens[s] = tok

        # --- per-stream asymmetric masks ---------------------------------- #
        masks = self.make_masks(B, device)

        # --- gather visible tokens of every stream ------------------------ #
        enc_parts = []
        info = {}
        for s in self.streams:
            mask_s = masks[s]                           # [B, N] 1 = masked
            ids_shuffle = torch.argsort(mask_s, dim=1, stable=True)
            k = int((mask_s == 0).sum(dim=1).max())     # visible per sample
            ids_keep = ids_shuffle[:, :k]
            tok = tokens[s]
            part = torch.gather(
                tok, 1, ids_keep.unsqueeze(-1).expand(B, k, tok.shape[2]))
            enc_parts.append(part)
            info[s] = (mask_s, ids_shuffle, k, tokens[s].shape[1])
        z = torch.cat(enc_parts, dim=1)                 # [B, sum_k, D]

        # --- shared encoder -------------------------------------------------
        for blk in self.enc_blocks:
            z = blk(z)
        z = self.enc_norm(z)

        # --- per-stream decode + masked reconstruction loss -----------------
        losses = {}          # per-modality weighted MSE (+ spectral) sum
        losses_mse = {}      # per-modality raw masked MSE (before weighting)
        losses_spectral = {}  # per-modality raw MR-STFT (only where enabled)
        preds = {}
        start = 0
        for s in self.streams:
            mask_s, ids_shuffle, k, n = info[s]
            enc_s = z[:, start:start + k]
            start += k
            ids_restore = torch.argsort(ids_shuffle, dim=1)     # [B, N]
            dec = torch.cat([enc_s, self.positions[s].mask_token
                             .expand(B, n - k, self.embed_dim)], dim=1)
            dec = torch.gather(dec, 1,
                               ids_restore.unsqueeze(-1).expand(B, n, dec.shape[2]))
            dec = dec + self.positions[s].pos_embed
            for blk in self.dec_blocks:
                dec = blk(dec)
            pred = self.heads[s](dec)                    # [B, N, flat]
            tgt = self._targets(xin[s], s)               # [B, N, flat]
            # masked MSE on this modality's MASKED positions only, then scaled
            # by its per-modality weight (losses[s] = weighted contribution)
            losses_mse[s] = self.loss_fn(pred, tgt, mask_s)
            contrib = self.loss_weights[s] * losses_mse[s]
            if s in self._spectral_streams:
                # Periodicity term on the ASSEMBLED clip waveform. SignalEmbed
                # uses kernel == stride, so the tokens are consecutive
                # non-overlapping windows and [B, N, sig_kernel] ->
                # [B, n_signal*sig_kernel] IS the clip waveform (target_norm
                # 'clip' guarantees both sides share one affine convention).
                # float32: torch.stft is not implemented for half precision.
                pred_w = pred.reshape(pred.shape[0], -1).float()
                tgt_w = tgt.reshape(tgt.shape[0], -1).float()
                # windows are PER STREAM (see __init__ / parse_fft_sizes): the
                # module for ``s`` already carries the window set resolved for
                # ``s``, so a BP and a RESP stream in the same run are scored at
                # their own resolutions instead of one shared compromise.
                losses_spectral[s] = self.spectral_fns[s](pred_w, tgt_w)
                contrib = contrib + (self.spectral_weights[s]
                                     * losses_spectral[s])
            losses[s] = contrib
            preds[s] = pred

        total = torch.stack(list(losses.values())).sum()
        return {'loss': total, 'losses': losses, 'losses_mse': losses_mse,
                'losses_spectral': losses_spectral, 'masks': masks,
                'preds': preds}


# --------------------------------------------------------------------------- #
# builder (mirrors run_pretrain argparse/YAML defaults)
# --------------------------------------------------------------------------- #
#: default span length (SECONDS) per 1-D stream, used when ``mask_span_s`` does
#: not name the stream. Anchored on the target's own period/correlation time so
#: that the gap is no longer fillable from its visible edges (see
#: ``code/SpanMask_PhysioSignals.md``): bp 1-2.5 Hz -> 1.0 s, resp 0.16-0.4 Hz
#: -> 4.0 s, eda aperiodic (2-10 s tonic scale) -> 8.0 s.
MASK_SPAN_DEFAULTS = {'bp': 1.0, 'resp': 4.0, 'eda': 8.0}


def parse_mask_span(spec) -> Dict[str, float]:
    """Normalise a ``mask_span_s`` argument to ``{stream: seconds}``.

    Accepts every form a runner/config can produce:

    * a number or numeric string (``4.0``, ``'4.0'``) -> keyed ``'*'``, i.e.
      every physio stream;
    * a mapping (YAML ``mask_span_s: {resp: 4.0, bp: 1.0}``);
    * a comma list of ``stream=seconds`` (``'resp=4.0,bp=1.0'``, the CLI form).

    ``None``/``''``/``0`` yield an empty mapping => per-stream defaults
    (:data:`MASK_SPAN_DEFAULTS`).
    """
    if spec is None:
        return {}
    if isinstance(spec, dict):
        return {str(k).strip(): float(v) for k, v in spec.items()}
    if isinstance(spec, (int, float)):
        return {'*': float(spec)}
    text = str(spec).strip()
    if not text:
        return {}
    if '=' in text:
        out = {}
        for part in text.split(','):
            part = part.strip()
            if not part:
                continue
            key, _, val = part.partition('=')
            out[key.strip()] = float(val)
        return out
    return {'*': float(text)}


#: default MR-STFT FFT window sizes (SAMPLES) per 1-D stream, applied when
#: ``spectral_fft_sizes`` is empty ("auto"). They mirror the Stage-3 finetune
#: configs ONE FOR ONE (``configs/finetune/{bp,resp,eda}.yaml`` -> the Stage-3
#: ``--fft_sizes``), so a given stream gets the SAME spectral objective in
#: Stage 2 and Stage 3 instead of the single global window set Stage 2 used to
#: force on every stream.
#:
#: Rationale (see ``code/TirROI_Resp_plan.md`` §9 and
#: ``code/SpanMask_PhysioSignals.md`` §4.1): a window must span at least ONE
#: period of the band it is meant to police, otherwise the magnitude term
#: measures local waveform shape rather than rate. At fs = 100 Hz:
#:   bp   1.0-2.5 Hz  -> period 0.4-1.0 s   -> 64/128/256 (1.56/0.78/0.39 Hz bins)
#:   resp 0.16-0.4 Hz -> period 2.5-6.25 s  -> 64/128/256 (the shipped Stage-3
#:        choice: it polices multi-scale SHAPE; 128/256/512 = 1.28/2.56/5.12 s
#:        is the ">= one breath per window" alternative -- if you switch, change
#:        it HERE *and* in configs/finetune/resp*.yaml so both stages stay
#:        aligned, and note that ``spec_resp`` is NOT comparable across sets)
#:   eda  aperiodic, tonic 2-10 s          -> 256/512/1024 (2.56/5.12/10.24 s)
#: ``eda`` needs a clip of >= 10.24 s: shorter clips DROP 1024 (logged) and a
#: clip shorter than 2.56 s leaves nothing and raises.
SPECTRAL_FFT_DEFAULTS = {'bp': (64, 128, 256), 'resp': (64, 128, 256),
                         'eda': (256, 512, 1024)}
#: windows for a physio stream the table above does not name
SPECTRAL_FFT_FALLBACK = (64, 128, 256)


def _window_set(v) -> Tuple[int, ...]:
    """``[128, 256]`` / ``'128/256'`` / ``'128,256'`` / ``256`` -> ``(128, 256)``."""
    if isinstance(v, (list, tuple, set)):
        items = list(v)
    elif isinstance(v, (int, float)):
        items = [v]
    else:
        items = [p for p in str(v).replace('/', ',').replace(';', ',')
                 .split(',') if p.strip()]
    try:
        sizes = sorted({int(round(float(x))) for x in items})
    except (TypeError, ValueError):
        raise ValueError(
            f'parse_fft_sizes: {v!r} is not a list of FFT window sizes')
    if not sizes or sizes[0] < 2:
        raise ValueError(
            f'parse_fft_sizes: {v!r} -> {sizes}; every FFT window must be an '
            f'integer number of samples >= 2')
    return tuple(sizes)


def parse_fft_sizes(spec) -> Dict[str, Tuple[int, ...]]:
    """Normalise a ``spectral_fft_sizes`` argument to ``{stream: windows}``.

    ``'*'`` means "every physio stream". Accepted forms:

    * ``None`` / ``''`` / ``'auto'`` -> ``{}``: the per-stream
      :data:`SPECTRAL_FFT_DEFAULTS` (the recommended form);
    * a plain comma string (``'64,128,256'``), a list/tuple, or a number ->
      keyed ``'*'``, i.e. the SAME windows for every physio stream -- the
      pre-2026-09 behaviour, kept so no existing config changes meaning;
    * a ``stream=w1/w2/...`` comma list (``'resp=128/256/512,bp=64/128/256'``,
      the CLI form: ``,`` separates STREAMS, ``/`` separates the windows of
      one stream -- a comma inside a value would be ambiguous);
    * a mapping (YAML ``spectral_fft_sizes: {resp: [128, 256, 512]}``); a value
      may also be a scalar or a string (``{resp: '128,256'}``).

    Window sets are de-duplicated and sorted ascending, so the printed order is
    the resolution order (coarsest bin first).
    """
    if spec is None:
        return {}
    if isinstance(spec, dict):
        return {str(k).strip(): _window_set(v) for k, v in spec.items()}
    if isinstance(spec, (list, tuple, set)):
        return {'*': _window_set(spec)}
    if isinstance(spec, (int, float)):
        return {'*': _window_set(spec)}
    text = str(spec).strip()
    if not text or text.lower() == 'auto':
        return {}
    if '=' in text:
        out = {}
        for part in text.split(','):
            part = part.strip()
            if not part:
                continue
            key, _, val = part.partition('=')
            if not val.strip():
                raise ValueError(
                    f'parse_fft_sizes: {part!r} names a stream but no '
                    f"windows. Use 'stream=128/256/512' -- the COMMA "
                    f"separates streams, '/' separates the windows of one "
                    f"stream.")
            out[key.strip()] = _window_set(val)
        if not out:
            raise ValueError(
                f'parse_fft_sizes: {text!r} does not name any stream')
        return out
    for part in text.split(','):
        if part.strip() and not part.strip().lstrip('+-').replace('.', '', 1).isdigit():
            raise ValueError(
                f'parse_fft_sizes: cannot parse {text!r}. Use either a plain '
                f'comma list of window sizes (64,128,256 = every physio '
                f"stream) or the per-stream form "
                f"(resp=128/256/512,bp=64/128/256).")
    return {'*': _window_set(text)}


def _fft_sizes_spec(args):
    """Resolve ``spectral_fft_sizes`` from a run_pretrain/args namespace.

    ``--fft_sizes`` is the **Stage-3 spelling** of the same knob (Stage 3's
    ``runners/run_waveform.py``), so a Stage-3 config line can be pasted into a
    Stage-2 config without silently doing nothing: it is accepted both as a CLI
    alias of ``--spectral_fft_sizes`` and as a YAML key (an unknown YAML key is
    absorbed by ``parser.set_defaults`` and would otherwise be ignored).

    Precedence: an explicit ``spectral_fft_sizes`` > ``fft_sizes`` > auto
    (:data:`SPECTRAL_FFT_DEFAULTS`). Returns ``None`` for auto, which is what
    :class:`MultiModalMAE` expects.
    """
    spec = getattr(args, 'spectral_fft_sizes', None)
    alias = getattr(args, 'fft_sizes', None)

    def _empty(v) -> bool:
        if v is None:
            return True
        if isinstance(v, str):
            return not v.strip() or v.strip().lower() == 'auto'
        return isinstance(v, (list, tuple, set)) and not len(v)

    if _empty(spec):
        spec = alias
    elif not _empty(alias) and str(alias).strip() != str(spec).strip():
        print(f"[spectral] both 'spectral_fft_sizes' ({spec!r}) and the "
              f"Stage-3 alias 'fft_sizes' ({alias!r}) are set and differ; "
              f'using spectral_fft_sizes')
    return None if _empty(spec) else spec


def _window_summary(sizes, fs: float) -> str:
    """``[64, 128]`` at fs 100 -> ``'64,128 samples = 0.64/1.28 s -> ...'``."""
    return (', '.join(str(n) for n in sizes)
            + ' samples = ' + '/'.join(f'{n / fs:.2f}' for n in sizes) + ' s'
            + ' -> ' + '/'.join(f'{fs / n:.2f}' for n in sizes) + ' Hz bins')


def _parse_int_csv(v, dtype=int):
    return tuple(dtype(x) for x in str(v).split(','))


def build_pretraining_model(args):
    """Construct the multimodal MAE from a run_pretrain ``args`` namespace."""
    streams = tuple(x.strip() for x in
                    str(getattr(args, 'streams', 'rgb,tir,bp')).split(',')
                    if x.strip())
    if not streams:
        raise ValueError(
            'build_pretraining_model: --streams must name at least two '
            'modalities (>=1 video rgb/tir and >=1 physiological 1-D '
            'bp/resp/eda).')
    if not any(s in _VISUAL_STREAMS for s in streams) or \
            not any(s in _SIGNAL_STREAMS for s in streams):
        raise ValueError(
            'build_pretraining_model: Stage-2 needs >=1 video stream '
            f'({", ".join(_VISUAL_STREAMS)}) AND >=1 physiological 1-D '
            f'stream ({", ".join(_SIGNAL_STREAMS)}); got --streams '
            f'"{",".join(streams)}".')
    tubelet = _parse_int_csv(getattr(args, 'tubelet', '2,16,16'))

    clip_duration = float(getattr(args, 'clip_duration', 4.0))
    fps = float(getattr(args, 'fps', 25.0))
    fs = float(getattr(args, 'fs', 100.0))
    # temporal decimation inside the window (1 = every frame). MUST mirror the
    # dataset's alignment.plan_clip so the video token geometry matches clip T.
    temporal_stride = max(1, int(getattr(args, 'temporal_stride', 1) or 1))
    num_frames = max(1, int(round(clip_duration * fps / temporal_stride)))
    # the Conv3d tubelet partitions time into ``t`` frames, so round the clip
    # length DOWN to a multiple of t (e.g. stride 4 @ 25 fps / 4 s -> 25 -> 24).
    # The dropped tail shortens the KEPT window, so the signal length must be
    # derived from the kept duration below -- otherwise the video and the 1-D
    # signals would cover different time spans (checked hard in the model).
    if num_frames % tubelet[0]:
        print(f'[geometry] num_frames {num_frames} is not a multiple of '
              f'tubelet_t {tubelet[0]}: keeping the first '
              f'{num_frames - num_frames % tubelet[0]} frames '
              f'({(num_frames - num_frames % tubelet[0]) * temporal_stride / fps:.3f} s)')
        num_frames -= num_frames % tubelet[0]
    num_frames = max(tubelet[0], num_frames)
    kept_seconds = num_frames * temporal_stride / fps
    seq_len = int(getattr(args, 'seq_len', 0) or 0) or max(
        1, int(round(kept_seconds * fs)))

    ratios = {'rgb': float(getattr(args, 'mask_ratio_rgb', 0.75)),
              'tir': float(getattr(args, 'mask_ratio_tir', 0.50)),
              'bp': float(getattr(args, 'mask_ratio_bp', 0.90)),
              'resp': float(getattr(args, 'mask_ratio_resp', 0.90)),
              'eda': float(getattr(args, 'mask_ratio_eda', 0.90))}

    # per-modality masked-MSE weights. ``signal_weight`` gives every physio
    # (non-visual) stream the same lambda (visual streams stay 1.0); a
    # non-empty ``loss_weights`` comma string (ONE value per stream, in
    # --streams order) overrides the whole policy per stream.
    signal_weight = float(getattr(args, 'signal_weight', 0.5))
    w_raw = str(getattr(args, 'loss_weights', '') or '').strip()
    loss_weights = None
    if w_raw:
        vals = [float(x.strip()) for x in w_raw.split(',') if x.strip()]
        if vals:
            assert len(vals) == len(streams), (
                f'--loss_weights expects one value per stream '
                f'({len(streams)}: {streams}), got {len(vals)}')
            loss_weights = dict(zip(streams, vals))

    # spectral (MR-STFT magnitude) term on the 1-D physio streams. OFF by
    # default (0.0) => the masked-MSE-only objective is reproduced exactly.
    # ``spectral_weight`` is the per-physio-stream lambda (video streams are
    # forced to 0.0); a non-empty ``spectral_weights`` comma string (ONE value
    # per stream, in --streams order) overrides it, like --loss_weights.
    spectral_weight = float(getattr(args, 'spectral_weight', 0.0) or 0.0)
    s_raw = str(getattr(args, 'spectral_weights', '') or '').strip()
    spectral_weights = None
    if s_raw:
        vals = [float(x.strip()) for x in s_raw.split(',') if x.strip()]
        if vals:
            assert len(vals) == len(streams), (
                f'--spectral_weights expects one value per stream '
                f'({len(streams)}: {streams}), got {len(vals)}')
            spectral_weights = dict(zip(streams, vals))

    # ViT geometry: the --model variant name sets it, and an explicit (>0)
    # enc_* flag/YAML value overrides it (ablation escape hatch).
    geom = multimae_variant(getattr(args, 'model', '')) or {}

    def _geo(arg_name: str, geom_key: str, default: int) -> int:
        return int(getattr(args, arg_name, 0) or 0) or geom.get(geom_key, default)

    return MultiModalMAE(
        streams=streams,
        stream_channels={'tir': int(getattr(args, 'tir_channels', 3))},
        embed_dim=_geo('enc_embed_dim', 'embed_dim', 768),
        enc_depth=_geo('enc_depth', 'enc_depth', 12),
        enc_num_heads=_geo('enc_num_heads', 'enc_num_heads', 12),
        mlp_ratio=float(getattr(args, 'mlp_ratio', 4.0)),
        dec_depth=int(getattr(args, 'dec_depth', 2)),
        tubelet=tubelet,
        input_size=int(getattr(args, 'input_size', 64)),
        num_frames=num_frames,
        sig_kernel=int(getattr(args, 'sig_kernel', 8)),
        seq_len=seq_len,
        fps=fps, fs=fs, temporal_stride=temporal_stride,
        mask_ratios=ratios,
        signal_weight=signal_weight,
        loss_weights=loss_weights,
        pos_init=str(getattr(args, 'pos_init', 'sincos3d')),
        target_norm=str(getattr(args, 'target_norm', 'token')),
        spectral_weight=spectral_weight,
        spectral_weights=spectral_weights,
        spectral_fft_sizes=_fft_sizes_spec(args),
        physio_mask=str(getattr(args, 'physio_mask', 'random') or 'random'),
        mask_span_s=getattr(args, 'mask_span_s', None),
        spectral_hop_ratio=float(getattr(args, 'spectral_hop_ratio', 0.25)))


# --------------------------------------------------------------------------- #
# weight inheritance: load an ImageNet / MAE / VideoMAE ViT encoder into the
# shared encoder of a MultiModalMAE (space-time priors, Stage 1 -> Stage 2)
# ---------------------------------------------------------------------------
#: HF ``transformers`` VideoMAE block sub-module -> ``core.blocks.Block`` name.
#: ``layernorm_before``/``layernorm_after`` are the pre-attention/pre-MLP norms
#: (= MAE's ``norm1``/``norm2``); Q/K/V are fused by
#: :func:`canonicalise_vit_state_dict`.
_VIDEOMAE_BLOCK_MAP = (
    ('layernorm_before.', 'norm1.'),
    ('layernorm_after.', 'norm2.'),
    ('attention.output.dense.', 'attn.proj.'),
    ('intermediate.dense.', 'mlp.fc1.'),
    ('output.dense.', 'mlp.fc2.'),
)

#: The five per-layer attention parameters of the HF VideoMAE layout (Q/K/V are
#: bias-free there and the bias is carried by separate ``q_bias``/``v_bias``).
_VIDEOMAE_QKV_PARTS = ('query', 'key', 'value', 'q_bias', 'v_bias')


def canonicalise_vit_state_dict(state: Dict[str, 'torch.Tensor']):
    """Normalise a Stage-1 checkpoint to the MAE-style key layout.

    Two layouts are understood, and both come out as ``blocks.N.*`` /
    ``norm.*`` / ``patch_embed.proj.*`` (which is what
    :func:`load_pretrained_encoder` and
    ``core.au_probe.load_au_probe_weights`` map from):

    * **official MAE / VideoMAE** (``dl.fbaipublicfiles.com``, the MCG-NJU
      releases) -- already canonical, returned unchanged.
    * **HF ``transformers`` VideoMAE** (the ``MCG-NJU/videomae-*`` Hub repos,
      a ``VideoMAEForPreTraining`` bundle) -- keys look like
      ``videomae.encoder.layer.0.layernorm_before.weight`` and Q/K/V are three
      separate bias-free projections plus ``q_bias``/``v_bias``. Those are
      fused back into one ``attn.qkv`` here using MAE's convention of a ZERO
      key bias (``bias = [q_bias, 0, v_bias]``). The decoder half of the
      pre-training bundle (``decoder.*``, ``encoder_to_decoder``,
      ``mask_token``, ``position_embeddings``) is dropped -- the encoder is the
      only part Stage 1 wants.

    :param state: loaded checkpoint state dict (tensors only).
    :return: ``(state, layout)`` with ``layout`` in ``{'mae', 'videomae-hf'}``.
    """
    if not any(k.startswith('videomae.') for k in state):
        return dict(state), 'mae'

    out: Dict[str, 'torch.Tensor'] = {}
    qkv: Dict[str, Dict[str, 'torch.Tensor']] = {}
    for key, v in state.items():
        name = key[len('videomae.'):] if key.startswith('videomae.') else key
        if name.startswith('embeddings.patch_embeddings.projection.'):
            # Conv3d tubelet filter -- same role as MAE's patch_embed.proj
            out['patch_embed.proj.' + name.rsplit('.', 1)[-1]] = v
        elif name in ('layernorm.weight', 'layernorm.bias'):
            out['norm.' + name.rsplit('.', 1)[-1]] = v
        elif name.startswith('encoder.layer.'):
            idx, _, tail = name[len('encoder.layer.'):].partition('.')
            mapped = False
            for src, dst in _VIDEOMAE_BLOCK_MAP:
                if tail.startswith(src):
                    out[f'blocks.{idx}.{dst}{tail[len(src):]}'] = v
                    mapped = True
                    break
            if not mapped:
                for part in _VIDEOMAE_QKV_PARTS:
                    if tail in (f'attention.attention.{part}',
                                f'attention.attention.{part}.weight'):
                        qkv.setdefault(idx, {})[part] = v
                        break
        # decoder.* / encoder_to_decoder / mask_token / position_embeddings:
        # dropped on purpose (not part of the encoder we reuse).

    for idx, parts in qkv.items():
        if all(p in parts for p in ('query', 'key', 'value')):
            out[f'blocks.{idx}.attn.qkv.weight'] = torch.cat(
                [parts['query'], parts['key'], parts['value']], dim=0)
        if 'q_bias' in parts and 'v_bias' in parts:
            out[f'blocks.{idx}.attn.qkv.bias'] = torch.cat(
                [parts['q_bias'], torch.zeros_like(parts['v_bias']),
                 parts['v_bias']], dim=0)
    return out, 'videomae-hf'


def fit_visual_patch_embed(model: 'MultiModalMAE',
                           tensor: 'torch.Tensor', dst_key: str,
                           target_shape, inflate: bool = True):
    """Adapt an RGB patch-embed tensor to this model's tubelet ``Conv3d``.

    * **5-D** ``[out, in, t, ph, pw]`` (VideoMAE): copied **verbatim**. Its
      temporal kernel is real prior knowledge -- re-averaging it would throw
      away exactly the part a 2-D source cannot provide, and with
      ``tubelet 2,16,16`` the shape matches the model's adapter exactly. A
      kernel that does NOT equal the model's ``tubelet`` raises, because the
      inherited temporal filter would then be meaningless; that case used to be
      counted as a quiet shape mismatch instead.
    * **4-D** ``[out, in, ph, pw]`` (MAE / timm): the only honest
      conversion is a boxcar -- broadcast along the tubelet axis and average.
      The model is then motion-blind at init (the documented Stage-1
      limitation), which is the whole reason a 3-D source is preferred.

    :param target_shape: the destination tensor's shape (``cur[dst_key].shape``).
    :param inflate: set ``False`` to skip the 4-D conversion entirely (ablation
        knob, ``--inflate_rgb_patch 0``): the adapter then stays random.
    :return: the tensor to store, or ``None`` when it cannot be adapted.

    .. note::
       This is for the conv **weight** only. The ``Conv3d`` bias is a plain
       ``[out]`` vector that needs no adaptation, so callers must not route it
       through here (a 1-D tensor would be rejected as unadaptable).
    """
    if tensor.ndim == 5:
        tubelet = tuple(int(x) for x in model.tubelet)
        if tuple(tensor.shape[2:]) != tubelet:
            raise ValueError(
                f'{dst_key}: checkpoint tubelet kernel '
                f'{tuple(tensor.shape[2:])} != model tubelet {tubelet}. Set '
                f'`tubelet` to the source geometry (VideoMAE and this repo\'s '
                f'default are 2,16,16) so the inherited temporal filter stays '
                f'meaningful.')
        return tensor
    if tensor.ndim == 4:
        if not inflate or len(tuple(target_shape)) != 5:
            return None
        t = int(tuple(model.tubelet)[0])
        # [out, in, ph, pw] -> [out, in, t, ph, pw], averaged over the tube
        return tensor.unsqueeze(2).expand(-1, -1, t, -1, -1).contiguous() / t
    return None


def load_pretrained_encoder(model: 'MultiModalMAE', path: str,
                            inflate_rgb_patch: bool = True,
                            patch_stream: str = '') -> Dict[str, int]:
    """Copy a (MAE / VideoMAE / timm) checkpoint's transformer weights in.

    Two source layouts are accepted (see :func:`canonicalise_vit_state_dict`),
    after which the mapping is always the same::

        blocks.{i}.*  -> enc_blocks.{i}.*
        norm.*        -> enc_norm.*
        patch_embed.proj.{weight,bias} -> adapters.rgb.patch_embed.*
                    (copied VERBATIM for a 3-D VideoMAE source; inflated from a
                    2-D Conv2d for MAE/timm -- see
                    :func:`fit_visual_patch_embed`)

    A VideoMAE source is preferred over a plain ImageNet/MAE one: its
    ``patch_embed.proj`` IS a ``Conv3d(3, D, (2,16,16))`` tubelet filter, so the
    temporal prior transfers instead of being faked by averaging two frames,
    and its objective (tube-masked video MAE) matches Stage 2.

    Everything else (cls_token / pos_embed / head / other adapters / signal
    streams) is left at its random initialisation. Encoder geometry must match
    the checkpoint (e.g. embed_dim=768, depth=12, heads=12 for ViT-Base).

    :param patch_stream: which VISUAL stream receives the tokenizer. ``''``
        (default) resolves to ``rgb`` when that adapter exists and otherwise to
        the first available visual adapter. So an ``rgb,...`` run behaves
        exactly as before, while a ``tir,...`` run (Stage 2 with the thermal ROI
        as the only visual stream) inherits the VideoMAE tubelet into
        ``adapters.tir`` instead of leaving its tokenizer random -- the
        checkpoint's patch embed used to be routed at a non-existent
        ``adapters.rgb`` and silently counted as skipped.
    :return: counts dict {loaded, skipped, shape_mismatch, layout,
        patch_stream}.
    """
    import torch

    ckpt = torch.load(path, map_location='cpu')
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ('model', 'state_dict', 'module'):
            if isinstance(ckpt.get(key), dict):
                state = ckpt[key]
                break
    if isinstance(state, dict) and isinstance(state.get('module'), dict):
        state = state['module']
    # MAE/VideoMAE-on-disk or the HF transformers VideoMAE layout -> MAE keys
    state, layout = canonicalise_vit_state_dict(state)

    cur = model.state_dict()
    new_state = {}
    loaded, skipped, shape_mismatch = [], [], []

    # --- which visual adapter receives the patch embed? -------------------- #
    if not patch_stream:
        for vis in _VISUAL_STREAMS:
            if f'adapters.{vis}.patch_embed.weight' in cur:
                patch_stream = vis
                break
    dst_patch = (f'adapters.{patch_stream}.patch_embed' if patch_stream
                 else 'adapters.rgb.patch_embed')

    # --- geometry guard (avoids silently loading nothing) ------------------ #
    src_qkv = state.get('blocks.0.attn.qkv.weight')
    tgt_qkv = cur.get('enc_blocks.0.attn.qkv.weight')
    if src_qkv is not None and tgt_qkv is not None \
            and tuple(src_qkv.shape) != tuple(tgt_qkv.shape):
        raise ValueError(
            f'Encoder geometry mismatch: checkpoint ViT embed dim '
            f'{src_qkv.shape[0]} != model enc_embed_dim {tgt_qkv.shape[0]}. '
            f'Set enc_embed_dim=768, enc_depth=12, enc_num_heads=12 for '
            f'ViT-Base (or match dims/depth/heads to the checkpoint).')

    for src_key, v in state.items():
        dst_key = None
        if src_key.startswith('blocks.'):
            dst_key = 'enc_blocks.' + src_key[len('blocks.'):]
        elif src_key.startswith('norm.'):
            dst_key = 'enc_norm.' + src_key[len('norm.'):]
        elif src_key in ('patch_embed.proj.weight', 'patch_embed.proj.bias'):
            name = src_key.rsplit('.', 1)[-1]
            dst_key = f'{dst_patch}.{name}'
            if name == 'weight' and dst_key in cur:
                # 3-D source -> verbatim; 2-D source -> boxcar inflation (or
                # left random when inflate_rgb_patch=False). Raises on a
                # tubelet-geometry mismatch rather than skipping quietly.
                v = fit_visual_patch_embed(model, v, dst_key,
                                           cur[dst_key].shape,
                                           inflate_rgb_patch)
                if v is None:
                    shape_mismatch.append(src_key)
                    continue
        # cls_token / pos_embed / head / others: intentionally not loaded
        if dst_key is None:
            continue
        if dst_key not in cur:
            skipped.append(src_key)
            continue
        if tuple(cur[dst_key].shape) != tuple(v.shape):
            shape_mismatch.append(src_key)
            continue
        new_state[dst_key] = v
        loaded.append(src_key)

    n_loaded = len(loaded)
    model.load_state_dict(new_state, strict=False)
    print(f'[pretrained] {path}: loaded {n_loaded} encoder tensors '
          f'(layout {layout}, patch embed -> '
          f'adapters.{patch_stream or "?"}), {len(skipped)} skipped, '
          f'{len(shape_mismatch)} shape-mismatched.')
    if shape_mismatch:
        print(f'[pretrained] first shape-mismatched keys: '
              f'{shape_mismatch[:5]}')
    return {'loaded': n_loaded, 'skipped': len(skipped),
            'shape_mismatch': len(shape_mismatch), 'layout': layout,
            'patch_stream': patch_stream}


# --------------------------------------------------------------------------- #
# registered entrypoints -- the NAME implies the geometry, exactly like the
# project_vit_* family in core/model.py (so `models.list_models()` lists them
# and `create_model('project_multimae_base')` builds a 768-d model).
# run_pretrain.py builds through build_pretraining_model(args), which resolves
# the same names from the run's args.
# --------------------------------------------------------------------------- #
def _build_multimae(geom: Dict[str, int], **kwargs):
    """``MultiModalMAE`` from a variant geometry + runtime kwargs."""
    kwargs.setdefault('streams', ('rgb', 'tir', 'bp'))
    return MultiModalMAE(embed_dim=geom['embed_dim'],
                         enc_depth=geom['enc_depth'],
                         enc_num_heads=geom['enc_num_heads'], **kwargs)


@register_model
def project_multimae_small(**kwargs):
    """384-d / 12-layer / 6-head (no Stage-1 checkpoint is shipped)."""
    return _build_multimae(MULTIMAE_VARIANTS['project_multimae_small'],
                           **kwargs)


@register_model
def project_multimae_base(**kwargs):
    """768-d / 12-layer / 12-head (MAE ViT-Base geometry)."""
    return _build_multimae(MULTIMAE_VARIANTS['project_multimae_base'], **kwargs)


@register_model
def project_multimae_large(**kwargs):
    """1024-d / 24-layer / 16-head (MAE ViT-Large geometry)."""
    return _build_multimae(MULTIMAE_VARIANTS['project_multimae_large'],
                           **kwargs)


@register_model
def project_multimae_huge(**kwargs):
    """1280-d / 32-layer / 16-head (no Stage-1 checkpoint is shipped)."""
    return _build_multimae(MULTIMAE_VARIANTS['project_multimae_huge'],
                           **kwargs)


# --------------------------------------------------------------------------- #
# masking self-test -- the span-mask invariants fail SILENTLY, so they are
# checked by a runnable self-test rather than by reading the code
# --------------------------------------------------------------------------- #
def _tiny_model(**over):
    """Smallest MultiModalMAE that satisfies the space-time alignment check.

    20 frames @25 fps = 0.8 s -> grid_t 10 (= 10 resp tokens of 8 samples at
    100 Hz), 1 spatial patch at input_size 16, tubelet_t 2 -> 0.08 s per token.
    """
    kw = dict(streams=('tir', 'resp'), embed_dim=64, enc_depth=1,
              enc_num_heads=4, dec_depth=1, mlp_ratio=4.0,
              num_frames=20, input_size=16, sig_kernel=8, seq_len=80,
              fps=25.0, fs=100.0, temporal_stride=1,
              mask_ratios={'tir': 0.5, 'resp': 0.5})
    kw.update(over)
    return MultiModalMAE(**kw)


def _longest_run(row) -> int:
    best = cur = 0
    for v in row:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def mask_self_test(verbose: bool = True) -> int:
    """Verify the 1-D masking invariants. Returns the number of failures.

    Covers (see ``code/SpanMask_PhysioSignals.md`` §4.3-§4.5):

    * ``_span_mask`` produces exactly ``min(n_spans * span, N - 1)`` masked
      tokens, **the same count for every sample of the batch** -- the batch-max
      visible gather in :meth:`MultiModalMAE.forward` silently leaks masked
      tokens into the encoder when counts differ;
    * runs are contiguous, at least ``span`` long, and never cover the whole
      stream (>= 1 visible token = the physio stream's only encoder path);
    * the gathered visible set really contains no masked token;
    * ``mask_span_s`` parses from a number, a ``stream=seconds`` CSV and a
      mapping, defaults per stream, and clamps an impossible span;
    * the default ``physio_mask='random'`` leaves the span tables empty (the
      historical path is untouched).
    """
    fails = []

    def check(name, cond, extra=''):
        if verbose:
            print(f'  [{"ok" if cond else "FAIL"}] {name}'
                  f'{(" -- " + str(extra)) if extra != "" else ""}')
        if not cond:
            fails.append(name)

    m = _tiny_model(physio_mask='span', mask_span_s={'resp': 0.32})
    N = m.n_signal
    if verbose:
        print(f'mask_self_test: N={N} tokens, span={m.mask_span_tokens}, '
              f'n_spans={m.mask_n_spans}')
    check('0.32 s span -> 4 tokens at fs 100 / sig_kernel 8',
          m.mask_span_tokens.get('resp') == 4, m.mask_span_tokens)

    dev = torch.device('cpu')
    for n_spans, span in ((1, 4), (2, 2), (1, 1)):
        B = 8
        mask = m._span_mask(B, dev, N, n_spans, span)
        exp = min(n_spans * span, N - 1)
        counts = [int(c) for c in mask.sum(dim=1)]
        check(f'span {n_spans}x{span}: count {exp} for all {B} samples',
              all(c == exp for c in counts), counts)
        check(f'span {n_spans}x{span}: runs >= span',
              all(_longest_run(mask[b].tolist()) >= span or exp == 0
                  for b in range(B)))
        check(f'span {n_spans}x{span}: <= N-1 masked (>= 1 visible)',
              all(c <= N - 1 for c in counts))
        # the encoder gather must not pick up a masked token
        ids = torch.argsort(mask, dim=1, stable=True)
        k = int((mask == 0).sum(dim=1).max())
        check(f'span {n_spans}x{span}: visible gather (k={k}) has no masked token',
              int(torch.gather(mask, 1, ids[:, :k]).sum()) == 0)

    check('scattered mask is NOT contiguous (contrast)',
          _longest_run(m._random_mask(1, dev, N, 0.5)[0].tolist()) < N // 2)

    # clamping + per-stream independence + spec forms
    m_big = _tiny_model(physio_mask='span', mask_span_s={'resp': 99.0})
    check('impossible span clamps to N-1', m_big.mask_span_tokens['resp'] == N - 1,
          m_big.mask_span_tokens)
    m_two = _tiny_model(streams=('tir', 'bp', 'resp'), physio_mask='span',
                        mask_span_s={'bp': 0.16, 'resp': 0.32},
                        mask_ratios={'tir': 0.5, 'bp': 0.5, 'resp': 0.5})
    check('per-stream spans differ (bp 2 tokens, resp 4)',
          m_two.mask_span_tokens == {'bp': 2, 'resp': 4}, m_two.mask_span_tokens)
    for form, label in ((0.32, 'number'), ('0.32', 'numeric string'),
                        ('resp=0.32', 'stream=seconds CSV'),
                        ({'resp': 0.32}, 'mapping')):
        mm = _tiny_model(physio_mask='span', mask_span_s=form)
        check(f'mask_span_s as {label} -> 4 tokens',
              mm.mask_span_tokens.get('resp') == 4)
    check('MASK_SPAN_DEFAULTS covers bp/resp/eda',
          set(MASK_SPAN_DEFAULTS) == {'bp', 'resp', 'eda'}, MASK_SPAN_DEFAULTS)

    # the historical path: no physio_mask -> scattered, no span state
    m_rand = _tiny_model()
    check("default physio_mask is 'random'", m_rand.physio_mask == 'random')
    check('random mode keeps the span tables empty',
          not m_rand.mask_span_tokens and not m_rand.mask_n_spans)
    check('random mode still masks round(ratio*N) tokens',
          int(m_rand.make_masks(4, dev)['resp'][0].sum()) == int(0.5 * N))

    if verbose:
        print('mask_self_test: ' + ('ALL PASS' if not fails
                                    else f'{len(fails)} FAILURE(S): {fails}'))
    return len(fails)


def spectral_self_test(verbose: bool = True) -> int:
    """Verify the PER-STREAM MR-STFT window plumbing. Returns failure count.

    The window resolution fails in the worst way when it is wrong: a wrong set
    still trains and still logs a plausible ``spec_<stream>``. So the
    invariants are checked by a runnable self-test -- the spec forms, the
    per-modality defaults, the ``<= clip`` filter, module sharing, and
    (crucially) that two streams with different windows really are scored at
    different resolutions instead of through one shared module.
    """
    from types import SimpleNamespace
    fails = []

    def check(name, cond, extra=''):
        if verbose:
            print(f'  [{"ok" if cond else "FAIL"}] {name}'
                  f'{(" -- " + str(extra)) if extra != "" else ""}')
        if not cond:
            fails.append(name)

    if verbose:
        print('spectral_self_test:')

    # ---- spec parsing ----------------------------------------------------- #
    for form, label, want in (
            (None, 'None -> auto', {}),
            ('', 'empty -> auto', {}),
            ('auto', "'auto' -> auto", {}),
            ('64,128,256', "comma string -> '*'", {'*': (64, 128, 256)}),
            ([64, 128, 256], "list -> '*'", {'*': (64, 128, 256)}),
            (256, "scalar -> '*'", {'*': (256,)}),
            ('256,64,128,64', 'sorted + de-duplicated', {'*': (64, 128, 256)}),
            ('resp=128/256/512,bp=64/128/256', 'CLI per-stream',
             {'resp': (128, 256, 512), 'bp': (64, 128, 256)}),
            ('resp=256', 'CLI single window', {'resp': (256,)}),
            ({'resp': [128, 256, 512]}, 'mapping of lists',
             {'resp': (128, 256, 512)}),
            ({'resp': '128/256'}, 'mapping of strings', {'resp': (128, 256)}),
            ({'resp': '128,256', 'bp': 64}, 'mixed mapping values',
             {'resp': (128, 256), 'bp': (64,)})):
        got = parse_fft_sizes(form)
        check(f'parse_fft_sizes {label} -> {want}', got == want, got)
    for bad, label in (('resp=128,256', 'a comma inside a per-stream value'),
                       ('resp=', 'a stream without windows'),
                       ('resp=1', 'a window < 2 samples'),
                       ('resp=128/abc', 'a non-numeric window')):
        try:
            parse_fft_sizes(bad)
            raised = False
        except ValueError:
            raised = True
        check(f'parse_fft_sizes rejects {label}', raised, repr(bad))

    check('SPECTRAL_FFT_DEFAULTS covers bp/resp/eda',
          set(SPECTRAL_FFT_DEFAULTS) == {'bp', 'resp', 'eda'},
          SPECTRAL_FFT_DEFAULTS)
    check('per-modality defaults mirror configs/finetune/*.yaml --fft_sizes',
          SPECTRAL_FFT_DEFAULTS == {'bp': (64, 128, 256),
                                    'resp': (64, 128, 256),
                                    'eda': (256, 512, 1024)},
          SPECTRAL_FFT_DEFAULTS)

    # ---- '--fft_sizes' = the Stage-3 alias of the same knob --------------- #
    def _ns(**kw):
        base = {'spectral_fft_sizes': '', 'fft_sizes': None}
        base.update(kw)
        return SimpleNamespace(**base)

    check('alias: an empty spectral_fft_sizes -> the fft_sizes value',
          _fft_sizes_spec(_ns(fft_sizes='64,128')) == '64,128')
    check('alias: an explicit spectral_fft_sizes wins',
          _fft_sizes_spec(_ns(spectral_fft_sizes='16,32',
                              fft_sizes='64,128')) == '16,32')
    check("alias: neither set -> None (=> per-modality defaults)",
          _fft_sizes_spec(_ns()) is None)
    check("alias: 'auto' -> None",
          _fft_sizes_spec(_ns(spectral_fft_sizes='auto')) is None)

    # ---- model-level resolution (toy clip: 80 samples = 0.8 s) ------------ #
    B = 2
    streams = ('tir', 'bp', 'resp')
    ratios = {'tir': 0.5, 'bp': 0.5, 'resp': 0.5}

    def _inputs(m):
        return {'tir': torch.randn(B, 3, m.num_frames, m.input_size,
                                   m.input_size),
                'bp': torch.randn(B, 1, m.seq_len),
                'resp': torch.randn(B, 1, m.seq_len)}

    m_uni = _tiny_model(streams=streams, mask_ratios=ratios,
                        target_norm='clip', spectral_weight=0.1,
                        spectral_fft_sizes='16,32')
    check('uniform spec -> the same windows for every physio stream',
          m_uni.spectral_fft_sizes == {'bp': [16, 32], 'resp': [16, 32]},
          m_uni.spectral_fft_sizes)
    check('uniform spec -> ONE shared module (the pre-2026-09 behaviour)',
          m_uni.spectral_fns['bp'] is m_uni.spectral_fns['resp'])
    check('video streams never get a spectral module',
          set(m_uni.spectral_fns) == set(m_uni.signal),
          set(m_uni.spectral_fns))
    out_uni = m_uni(_inputs(m_uni))
    check('uniform windows train: finite spec_bp/spec_resp',
          torch.isfinite(out_uni['loss'])
          and all(torch.isfinite(out_uni['losses_spectral'][s])
                  for s in m_uni.signal),
          {k: round(v.item(), 4)
           for k, v in out_uni['losses_spectral'].items()})

    m_sep = _tiny_model(streams=streams, mask_ratios=ratios,
                        target_norm='clip', spectral_weight=0.1,
                        spectral_fft_sizes='bp=8/16,resp=32/64')
    check('per-stream spec resolves independently (not one global set)',
          m_sep.spectral_fft_sizes == {'bp': [8, 16], 'resp': [32, 64]},
          m_sep.spectral_fft_sizes)
    check('different windows -> different modules',
          m_sep.spectral_fns['bp'] is not m_sep.spectral_fns['resp'])
    check('each module carries ITS OWN stream windows',
          m_sep.spectral_fns['bp'].fft_sizes == [8, 16]
          and m_sep.spectral_fns['resp'].fft_sizes == [32, 64],
          {s: m.fft_sizes for s, m in m_sep.spectral_fns.items()})
    probe = torch.randn(4, m_sep.n_signal * m_sep.sig_kernel)
    ref = probe.roll(3, dims=-1) * 0.7
    check('the two window sets really differ numerically on one waveform',
          not torch.allclose(m_sep.spectral_fns['bp'](probe, ref),
                             m_sep.spectral_fns['resp'](probe, ref)))
    check('hop_ratio is shared by every per-stream module',
          {m.hop_ratio for m in m_sep.spectral_fns.values()}
          == {m_sep.spectral_hop_ratio},
          {s: m.hop_ratio for s, m in m_sep.spectral_fns.items()})
    out_sep = m_sep(_inputs(m_sep))
    check('heterogeneous windows train: finite spec_bp/spec_resp',
          all(torch.isfinite(out_sep['losses_spectral'][s]) for s in m_sep.signal),
          {k: round(v.item(), 4)
           for k, v in out_sep['losses_spectral'].items()})
    check('total = sum of the per-stream contributions',
          torch.allclose(out_sep['loss'],
                         torch.stack(list(out_sep['losses'].values())).sum()))

    m_drop = _tiny_model(streams=streams, mask_ratios=ratios,
                         target_norm='clip', spectral_weight=0.1,
                         spectral_fft_sizes='16,1024')
    check('windows longer than the clip are DROPPED, not fatal',
          m_drop.spectral_fft_sizes == {'bp': [16], 'resp': [16]},
          m_drop.spectral_fft_sizes)
    try:
        _tiny_model(streams=streams, mask_ratios=ratios, target_norm='clip',
                    spectral_weight=0.1, spectral_fft_sizes='1024,2048')
        raised = False
    except ValueError:
        raised = True
    check('no window fits the clip -> raises instead of training blind',
          raised)

    # ---- auto spec: the per-modality table (needs a 10.24 s toy clip) ----- #
    m_auto = _tiny_model(streams=('tir', 'bp', 'resp', 'eda'),
                         mask_ratios=dict(ratios, eda=0.5),
                         num_frames=256, seq_len=1024,
                         target_norm='clip', spectral_weight=0.1,
                         spectral_fft_sizes='')
    check('empty spec -> SPECTRAL_FFT_DEFAULTS per stream',
          m_auto.spectral_fft_sizes == {'bp': [64, 128, 256],
                                        'resp': [64, 128, 256],
                                        'eda': [256, 512, 1024]},
          m_auto.spectral_fft_sizes)
    check('bp/resp share a module, eda gets its own',
          m_auto.spectral_fns['bp'] is m_auto.spectral_fns['resp']
          and m_auto.spectral_fns['eda'] is not m_auto.spectral_fns['bp'])
    try:
        _tiny_model(streams=('tir', 'eda'), target_norm='clip',
                    spectral_weight=0.1, spectral_fft_sizes='')
        raised = False
    except ValueError:
        raised = True
    check('eda defaults need a >= 2.56 s clip (raise on the 0.8 s toy clip)',
          raised)

    # ---- OFF: the historical objective, bit for bit ---------------------- #
    m_off = _tiny_model(streams=streams, mask_ratios=ratios)
    check('spectral OFF: no windows resolved and no modules built',
          m_off.spectral_fns == {} and m_off.spectral_fft_sizes == {})
    out_off = m_off(_inputs(m_off))
    check('spectral OFF: losses_spectral stays empty',
          out_off['losses_spectral'] == {} and torch.isfinite(out_off['loss']))
    check('spectral OFF: total is exactly the weighted masked MSE',
          torch.allclose(out_off['loss'], torch.stack(
              [m_off.loss_weights[s] * v
               for s, v in out_off['losses_mse'].items()]).sum()))

    # ---- the two guards on the term itself ------------------------------- #
    for kw, label in (({'streams': ('tir', 'bp'), 'spectral_weight': 0.1,
                        'target_norm': 'token'},
                       "spectral_weight > 0 with target_norm 'token'"),
                      ({'streams': ('tir', 'bp'), 'target_norm': 'clip',
                        'spectral_weights': {'tir': 0.1}},
                       'a positive weight on a VIDEO stream')):
        try:
            _tiny_model(**kw)
            raised = False
        except ValueError:
            raised = True
        check(f'still rejected: {label}', raised)

    # ---- the run_pretrain builder path ----------------------------------- #
    def _build_ns(**kw):
        base = dict(streams='tir,bp,resp', clip_duration=0.96, fps=25.0,
                    fs=100.0, seq_len=0, sig_kernel=8, input_size=16,
                    target_norm='clip', spectral_weight=0.1,
                    enc_embed_dim=64, enc_depth=1, enc_num_heads=4,
                    spectral_fft_sizes='resp=32/64,bp=16/32')
        base.update(kw)
        return SimpleNamespace(**base)

    m_built = build_pretraining_model(_build_ns())
    check('build_pretraining_model plumbs the per-stream spec',
          m_built.spectral_fft_sizes == {'bp': [16, 32], 'resp': [32, 64]},
          m_built.spectral_fft_sizes)
    m_alias = build_pretraining_model(
        _build_ns(spectral_fft_sizes='', fft_sizes='16,32'))
    check("build_pretraining_model accepts the Stage-3 'fft_sizes' alias",
          m_alias.spectral_fft_sizes == {'bp': [16, 32], 'resp': [16, 32]},
          m_alias.spectral_fft_sizes)
    m_auto_built = build_pretraining_model(_build_ns(spectral_fft_sizes=''))
    check('build_pretraining_model with an empty spec -> per-modality default',
          m_auto_built.spectral_fft_sizes == {'bp': [64], 'resp': [64]},
          m_auto_built.spectral_fft_sizes)

    if verbose:
        print('spectral_self_test: ' + ('ALL PASS' if not fails
                                        else f'{len(fails)} FAILURE(S): '
                                             f'{fails}'))
    return len(fails)


if __name__ == '__main__':
    raise SystemExit(1 if (mask_self_test() + spectral_self_test()) else 0)
