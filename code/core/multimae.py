"""Multimodal masked autoencoder for Stage-2 pre-training.

Streams: ``rgb`` + ``tir`` (3-D tubelet spatio-temporal video) plus any
physiological 1-D signals (``bp`` / ``resp``; milestone runs
``rgb,tir,bp``). STAGE-2 CONTRACT: at least TWO streams -- >=1 video
(rgb/tir) AND >=1 physiological 1-D signal (bp/resp, the waveform
later regressed in Stage 3). All-visual or all-signal stream lists are
rejected. Pipeline inside ``MultiModalMAE.forward``:

  * per-stream adapters -> tokens (+ learned positional embedding)
  * ONE masking policy (ADR-002, ``core.budget_masking``): every step draws a
    mode and one kept-token count per stream, and the counts of all streams sum
    to the CONSTANT budget ``n_budget``; placement (which tubes / which tokens)
    is drawn per sample by ``core.mask_placement``. masks == 1 => "masked / to
    reconstruct" (matches ``MaskedMSELoss``)
  * visible tokens of ALL streams are concatenated into ONE shared transformer
  * a shared lightweight decoder re-inserts a learned [MASK] token at every
    masked position and each stream reconstructs its NORMALIZED tubelet/window
    patches (VideoMAE-style ``norm_pix``) through a per-stream linear head;
    the loss is a masked MSE computed INDEPENDENTLY per modality on its masked
    positions only, then combined as a WEIGHTED sum: visual streams (rgb/tir)
    keep lambda = 1.0 while physio (1-D) streams get lambda = ``signal_weight``
    (default 0.5; ``loss_weights`` gives a full per-stream override).

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
* VISUAL INPUT PREPROCESSING (design
  ``code/notes/design/Video_input_Preprocessing.md``): after ``_fit_time`` and
  BEFORE patch division, every visual clip is
  z-scored per channel over ``(T, H, W)`` per sample
  (``core.input_adapters.zscore_clip``). The datasets keep returning ``[0, 1]``
  -- that is their contract, not the network's input space. Forward order is
  ``_fit_time -> zscore_clip (visual) -> adapters -> positions -> masks ->
  encoder -> decoder``. The reconstruction targets follow the same split: a
  MASKED cube is standardised with its own per-cube mean/var, an UNMASKED cube
  carries the clip-normalised values (:meth:`_targets`).
* ``_init_pos_embeds`` initialises the (learned) positions with a 3-D
  ``(t, h, w)`` sincos grid for the videos and the SAME 1-D temporal sincos for
  the physio streams, so the space-time prior exists at init -- the Stage-1
  MAE/ViT checkpoint cannot provide it (its 2-D ``pos_embed`` has a different
  token count and is dropped by ``load_pretrained_encoder``). Disable with
  ``pos_init='random'``.

Tensor layouts::

    x = {'rgb': [B, 3, T, H, W],
         'tir': [B, 3, T, H, W],   # false-colour thermal rendering (3 ch)
         'bp': [B, 1, S]}         # any physio stream (resp shares this layout)

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
from torch import Tensor

from .blocks import Block, trunc_normal_
from .budget_masking import SIGNAL as _SIGNAL_KIND
from .budget_masking import VISUAL as _VISUAL_KIND
from .budget_masking import BudgetPlan, ConstantBudgetApportioner, StreamSpec
from .criterion import MaskedMSELoss
from .input_adapters import zscore_clip
from .registry import register_model
from . import mask_placement as placement

__all__ = ['TubeletEmbed', 'MultiModalMAE', 'build_pretraining_model',
           'load_pretrained_encoder', 'canonicalise_vit_state_dict',
           'fit_visual_patch_embed', 'MULTIMAE_VARIANTS', 'multimae_variant',
           'STREAM_CHANNELS']

_EPS = 1e-6
#: ``rgb2d`` (2026-11-10) is the BP4D+ 2D-TEXTURE face ROI served from the
#: offline cache (``data/rgb2d_cache.py``). It is a distinct name from ``rgb``
#: (the 224 px ``rgb_roi`` + BP arm) so a checkpoint can never be paired with the
#: wrong front-end. Both visual streams must share ``input_size`` / ``tubelet`` /
#: ``num_frames`` -- the model has ONE patch grid.
_VISUAL_STREAMS = ('rgb', 'tir', 'rgb2d')
_SIGNAL_STREAMS = ('bp', 'resp')

#: Default input channels per stream. ``tir`` is 3, NOT 1: the BP4D thermal
#: stream is a false-colour (rainbow) rendering with a burned-in degC legend --
#: verified with two independent decoders (OpenCV + PyAV): ``wmv3``/``yuv420p``,
#: 3 planes, mean |U-128| ~ 23-27 on every session, chroma following the scene.
#: Feeding luma only would discard the palette the camera wrote. Override per
#: run with ``--tir_channels 1`` (legacy), but note that changes the TIR
#: adapter geometry: Stage-2 checkpoints are only compatible within one setting.
STREAM_CHANNELS = {'rgb': 3, 'tir': 3, 'rgb2d': 3, 'bp': 1, 'resp': 1}

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
                 n_budget: int = 350,
                 video_cap_ratio: float = 0.25,
                 physio_cap_ratio: float = 0.50,
                 video_min_tubes: int = 2,
                 p_virtual_sensing: float = 0.50,
                 virtual_sensing_warmup_p: float = 0.10,
                 virtual_sensing_warmup_epochs: int = 0,
                 spatial_patches: Optional[int] = None,
                 physio_tokens: Optional[int] = None,
                 signal_weight: float = 0.5,
                 loss_weights: Optional[Dict[str, float]] = None,
                 pos_init: str = 'sincos3d',
                 target_norm: str = 'token',
                 physio_placement: str = 'scattered',
                 physio_span_tokens: int = 0):
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
        # (bp/resp) -- the physio stream(s) are the Stage-3 regression
        # targets, so video-only or signal-only runs are not allowed.
        if not self.visual or not self.signal:
            raise ValueError(
                'MultiModalMAE: Stage-2 requires >=1 video stream '
                f'({", ".join(_VISUAL_STREAMS)}) AND >=1 physiological 1-D '
                f'stream ({", ".join(_SIGNAL_STREAMS)}); got '
                f'visual={self.visual}, signal={self.signal}.')

        # EVERY stream is fed to the shared encoder: there is exactly ONE
        # architecture (the shared-encoder masked autoencoder). The asymmetric
        # cross-MAE variant ("Solution A": physio stream kept OUT of the
        # encoder and decoded by target-driven cross-attention queries) was
        # retired on 2026-11-07 when the research focus moved to the
        # reconstruction benchmark.

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
        #   the token windows concatenate back into a coherent waveform and the
        #   Stage-2 target space matches Stage 3. Per-token rescaling puts a
        #   step at every token boundary instead.
        # 'none' : IDENTITY -- the 1-D target is used EXACTLY as the dataset
        #   returns it. This is the ROI-lineage default: the dataset already
        #   applied the pipeline's ONE label normalisation (a per-clip z-score,
        #   ``data.tir_resp_dataset.TARGET_NORM``), so the model must NOT
        #   re-normalise per clip -- doing both would rescale an already
        #   normalised label. The VISUAL streams are unaffected: they always
        #   take their own per-token z-score below.
        if target_norm not in ('token', 'clip', 'none'):
            raise ValueError(
                f"MultiModalMAE: target_norm must be 'token', 'clip' or "
                f"'none'; got {target_norm!r}")
        self.target_norm = target_norm

        # ---- masking policy: ONE constant-budget apportioner (ADR-005) ----- #
        # There is exactly one policy. Each step samples a mode and one
        # kept-token count per stream such that the visible tokens of all
        # streams sum to ``n_budget``; ``core.mask_placement`` then draws which
        # tubes / which temporal tokens per sample. Design of record:
        # ``code/notes/design/ADR-005-asymmetric-multimodal-masking-3streams.md``.
        #
        # ``virtual_sensing_warmup_p`` / ``virtual_sensing_warmup_epochs`` are
        # the ADR-005 §2.7 curriculum (R6). ``E_warm = 0`` is the neutral
        # default: ``p(e)`` is then the constant ``p_virtual_sensing`` and the
        # mask sequence of every pre-ADR-005 run is unchanged.
        #
        # ``spatial_patches`` and ``physio_tokens`` are TOKENIZER-DERIVED: they
        # come from ``input_size``/``tubelet`` and ``seq_len``/``sig_kernel``.
        # Passing them explicitly is allowed only when it agrees, so a stale
        # config fails loudly instead of silently masking a different grid.
        self.n_budget = int(n_budget)
        self.video_cap_ratio = float(video_cap_ratio)
        self.physio_cap_ratio = float(physio_cap_ratio)
        self.video_min_tubes = int(video_min_tubes)
        self.p_virtual_sensing = float(p_virtual_sensing)
        self.virtual_sensing_warmup_p = float(virtual_sensing_warmup_p)
        self.virtual_sensing_warmup_epochs = int(virtual_sensing_warmup_epochs)
        self.spatial_patches = int(self.grid_h * self.grid_w)
        self.physio_tokens = int(self.n_signal)
        if spatial_patches is not None and int(spatial_patches) != self.spatial_patches:
            raise ValueError(
                f'MultiModalMAE: spatial_patches={int(spatial_patches)} but the '
                f'tokenizer produces (input_size/ph)*(input_size/pw) = '
                f'{self.input_size}/{ph} x {self.input_size}/{pw} = '
                f'{self.spatial_patches}. Drop the override or fix the '
                f'geometry.')
        if physio_tokens is not None and int(physio_tokens) != self.physio_tokens:
            raise ValueError(
                f'MultiModalMAE: physio_tokens={int(physio_tokens)} but the '
                f'tokenizer produces seq_len/sig_kernel = {self.seq_len}/'
                f'{self.sig_kernel} = {self.physio_tokens}. Drop the override '
                f'or fix the geometry.')

        # 1-D placement: scattered tokens by default; 'span' keeps contiguous
        # blocks (the ablation that removes the local-interpolation shortcut)
        # and is a run setting RECORDED with the run.
        self.physio_placement = str(physio_placement or placement.SCATTERED)
        self.physio_placement = self.physio_placement.strip().lower()
        if self.physio_placement not in placement.PLACEMENT_MODES:
            raise ValueError(
                f'MultiModalMAE: physio_placement must be one of '
                f'{placement.PLACEMENT_MODES}; got {self.physio_placement!r}.')
        self.physio_span_tokens = int(physio_span_tokens or 0)
        if self.physio_placement == placement.SPAN and self.physio_span_tokens < 1:
            raise ValueError(
                "MultiModalMAE: physio_placement='span' needs "
                'physio_span_tokens >= 1 (an explicit length, never a hidden '
                'default); got '
                f'{self.physio_span_tokens}.')

        # one StreamSpec per declared stream: the engine sees only unit counts
        # and how many tokens a unit carries, never a modality name.
        self.specs: Tuple[StreamSpec, ...] = tuple(
            StreamSpec(s, _VISUAL_KIND, self.spatial_patches,
                       tokens_per_unit=self.grid_t)
            if s in self.visual else
            StreamSpec(s, _SIGNAL_KIND, self.physio_tokens, tokens_per_unit=1)
            for s in self.streams)
        self.apportioner = ConstantBudgetApportioner(
            self.specs, n_budget=self.n_budget,
            video_cap_ratio=self.video_cap_ratio,
            physio_cap_ratio=self.physio_cap_ratio,
            video_min_tubes=self.video_min_tubes,
            p_virtual_sensing=self.p_virtual_sensing,
            virtual_sensing_warmup_p=self.virtual_sensing_warmup_p,
            virtual_sensing_warmup_epochs=self.virtual_sensing_warmup_epochs)
        self._last_plan: Optional[BudgetPlan] = None

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
        tubelet token and a physio token are only interchangeable if one signal
        token is an integer fraction of the video token's span and the two
        streams cover the same window. Conditions enforced (all hard errors --
        a silent mismatch would let the model "learn" a cross-modal relation
        that does not exist):

        1. the tubelet grid tiles the clip exactly (``tubelet_t | num_frames``);
        2. the signal grid tiles the window exactly (``sig_kernel | seq_len``);
        3. one tubelet token and one signal token start from the same geometry:
           the signal rate ``n_signal / Gt`` -- the number of signal tokens per
           visual token -- must be an INTEGER ``R``, and
           ``R * sig_kernel/fs == tubelet_t*temporal_stride/fps``. ``R = 1`` is
           the special case where one token of each stream spans the same
           window; ``R = 10`` (320 ms tubelet vs 32 ms signal token) is the
           ADR-002 geometry.
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

        ratio = self.n_signal / self.grid_t
        if ratio < 1.0 or not math.isclose(ratio, round(ratio),
                                           rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f'Space-time alignment: the signal grid has {self.n_signal} '
                f'token(s) over the {self.grid_t} tubelet step(s) of the '
                f'video, i.e. {ratio:g} signal token(s) per visual token -- '
                f'not an integer. Set seq_len = R * '
                f'{self.grid_t} * {self.sig_kernel} for an integer R (one '
                f'signal token must tile a visual token evenly).')
        #: signal tokens per visual token (the ADR's ``R``).
        self.signal_tokens_per_visual = int(round(ratio))

        if not math.isclose(
                self.sec_per_signal_token * self.signal_tokens_per_visual,
                self.sec_per_visual_token,
                rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(
                f'Space-time alignment: one video tubelet covers '
                f'tubelet_t*temporal_stride/fps = '
                f'{t}*{self.temporal_stride}/{self.fps} = '
                f'{self.sec_per_visual_token * 1000:.2f} ms but '
                f'{self.signal_tokens_per_visual} signal token(s) cover '
                f'{self.signal_tokens_per_visual}*sig_kernel/fs = '
                f'{self.signal_tokens_per_visual}*{self.sig_kernel}/'
                f'{self.fs} = '
                f'{self.signal_tokens_per_visual * self.sec_per_signal_token * 1000:.2f} '
                f'ms, so the token grids drift. Set sig_kernel = '
                f'{int(round(self.sec_per_visual_token * self.fs / self.signal_tokens_per_visual))} '
                f'(recommended) or temporal_stride = 1 / tune fps-fs.')

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
    def plan_from_counts(self, kept_tokens: Dict[str, int]) -> BudgetPlan:
        """Validate an EXPLICIT allocation (evaluation pins, reconstructions).

        The policy owns the training draws; this entry point lets a caller that
        must reproduce ONE specific masking state the counts and have the same
        invariants enforced (exact budget, caps, floors, whole tubes).
        """
        return self.apportioner.plan_from_counts(kept_tokens)

    def make_masks(self, B: int, device,
                   counts_generator: Optional[torch.Generator] = None,
                   placement_generator: Optional[torch.Generator] = None,
                   plan: Optional[BudgetPlan] = None,
                   epoch: Optional[int] = None):
        """Per-stream 1=masked tensors and the step's plan.

        One ``sample_step()`` fixes the counts of the whole batch (ADR-002
        contract 6); the kept INDICES are drawn per sample, so two samples of a
        step share the counts and differ in placement.

        :param counts_generator: RNG of the policy draw. ``None`` uses the
            global RNG (the training path uses a rank-independent generator, so
            every rank derives the same plan -- see
            ``core.mask_placement.counts_generator``).
        :param placement_generator: RNG of the per-sample placement. ``None``
            uses the global RNG.
        :param plan: an explicit plan (evaluation pin); no count is drawn.
        :param epoch: 0-based epoch index selecting ``p(e)`` (ADR-005 §2.7).
            ``None`` uses ``p_target``, so an old caller is unchanged.
        :return: ``(masks, plan, indices)`` — masks are ``{stream: [B, N]}``
            with 1 = masked, ``indices`` are the kept encoder indices per
            stream, ascending.
        """
        if plan is None:
            plan = self.apportioner.sample_step(
                generator=counts_generator, epoch=epoch)
        self._last_plan = plan
        masks: Dict[str, Tensor] = {}
        indices: Dict[str, Tensor] = {}
        for spec in self.specs:
            kept = int(plan.kept_tokens[spec.name])
            if spec.kind == _VISUAL_KIND:
                idx = placement.place_video(
                    kept // int(spec.tokens_per_unit), spec.n_units,
                    spec.tokens_per_unit, generator=placement_generator,
                    batch_size=B, device=device)
            else:
                idx = placement.place_signal(
                    kept, spec.n_units, generator=placement_generator,
                    mode=self.physio_placement,
                    span_tokens=self.physio_span_tokens or None,
                    batch_size=B, device=device)
            keep = torch.zeros(B, spec.n_tokens, device=device,
                               dtype=torch.long)
            keep.scatter_(1, idx, 1)
            masks[spec.name] = 1 - keep
            indices[spec.name] = idx
        return masks, plan, indices

    # ------------------------------------------------------------------ #
    # reconstruction targets (normalized flat patches)
    # ------------------------------------------------------------------ #
    def _targets(self, x, stream: str, mask: Optional[Tensor] = None):
        """Reconstruction targets ``[B, N, flat]`` -- masked cubes only.

        VISUAL streams (design ``Video_input_Preprocessing.md``):

        * a MASKED cube (``mask == 1``) is standardised with that cube's OWN
          mean/var over its ``C*t*ph*pw`` values -- the classic VideoMAE
          ``norm_pix`` target, local to the cube;
        * an UNMASKED cube carries the CLIP-normalised values it was patched
          from (``core.input_adapters.zscore_clip`` applied in ``forward``),
          i.e. NO local per-cube map -- the contract says unmasked cubes only
          ever receive clip-level normalisation. The loss ignores those
          positions anyway (``MaskedMSELoss`` keeps ``mask == 1``), but the
          returned tensor is honest.

        ``mask=None`` (D7) means "every cube masked" -- only for unit tests and
        diagnostics; :meth:`forward` always passes the real ``loss_mask``.

        1-D streams are untouched: their label normalisation is the dataset's
        per-clip z-score (``data.tir_resp_dataset.TARGET_NORM``) and
        ``target_norm`` decides whether the model re-normalises; ``mask`` is
        accepted and ignored.
        """
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
            zcube = (x - mean) / torch.sqrt(var + _EPS)
            if mask is None:
                return zcube
            return torch.where(mask.unsqueeze(-1).bool(), zcube, x)
        x = x.reshape(x.shape[0], self.n_signal, self.sig_kernel)
        if self.target_norm == 'none':
            # IDENTITY: the dataset already normalised this 1-D stream (the
            # per-clip z-score, ``data.tir_resp_dataset.TARGET_NORM``); use it
            # verbatim. The reshape is the exact inverse of the tokenisation,
            # so the assembled target IS the waveform the dataset produced.
            return x
        if self.target_norm == 'clip':
            # ONE mean/std per (sample, stream) -- the flattening inverse of
            # this (tokens are consecutive non-overlapping windows) reproduces
            # exactly the z-scored clip waveform, so the Stage-2 target space
            # matches the Stage-3 one.
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

    def forward(self, x, counts_generator=None, placement_generator=None,
                plan=None, epoch: Optional[int] = None):
        """Masked multimodal forward.

        :param counts_generator: RNG of the step's PLAN (mode + counts). ``None``
            draws from the global RNG; the runners pass a rank-independent
            generator so every rank derives the same plan without a collective.
        :param placement_generator: RNG of the per-sample placement. ``None``
            draws from the global RNG; the runners pass a rank-distinct one.
        :param plan: an explicit :class:`BudgetPlan` (evaluation pin). When
            given, no count is drawn.
        :param epoch: 0-based epoch index selecting ``p(e)`` (ADR-005 §2.7).
            ``None`` uses ``p_target`` and reproduces the pre-curriculum draw.
        """
        B = next(iter(x.values())).shape[0]
        device = next(iter(x.values())).device

        # Enforce the configured time/signal lengths. Both streams are trimmed
        # from the END, which keeps the shared time origin and therefore the
        # tubelet <-> signal token alignment; a too-short stream raises.
        #
        # VISUAL INPUT PREPROCESSING (design ``Video_input_Preprocessing.md``):
        # after the trim (so the statistics describe the exact clip the
        # adapters see) and BEFORE patch division, the whole clip is z-scored
        # per channel. ``_targets`` below reads ``xin`` too, which is what makes
        # "unmasked cubes receive clip-level normalisation" true of the target
        # tensor as well. The 1-D streams are NOT touched: their normalisation
        # is the dataset's per-clip z-score.
        xin = {}
        for s in self.streams:
            n = self.num_frames if s in self.visual else self.seq_len
            xin[s] = self._fit_time(x[s], n, s)
            if s in self.visual:
                xin[s] = zscore_clip(xin[s])

        # --- counts + placement (ADR-002) --------------------------------- #
        # Computed BEFORE tokenisation so a stream with no visible unit can skip
        # its adapter entirely: a fully masked video would otherwise pay a full
        # Conv3d and receive a ZERO (not absent) gradient. The adapters consume
        # no randomness, so the order does not change the RNG stream.
        masks, plan, ids_keep = self.make_masks(
            B, device, counts_generator=counts_generator,
            placement_generator=placement_generator, plan=plan, epoch=epoch)

        # --- tokenize + add positional embedding -------------------------- #
        tokens = {}
        for s in self.streams:
            if int(plan.kept_tokens[s]) == 0:    # nothing to encode
                tokens[s] = None
                continue
            tok = self.adapters[s](xin[s])           # [B, N, D]
            tok = tok + self.positions[s].pos_embed
            tokens[s] = tok

        # --- gather the VISIBLE tokens of every stream -------------------- #
        # The plan's counts are identical for every sample of the step, so the
        # gather is exact (no batch-max) and the concatenated length is the
        # CONSTANT budget by construction.
        enc_parts = []
        for s in self.streams:
            if tokens[s] is None:
                # 0 kept: no visible token and therefore no gradient to this
                # stream's adapter. Its loss is still computed (and reaches the
                # encoder through the decoder conditioning below).
                continue
            keep = ids_keep[s]
            tok = tokens[s]
            enc_parts.append(torch.gather(
                tok, 1, keep.unsqueeze(-1).expand(B, keep.shape[1],
                                                  tok.shape[2])))
        z = torch.cat(enc_parts, dim=1)                 # [B, n_budget, D]
        assert tuple(z.shape) == (B, self.n_budget, self.embed_dim), (
            f'MultiModalMAE.forward: the encoder input is {tuple(z.shape)} but '
            f'the constant budget requires {(B, self.n_budget, self.embed_dim)} '
            f'(plan {plan.kept_tokens}, mode {plan.mode}).')

        # --- shared encoder -------------------------------------------------
        for blk in self.enc_blocks:
            z = blk(z)
        z = self.enc_norm(z)

        # --- per-stream decode + masked reconstruction loss -----------------
        # DECODER CONDITIONING ON z (ADR-002 contract 7): every stream decodes
        # from the encoder latents plus its own [MASK] tokens at the masked
        # positions. Prefixing z is what gives a 0-kept stream a gradient path
        # to the encoder -- without it Mode 1 would reconstruct the waveform
        # with no visual evidence and contribute nothing to z.
        losses = {}          # per-modality weighted masked-MSE sum
        losses_mse = {}      # per-modality raw masked MSE (before weighting)
        preds = {}
        for s in self.streams:
            mask_s = masks[s]                            # [B, N] 1 = masked
            n = self.n_visual if s in self.visual else self.n_signal
            embs = self.positions[s]
            n_masked = int(mask_s[0].sum())
            if not bool((mask_s.sum(dim=1) == n_masked).all()):
                raise RuntimeError(
                    f'MultiModalMAE.forward: stream {s!r} has a per-sample '
                    f'COUNT of masked tokens, but the policy guarantees one '
                    f'count per step (ADR-002 contract 6). Only the PLACEMENT '
                    f'may vary across samples.')
            tgt = self._targets(xin[s], s, mask_s)           # [B, N, flat]

            if n_masked == 0:
                # fully visible stream (legal only with a cap of 1.0): its loss
                # is an explicit zero, which keeps the loss-key set constant.
                zero = torch.zeros((), device=device)
                losses_mse[s] = zero
                losses[s] = zero * self.loss_weights[s]
                preds[s] = torch.zeros(B, n, self._flat[s], device=device)
                continue

            # masked positions are drawn per SAMPLE, so their indices differ
            # across the batch even though their number does not.
            ids_masked = torch.argsort(mask_s, dim=1, descending=True,
                                       stable=True)[:, :n_masked]
            dec = torch.cat([z, embs.mask_token.expand(B, n_masked,
                                                       self.embed_dim)], dim=1)
            mask_pos = embs.pos_embed.expand(B, n, self.embed_dim).gather(
                1, ids_masked.unsqueeze(-1).expand(B, n_masked,
                                                   self.embed_dim))
            dec_pos = torch.cat(
                [z.new_zeros(B, z.shape[1], self.embed_dim), mask_pos], dim=1)
            dec = dec + dec_pos
            for blk in self.dec_blocks:
                dec = blk(dec)
            pred_masked = self.heads[s](dec[:, z.shape[1]:])
            pred = torch.zeros(B, n, self._flat[s], device=device)
            pred.scatter_(1, ids_masked.unsqueeze(-1).expand(
                B, n_masked, self._flat[s]), pred_masked)
            losses_mse[s] = self.loss_fn(pred, tgt, mask_s)
            losses[s] = self.loss_weights[s] * losses_mse[s]
            preds[s] = pred

        total = torch.stack(list(losses.values())).sum()
        return {'loss': total, 'losses': losses, 'losses_mse': losses_mse,
                'masks': masks, 'preds': preds,
                # STEP diagnostics (counts are batch-uniform, so they are a
                # property of the draw, not of the sample). The key set is a
                # function of the CONFIG: every declared stream has an entry on
                # every step, whatever the mode.
                'mode': int(plan.mode),
                # ADR-005 §2.8: the contract-4 resolved mode (differs from the
                # draw on the K_tilde=0 step) and the schedule value that
                # produced it. Both are properties of the step's plan.
                'mode_resolved': int(plan.mode_by_counts),
                'p_virtual': float(self.apportioner.p_at(epoch)),
                'kept': {s: int(plan.kept_tokens[s]) for s in self.streams},
                'kept_units': {s: int(plan.kept_units[s])
                               for s in self.streams},
                'n_batch': int(z.shape[1])}


# --------------------------------------------------------------------------- #
# builder (mirrors run_pretrain argparse/YAML defaults)
# --------------------------------------------------------------------------- #
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
            'bp/resp).')
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

    # ViT geometry: the --model variant name sets it, and an explicit (>0)
    # enc_* flag/YAML value overrides it (ablation escape hatch).
    geom = multimae_variant(getattr(args, 'model', '')) or {}

    def _geo(arg_name: str, geom_key: str, default: int) -> int:
        return int(getattr(args, arg_name, 0) or 0) or geom.get(geom_key, default)

    model = MultiModalMAE(
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
        # ADR-002 masking policy: the seven exposed parameters. The two unit
        # counts are tokenizer-derived; passing them is allowed only when they
        # agree (the model asserts the identity).
        n_budget=int(getattr(args, 'n_budget', 350)),
        video_cap_ratio=float(getattr(args, 'video_cap_ratio', 0.25)),
        physio_cap_ratio=float(getattr(args, 'physio_cap_ratio', 0.50)),
        video_min_tubes=int(getattr(args, 'video_min_tubes', 2)),
        p_virtual_sensing=float(getattr(args, 'p_virtual_sensing', 0.50)),
        # ADR-005 §2.7 curriculum. Both defaults are NEUTRAL: E_warm=0 makes
        # p(e) the constant p_target, so a config that omits them behaves
        # exactly as before. The shipped arm's (0.10, 50) lives in its YAML.
        virtual_sensing_warmup_p=float(
            getattr(args, 'virtual_sensing_warmup_p', 0.10)),
        virtual_sensing_warmup_epochs=int(
            getattr(args, 'virtual_sensing_warmup_epochs', 0)),
        spatial_patches=getattr(args, 'spatial_patches', None) or None,
        physio_tokens=getattr(args, 'physio_tokens', None) or None,
        signal_weight=signal_weight,
        loss_weights=loss_weights,
        pos_init=str(getattr(args, 'pos_init', 'sincos3d')),
        target_norm=str(getattr(args, 'target_norm', 'token')),
        physio_placement=str(getattr(args, 'physio_placement', 'scattered')
                             or 'scattered'),
        physio_span_tokens=int(getattr(args, 'physio_span_tokens', 0) or 0))
    print(f'[encoder] streams: {model.streams} (all fed to the shared '
          f'encoder); target streams: {model.signal}')
    print(f'[mask] policy=constant-budget  n_budget={model.n_budget}  '
          f'video_cap={model.video_cap_ratio:g} '
          f'({model.apportioner.video_cap_units} tubes)  '
          f'physio_cap={model.physio_cap_ratio:g} '
          f'({model.apportioner.physio_cap_tokens} tokens)  '
          f'video_min_tubes={model.video_min_tubes}  '
          f'p_virtual_sensing={model.p_virtual_sensing:g}  '
          f'p_warm={model.virtual_sensing_warmup_p:g}  '
          f'E_warm={model.virtual_sensing_warmup_epochs}  '
          f'units={[(s.name, s.n_units, s.tokens_per_unit) for s in model.specs]}'
          f'  physio_placement={model.physio_placement}')
    return model


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
                           tensor: 'torch.Tensor', dst_key: str):
    """Validate a VideoMAE patch-embed tensor against this model's tubelet.

    Only a **5-D** ``[out, in, t, ph, pw]`` ``Conv3d`` source is accepted, and
    it is copied **verbatim**: its temporal kernel is real prior knowledge, and
    with ``tubelet 2,16,16`` the shape matches the model's adapter exactly. A
    kernel that does NOT equal the model's ``tubelet`` raises, because the
    inherited temporal filter would then be meaningless.

    The legacy 2-D ``[out, in, ph, pw]`` boxcar inflation was removed together
    with the ImageNet-MAE / timm sources: Stage 1 is VideoMAE-only, so every
    built-in checkpoint already carries a 3-D tubelet.

    :return: the tensor to store, or ``None`` when it cannot be adapted (the
        caller counts that as a shape mismatch).

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
    return None


def load_pretrained_encoder(model: 'MultiModalMAE', path: str,
                            patch_stream: str = '') -> Dict[str, int]:
    """Copy a VideoMAE ViT-Base checkpoint's transformer weights in.

    Two on-disk layouts are accepted (see :func:`canonicalise_vit_state_dict`),
    after which the mapping is always the same::

        blocks.{i}.*  -> enc_blocks.{i}.*
        norm.*        -> enc_norm.*
        patch_embed.proj.{weight,bias} -> adapters.<visual>.patch_embed.*
                    (copied VERBATIM: a VideoMAE patch embed IS a 3-D
                    Conv3d(3, D, (2,16,16)) tubelet -- see
                    :func:`fit_visual_patch_embed`)

    Stage 1 is VideoMAE-only, so the source ``patch_embed.proj`` IS a
    ``Conv3d(3, D, (2,16,16))`` tubelet filter: the temporal prior transfers
    verbatim instead of being faked by averaging two frames.

    Everything else (cls_token / pos_embed / head / other adapters / signal
    streams) is left at its random initialisation. Encoder geometry must match
    the checkpoint (e.g. embed_dim=768, depth=12, heads=12 for ViT-Base).

    A checkpoint that contributes **zero** encoder tensors raises: this catches
    a Stage-2 ``MultiModalMAE`` checkpoint (``enc_blocks.*`` keys, which this
    Stage-1 mapping cannot read) or any unsupported layout, instead of leaving
    the encoder silently random.

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
                # VideoMAE 3-D Conv3d source -> verbatim; raises on a
                # tubelet-geometry mismatch rather than skipping quietly.
                v = fit_visual_patch_embed(model, v, dst_key)
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
    if n_loaded == 0:
        raise ValueError(
            f'{path}: no encoder tensors could be mapped into '
            f'enc_blocks.* / enc_norm.* / adapters.* -- this is not a Stage-1 '
            f'VideoMAE/ViT checkpoint (layout {layout!r}, {len(skipped)} keys '
            f'skipped, {len(shape_mismatch)} shape-mismatched). A Stage-2 '
            f'MultiModalMAE checkpoint (`enc_blocks.*`) cannot initialise '
            f'Stage 1. Pass a VideoMAE ViT-Base file (`videomae:k400` / '
            f'`videomae:ssv2`) or a compatible local ViT state dict.')
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
# masking self-test -- a masking policy fails SILENTLY, so its invariants are
# checked by a runnable self-test rather than by reading the code
# --------------------------------------------------------------------------- #
def _tiny_model(**over):
    """Smallest MultiModalMAE that satisfies ADR-002's feasibility conditions.

    40 frames @25 fps / tubelet_t 4 = 10 visual steps of 0.16 s; 160 signal
    samples / sig_kernel 8 = 20 tokens of 0.08 s, i.e. R = 2 signal tokens per
    visual token. ``input_size 32`` / patch 8 -> 4x4 = 16 spatial tubes, so the
    25 % visual cap is 4 tubes and the 2-tube floor is satisfiable.
    """
    kw = dict(streams=('tir', 'rgb2d', 'resp'), embed_dim=64, enc_depth=1,
              enc_num_heads=4, dec_depth=1, mlp_ratio=4.0,
              tubelet=(4, 8, 8), num_frames=40, input_size=32,
              sig_kernel=8, seq_len=160, fps=25.0, fs=100.0,
              temporal_stride=1, n_budget=80)
    kw.update(over)
    return MultiModalMAE(**kw)


def mask_self_test(verbose: bool = True) -> int:
    """Verify the constant-budget masking invariants. Returns #failures.

    A runnable check of ADR-002 through the REAL model, because a masking
    policy fails silently: a wrong count still produces a plausible loss.

    * every step draws one count per stream and the counts sum to ``n_budget``;
    * the encoder sees exactly ``n_budget`` tokens (Criterion 3);
    * Mode 1 keeps no physiological token and still backpropagates into the
      visual adapters and the encoder (contracts 4 and 7);
    * a 0-kept stream's adapter receives no gradient, while its loss still
      reaches ``z`` (the decoder conditioning of ADR-002 §6);
    * placements differ across the samples of one step while counts agree.
    """
    fails = []

    def check(name, cond, extra=''):
        if verbose:
            print(f'  [{"ok" if cond else "FAIL"}] {name}'
                  f'{(" -- " + str(extra)) if extra != "" else ""}')
        if not cond:
            fails.append(name)

    m = _tiny_model(p_virtual_sensing=0.5)
    dev = torch.device('cpu')
    B = 4
    x = {
        'tir': torch.randn(B, 3, m.num_frames, m.input_size, m.input_size),
        'rgb2d': torch.randn(B, 3, m.num_frames, m.input_size, m.input_size),
        'resp': torch.randn(B, 1, m.seq_len),
    }
    if verbose:
        print(f'mask_self_test: n_budget={m.n_budget} '
              f'units={[(s.name, s.n_units, s.tokens_per_unit) for s in m.specs]}')

    modes = set()
    for step in range(20):
        cg = placement.counts_generator(7, 0, step)
        pg = placement.placement_generator(7, 0, step, rank=0)
        out = m(x, counts_generator=cg, placement_generator=pg)
        modes.add(out['mode'])
        check(f'step {step}: encoder length == n_budget',
              out['n_batch'] == m.n_budget, out['n_batch'])
        check(f'step {step}: kept counts sum to n_budget',
              sum(out['kept'].values()) == m.n_budget, out['kept'])
        if out['mode'] == 1:
            check(f'step {step}: Mode 1 keeps no physiological token',
                  all(out['kept'][s] == 0 for s in m.signal), out['kept'])
    check('both modes occur at p=0.5 over 20 steps', modes == {1, 2}, modes)

    # --- Mode 1: 0-kept physiology still trains the encoder -------------- #
    plan1 = m.plan_from_counts({'tir': 40, 'rgb2d': 40, 'resp': 0})
    m.zero_grad(set_to_none=True)
    captured = {}

    def _grab_z(_mod, _inp, out):
        out.retain_grad()
        captured['z'] = out

    handle = m.enc_norm.register_forward_hook(_grab_z)
    try:
        out = m(x, plan=plan1)
        out['losses']['resp'].backward()
    finally:
        handle.remove()
    z = captured['z']
    check('Mode 1: d(loss_resp)/d(z) is non-zero (contract 7)',
          z.grad is not None and float(z.grad.abs().sum()) > 0, None)
    check('Mode 1: visual adapter receives gradient',
          m.adapters['tir'].patch_embed.weight.grad is not None
          and float(m.adapters['tir'].patch_embed.weight.grad.abs().sum()) > 0)
    check('Mode 1: physiological adapter receives NO gradient',
          m.adapters['resp'].embed.weight.grad is None)

    # --- placements vary across the batch, counts do not ----------------- #
    masks, plan, idx = m.make_masks(
        B, dev, counts_generator=placement.counts_generator(7, 0, 0),
        placement_generator=placement.placement_generator(7, 0, 0, rank=0))
    check('counts are batch-uniform',
          all(int(masks[s][b].sum()) == int(masks[s][0].sum())
              for s in masks for b in range(B)))
    check('placements differ across the batch',
          len({tuple(idx['tir'][b].tolist()) for b in range(B)}) > 1)

    if verbose:
        print('mask_self_test: ' + ('ALL PASS' if not fails
                                    else f'{len(fails)} FAILURE(S): {fails}'))
    return len(fails)


def stft_self_test(verbose: bool = True) -> int:
    """Verify ``MultiResolutionSTFTLoss``'s degenerate-target guard.

    The guard belongs to the Stage-3 criterion (``core.waveform_losses``) and
    fails in the worst way when it is wrong: a wildly wrong loss still trains.
    A per-clip z-scored target that is CONSTANT is exactly zero, so ``||T|| = 0``
    and the scale-invariant spectral-convergence ratio explodes (measured
    4.3e10 on a real railed clip). Samples with no spectral energy are
    therefore excluded from the term.
    """
    from .waveform_losses import MultiResolutionSTFTLoss

    fails = []

    def check(name, cond, extra=''):
        if verbose:
            print(f'  [{"ok" if cond else "FAIL"}] {name}'
                  f'{(" -- " + str(extra)) if extra != "" else ""}')
        if not cond:
            fails.append(name)

    if verbose:
        print('stft_self_test:')

    # ---- the SHARED loss's degenerate-target guard ----------------------- #
    # A z-scored target that is CONSTANT is exactly zero -> ||T|| = 0 -> the
    # scale-invariant ratio explodes. Measured 4.3e10 on a real railed clip;
    # these checks pin the guard.
    stft = MultiResolutionSTFTLoss(fft_sizes=(32, 64), hop_ratio=0.25)
    torch.manual_seed(0)
    p_probe = torch.randn(4, 256)
    t_ok = torch.randn(4, 256)
    t_flat = torch.zeros(4, 256)

    def _unguarded(pred, tgt):
        tot = torch.tensor(0.0)
        for n in stft.fft_sizes:
            pm, tm = stft._magnitude(pred, n), stft._magnitude(tgt, n)
            tot = tot + ((torch.norm(pm - tm, dim=(-1, -2))
                          / (torch.norm(tm, dim=(-1, -2)) + 1e-8)).mean()
                         + torch.mean(torch.abs(torch.log(pm + 1e-6)
                                                - torch.log(tm + 1e-6))))
        return tot / len(stft.fft_sizes)

    check('MR-STFT on an all-valid batch is bit-identical to the unguarded '
          'formula',
          torch.equal(stft(p_probe, t_ok), _unguarded(p_probe, t_ok)),
          f'{float(stft(p_probe, t_ok)):.6g} vs {float(_unguarded(p_probe, t_ok)):.6g}')
    v_flat = float(stft(p_probe, t_flat))
    check('MR-STFT on an all-flat target returns 0 instead of ~1e10',
          v_flat == 0.0, f'{v_flat:.6g} (unguarded would be '
                         f'{float(_unguarded(p_probe, t_flat)):.3g})')
    check('the guard reports the degenerate sample (once)',
          stft._warned_degenerate is True)
    t_mix = t_ok.clone()
    t_mix[0] = 0.0
    v_mix = float(stft(p_probe, t_mix))
    check('MR-STFT with 1 of 4 samples flat stays bounded',
          v_mix < 10.0, f'{v_mix:.4g} (unguarded '
                        f'{float(_unguarded(p_probe, t_mix)):.3g})')

    if verbose:
        print('stft_self_test: ' + ('ALL PASS' if not fails
                                    else f'{len(fails)} FAILURE(S): '
                                         f'{fails}'))
    return len(fails)


if __name__ == '__main__':
    raise SystemExit(1 if (mask_self_test() + stft_self_test()) else 0)
