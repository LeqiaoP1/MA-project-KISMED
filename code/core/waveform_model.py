"""Stage-3 downstream model: waveform regression on top of the Stage-2 encoder.

``docs/ImplementationPlan.md`` Stage 3 simulates a COMPLETE 1-D sensor failure:
no BP/RESP/EDA is fed, only the visual stream(s) drive the prediction. This
module therefore

* re-uses the Stage-2 multimodal *encoder* verbatim -- ``adapters.<s>``
  (Conv3d tubelet embed), ``positions.<s>`` (the space-time positional embed,
  incl. the trained ``mask_token`` params which are simply unused here),
  ``enc_blocks.*`` and ``enc_norm`` keep the EXACT Stage-2 key names, so a
  Stage-2 checkpoint loads with ``load_state_dict(..., strict=False)`` and
  ``load_stage2_encoder`` refuses to pretend a load happened, and
* appends a *temporal* regression head: the encoded tubelet tokens are mean
  pooled over the spatial grid (per time step) and one linear head predicts the
  ``sig_kernel`` waveform samples belonging to that tubelet's time window.
  Because Stage 2 enforces ``tubelet_t*temporal_stride/fps == sig_kernel/fs``
  (see ``core.multimae.MultiModalMAE._check_space_time_alignment``), output
  token ``i`` covers exactly the same window as the waveform samples it
  predicts -- the head is time-aligned by construction, not by luck.

This is the difference from ``core.model.ProjectViT`` (a 2-D ViT with a 4-D
``Conv2d`` patch embed), which cannot consume the dataset's
``[B, C, T, H, W]`` clips at all and shares no keys with the Stage-2 encoder.
"""
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .blocks import Block
from .multimae import STREAM_CHANNELS, TubeletEmbed, _PosMask, multimae_variant

__all__ = ['MultiModalWaveformRegressor', 'build_waveform_model',
           'load_stage2_encoder']

_VISUAL_STREAMS = ('rgb', 'tir')


def _parse_int_csv(v, dtype=int):
    return tuple(dtype(x) for x in str(v).split(','))


def _unwrap_state(ckpt):
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ('model', 'state_dict', 'module'):
            if isinstance(ckpt.get(key), dict):
                state = ckpt[key]
                break
    if isinstance(state, dict) and isinstance(state.get('module'), dict):
        state = state['module']
    return state


class MultiModalWaveformRegressor(nn.Module):
    """Stage-2 visual encoder + per-time-step waveform regression head.

    :param streams: visual streams to feed (``rgb`` / ``rgb,tir``). 1-D physio
        streams are NOT allowed: Stage 3 is the sensor-failure scenario.
    :param output_len: length of the predicted waveform in samples; must be a
        multiple of the number of tubelet time steps so every output segment
        belongs to one tubelet window. With the plan's default geometry
        (4 s clip, fs 100, tubelet_t 2 @ 25 fps) that is ``50 * sig_kernel``.
    :param head_hidden: optional hidden width of a 2-layer head (0 = linear).
    """

    def __init__(self, streams: Sequence[str] = ('rgb',),
                 stream_channels: Optional[Dict[str, int]] = None,
                 embed_dim: int = 768, enc_depth: int = 12,
                 enc_num_heads: int = 12, mlp_ratio: float = 4.0,
                 tubelet: Tuple[int, int, int] = (2, 16, 16),
                 input_size: int = 224, num_frames: int = 100,
                 output_len: int = 400, sig_kernel: int = 8,
                 fps: float = 25.0, fs: float = 100.0,
                 temporal_stride: int = 1, head_hidden: int = 0,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0, pos_init: str = 'sincos3d'):
        super().__init__()
        self.streams = list(streams)
        if not self.streams:
            raise ValueError(
                'MultiModalWaveformRegressor: needs >=1 visual stream '
                f'({", ".join(_VISUAL_STREAMS)}).')
        unknown = [s for s in self.streams if s not in _VISUAL_STREAMS]
        if unknown:
            raise ValueError(
                f'MultiModalWaveformRegressor: Stage 3 feeds VISUAL streams '
                f'only (simulated sensor failure), got {unknown}; allowed '
                f'{_VISUAL_STREAMS}.')

        self.stream_channels = dict(STREAM_CHANNELS)
        if stream_channels:
            self.stream_channels.update(
                {k: int(v) for k, v in stream_channels.items()})

        t, ph, pw = tubelet
        assert input_size % ph == 0 and input_size % pw == 0, \
            f'patch {tubelet} must divide input_size {input_size}'
        if num_frames % t:
            raise ValueError(
                f'MultiModalWaveformRegressor: tubelet_t {t} must divide '
                f'num_frames {num_frames}.')

        self.tubelet = tubelet
        self.embed_dim = embed_dim
        self.input_size = input_size
        self.num_frames = num_frames
        self.output_len = output_len
        self.sig_kernel = int(sig_kernel)
        self.fps = float(fps)
        self.fs = float(fs)
        self.temporal_stride = max(1, int(temporal_stride or 1))
        self.pos_init = pos_init

        self.grid_t = num_frames // t
        self.grid_h = self.grid_w = input_size // ph
        self.n_visual = self.grid_t * self.grid_h * self.grid_w

        # one output segment per tubelet time step -> exact temporal alignment
        if output_len % self.grid_t:
            raise ValueError(
                f'MultiModalWaveformRegressor: output_len {output_len} is not '
                f'a multiple of the {self.grid_t} tubelet time steps '
                f'(num_frames {num_frames} / tubelet_t {t}), so the predicted '
                f'waveform cannot be laid out one segment per tubelet window. '
                f'Use output_len = {self.grid_t} * sig_kernel = '
                f'{self.grid_t * self.sig_kernel} for the plan geometry.')
        self.samples_per_token = output_len // self.grid_t
        self.sec_per_visual_token = t * self.temporal_stride / self.fps

        # --- Stage-2 encoder (identical module names!) --------------------- #
        self.adapters = nn.ModuleDict()
        self.positions = nn.ModuleDict()
        for s in self.streams:
            in_ch = int(self.stream_channels.get(s, 3))
            self.adapters[s] = TubeletEmbed(in_ch, embed_dim, tubelet)
            self.positions[s] = _PosMask(self.n_visual, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, enc_depth)]
        self.enc_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=enc_num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=nn.LayerNorm)
            for i in range(enc_depth)
        ])
        self.enc_norm = nn.LayerNorm(embed_dim)

        # --- temporal waveform head ---------------------------------------- #
        if head_hidden and int(head_hidden) > 0:
            self.waveform_head = nn.Sequential(
                nn.Linear(embed_dim, int(head_hidden)), nn.GELU(),
                nn.Linear(int(head_hidden), self.samples_per_token))
        else:
            self.waveform_head = nn.Linear(embed_dim, self.samples_per_token)

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

    def _init_pos_embeds(self):
        """Same space-time prior as Stage 2 (overwritten by the ckpt anyway)."""
        from utils.pos_embed import get_3d_sincos_pos_embed
        vis = torch.from_numpy(get_3d_sincos_pos_embed(
            self.embed_dim, (self.grid_t, self.grid_h, self.grid_w))).float()
        for s in self.streams:
            with torch.no_grad():
                self.positions[s].pos_embed.copy_(vis.unsqueeze(0))

    # ------------------------------------------------------------------ #
    def _fit_time(self, x, n, stream):
        T = x.shape[2]
        if T == n:
            return x
        if T > n:
            return x[:, :, :n]
        raise ValueError(
            f'Stage-3 geometry: stream "{stream}" provides T={T} but this '
            f'checkpoint needs T={n} (num_frames). The Stage-3 dataset must '
            f'use the SAME clip geometry as Stage 2 (clip_duration/fps/'
            f'temporal_stride/tubelet) so the encoder sees the token grid it '
            f'was pre-trained on.')

    def freeze_encoder(self, freeze: bool = True):
        """Optional feature-freezing escape hatch (linear-probe style)."""
        for name, p in self.named_parameters():
            if not name.startswith('waveform_head.'):
                p.requires_grad = not freeze

    def forward(self, x):
        """``x``: dict {stream: [B, C, T, H, W]} or a single rgb tensor.

        ``PairedSessionDataset`` returns a plain ``[B, C, T, H, W]`` clip when
        only one visual stream is configured, which is accepted directly.
        """
        if torch.is_tensor(x):
            if len(self.streams) != 1:
                raise ValueError(
                    f'MultiModalWaveformRegressor expects a dict with streams '
                    f'{self.streams}; got a single tensor. Use streams="rgb" '
                    f'or pass a dict.')
            x = {self.streams[0]: x}
        for s in self.streams:
            if s not in x:
                raise ValueError(
                    f'MultiModalWaveformRegressor: stream "{s}" missing from '
                    f'input; got keys {list(x)}')

        parts = []
        for s in self.streams:
            xt = self._fit_time(x[s], self.num_frames, s)
            tok = self.adapters[s](xt)                     # [B, n_visual, D]
            if tok.shape[1] != self.n_visual:
                raise ValueError(
                    f'Stage-3 geometry: stream "{s}" tokenized to '
                    f'{tok.shape[1]} tokens but this checkpoint expects '
                    f'{self.n_visual} (input_size {self.input_size}, '
                    f'num_frames {self.num_frames}, tubelet {self.tubelet}).')
            tok = tok + self.positions[s].pos_embed
            parts.append(tok)

        z = torch.cat(parts, dim=1)
        for blk in self.enc_blocks:
            z = blk(z)
        z = self.enc_norm(z)

        # [B, S*n_visual, D] -> [B, grid_t, D]: mean over the spatial grid of
        # every visual stream (the streams share the time grid, and each time
        # step keeps its own token -> the head stays temporally resolved).
        feats = []
        for i, s in enumerate(self.streams):
            zi = z[:, i * self.n_visual:(i + 1) * self.n_visual]
            zi = zi.reshape(zi.shape[0], self.grid_t,
                            self.grid_h * self.grid_w, zi.shape[-1])
            feats.append(zi.mean(dim=2))
        h = torch.stack(feats, dim=0).mean(dim=0)          # [B, grid_t, D]

        pred = self.waveform_head(h)                       # [B, grid_t, k]
        return pred.reshape(pred.shape[0], -1)             # [B, output_len]


# --------------------------------------------------------------------------- #
# builder + Stage-2 weight transfer
# --------------------------------------------------------------------------- #
def build_waveform_model(args):
    """Construct the Stage-3 model from a ``run_waveform`` args namespace.

    Geometry comes from the ``--model`` variant name (``project_multimae_*``) so
    it can be kept identical to the Stage-2 run; ``enc_*`` > 0 overrides it.
    """
    streams = tuple(s.strip() for s in
                    str(getattr(args, 'streams', 'rgb')).split(',') if s.strip())
    tubelet = _parse_int_csv(getattr(args, 'tubelet', '2,16,16'))
    fps = float(getattr(args, 'fps', 25.0))
    temporal_stride = max(1, int(getattr(args, 'temporal_stride', 1) or 1))
    clip_duration = float(getattr(args, 'clip_duration', 4.0))
    num_frames = int(getattr(args, 'num_frames', 0) or 0) or max(
        1, int(round(clip_duration * fps / temporal_stride)))
    if num_frames % tubelet[0]:
        num_frames -= num_frames % tubelet[0]

    geom = multimae_variant(getattr(args, 'model', '')) or {}

    def _geo(arg_name: str, geom_key: str, default: int) -> int:
        return int(getattr(args, arg_name, 0) or 0) or geom.get(geom_key,
                                                                default)

    output_len = int(getattr(args, 'seq_len', 0) or 0) or max(
        1, int(round(num_frames * temporal_stride / fps
                     * float(getattr(args, 'fs', 100.0)))))

    return MultiModalWaveformRegressor(
        streams=streams,
        stream_channels={'tir': int(getattr(args, 'tir_channels', 3))},
        embed_dim=_geo('enc_embed_dim', 'embed_dim', 768),
        enc_depth=_geo('enc_depth', 'enc_depth', 12),
        enc_num_heads=_geo('enc_num_heads', 'enc_num_heads', 12),
        mlp_ratio=float(getattr(args, 'mlp_ratio', 4.0)),
        tubelet=tubelet,
        input_size=int(getattr(args, 'input_size', 224)),
        num_frames=num_frames,
        output_len=output_len,
        sig_kernel=int(getattr(args, 'sig_kernel', 8)),
        fps=fps, fs=float(getattr(args, 'fs', 100.0)),
        temporal_stride=temporal_stride,
        head_hidden=int(getattr(args, 'head_hidden', 0) or 0),
        drop_rate=float(getattr(args, 'drop_rate', 0.0)),
        attn_drop_rate=float(getattr(args, 'attn_drop_rate', 0.0)),
        drop_path_rate=float(getattr(args, 'drop_path_rate', 0.0)),
        pos_init=str(getattr(args, 'pos_init', 'sincos3d')))


def load_stage2_encoder(model: MultiModalWaveformRegressor, path: str):
    """Load a Stage-2 (or Stage-1 MAE/ViT) checkpoint into the encoder.

    Accepts BOTH layouts, like ``core.au_probe.load_au_probe_weights``:

    * Stage-2 ``MultiModalMAE`` -> ``adapters.*`` / ``positions.*`` /
      ``enc_blocks.*`` / ``enc_norm.*`` are copied directly (decoder ``dec_*``,
      per-stream ``heads.*`` and the unused physio streams are skipped);
    * MAE/timm ViT -> ``core.multimae.load_pretrained_encoder`` maps
      ``blocks.*->enc_blocks.*`` and inflates the 2-D patch embed into the
      rgb Conv3d.

    Unlike the old ``load_state_dict(..., strict=False)`` call in
    ``run_waveform`` (which silently loaded NOTHING because Stage-2 keys are
    ``enc_blocks.*`` while ``ProjectViT`` uses ``blocks.*``), this RAISES when
    no encoder tensor matches.
    """
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = _unwrap_state(ckpt)

    if any(k.startswith('waveform_head.') for k in state):
        raise ValueError(
            f'{path} looks like a STAGE-3 checkpoint (it contains '
            f'waveform_head.*). Fine-tuning a fine-tuned head defeats the '
            f'purpose; pass a Stage-2 pre-training checkpoint (or a Stage-1 '
            f'MAE/ViT init) instead, or start from scratch by dropping '
            f'--finetune.')

    is_stage2 = any(k.startswith(('enc_blocks.', 'enc_norm.')) for k in state)

    if not is_stage2:
        print(f'[stage3] {path} has no enc_blocks.* keys -> treating it as a '
              f'Stage-1 MAE/timm ViT init (blocks.* -> enc_blocks.*, 2-D '
              f'patch embed inflated into the rgb Conv3d tubelet). If this was '
              f'a Stage-2 checkpoint, it is in the wrong format.')
        from .multimae import load_pretrained_encoder
        info = load_pretrained_encoder(model, path)
        if info['loaded'] == 0:
            raise RuntimeError(
                f'{path}: no encoder tensors matched -- the file is neither a '
                f'Stage-2 multimodal MAE checkpoint (enc_blocks.*) nor a '
                f'MAE/timm ViT checkpoint (blocks.*). Loaded nothing, so the '
                f'encoder would silently start from random weights.')
        return info

    cur = model.state_dict()
    src_qkv = state.get('enc_blocks.0.attn.qkv.weight')
    tgt_qkv = cur.get('enc_blocks.0.attn.qkv.weight')
    if src_qkv is not None and tgt_qkv is not None and \
            tuple(src_qkv.shape) != tuple(tgt_qkv.shape):
        raise ValueError(
            f'Stage-2 -> Stage-3 geometry mismatch: checkpoint embed dim '
            f'{src_qkv.shape[0]} != model enc_embed_dim {tgt_qkv.shape[0]}. '
            f'Mirror the Stage-2 --model variant / enc_* geometry in the '
            f'Stage-3 config.')

    new_state, loaded, skipped, mism = {}, [], [], []
    for k, v in state.items():
        if not k.startswith(('enc_blocks.', 'enc_norm.', 'adapters.',
                             'positions.')):
            continue                        # decoder / heads / loss buffers
        if k not in cur:
            skipped.append(k)               # e.g. positions.*.mask_token
            continue
        if tuple(cur[k].shape) != tuple(v.shape):
            mism.append(k)
            continue
        new_state[k] = v
        loaded.append(k)

    if not loaded:
        raise RuntimeError(
            f'{path}: a Stage-2 checkpoint was detected but no tensor was '
            f'copied (0 loaded, {len(skipped)} skipped, {len(mism)} shape-'
            f'mismatched) -- check --streams (the Stage-3 visual streams must '
            f'exist in the Stage-2 run) and the geometry.')
    if mism:
        raise ValueError(
            f'{path}: {len(mism)} of the Stage-2 tensors have the wrong shape '
            f'for this Stage-3 geometry (e.g. {mism[:3]}). A partially loaded '
            f'encoder is worse than a loud failure -- reproduce the Stage-2 '
            f'geometry exactly (input_size/tubelet/clip_duration/fps/'
            f'temporal_stride/enc_*).')

    model.load_state_dict(new_state, strict=False)
    print(f'[stage3] loaded {len(loaded)} Stage-2 encoder tensors from {path}, '
          f'{len(skipped)} skipped (decoder/heads/unused physio streams), '
          f'{len(mism)} shape-mismatched.')
    return {'loaded': len(loaded), 'skipped': len(skipped),
            'shape_mismatch': len(mism)}
