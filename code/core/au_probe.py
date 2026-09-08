"""AU-occurrence probe over the Stage-2 encoder -- ADD-ON / diagnostic.

This module does NOT touch ``core/multimae.MultiModalMAE`` or any Stage-1/2/3
path. It re-uses the Stage-2 multimodal MAE *encoder* verbatim -- identical
sub-modules and key names (``adapters.<s>.*``, ``positions.<s>.*``,
``enc_blocks.*``, ``enc_norm``) -- with the masking/decoder removed and a
multi-label classification head appended. Because the key names and geometry
are identical, a Stage-2 pre-training checkpoint (or the Stage-1 MAE/ImageNet
checkpoint) loads straight into it with ``load_state_dict(..., strict=False)``.

Probe semantics (semantic-representation-quality control)
---------------------------------------------------------
* Input: visual stream(s) ONLY (rgb / rgb+tir), all tokens visible.
  Physiological 1-D signals are intentionally NOT fed (they would leak
  autonomic state into the AU prediction and break the "visual semantics"
  claim).
* The Stage-2 encoder has no CLS token, so the read-out is mean-pooling over
  the encoded visual tokens (MAE-style), followed by one linear head.
* ``--probe linear`` freezes the encoder (the formal diagnostic: are AUs
  linearly decodable from the frozen features?); ``--probe ft`` fine-tunes.
"""
from typing import Dict, Optional, Sequence, Tuple

import math

import torch
import torch.nn as nn

from .blocks import Block, trunc_normal_
from .multimae import TubeletEmbed, _PosMask

__all__ = ['MultiModalMAEProbe', 'build_au_probe_model', 'load_au_probe_weights']

_VISUAL_STREAMS = ('rgb', 'tir')


def _parse_int_csv(v, dtype=int):
    return tuple(dtype(x) for x in str(v).split(','))


def _resize_t(x: torch.Tensor, n: int) -> torch.Tensor:
    """Slice/pad the time dim (dim 2) of a [B, C, T, ...] tensor to ``n``."""
    T = x.shape[2]
    if T == n:
        return x
    if T > n:
        return x[:, :, :n]
    idx = torch.arange(n, device=x.device) % T
    return x.index_select(2, idx)


class MultiModalMAEProbe(nn.Module):
    """Stage-2 shared encoder + mean-pool + AU classification head.

    Constructor geometry MUST reproduce the pre-training checkpoint geometry
    (``embed_dim``/``enc_depth``/``enc_num_heads``/``tubelet``/``input_size``/
    ``num_frames``); a mismatch is detected and raises in
    :func:`load_au_probe_weights`.
    """

    def __init__(self, streams: Sequence[str] = ('rgb',),
                 embed_dim: int = 192, enc_depth: int = 6,
                 enc_num_heads: int = 6, mlp_ratio: float = 4.0,
                 num_classes: int = 12, pool: str = 'mean',
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 tubelet: Tuple[int, int, int] = (2, 16, 16),
                 input_size: int = 64, num_frames: int = 100):
        super().__init__()
        self.streams = list(streams)
        if not self.streams:
            raise ValueError('MultiModalMAEProbe: needs >=1 visual stream.')
        unknown = [s for s in self.streams if s not in _VISUAL_STREAMS]
        if unknown:
            raise ValueError(
                f'MultiModalMAEProbe: unknown stream(s) {unknown}; allowed '
                f'{_VISUAL_STREAMS}')
        if pool != 'mean':
            raise NotImplementedError(
                f'MultiModalMAEProbe: only pool="mean" is implemented '
                f'(Stage-2 encoder has no CLS token); got pool={pool}')
        self.pool = pool
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.tubelet = tubelet
        self.linear_probe = False

        t, ph, pw = tubelet
        assert input_size % ph == 0 and input_size % pw == 0, \
            f'patch {tubelet} must divide input_size {input_size}'
        assert num_frames % t == 0, \
            f'tubelet_t {t} must divide num_frames {num_frames}'
        Gh = Gw = input_size // ph
        Gt = num_frames // t
        self.n_visual = Gt * Gh * Gw

        # --- visual adapters + per-stream positions (identical to Stage 2) - #
        self.adapters = nn.ModuleDict()
        self.positions = nn.ModuleDict()
        for s in self.streams:
            in_ch = 3 if s == 'rgb' else 1
            self.adapters[s] = TubeletEmbed(in_ch, embed_dim, tubelet)
            self.positions[s] = _PosMask(self.n_visual, embed_dim)

        # --- shared encoder (identical key layout to Stage 2) -------------- #
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, enc_depth)]
        self.enc_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=enc_num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=nn.LayerNorm)
            for i in range(enc_depth)
        ])
        self.enc_norm = nn.LayerNorm(embed_dim)

        # --- multi-label head ---------------------------------------------- #
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------------ #
    def freeze_features(self):
        """Linear-probe mode: freeze every feature layer, head stays trainable."""
        self.linear_probe = True
        for name, p in self.named_parameters():
            if not name.startswith('head.'):
                p.requires_grad = False

    def set_train(self, mode: bool = True):
        """Train/eval switch. In linear-probe mode the frozen feature layers
        (incl. their dropout) stay in eval; only the head trains."""
        if self.linear_probe:
            for m in (self.adapters, self.enc_blocks, self.enc_norm):
                m.train(False)
            self.head.train(mode)
        else:
            self.train(mode)

    def forward(self, x):
        """``x``: dict {stream: [B, C, T, H, W]} or a single rgb [B, 3, T, H, W]."""
        if torch.is_tensor(x):
            x = {'rgb': x}
        for s in self.streams:
            if s not in x:
                raise ValueError(
                    f'MultiModalMAEProbe: stream "{s}" missing from input; '
                    f'got keys {list(x)}')
        parts = []
        for s in self.streams:
            xt = _resize_t(x[s], self.num_frames)     # [B, C, T, H, W]
            tok = self.adapters[s](xt)                # [B, N, D]
            tok = tok + self.positions[s].pos_embed
            parts.append(tok)
        z = torch.cat(parts, dim=1)
        for blk in self.enc_blocks:
            z = blk(z)
        z = self.enc_norm(z)
        z = z.mean(dim=1)                             # [B, D] (mean-pool)
        return self.head(z)                           # [B, num_classes]


# --------------------------------------------------------------------------- #
# builder (mirrors run_pretrain geometry arg names)
# --------------------------------------------------------------------------- #
def build_au_probe_model(args, num_classes: int):
    """Construct the probe from a run_au_probe ``args`` namespace."""
    streams = tuple(s.strip() for s in
                    str(getattr(args, 'streams', 'rgb')).split(',') if s.strip())
    if not streams or any(s not in _VISUAL_STREAMS for s in streams):
        raise ValueError(
            f'build_au_probe_model: --streams must be a non-empty visual '
            f'subset of {_VISUAL_STREAMS} (AU probe is visual-only); '
            f'got {streams}')
    tubelet = _parse_int_csv(getattr(args, 'tubelet', '2,16,16'))
    fps = float(getattr(args, 'fps', 25.0))
    clip_duration = float(getattr(args, 'clip_duration', 4.0))
    num_frames = int(getattr(args, 'num_frames', 0)) or max(
        1, int(round(clip_duration * fps)))
    pool = getattr(args, 'pool', 'mean')
    return MultiModalMAEProbe(
        streams=streams,
        embed_dim=int(getattr(args, 'enc_embed_dim', 192)),
        enc_depth=int(getattr(args, 'enc_depth', 6)),
        enc_num_heads=int(getattr(args, 'enc_num_heads', 6)),
        mlp_ratio=float(getattr(args, 'mlp_ratio', 4.0)),
        num_classes=int(num_classes), pool=pool,
        drop_rate=float(getattr(args, 'drop_rate', 0.0)),
        attn_drop_rate=float(getattr(args, 'attn_drop_rate', 0.0)),
        drop_path_rate=float(getattr(args, 'drop_path_rate', 0.0)),
        tubelet=tubelet,
        input_size=int(getattr(args, 'input_size', 64)),
        num_frames=num_frames)


# --------------------------------------------------------------------------- #
# checkpoint loading (Stage-2 multimodal OR Stage-1 MAE/ViT)
# --------------------------------------------------------------------------- #
def load_au_probe_weights(model: MultiModalMAEProbe, path: str):
    """Load encoder weights from either checkpoint layout:

    * Stage-2 multimodal MAE (``enc_blocks.*`` / ``enc_norm`` / ``adapters.*``
      / ``positions.*``): matched directly.
    * MAE / timm ViT (``blocks.*`` / ``norm.*`` / ``patch_embed.proj.*``): keys
      mapped ``blocks->enc_blocks``, ``norm->enc_norm``; the 2-D rgb Conv2d is
      inflated along the tubelet time axis into the rgb Conv3d.

    The head, decoder, and any non-visual stream weights are not used.
    """
    # Stage checkpoints carry an 'args' (argparse.Namespace) field, which
    # torch >=2.6 rejects under the default weights_only=True. The files are
    # produced by this project, so load with weights_only=False.
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ('model', 'state_dict', 'module'):
            if isinstance(ckpt.get(key), dict):
                state = ckpt[key]
                break
    if isinstance(state, dict) and isinstance(state.get('module'), dict):
        state = state['module']

    cur = model.state_dict()
    # --- geometry guard (never silently load nothing) ---------------------- #
    src_key = ('enc_blocks.0.attn.qkv.weight'
               if 'enc_blocks.0.attn.qkv.weight' in state
               else 'blocks.0.attn.qkv.weight')
    if src_key in state and 'enc_blocks.0.attn.qkv.weight' in cur:
        if tuple(state[src_key].shape) != tuple(
                cur['enc_blocks.0.attn.qkv.weight'].shape):
            raise ValueError(
                f'Probe geometry mismatch: checkpoint embed dim '
                f'{state[src_key].shape[0]} != probe enc_embed_dim '
                f'{cur["enc_blocks.0.attn.qkv.weight"].shape[0]}. Reproduce '
                f'the Stage-2 geometry (enc_embed_dim/enc_depth/enc_num_heads/'
                f'tubelet/input_size/clip_duration) in the AU config.')

    new_state = {}
    loaded, skipped, mism = [], [], []
    for src_key, v in state.items():
        dst_key = None
        if src_key.startswith(('enc_blocks.', 'enc_norm.', 'adapters.',
                               'positions.')):
            dst_key = src_key
        elif src_key.startswith('blocks.'):
            dst_key = 'enc_blocks.' + src_key[len('blocks.'):]
        elif src_key.startswith('norm.'):
            dst_key = 'enc_norm.' + src_key[len('norm.'):]
        elif src_key in ('patch_embed.proj.weight', 'patch_embed.proj.bias'):
            name = src_key.rsplit('.', 1)[-1]
            dst_key = f'adapters.rgb.patch_embed.{name}'
            if name == 'weight' and dst_key in cur and v.ndim == 4 \
                    and cur[dst_key].ndim == 5:
                t = model.tubelet[0]
                # [out, in, ph, pw] -> [out, in, t, ph, pw] (averaged tube)
                v = v.unsqueeze(2).expand(-1, -1, t, -1, -1).contiguous() / t
        # cls_token / pos_embed(MAE) / decoder / heads / signals: unused
        if dst_key is None:
            continue
        if dst_key not in cur:
            skipped.append(src_key)
            continue
        if tuple(cur[dst_key].shape) != tuple(v.shape):
            mism.append(src_key)
            continue
        new_state[dst_key] = v
        loaded.append(src_key)

    model.load_state_dict(new_state, strict=False)
    print(f'[au_probe] loaded {len(loaded)} encoder tensors from {path}, '
          f'{len(skipped)} skipped, {len(mism)} shape-mismatched.')
    if mism:
        print(f'[au_probe] shape-mismatched keys (first 5): {mism[:5]}')
    return {'loaded': len(loaded), 'skipped': len(skipped),
            'shape_mismatch': len(mism)}
