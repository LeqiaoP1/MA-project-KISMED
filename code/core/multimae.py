"""Multimodal masked autoencoder for Stage-2 pre-training (first local milestone).

Streams (milestone): ``rgb`` + ``tir`` (3-D tubelet spatio-temporal video) and
``bvp`` (1-D physiological signal). Pipeline inside ``MultiModalMAE.forward``:

  * per-stream adapters -> tokens (+ learned positional embedding)
  * per-stream ASYMMETRIC masks: tube masks for the videos, random-window for
    the signal (visual 50-75 %, signals 90 %+); masks == 1 => "masked / to
    reconstruct" (matches ``MaskedL1Loss``)
  * visible tokens of ALL streams are concatenated into ONE shared transformer
  * a shared lightweight decoder re-inserts a learned [MASK] token at every
    masked position and each stream reconstructs its NORMALIZED tubelet/window
    patches (VideoMAE-style ``norm_pix``) through a per-stream linear head;
    masked L1 is summed over the streams.

Tensor layouts::

    x = {'rgb': [B, 3, T, H, W], 'tir': [B, 1, T, H, W], 'bvp': [B, 1, S]}

TODO(extend): resp/eda streams, MultiMAE-style prediction-task sampling,
separate deeper decoders, 3-D sincos pos-embed, visual-stream weight sharing.
"""
from typing import Dict, Optional, Sequence, Tuple

import math

import torch
import torch.nn as nn

from .blocks import Block, trunc_normal_
from .criterion import MaskedL1Loss

__all__ = ['TubeletEmbed', 'MultiModalMAE', 'build_pretraining_model',
           'load_pretrained_encoder']

_EPS = 1e-6
_VISUAL_STREAMS = ('rgb', 'tir')


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
    """Learned 1-D position embedding + single [MASK] token for one stream."""

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

    def __init__(self, streams: Sequence[str] = ('rgb', 'tir', 'bvp'),
                 embed_dim: int = 192, enc_depth: int = 6,
                 enc_num_heads: int = 6, mlp_ratio: float = 4.0,
                 dec_depth: int = 2, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0, drop_path_rate: float = 0.0,
                 tubelet: Tuple[int, int, int] = (2, 16, 16),
                 input_size: int = 64, num_frames: int = 100,
                 sig_kernel: int = 8, seq_len: int = 400,
                 mask_ratios: Optional[Dict[str, float]] = None):
        super().__init__()
        self.streams = list(streams)
        self.visual = [s for s in self.streams if s in _VISUAL_STREAMS]
        self.signal = [s for s in self.streams if s not in _VISUAL_STREAMS]

        t, ph, pw = tubelet
        assert input_size % ph == 0 and input_size % pw == 0, \
            f'patch {tubelet} must divide input_size {input_size}'
        assert num_frames % t == 0, \
            f'tubelet_t {t} must divide num_frames {num_frames}'
        self.tubelet = tubelet
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.seq_len = seq_len

        Gh = Gw = input_size // ph
        Gt = num_frames // t
        self.n_visual = Gt * Gh * Gw            # tokens per visual stream
        self.n_signal = seq_len // sig_kernel   # tokens for a signal stream
        assert self.n_signal > 0, 'seq_len < sig_kernel'
        self.sig_kernel = sig_kernel

        # default asymmetric ratios (visual 50-75 %, signals 90 %+)
        ratios = {'rgb': 0.75, 'tir': 0.50, 'bvp': 0.90}
        if mask_ratios:
            ratios.update(mask_ratios)
        self.mask_ratios = {s: ratios.get(s, 0.90) for s in self.streams}

        # --- adapters ----------------------------------------------------- #
        self.adapters = nn.ModuleDict()
        for s in self.streams:
            if s in self.visual:
                in_ch = 3 if s == 'rgb' else 1
                self.adapters[s] = TubeletEmbed(in_ch, embed_dim, tubelet)
            else:
                self.adapters[s] = SignalEmbed(1, embed_dim, sig_kernel)

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

        # flat reconstruction dim per stream (normalized patch of raw values)
        self.heads = nn.ModuleDict()
        self._flat = {}
        for s in self.streams:
            if s in self.visual:
                in_ch = 3 if s == 'rgb' else 1
                f = t * ph * pw * in_ch
            else:
                f = sig_kernel
            self._flat[s] = f
            self.heads[s] = nn.Linear(embed_dim, f)

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
    # masking (masks == 1 => masked / to reconstruct)
    # ------------------------------------------------------------------ #
    def _tube_mask(self, B: int, device, mask_ratio: float):
        """Random subset of spatial patches, replicated across all frames."""
        Gt = self.num_frames // self.tubelet[0]
        Gh = Gw = math.isqrt(self.n_visual // Gt)   # square spatial grid
        n_spatial = Gh * Gw
        k = max(1, min(n_spatial - 1, int(mask_ratio * n_spatial)))
        perm = torch.rand(B, n_spatial, device=device).argsort(dim=1)
        hidden = perm[:, :k]                                   # [B, k]
        m = torch.zeros(B, n_spatial, device=device, dtype=torch.long)
        m.scatter_(1, hidden, 1)                               # [B, n_spatial]
        mask = m.unsqueeze(1).expand(B, Gt, n_spatial)
        return mask.reshape(B, self.n_visual)

    def _random_mask(self, B: int, device, N: int, mask_ratio: float):
        k = max(1, min(N - 1, int(mask_ratio * N)))
        perm = torch.rand(B, N, device=device).argsort(dim=1)
        hidden = perm[:, :k]
        m = torch.zeros(B, N, device=device, dtype=torch.long)
        m.scatter_(1, hidden, 1)
        return m

    def make_masks(self, B: int, device):
        return {
            s: (self._tube_mask(B, device, self.mask_ratios[s])
                if s in self.visual
                else self._random_mask(B, device, self.n_signal,
                                       self.mask_ratios[s]))
            for s in self.streams
        }

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
        else:
            x = x.reshape(x.shape[0], self.n_signal, self.sig_kernel)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + _EPS)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resize_t(x, n: int):
        """Slice/pad the time dim (dim 2) of a [B, C, T, ...] tensor to ``n``."""
        T = x.shape[2]
        if T == n:
            return x
        if T > n:
            return x[:, :, :n]
        idx = torch.arange(n, device=x.device) % T   # replicate last frames
        return x.index_select(2, idx)

    def forward(self, x):
        B = next(iter(x.values())).shape[0]
        device = next(iter(x.values())).device

        # enforce the configured time/signal lengths (token geometry contract)
        xin = {}
        for s in self.streams:
            if s in self.visual:
                xin[s] = self._resize_t(x[s], self.num_frames)
            else:
                xin[s] = self._resize_t(x[s], self.seq_len)

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
        losses = {}
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
            losses[s] = MaskedL1Loss()(pred, tgt, mask_s)
            preds[s] = pred

        total = torch.stack(list(losses.values())).sum()
        return {'loss': total, 'losses': losses, 'masks': masks, 'preds': preds}


# --------------------------------------------------------------------------- #
# builder (mirrors run_pretrain argparse/YAML defaults)
# --------------------------------------------------------------------------- #
def _parse_int_csv(v, dtype=int):
    return tuple(dtype(x) for x in str(v).split(','))


def build_pretraining_model(args):
    """Construct the multimodal MAE from a run_pretrain ``args`` namespace."""
    streams = tuple(x.strip() for x in
                    str(getattr(args, 'streams', 'rgb,tir,bvp')).split(',')
                    if x.strip())
    tubelet = _parse_int_csv(getattr(args, 'tubelet', '2,16,16'))

    clip_duration = float(getattr(args, 'clip_duration', 4.0))
    fps = float(getattr(args, 'fps', 25.0))
    fs = float(getattr(args, 'fs', 100.0))
    num_frames = max(1, int(round(clip_duration * fps)))
    seq_len = int(getattr(args, 'seq_len', 0)) or max(1, int(round(clip_duration * fs)))

    ratios = {'rgb': float(getattr(args, 'mask_ratio_rgb', 0.75)),
              'tir': float(getattr(args, 'mask_ratio_tir', 0.50)),
              'bvp': float(getattr(args, 'mask_ratio_bvp', 0.90))}

    return MultiModalMAE(
        streams=streams,
        embed_dim=int(getattr(args, 'enc_embed_dim', 192)),
        enc_depth=int(getattr(args, 'enc_depth', 6)),
        enc_num_heads=int(getattr(args, 'enc_num_heads', 6)),
        mlp_ratio=float(getattr(args, 'mlp_ratio', 4.0)),
        dec_depth=int(getattr(args, 'dec_depth', 2)),
        tubelet=tubelet,
        input_size=int(getattr(args, 'input_size', 64)),
        num_frames=num_frames,
        sig_kernel=int(getattr(args, 'sig_kernel', 8)),
        seq_len=seq_len,
        mask_ratios=ratios)


# --------------------------------------------------------------------------- #
# weight inheritance: load an ImageNet / MAE ViT encoder into the shared
# encoder of a MultiModalMAE (spatial priors, plan Stage 1 -> Stage 2)
# --------------------------------------------------------------------------- #
def load_pretrained_encoder(model: 'MultiModalMAE', path: str,
                            inflate_rgb_patch: bool = True) -> Dict[str, int]:
    """Copy a (MAE/ViT) checkpoint's transformer weights into the encoder.

    Handles MAE/timm-style key layouts::

        blocks.{i}.*  -> enc_blocks.{i}.*
        norm.*        -> enc_norm.*
        patch_embed.proj.{weight,bias} -> adapters.rgb.patch_embed.*  (2-D
                    Conv2d inflated along the tubelet time axis into Conv3d)

    Everything else (cls_token / pos_embed / head / non-rgb adapters / signal
    streams) is left at its random initialisation. Encoder geometry must match
    the checkpoint (e.g. embed_dim=768, depth=12, heads=12 for ViT-Base).

    :return: counts dict {loaded, skipped, mismatch_shapes, keys}.
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

    cur = model.state_dict()
    new_state = {}
    loaded, skipped, shape_mismatch = [], [], []

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
            dst_key = f'adapters.rgb.patch_embed.{name}'
            if name == 'weight' and inflate_rgb_patch and dst_key in cur:
                t = model.tubelet[0]
                # [out, in, ph, pw] -> [out, in, t, ph, pw] (averaged tube)
                v = v.unsqueeze(2).expand(-1, -1, t, -1, -1).contiguous() / t
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
    print(f'[pretrained] {path}: loaded {n_loaded} encoder tensors, '
          f'{len(skipped)} skipped, {len(shape_mismatch)} shape-mismatched.')
    if shape_mismatch:
        print(f'[pretrained] first shape-mismatched keys: '
              f'{shape_mismatch[:5]}')
    return {'loaded': n_loaded, 'skipped': len(skipped),
            'shape_mismatch': len(shape_mismatch)}
