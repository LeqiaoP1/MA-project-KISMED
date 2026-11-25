"""Symmetric cross-modal InfoNCE alignment for Stage-2 pre-training (ADR-008 §2.3).

The regularizer aligns pooled thermal (`tir`) and physiological (`resp`) encoder
latents **before** either decoder, so respiratory dynamics inject a direct
gradient into the shared visual Transformer blocks instead of only through the
pixel-reconstruction path (ADR-008 §2.3.3, "gradient directivity").

Three pieces:

``AlignProjection``
    the per-modality 2-layer MLP ``D -> D -> d_proj``. It exists so contrastive
    dimensional collapse cannot erode the spatial detail the visual decoder
    needs — the projection is discarded at Stage 3 (its ``align_proj.`` key
    prefix is outside ``core.waveform_model.load_stage2_encoder``'s whitelist).

``anatomy_weighted_pool``
    mean pooling over a modality's visible encoder tokens, with the anatomical
    core tubes (ADR-007 ``A_core``) weighted by ``align_core_weight``. It is
    deliberately branch-free: an empty ``A_core`` (the ``uniform`` ablation) and
    ``core_weight = 1.0`` produce the unweighted mean through the *same*
    arithmetic, so the neutrality claim is testable rather than asserted.

``symmetric_infonce``
    ``0.5 * (CE(tir->resp) + CE(resp->tir))`` over FP32 cosine logits, with
    optional cross-rank negatives.

FP32 (§2.4 contract 4)
----------------------
Normalisation, the cosine matmul and the cross-entropy all run in float32, with
no cast back to the autocast compute dtype. A half-precision cosine logit at
``tau = 0.07`` loses enough mantissa to change the ranking of negatives.

Distributed gathering (§2.3.3, §2.4 contract 6)
-----------------------------------------------
``gather`` is decided by the CALLER as ``align_gather_ddp and model.training``:
``dist.all_gather`` requires identically shaped tensors on every rank, and the
validation loader runs ``drop_last=False``, so an unpadded final validation
batch would otherwise deadlock. Remote slices are detached — only the local
rank's rows carry gradient — exactly as §2.3.3 specifies.
"""
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AlignProjection', 'anatomy_weighted_pool', 'symmetric_infonce']

_EPS = 1e-8


class AlignProjection(nn.Module):
    """``Linear(D, D) -> GELU -> Linear(D, d_proj)`` (ADR-008 §2.3.2).

    No normalisation inside: the L2 normalisation is a separate FP32 step in
    :func:`symmetric_infonce`, which keeps this module dtype-transparent and
    keeps the unit-sphere constraint in one place.

    :param embed_dim: the shared encoder width ``D``.
    :param proj_dim: ``d_proj`` (ADR-008 §2.8, default 128, minimum 16).
    """

    def __init__(self, embed_dim: int, proj_dim: int = 128):
        super().__init__()
        embed_dim, proj_dim = int(embed_dim), int(proj_dim)
        if embed_dim < 1:
            raise ValueError(
                f'AlignProjection: embed_dim must be >= 1, got {embed_dim}')
        if proj_dim < 16:
            raise ValueError(
                f'AlignProjection: proj_dim must be >= 16 (ADR-008 §2.6 rule '
                f'4), got {proj_dim}')
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(embed_dim, proj_dim)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """``[B, D]`` pooled latent -> ``[B, d_proj]`` pre-normalisation."""
        if u.dim() != 2 or u.shape[-1] != self.embed_dim:
            raise ValueError(
                f'AlignProjection: expected [B, {self.embed_dim}], got '
                f'{tuple(u.shape)}.')
        return self.fc2(self.act(self.fc1(u)))

    def extra_repr(self) -> str:
        return f'{self.embed_dim}->{self.embed_dim}->{self.proj_dim}'


def anatomy_weighted_pool(z: torch.Tensor, ids_keep: torch.Tensor,
                          core_indices: Iterable[int], n_spatial: int,
                          core_weight: float = 1.0) -> torch.Tensor:
    """Anatomy-weighted mean pooling of one visual stream (ADR-008 §2.3.1).

    ``u_b = sum_i w_bi * Z_bi / sum_i w_bi`` with ``w_bi = core_weight`` when the
    token's spatial tube lies in ``A_core`` and ``1.0`` otherwise. The spatial
    tube is recovered as ``ids_keep % n_spatial`` because both placement
    routines emit the encoder index ``t * n_units + s``.

    :param z: ``[B, K]``-indexed latents, i.e. ``[B, K_tir, D]``.
    :param ids_keep: ``[B, K_tir]`` kept encoder indices (ascending).
    :param core_indices: the flat tube indices of ``A_core`` (empty under
        ``spatial_sampling = 'uniform'``).
    :param n_spatial: the number of spatial tubes (``Gh * Gw``).
    :param core_weight: ``w_core`` (ADR-008 §2.8). ``1.0`` or an empty
        ``A_core`` reduces to the plain mean through the same arithmetic.
    """
    if z.dim() != 3:
        raise ValueError(
            f'anatomy_weighted_pool: expected [B, K, D] latents, got '
            f'{tuple(z.shape)}.')
    if ids_keep.dim() != 2 or tuple(ids_keep.shape) != tuple(z.shape[:2]):
        raise ValueError(
            f'anatomy_weighted_pool: ids_keep {tuple(ids_keep.shape)} does not '
            f'match the latent grid {tuple(z.shape[:2])}.')
    if int(n_spatial) < 1:
        raise ValueError(
            f'anatomy_weighted_pool: n_spatial must be >= 1, got {n_spatial}')
    if float(core_weight) <= 0.0:
        raise ValueError(
            'anatomy_weighted_pool: core_weight must be > 0 (ADR-008 §2.6 '
            f'rule 4), got {core_weight}')

    core = torch.as_tensor(list(core_indices), dtype=torch.long,
                           device=z.device)
    spatial = ids_keep.to(torch.long) % int(n_spatial)
    # empty `core` -> isin is all-False -> the weights stay 1.0 (branch-free)
    hit = torch.isin(spatial, core)
    w = torch.ones_like(spatial, dtype=z.dtype)
    w = torch.where(hit, torch.full_like(w, float(core_weight)), w)
    return (w.unsqueeze(-1) * z).sum(dim=1) / w.sum(dim=1, keepdim=True)


def _l2_normalize(x: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """FP32 L2 normalisation onto the unit hypersphere, ``max(norm, eps)``."""
    x = x.float()
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def _gather_keep_local_grad(v: torch.Tensor, world: int,
                            rank: int) -> torch.Tensor:
    """All-gather ``v`` with gradient preserved ONLY on this rank's slice.

    ``dist.all_gather`` itself is not differentiable, so the remote rows are
    detached and the local rows are re-inserted from the original tensor.
    """
    outs = [torch.zeros_like(v) for _ in range(world)]
    torch.distributed.all_gather(outs, v.detach().contiguous())
    outs[rank] = v
    return torch.cat(outs, dim=0)


def symmetric_infonce(v_tir: torch.Tensor, v_resp: torch.Tensor,
                      temperature: float = 0.07, gather: bool = True,
                      eps: float = _EPS) -> torch.Tensor:
    """Symmetric InfoNCE over cosine similarity (ADR-008 §2.3.3).

    :param v_tir: ``[B, d_proj]`` thermal projection (pre-normalisation).
    :param v_resp: ``[B, d_proj]`` respiratory projection (pre-normalisation).
    :param temperature: ``tau > 0``.
    :param gather: gather embeddings across ranks for global negatives. Pass
        ``align_gather_ddp and model.training`` — never the raw knob.
    :param eps: normalisation stabiliser.
    :returns: scalar loss, or an explicit ``0.0`` when fewer than two rows are
        available (e.g. ``B = 1`` on a single device).
    """
    if float(temperature) <= 0.0:
        raise ValueError(
            f'symmetric_infonce: temperature must be > 0, got {temperature}')
    if v_tir.dim() != 2 or tuple(v_tir.shape) != tuple(v_resp.shape):
        raise ValueError(
            f'symmetric_infonce: expected matching [B, d] embeddings, got '
            f'{tuple(v_tir.shape)} / {tuple(v_resp.shape)}.')

    a = _l2_normalize(v_tir, eps)
    b = _l2_normalize(v_resp, eps)

    world, rank = 1, 0
    if (gather and torch.distributed.is_available()
            and torch.distributed.is_initialized()):
        world = int(torch.distributed.get_world_size())
        rank = int(torch.distributed.get_rank())
    if world > 1:
        gathered_tir = _gather_keep_local_grad(a, world, rank)
        gathered_resp = _gather_keep_local_grad(b, world, rank)
    else:
        gathered_tir, gathered_resp, rank = a, b, 0

    if gathered_tir.shape[0] < 2:
        # a single row has no negative and no defined positive rank (§2.3.3)
        return a.new_zeros(())

    labels = (torch.arange(a.shape[0], device=a.device, dtype=torch.long)
              + rank * a.shape[0])
    logits_t2r = (a @ gathered_resp.t().float()) / float(temperature)
    logits_r2t = (b @ gathered_tir.t().float()) / float(temperature)
    return 0.5 * (F.cross_entropy(logits_t2r, labels)
                  + F.cross_entropy(logits_r2t, labels))
