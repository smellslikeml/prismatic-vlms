"""
visual_token_pruning.py

Training-free, spatially-structured pruning of the projected visual-token grid, adapted from S^2Prune
("Spatially Structured Visual Token Pruning for Multimodal Large Language Models", arXiv:2609.01224).

The core insight of S^2Prune is that *broad spatial coverage* matters more than raw importance/redundancy
rankings: uniform-grid sampling is a strong baseline, so a good pruner should (1) guarantee every image region
keeps at least one token, then (2) spend the remaining budget where local structure is richest.

This module keeps that mechanism at full fidelity while substituting two auxiliary components with
parameter-free, target-native proxies (the projector-output grid is all we have inside `PrismaticVLM.forward`):

  * Laplacian variation is computed on the *projected patch-embedding grid* rather than on raw pixels. It still
    measures local structural richness, but from features the VLM already produces (no image handle required).
  * Early Representation Change (ERC), which S^2Prune derives from the first decoder block, is replaced by the
    same parameter-free Laplacian saliency for intra-region representative selection. This removes the dependency
    on running a partial LLM forward pass while preserving the "keep the tokens carrying local structure" intent.

Everything here is stateless and holds no learnable parameters, so it is safe to toggle at inference time on an
already-trained checkpoint without altering the model architecture.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def laplacian_saliency(grid: torch.Tensor) -> torch.Tensor:
    """Per-token structural richness via a fixed 4-neighbor discrete Laplacian over the H x W feature grid.

    :param grid: Feature grid of shape [B, H, W, D].
    :return: Non-negative saliency map of shape [B, H, W] (L2 norm of the per-token Laplacian response).
    """
    # [B, H, W, D] -> [B, D, H, W] for spatial convolution-style neighbor access
    x = grid.permute(0, 3, 1, 2).float()

    # Replicate padding so border tokens get a well-defined (attenuated) Laplacian instead of wrap-around artifacts
    padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
    center = padded[..., 1:-1, 1:-1]
    up = padded[..., 0:-2, 1:-1]
    down = padded[..., 2:, 1:-1]
    left = padded[..., 1:-1, 0:-2]
    right = padded[..., 1:-1, 2:]

    laplacian = 4.0 * center - up - down - left - right
    return laplacian.norm(dim=1)


def _allocate_region_budget(
    region_weight: torch.Tensor, region_capacity: torch.Tensor, keep_tokens: int
) -> torch.Tensor:
    """Allocate `keep_tokens` across regions: >=1 per region (coverage), remainder by weight (structure).

    Uses the largest-remainder (Hamilton) method, capped by per-region token capacity with redistribution of any
    overflow onto non-saturated regions. Falls back to uniform weighting when total structural weight is zero.

    :param region_weight: Aggregated saliency per region, shape [R].
    :param region_capacity: Number of available tokens per region, shape [R].
    :param keep_tokens: Total tokens to keep across all regions.
    :return: Integer allocation per region, shape [R], summing to `keep_tokens`.
    """
    num_regions = region_weight.shape[0]
    alloc = torch.zeros(num_regions, dtype=torch.long, device=region_weight.device)

    # Budget too small to cover every region: hand a single token to the highest-weight regions only.
    if keep_tokens <= num_regions:
        top = torch.topk(region_weight, k=keep_tokens).indices
        alloc[top] = 1
        return alloc

    # Coverage guarantee: one token per region, then distribute the remainder by structural weight.
    alloc[:] = 1
    remaining = keep_tokens - num_regions

    while remaining > 0:
        headroom = region_capacity - alloc
        active = headroom > 0
        if not bool(active.any()):
            break

        weight = region_weight.clone()
        weight[~active] = 0.0
        total = float(weight.sum())
        if total <= 0.0:
            # No structure signal (or all saturated regions carry the weight) -> spread uniformly over active regions.
            weight = active.float()
            total = float(weight.sum())

        raw = weight / total * remaining
        floor = torch.floor(raw).long()
        floor = torch.minimum(floor, headroom)
        alloc += floor
        granted = int(floor.sum())
        remaining -= granted

        if granted == 0:
            # Fractional round-down stalled progress: award the largest remainders one token at a time.
            headroom = region_capacity - alloc
            frac = torch.where(headroom > 0, raw - torch.floor(raw), torch.full_like(raw, -1.0))
            order = torch.argsort(frac, descending=True)
            for idx in order.tolist():
                if remaining <= 0:
                    break
                if (region_capacity[idx] - alloc[idx]) > 0:
                    alloc[idx] += 1
                    remaining -= 1
            break

    return alloc


def prune_visual_tokens(
    patch_embeddings: torch.Tensor,
    keep_tokens: int,
    num_regions_per_side: int = 4,
) -> torch.Tensor:
    """Spatially-structured pruning of a projected visual-token grid.

    Preserves broad spatial coverage (>=1 token per region) and allocates the remaining budget to regions with
    richer local structure, selecting the most salient tokens within each region.

    :param patch_embeddings: Projected patch tokens, shape [B, N, D], where N is a perfect square (H == W grid).
    :param keep_tokens: Target number of tokens to retain per image. Values >= N are a no-op.
    :param num_regions_per_side: Region grid is `num_regions_per_side` x `num_regions_per_side`.
    :return: Pruned tokens, shape [B, keep_tokens, D], with per-image tokens re-ordered into ascending grid order.
    """
    bsz, num_patches, dim = patch_embeddings.shape

    if keep_tokens >= num_patches:
        return patch_embeddings
    if keep_tokens <= 0:
        raise ValueError(f"`keep_tokens` must be positive, got {keep_tokens}")

    side = int(round(math.sqrt(num_patches)))
    if side * side != num_patches:
        raise ValueError(f"S^2Prune expects a square token grid; got num_patches={num_patches} (not a perfect square)")

    regions_per_side = min(num_regions_per_side, side)
    grid = patch_embeddings.view(bsz, side, side, dim)
    saliency = laplacian_saliency(grid)  # [B, side, side]

    # Map every grid cell to a region id in [0, regions_per_side**2); blocks are as even as possible.
    row_ids = (torch.arange(side, device=patch_embeddings.device) * regions_per_side) // side
    col_ids = (torch.arange(side, device=patch_embeddings.device) * regions_per_side) // side
    region_map = (row_ids.view(side, 1) * regions_per_side + col_ids.view(1, side)).reshape(-1)  # [N]
    num_regions = regions_per_side * regions_per_side
    region_capacity = torch.bincount(region_map, minlength=num_regions)

    flat_saliency = saliency.view(bsz, num_patches)
    keep_indices = torch.empty(bsz, keep_tokens, dtype=torch.long, device=patch_embeddings.device)

    for b in range(bsz):
        # Aggregate saliency per region for this image, then decide how many tokens each region earns.
        region_weight = torch.zeros(num_regions, device=patch_embeddings.device)
        region_weight.scatter_add_(0, region_map, flat_saliency[b])
        alloc = _allocate_region_budget(region_weight, region_capacity, keep_tokens)

        selected = []
        for r in range(num_regions):
            budget = int(alloc[r])
            if budget == 0:
                continue
            member_idx = (region_map == r).nonzero(as_tuple=True)[0]
            # ERC-substitute: keep the `budget` most structurally salient tokens in the region.
            local_saliency = flat_saliency[b][member_idx]
            top_local = torch.topk(local_saliency, k=budget).indices
            selected.append(member_idx[top_local])

        # Re-order into ascending grid position so the pruned sequence keeps a stable raster layout.
        keep_indices[b] = torch.cat(selected).sort().values

    gather_index = keep_indices.unsqueeze(-1).expand(-1, -1, dim)
    return torch.gather(patch_embeddings, dim=1, index=gather_index)


class SpatialTokenPruner:
    """Callable, parameter-free S^2Prune pruner for the projected visual-token grid.

    Held as a plain attribute (not an ``nn.Module``) so it never enters the model's ``state_dict`` or FSDP wrapping
    policy — enabling/disabling it is a pure inference-time toggle that does not touch trained weights.
    """

    def __init__(self, keep_tokens: int, num_regions_per_side: int = 4) -> None:
        if keep_tokens <= 0:
            raise ValueError(f"`keep_tokens` must be positive, got {keep_tokens}")
        if num_regions_per_side <= 0:
            raise ValueError(f"`num_regions_per_side` must be positive, got {num_regions_per_side}")
        self.keep_tokens = keep_tokens
        self.num_regions_per_side = num_regions_per_side

    def __call__(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        return prune_visual_tokens(patch_embeddings, self.keep_tokens, self.num_regions_per_side)

    def __repr__(self) -> str:
        return f"SpatialTokenPruner(keep_tokens={self.keep_tokens}, num_regions_per_side={self.num_regions_per_side})"
