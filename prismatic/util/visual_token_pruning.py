"""
visual_token_pruning.py

Training-free, spatially structured pruning of the post-projection visual-token sequence.

Adapted from "S^2Prune: Spatially Structured Visual Token Pruning for Multimodal Large Language
Models" (arXiv:2609.01224). The paper observes that importance/redundancy scores carry stable
spatial biases and often lose to plain Uniform-Grid sampling, so it prunes while *preserving spatial
coverage* and then adapts token density to local structure:

    1. Divide the patch grid into regions and guarantee >= 1 kept token per region (coverage).
    2. Distribute the remaining budget across regions in proportion to a structure signal
       (Laplacian variation), so richer regions keep more tokens.
    3. Select representative tokens within each region.

This is a Mode-2 adaptation. The core mechanism above is implemented at full fidelity, operating on
the projected patch embeddings already in hand at the projector output (`[bsz, num_patches, dim]`).
Two auxiliary signals are replaced with parameter-free, target-native proxies so the method stays
training-free and needs no extra hooks:

    - S^2Prune computes Laplacian variation on the *raw image*. Here it is computed on the projected
      token-feature grid (a discrete Laplacian over per-token feature energy). Same "more structure =>
      more tokens" signal, from tensors already available post-projection.
    - S^2Prune ranks within-region tokens with Early Representation Change (ERC), which requires
      hooking the LLM's first decoder block. That hook is out of scope for a projector-output
      integration, so representativeness is proxied by each token's feature deviation from its region
      mean (the most distinctive token summarizes the region). No decoder-block read is needed.

The paper's separate benchmark/eval framework (VQAv2 etc.) is intentionally not ported; evaluation
belongs in a downstream change.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def _region_bounds(length: int, splits: int) -> list[Tuple[int, int]]:
    """Split ``range(length)`` into ``splits`` contiguous, non-empty spans (numpy.array_split style)."""
    splits = max(1, min(splits, length))
    base, extra = divmod(length, splits)
    bounds, start = [], 0
    for i in range(splits):
        stop = start + base + (1 if i < extra else 0)
        bounds.append((start, stop))
        start = stop
    return bounds


def _allocate_budget(region_scores: torch.Tensor, region_sizes: list[int], budget: int) -> list[int]:
    """Give each region >= 1 token, then split the remainder proportional to ``region_scores``.

    Allocations are clamped to each region's token count and any spillover is redistributed to
    regions with spare capacity, so the returned counts always sum to exactly ``budget``.
    """
    num_regions = len(region_sizes)
    alloc = [1] * num_regions
    remaining = budget - num_regions

    weights = torch.clamp(region_scores, min=0.0)
    if float(weights.sum()) <= 0.0:
        weights = torch.ones_like(weights)
    weights = weights / weights.sum()

    # Largest-remainder apportionment of the leftover budget.
    exact = [float(weights[i]) * remaining for i in range(num_regions)]
    extra = [int(math.floor(e)) for e in exact]
    leftover = remaining - sum(extra)
    for i in sorted(range(num_regions), key=lambda j: exact[j] - extra[j], reverse=True)[:leftover]:
        extra[i] += 1
    alloc = [a + e for a, e in zip(alloc, extra)]

    # Clamp to capacity, then push spilled tokens onto regions that still have room.
    alloc = [min(a, s) for a, s in zip(alloc, region_sizes)]
    spill = budget - sum(alloc)
    while spill > 0:
        room = [s - a for a, s in zip(alloc, region_sizes)]
        if sum(room) <= 0:
            break
        order = sorted(range(num_regions), key=lambda j, r=room: (r[j], float(region_scores[j])), reverse=True)
        for i in order:
            if spill == 0:
                break
            if room[i] > 0:
                alloc[i] += 1
                spill -= 1
    return alloc


def prune_visual_tokens(
    tokens: torch.Tensor,
    budget: int,
    region_grid: Tuple[int, int] = (2, 2),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prune ``[bsz, num_patches, dim]`` visual tokens to ``budget`` per example, preserving coverage.

    The patch sequence is assumed to be a flattened square grid (row-major). If it is not a perfect
    square, or is already within ``budget``, the tokens are returned unchanged so the call is safe to
    place unconditionally after the projector.

    Returns the pruned tokens ``[bsz, budget, dim]`` and the kept indices ``[bsz, budget]`` (long),
    with each example's indices sorted so original patch order is preserved downstream.
    """
    if budget <= 0:
        raise ValueError(f"`budget` must be positive, got {budget}.")

    bsz, num_patches, dim = tokens.shape
    side = int(round(math.sqrt(num_patches)))
    if side * side != num_patches or num_patches <= budget:
        idx = torch.arange(num_patches, device=tokens.device).unsqueeze(0).expand(bsz, -1)
        return tokens, idx

    rows, cols = _region_bounds(side, region_grid[0]), _region_bounds(side, region_grid[1])
    regions = [(r0, r1, c0, c1) for (r0, r1) in rows for (c0, c1) in cols]
    if budget < len(regions):
        # Cannot honor >= 1 token per region under this budget => coarsen to a single region.
        regions = [(0, side, 0, side)]

    grid = tokens.reshape(bsz, side, side, dim)

    # Per-token feature energy, then a 4-neighbour discrete Laplacian => local structure magnitude.
    energy = grid.float().pow(2).mean(dim=-1)  # [bsz, side, side]
    lap = torch.zeros_like(energy)
    lap[:, 1:, :] += energy[:, 1:, :] - energy[:, :-1, :]
    lap[:, :-1, :] += energy[:, :-1, :] - energy[:, 1:, :]
    lap[:, :, 1:] += energy[:, :, 1:] - energy[:, :, :-1]
    lap[:, :, :-1] += energy[:, :, :-1] - energy[:, :, 1:]
    variation = lap.abs()  # [bsz, side, side]

    flat_index = torch.arange(num_patches, device=tokens.device).reshape(side, side)
    kept = torch.empty((bsz, budget), dtype=torch.long, device=tokens.device)
    for b in range(bsz):
        region_var = torch.stack([variation[b, r0:r1, c0:c1].sum() for (r0, r1, c0, c1) in regions])
        region_sizes = [(r1 - r0) * (c1 - c0) for (r0, r1, c0, c1) in regions]
        alloc = _allocate_budget(region_var, region_sizes, budget)

        picks: list[torch.Tensor] = []
        for (r0, r1, c0, c1), k in zip(regions, alloc):
            if k <= 0:
                continue
            patch_ids = flat_index[r0:r1, c0:c1].reshape(-1)
            feats = grid[b, r0:r1, c0:c1, :].reshape(-1, dim).float()
            # ERC proxy: distance from region mean => most distinctive/representative tokens first.
            score = (feats - feats.mean(dim=0, keepdim=True)).norm(dim=-1)
            sel = torch.topk(score, k=min(k, patch_ids.numel())).indices
            picks.append(patch_ids[sel])
        kept[b] = torch.sort(torch.cat(picks))[0]

    pruned = torch.gather(tokens, 1, kept.unsqueeze(-1).expand(-1, -1, dim))
    return pruned, kept


class SpatialTokenPruner:
    """Callable wrapper storing a pruning ``budget`` and ``region_grid`` (see :func:`prune_visual_tokens`)."""

    def __init__(self, budget: int, region_grid: Tuple[int, int] = (2, 2)) -> None:
        self.budget = budget
        self.region_grid = region_grid

    def __call__(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return prune_visual_tokens(tokens, self.budget, self.region_grid)

    def __repr__(self) -> str:
        return f"SpatialTokenPruner(budget={self.budget}, region_grid={self.region_grid})"
