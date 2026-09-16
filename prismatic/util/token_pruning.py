"""
token_pruning.py

Training-free visual-token pruning utilities for reducing the number of projected patch embeddings fed to the LLM
backbone, cutting inference overhead while preserving broad spatial coverage.

The scheme implemented here is `spatially_structured_prune`, an adaptation of S^2Prune (Spatially Structured Visual
Token Pruning for Multimodal Large Language Models, https://arxiv.org/abs/2609.01224). We keep the paper's core
mechanism at full fidelity:

    1. Reshape the visual tokens onto their 2D grid and split the grid into a small set of regions.
    2. Preserve spatial coverage by assigning at least one token to every region.
    3. Distribute the remaining token budget by *Laplacian variation*, giving more tokens to regions with richer
       local structure.
    4. Select representative tokens *within* each region.

Two auxiliary components of the paper are substituted with parameter-free, target-native equivalents so the method
can run directly on `PrismaticVLM`'s projected patch embeddings without extra infrastructure:

    * Laplacian variation is computed over the *projected patch-embedding grid* rather than the raw image — at this
      call site only the embeddings are available, and their spatial gradient is a faithful stand-in for local
      image structure.
    * The paper's Early Representation Change (ERC) signal — which requires running the first decoder block — is
      replaced by a within-region representativeness proxy: tokens closest to their region's mean embedding are
      kept. This approximates ERC's "pick representative tokens" objective without an extra forward pass.

The paper's separate benchmark / evaluation framework is intentionally out of scope (that belongs downstream).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch


def _infer_square_grid(num_tokens: int) -> Optional[Tuple[int, int]]:
    """Return (H, W) for a square token grid, or None if `num_tokens` is not a perfect square."""
    side = int(round(math.sqrt(num_tokens)))
    if side * side == num_tokens and side > 0:
        return side, side
    return None


def _region_assignment(grid_hw: Tuple[int, int], num_regions_hw: Tuple[int, int]) -> Tuple[List[List[int]], int]:
    """Map each flattened (row-major) token index to a region; return per-region token-index lists + region count."""
    height, width = grid_hw
    n_rows = max(1, min(num_regions_hw[0], height))
    n_cols = max(1, min(num_regions_hw[1], width))
    num_regions = n_rows * n_cols

    region_tokens: List[List[int]] = [[] for _ in range(num_regions)]
    for row in range(height):
        region_row = (row * n_rows) // height
        for col in range(width):
            region_col = (col * n_cols) // width
            region_id = region_row * n_cols + region_col
            region_tokens[region_id].append(row * width + col)
    return region_tokens, num_regions


def _laplacian_variation(tokens: torch.Tensor, grid_hw: Tuple[int, int]) -> torch.Tensor:
    """Per-token local structure score via a 4-neighbour discrete Laplacian over the embedding grid.

    Args:
        tokens: [B, N, D] projected patch embeddings.
        grid_hw: (H, W) with H * W == N.

    Returns:
        [B, N] non-negative variation scores (edges use replicate padding).
    """
    bsz, _, dim = tokens.shape
    height, width = grid_hw
    grid = tokens.reshape(bsz, height, width, dim).float()

    up = torch.cat([grid[:, :1], grid[:, :-1]], dim=1)
    down = torch.cat([grid[:, 1:], grid[:, -1:]], dim=1)
    left = torch.cat([grid[:, :, :1], grid[:, :, :-1]], dim=2)
    right = torch.cat([grid[:, :, 1:], grid[:, :, -1:]], dim=2)

    laplacian = 4.0 * grid - up - down - left - right
    return laplacian.norm(dim=-1).reshape(bsz, height * width)


def _allocate_budget(scores: List[float], sizes: List[int], budget: int) -> List[int]:
    """Split `budget` tokens across regions: coverage floor first, then Laplacian-weighted, capped by region size."""
    num_regions = len(scores)
    alloc = [0] * num_regions
    total_capacity = sum(sizes)
    budget = min(budget, total_capacity)
    if budget <= 0:
        return alloc

    # Not enough budget to cover every region --> keep the highest-structure regions only.
    if budget <= num_regions:
        order = sorted(range(num_regions), key=lambda i: scores[i], reverse=True)
        for i in order[:budget]:
            alloc[i] = 1
        return alloc

    # Coverage: at least one token per region (every region holds >= 1 token by construction).
    for i in range(num_regions):
        alloc[i] = 1
    remaining = budget - num_regions

    # Distribute the remainder proportionally to region structure (largest-remainder rounding).
    score_sum = sum(scores)
    weights = [1.0 / num_regions] * num_regions if score_sum <= 0 else [s / score_sum for s in scores]
    exact = [w * remaining for w in weights]
    base = [int(math.floor(e)) for e in exact]
    for i in range(num_regions):
        alloc[i] += base[i]

    leftover = remaining - sum(base)
    frac_order = sorted(range(num_regions), key=lambda i: exact[i] - base[i], reverse=True)
    cursor = 0
    while leftover > 0:
        alloc[frac_order[cursor % num_regions]] += 1
        leftover -= 1
        cursor += 1

    # Cap each region by how many tokens it actually has, then redistribute any overflow to regions with headroom.
    overflow = 0
    for i in range(num_regions):
        if alloc[i] > sizes[i]:
            overflow += alloc[i] - sizes[i]
            alloc[i] = sizes[i]

    if overflow > 0:
        capacity_order = sorted(range(num_regions), key=lambda i: scores[i], reverse=True)
        while overflow > 0:
            progressed = False
            for i in capacity_order:
                if overflow <= 0:
                    break
                if alloc[i] < sizes[i]:
                    alloc[i] += 1
                    overflow -= 1
                    progressed = True
            if not progressed:
                break

    return alloc


def _select_indices_for_sample(
    sample_tokens: torch.Tensor,
    variation: torch.Tensor,
    region_tokens: List[List[int]],
    budget: int,
) -> torch.Tensor:
    """Pick `budget` token indices for one sample: allocate per region, then keep the most representative tokens."""
    scores = [float(variation[idxs].sum()) for idxs in region_tokens]
    sizes = [len(idxs) for idxs in region_tokens]
    alloc = _allocate_budget(scores, sizes, budget)

    selected: List[int] = []
    for region_idx, keep in enumerate(alloc):
        if keep <= 0:
            continue
        idxs = region_tokens[region_idx]
        embeddings = sample_tokens[idxs].float()
        centroid = embeddings.mean(dim=0, keepdim=True)
        # Representative == closest to the region centroid (parameter-free ERC proxy).
        distances = (embeddings - centroid).norm(dim=-1)
        chosen = torch.topk(distances, k=min(keep, len(idxs)), largest=False).indices
        selected.extend(int(idxs[c]) for c in chosen.tolist())

    # Safety net: top up (or trim) to exactly `budget` using global variation ranking.
    if len(selected) < budget:
        chosen_set = set(selected)
        order = torch.argsort(variation, descending=True).tolist()
        for idx in order:
            if idx not in chosen_set:
                selected.append(idx)
                chosen_set.add(idx)
                if len(selected) == budget:
                    break
    selected = selected[:budget]

    return torch.tensor(sorted(selected), dtype=torch.long, device=sample_tokens.device)


def spatially_structured_prune(
    visual_tokens: torch.Tensor,
    budget: int,
    grid_hw: Optional[Tuple[int, int]] = None,
    num_regions_hw: Tuple[int, int] = (2, 2),
) -> torch.Tensor:
    """Prune `visual_tokens` to `budget` tokens per sample with spatial-coverage-preserving structure.

    Args:
        visual_tokens: [B, N, D] projected patch embeddings on a 2D grid (row-major).
        budget: Target number of visual tokens to retain per sample.
        grid_hw: (H, W) of the token grid with H * W == N. Inferred as a square grid when None.
        num_regions_hw: Coarse region partition (rows, cols) used for coverage + density allocation.

    Returns:
        [B, budget, D] pruned embeddings. Returns the input unchanged when pruning is a no-op or the grid cannot
        be resolved (keeps the call site non-disruptive for non-square / already-small token layouts).
    """
    if visual_tokens.dim() != 3:
        raise ValueError(f"Expected `visual_tokens` of shape [B, N, D]; got {tuple(visual_tokens.shape)}")

    bsz, num_tokens, _ = visual_tokens.shape
    if budget <= 0 or budget >= num_tokens:
        return visual_tokens

    grid = grid_hw if grid_hw is not None else _infer_square_grid(num_tokens)
    if grid is None or grid[0] * grid[1] != num_tokens:
        # Unknown / non-square spatial layout (e.g. includes a CLS token) --> skip rather than guess.
        return visual_tokens

    region_tokens, _ = _region_assignment(grid, num_regions_hw)
    variation = _laplacian_variation(visual_tokens, grid)

    keep_indices = torch.stack(
        [_select_indices_for_sample(visual_tokens[b], variation[b], region_tokens, budget) for b in range(bsz)],
        dim=0,
    )
    gather_index = keep_indices.unsqueeze(-1).expand(-1, -1, visual_tokens.shape[-1])
    return torch.gather(visual_tokens, dim=1, index=gather_index)
