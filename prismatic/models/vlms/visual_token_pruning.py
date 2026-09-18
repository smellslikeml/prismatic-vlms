"""
visual_token_pruning.py

Training-free spatial visual-token pruning for Prismatic VLMs.

Adapted from S^2Prune (*Spatially Structured Visual Token Pruning for Multimodal
Large Language Models*, arXiv:2609.01224). The paper's central observation is
that importance/redundancy-based pruners develop stable spatial biases and often
fail to beat plain Uniform-Grid sampling -- broad *spatial coverage* is what
matters. S^2Prune therefore (1) guarantees coverage by keeping at least one token
per region, (2) spends the remaining budget where local structure is richest, and
(3) picks representative tokens inside each region.

We keep that three-part mechanism at full fidelity while substituting the
paper's auxiliary components with parameter-free, target-native proxies
(Mode 2 adapted port):

  * **Laplacian variation** -- the paper measures it on the raw image; we measure
    it directly on the projected patch-embedding grid, so no extra image plumbing
    is threaded through `forward()`.
  * **Early Representation Change (ERC)** -- the paper ranks in-region tokens by
    how much their representation shifts through the first decoder block, which
    needs a decoder round-trip. We substitute a parameter-free embedding-saliency
    proxy: a token's distance from its region centroid (tokens that stand out
    from their neighbourhood carry the most region-specific information).
  * The paper's separate benchmark / evaluation framework is out of scope here.

The public entry point is :func:`prune_visual_tokens`, which maps a
``[bsz, N, D]`` block of projected patch embeddings to ``[bsz, N', D]`` with
exactly ``N'`` tokens retained per image (batch-consistent, so every downstream
fusion path in ``PrismaticVLM.forward`` stays correctly sized).
"""

from __future__ import annotations

import math

import torch

__all__ = ["prune_visual_tokens"]


def _spatial_structure(tokens: torch.Tensor, grid: int) -> torch.Tensor:
    """Per-token Laplacian variation over a `grid x grid` embedding map -> [N]."""
    feats = tokens.reshape(grid, grid, -1)
    # Discrete Laplacian with replicate ("edge") padding on the spatial dims.
    up = torch.cat([feats[:1], feats[:-1]], dim=0)
    down = torch.cat([feats[1:], feats[-1:]], dim=0)
    left = torch.cat([feats[:, :1], feats[:, :-1]], dim=1)
    right = torch.cat([feats[:, 1:], feats[:, -1:]], dim=1)
    laplacian = (4.0 * feats) - up - down - left - right
    return laplacian.norm(dim=-1).reshape(-1)


def _sequential_structure(tokens: torch.Tensor) -> torch.Tensor:
    """Fallback 1-D Laplacian variation for non-square token counts -> [N]."""
    prev = torch.cat([tokens[:1], tokens[:-1]], dim=0)
    nxt = torch.cat([tokens[1:], tokens[-1:]], dim=0)
    return ((2.0 * tokens) - prev - nxt).norm(dim=-1)


def _region_ids(num_tokens: int, grid: int, region_grid: int, square: bool, device: torch.device) -> torch.Tensor:
    """Assign every token to one of (region_grid x region_grid) coverage regions -> [N]."""
    if square:
        rows = torch.arange(grid, device=device)
        row_region = (rows * region_grid) // grid
        col_region = (rows * region_grid) // grid
        ids = row_region[:, None] * region_grid + col_region[None, :]
        return ids.reshape(-1)

    # Non-square fallback: contiguous 1-D bins preserve positional coverage.
    positions = torch.arange(num_tokens, device=device)
    return (positions * (region_grid * region_grid)) // num_tokens


def _allocate_budget(region_scores: list[float], region_capacity: list[int], keep_tokens: int) -> list[int]:
    """Coverage-first, structure-weighted apportionment respecting per-region capacity.

    Every region with capacity receives one token first (spatial coverage); the
    remainder is handed out one token at a time by a highest-averages rule
    (priority = structure_score / (current_alloc + 1)), which spends budget where
    Laplacian variation is greatest while never exceeding a region's token count.
    """
    num_regions = len(region_scores)
    alloc = [0] * num_regions

    coverable = [r for r in range(num_regions) if region_capacity[r] > 0]
    remaining = keep_tokens
    if keep_tokens >= len(coverable):
        for r in coverable:
            alloc[r] = 1
            remaining -= 1

    while remaining > 0:
        best, best_priority = -1, -math.inf
        for r in range(num_regions):
            if alloc[r] >= region_capacity[r]:
                continue
            priority = region_scores[r] / (alloc[r] + 1)
            if priority > best_priority:
                best, best_priority = r, priority
        if best < 0:
            break
        alloc[best] += 1
        remaining -= 1

    return alloc


def _select_indices(
    tokens: torch.Tensor, region_ids: torch.Tensor, structure: torch.Tensor, keep_tokens: int
) -> torch.Tensor:
    """Pick `keep_tokens` retained token indices (sorted, original order preserved)."""
    num_regions = int(region_ids.max().item()) + 1
    region_scores, region_capacity, region_members = [], [], []
    for r in range(num_regions):
        members = torch.nonzero(region_ids == r, as_tuple=False).reshape(-1)
        region_members.append(members)
        region_capacity.append(int(members.numel()))
        region_scores.append(float(structure[members].sum().item()) if members.numel() else 0.0)

    alloc = _allocate_budget(region_scores, region_capacity, keep_tokens)

    selected = []
    for r in range(num_regions):
        take = alloc[r]
        members = region_members[r]
        if take <= 0 or members.numel() == 0:
            continue
        if take >= members.numel():
            selected.append(members)
            continue
        # Saliency proxy for ERC: distance from the region centroid.
        centroid = tokens[members].mean(dim=0, keepdim=True)
        saliency = (tokens[members] - centroid).norm(dim=-1)
        top = torch.topk(saliency, take).indices
        selected.append(members[top])

    keep_idx = torch.cat(selected) if selected else torch.arange(keep_tokens, device=tokens.device)
    return torch.sort(keep_idx).values


def prune_visual_tokens(patch_embeddings: torch.Tensor, keep_tokens: int, region_grid: int = 4) -> torch.Tensor:
    """Spatially structured, training-free pruning of projected visual tokens.

    Args:
        patch_embeddings: Projected patch embeddings, shape ``[bsz, N, D]``.
        keep_tokens: Number of visual tokens to retain per image. A no-op when
            ``keep_tokens`` is falsy, ``<= 0``, or ``>= N``.
        region_grid: Side length of the coverage-region grid (``region_grid**2``
            regions). Clamped so it never exceeds the token grid resolution.

    Returns:
        Pruned embeddings of shape ``[bsz, keep_tokens, D]`` (unchanged input when
        pruning is a no-op). Every image retains exactly ``keep_tokens`` tokens, so
        the result drops in directly where ``projected_patch_embeddings`` is used.
    """
    if not keep_tokens or keep_tokens <= 0:
        return patch_embeddings

    bsz, num_tokens, _ = patch_embeddings.shape
    if keep_tokens >= num_tokens:
        return patch_embeddings

    grid = int(math.isqrt(num_tokens))
    square = grid * grid == num_tokens
    effective_grid = max(1, min(region_grid, grid)) if square else max(1, region_grid)
    region_ids = _region_ids(num_tokens, grid, effective_grid, square, patch_embeddings.device)

    pruned = []
    for b in range(bsz):
        tokens = patch_embeddings[b]
        structure = _spatial_structure(tokens, grid) if square else _sequential_structure(tokens)
        keep_idx = _select_indices(tokens, region_ids, structure, keep_tokens)
        pruned.append(tokens.index_select(0, keep_idx))

    return torch.stack(pruned, dim=0)
