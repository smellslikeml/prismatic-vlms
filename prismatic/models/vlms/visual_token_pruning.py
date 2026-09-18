"""
visual_token_pruning.py

Training-free spatial visual-token pruning for Prismatic VLMs.

Adapted from S^2Prune (*Spatially Structured Visual Token Pruning for Multimodal
Large Language Models*, arXiv:2609.01224). The paper's central observation is
that importance/redundancy-based pruners develop stable spatial biases and often
fail to beat plain Uniform-Grid sampling -- broad *spatial coverage* is what
matters. S^2Prune therefore (1) guarantees coverage by keeping at least one token
per region, (2) spends the remaining budget where local structure is richest via
capacity-aware largest-remainder apportionment over per-image min-max-normalized
complexity, and (3) keeps one representative per deterministic local cell so the
retained tokens stay spatially spread inside each region.

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

Like the reference, the pruner operates on a square token grid; the coarse grid
is tied to the retained-token budget (32->4x4, 64->5x5, 128->8x8, 192->9x9).

The public entry point is :func:`prune_visual_tokens`, which maps a
``[bsz, N, D]`` block of projected patch embeddings to ``[bsz, N', D]`` with
exactly ``N'`` tokens retained per image (batch-consistent, so every downstream
fusion path in ``PrismaticVLM.forward`` stays correctly sized).
"""

from __future__ import annotations

import math

import torch

__all__ = ["prune_visual_tokens"]

# Paper configuration: the coarse grid is tied to the visual-token budget.
_DEFAULT_GRIDS = {32: 4, 64: 5, 128: 8, 192: 9}
_EPS = 1e-12


def _default_grid_size(keep_tokens: int) -> int:
    """Return the paper's coarse-grid side length for a supported budget."""
    try:
        return _DEFAULT_GRIDS[int(keep_tokens)]
    except KeyError as exc:
        raise ValueError(
            f"No paper-default coarse grid is defined for keep_tokens={keep_tokens}; "
            "pass region_grid explicitly."
        ) from exc


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


def _coarse_regions(grid: int, region_grid: int) -> list[tuple[int, int, int, int]]:
    """Partition a `grid x grid` token map into near-equal non-overlapping rectangles.

    Boundaries use ``round(i * grid / parts)`` exactly as in the reference, so the
    coarse regions tile the grid once with no gaps or overlaps.
    """
    edges = [round(i * grid / region_grid) for i in range(region_grid + 1)]
    regions: list[tuple[int, int, int, int]] = []
    for row in range(region_grid):
        for col in range(region_grid):
            r0, r1 = edges[row], edges[row + 1]
            c0, c1 = edges[col], edges[col + 1]
            if r1 > r0 and c1 > c0:
                regions.append((r0, r1, c0, c1))
    return regions


def _min_max_normalize(scores: list[float]) -> list[float]:
    """Per-image min-max normalization of the region complexity scores."""
    lo, hi = min(scores), max(scores)
    denom = max(hi - lo, _EPS)
    return [(s - lo) / denom for s in scores]


def _largest_remainder_allocation(
    region_scores: list[float], region_sizes: list[int], keep_tokens: int, minimum: int = 1
) -> list[int]:
    """Capacity-aware largest-remainder (Hamilton) apportionment of the budget.

    Every region receives ``minimum`` tokens first (spatial coverage); the
    remainder is distributed proportionally to structural complexity via capped
    largest fractional remainders, never exceeding a region's token count. This
    matches the reference allocator exactly.
    """
    num_regions = len(region_scores)
    if keep_tokens < minimum * num_regions or keep_tokens > sum(region_sizes):
        raise ValueError(
            f"keep_tokens={keep_tokens} is infeasible for {num_regions} regions with "
            f"minimum={minimum}"
        )

    alloc = [minimum] * num_regions
    remaining = keep_tokens - minimum * num_regions
    while remaining > 0:
        available = [region_sizes[i] - alloc[i] for i in range(num_regions)]
        eligible = [i for i in range(num_regions) if available[i] > 0]
        weights = [region_scores[i] for i in eligible]
        total = sum(weights)
        if total <= _EPS:
            weights = [1.0] * len(eligible)
            total = float(len(eligible))

        quotas = {i: w / total * remaining for i, w in zip(eligible, weights)}
        added = 0
        for i in eligible:
            gain = min(int(math.floor(quotas[i])), region_sizes[i] - alloc[i])
            alloc[i] += gain
            added += gain
        remaining -= added
        if remaining == 0:
            break

        fractions = {i: quotas[i] - math.floor(quotas[i]) for i in eligible}
        order = sorted(
            (i for i in range(num_regions) if region_sizes[i] - alloc[i] > 0),
            key=lambda i: (fractions.get(i, 0.0), region_scores[i], -i),
            reverse=True,
        )
        for i in order:
            if remaining == 0:
                break
            if alloc[i] < region_sizes[i]:
                alloc[i] += 1
                remaining -= 1

    return alloc


def _recursive_region_cells(
    region: tuple[int, int, int, int], budget: int
) -> list[tuple[int, int, int, int]]:
    """Split a region into exactly ``budget`` deterministic rectangular cells.

    The largest splittable rectangle is bisected along its longer dimension.
    Rows win dimension ties, and integer floor midpoints produce floor/ceil
    children when a side length is odd. Guarantees intra-region spatial spread.
    """
    r0, r1, c0, c1 = region
    capacity = (r1 - r0) * (c1 - c0)
    if budget < 1 or budget > capacity:
        raise ValueError(f"Cannot form {budget} non-empty cells for region={region}")

    cells: list[tuple[int, int, int, int]] = [(r0, r1, c0, c1)]
    while len(cells) < budget:
        candidates = [
            (-(b - a) * (d - c), index, a, b, c, d)
            for index, (a, b, c, d) in enumerate(cells)
            if (b - a) > 1 or (d - c) > 1
        ]
        _negative_area, index, a, b, c, d = min(candidates)
        if (b - a) >= (d - c) and (b - a) > 1:
            midpoint = a + (b - a) // 2
            children = [(a, midpoint, c, d), (midpoint, b, c, d)]
        else:
            midpoint = c + (d - c) // 2
            children = [(a, b, c, midpoint), (a, b, midpoint, d)]
        cells[index : index + 1] = children
    return cells


def _select_indices(
    tokens: torch.Tensor,
    grid: int,
    structure: torch.Tensor,
    regions: list[tuple[int, int, int, int]],
    keep_tokens: int,
) -> torch.Tensor:
    """Pick `keep_tokens` retained token indices (sorted, original order preserved)."""
    device = tokens.device
    region_flat, region_scores, region_sizes = [], [], []
    for r0, r1, c0, c1 in regions:
        rows = torch.arange(r0, r1, device=device)
        cols = torch.arange(c0, c1, device=device)
        idx = (rows[:, None] * grid + cols[None, :]).reshape(-1)
        region_flat.append(idx)
        region_sizes.append(int(idx.numel()))
        region_scores.append(float(structure.index_select(0, idx).sum().item()))

    normalized = _min_max_normalize(region_scores)
    alloc = _largest_remainder_allocation(normalized, region_sizes, keep_tokens)

    selected = []
    for region, take, idx in zip(regions, alloc, region_flat):
        if take <= 0:
            continue
        if take >= idx.numel():
            selected.append(idx)
            continue
        r0, _r1, c0, c1 = region
        width = c1 - c0
        # Saliency proxy for ERC: distance from the region centroid, aligned to `idx`.
        centroid = tokens.index_select(0, idx).mean(dim=0, keepdim=True)
        saliency = (tokens.index_select(0, idx) - centroid).norm(dim=-1)
        # Keep the most salient token in each deterministic non-overlapping cell.
        for a, b, c, d in _recursive_region_cells(region, take):
            prows = torch.arange(a - r0, b - r0, device=device)
            pcols = torch.arange(c - c0, d - c0, device=device)
            positions = (prows[:, None] * width + pcols[None, :]).reshape(-1)
            best = positions[int(torch.argmax(saliency.index_select(0, positions)).item())]
            selected.append(idx[best].reshape(1))

    keep_idx = torch.cat(selected) if selected else torch.arange(keep_tokens, device=device)
    return torch.sort(keep_idx).values


def prune_visual_tokens(
    patch_embeddings: torch.Tensor, keep_tokens: int, region_grid: int | None = None
) -> torch.Tensor:
    """Spatially structured, training-free pruning of projected visual tokens.

    Args:
        patch_embeddings: Projected patch embeddings, shape ``[bsz, N, D]``. ``N``
            must be a perfect square (the reference operates on a square token grid).
        keep_tokens: Number of visual tokens to retain per image. A no-op when
            ``keep_tokens`` is falsy, ``<= 0``, or ``>= N``.
        region_grid: Side length of the coarse-region grid (``region_grid**2``
            regions). Defaults to the paper's budget-tied configuration
            (32->4, 64->5, 128->8, 192->9); pass a value to override. Clamped so it
            never exceeds the token grid resolution.

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
    if grid * grid != num_tokens:
        raise ValueError(
            f"S^2Prune requires a square visual-token grid; got {num_tokens} tokens"
        )

    if region_grid is None:
        region_grid = _default_grid_size(keep_tokens)
    effective_grid = max(1, min(int(region_grid), grid))
    regions = _coarse_regions(grid, effective_grid)

    pruned = []
    for b in range(bsz):
        tokens = patch_embeddings[b]
        structure = _spatial_structure(tokens, grid)
        keep_idx = _select_indices(tokens, grid, structure, regions, keep_tokens)
        pruned.append(tokens.index_select(0, keep_idx))

    return torch.stack(pruned, dim=0)
