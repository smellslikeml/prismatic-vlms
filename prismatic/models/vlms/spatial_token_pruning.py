"""
spatial_token_pruning.py

Training-free spatial visual-token pruning for Prismatic VLMs.

Reduces the number of projected visual tokens fed into the LLM backbone from `N` to a smaller budget
`K < N`, shrinking sequence length (and thus KV-cache / attention cost) during `forward()` and
`generate()` without touching the vision backbone, projector, or LLM weights.

Method (adapted from "S^2Prune: Spatially Structured Visual Token Pruning for Multimodal Large Language
Models", https://arxiv.org/abs/2609.01224):

    1. Partition the patch grid into `region_grid x region_grid` regions.
    2. Guarantee spatial *coverage* by reserving at least one token for every region.
    3. Distribute the remaining budget by *Laplacian variation* -- regions whose embeddings vary more
       (richer local structure) receive proportionally more tokens (D'Hondt / largest-averages).
    4. Within each region, keep the highest-variation ("most representative") tokens.

Fidelity note (this is an *adapted* port, not a 1:1 reproduction):
    - Full fidelity: the region partition + guaranteed-coverage allocation and the Laplacian-variation
      density adaptation -- the paper's core contribution -- are implemented as described.
    - Substituted auxiliary: the paper's Early Representation Change (ERC) selector, which requires a
      forward pass through the model's *first decoder block*, is replaced by a parameter-free proxy that
      ranks tokens by the L2 magnitude of the discrete Laplacian of the *projected patch embeddings*
      (already available at the call site). This keeps the method training-free and self-contained.
    - Out of scope (belongs downstream): the paper's benchmark / accuracy-vs-budget evaluation suite.
"""

from __future__ import annotations

from math import isqrt
from typing import List, Optional

import torch
import torch.nn.functional as F

__all__ = ["SpatialTokenPruner", "spatially_structured_prune"]


def _infer_grid(num_tokens: int) -> tuple[int, bool]:
    """Return `(side, has_cls_token)` for a token count, or raise if no square grid can be inferred."""
    side = isqrt(num_tokens)
    if side * side == num_tokens:
        return side, False
    side = isqrt(num_tokens - 1)
    if side * side == num_tokens - 1:
        return side, True
    raise ValueError(f"Cannot infer a square patch grid from {num_tokens} tokens.")


def _laplacian_scores(spatial: torch.Tensor, side: int) -> torch.Tensor:
    """Per-patch structure score = L2 norm of the discrete grid Laplacian of the embeddings.

    `spatial`: [bsz, side * side, dim] in raster order. Returns [bsz, side * side] (float32).
    """
    bsz, _, dim = spatial.shape
    grid = spatial.float().reshape(bsz, side, side, dim).permute(0, 3, 1, 2)  # [bsz, dim, side, side]
    padded = F.pad(grid, (1, 1, 1, 1), mode="replicate")
    lap = (
        4.0 * padded[:, :, 1:-1, 1:-1]
        - padded[:, :, :-2, 1:-1]
        - padded[:, :, 2:, 1:-1]
        - padded[:, :, 1:-1, :-2]
        - padded[:, :, 1:-1, 2:]
    )
    return lap.norm(dim=1).reshape(bsz, side * side)  # [bsz, side * side]


def _region_of_patch(side: int, region_grid: int) -> List[List[int]]:
    """Map each region id -> list of flat patch indices (raster order) belonging to it."""
    regions: List[List[int]] = [[] for _ in range(region_grid * region_grid)]
    for row in range(side):
        for col in range(side):
            region_id = (row * region_grid // side) * region_grid + (col * region_grid // side)
            regions[region_id].append(row * side + col)
    return regions


def _allocate_budget(region_scores: List[float], capacities: List[int], budget: int) -> List[int]:
    """Allocate `budget` tokens across regions: >=1 per region, remainder by score (D'Hondt averages).

    `capacities[r]` bounds how many tokens region `r` can supply. Assumes `sum(capacities) >= budget`
    and `budget >= len(region_scores)` (both guaranteed by the caller).
    """
    num_regions = len(region_scores)
    alloc = [1 if capacities[r] > 0 else 0 for r in range(num_regions)]
    eps = 1e-6
    for _ in range(budget - sum(alloc)):
        best, best_quotient = -1, -1.0
        for r in range(num_regions):
            if alloc[r] >= capacities[r]:
                continue
            quotient = (region_scores[r] + eps) / (alloc[r] + 1)
            if quotient > best_quotient:
                best, best_quotient = r, quotient
        if best < 0:
            break
        alloc[best] += 1
    return alloc


def spatially_structured_prune(
    patch_embeddings: torch.Tensor,
    keep_tokens: int,
    region_grid: int = 2,
    has_cls_token: Optional[bool] = None,
) -> torch.Tensor:
    """Prune projected visual tokens `[bsz, N, d]` down to `[bsz, keep_tokens, d]`.

    Selection preserves broad spatial coverage (>=1 token per region) while spending the remaining
    budget on high-Laplacian-variation regions. A leading CLS token (auto-detected, or forced via
    `has_cls_token`) is always retained. If the grid cannot be inferred, falls back to evenly strided
    subsampling. Returns the input unchanged when `keep_tokens >= N`.
    """
    bsz, num_tokens, dim = patch_embeddings.shape
    if keep_tokens is None or keep_tokens >= num_tokens or keep_tokens <= 0:
        return patch_embeddings

    if has_cls_token is None:
        try:
            side, has_cls_token = _infer_grid(num_tokens)
        except ValueError:
            stride_idx = torch.linspace(0, num_tokens - 1, keep_tokens, device=patch_embeddings.device).long()
            return patch_embeddings.index_select(1, stride_idx)
    else:
        side = isqrt(num_tokens - 1 if has_cls_token else num_tokens)

    cls = patch_embeddings[:, :1, :] if has_cls_token else None
    spatial = patch_embeddings[:, 1:, :] if has_cls_token else patch_embeddings
    spatial_budget = keep_tokens - (1 if has_cls_token else 0)

    # Clamp the region grid so every region can be guaranteed at least one token.
    grid = region_grid
    while grid > 1 and grid * grid > spatial_budget:
        grid -= 1
    regions = _region_of_patch(side, grid)
    capacities = [len(r) for r in regions]

    scores = _laplacian_scores(spatial, side)  # [bsz, side * side]

    kept_per_image = []
    for b in range(bsz):
        region_scores = [float(scores[b, torch.tensor(r, device=scores.device)].sum()) for r in regions]
        alloc = _allocate_budget(region_scores, capacities, spatial_budget)

        selected: List[int] = []
        for region_idx, take in zip(regions, alloc):
            if take <= 0:
                continue
            idx_tensor = torch.tensor(region_idx, device=scores.device)
            region_vals = scores[b].index_select(0, idx_tensor)
            top = torch.topk(region_vals, k=min(take, len(region_idx))).indices
            selected.extend(idx_tensor.index_select(0, top).tolist())

        selected = sorted(selected)[:spatial_budget]
        keep_idx = torch.tensor(selected, device=spatial.device)
        kept = spatial[b].index_select(0, keep_idx)
        if cls is not None:
            kept = torch.cat([cls[b], kept], dim=0)
        kept_per_image.append(kept)

    return torch.stack(kept_per_image, dim=0)


class SpatialTokenPruner:
    """Callable config wrapper around :func:`spatially_structured_prune`.

    Holds the token budget so it can be attached to a model and invoked as `pruner(patch_embeddings)`
    inside the forward pass.
    """

    def __init__(self, keep_tokens: int, region_grid: int = 2, has_cls_token: Optional[bool] = None) -> None:
        self.keep_tokens = keep_tokens
        self.region_grid = region_grid
        self.has_cls_token = has_cls_token

    def __call__(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        return spatially_structured_prune(
            patch_embeddings,
            keep_tokens=self.keep_tokens,
            region_grid=self.region_grid,
            has_cls_token=self.has_cls_token,
        )

    def __repr__(self) -> str:
        return (
            f"SpatialTokenPruner(keep_tokens={self.keep_tokens}, region_grid={self.region_grid}, "
            f"has_cls_token={self.has_cls_token})"
        )
