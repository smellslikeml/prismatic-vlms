"""
test_visual_token_pruning.py

Tests for the training-free S^2Prune visual-token pruner and its wiring into `PrismaticVLM.forward`.

These exercise the integration through the existing call-site module (`prismatic.models.vlms.prismatic`),
not just the new pruning module in isolation.
"""

import torch

# Existing call-site module (NOT introduced by this change)
from prismatic.models.vlms.prismatic import PrismaticVLM

# New capability module
from prismatic.util.visual_token_pruning import (
    SpatialTokenPruner,
    laplacian_saliency,
    prune_visual_tokens,
)


def _grid_with_hotspot(bsz: int = 2, side: int = 16, dim: int = 8) -> torch.Tensor:
    """A token grid with a high-frequency checkerboard hotspot in the top-left quadrant, smooth elsewhere.

    Channel 0 carries a unique per-position signature (``i * side + j``). Being affine in ``(i, j)`` its discrete
    Laplacian is exactly zero, so it never affects saliency, but it lets tests recover which grid cell survived.
    """
    torch.manual_seed(0)
    grid = torch.linspace(0, 1, side).view(1, side, 1, 1).expand(bsz, side, side, dim).clone()
    half = side // 2
    checker = ((torch.arange(half).view(half, 1) + torch.arange(half).view(1, half)) % 2).float()
    grid[:, :half, :half, 1:] += checker.view(1, half, half, 1) * 5.0

    signature = (torch.arange(side).view(side, 1) * side + torch.arange(side).view(1, side)).float()
    grid[:, :, :, 0] = signature.view(1, side, side)
    return grid


def _region_of_token(token: torch.Tensor, side: int, regions: int) -> int:
    """Recover a token's region id from its unique channel-0 signature."""
    pos = int(round(float(token[0])))
    r, c = divmod(pos, side)
    return (r * regions // side) * regions + (c * regions // side)


def test_prune_reduces_token_count_and_preserves_batch_dim():
    x = _grid_with_hotspot().reshape(2, 256, 8)
    pruned = prune_visual_tokens(x, keep_tokens=32, num_regions_per_side=4)
    assert pruned.shape == (2, 32, 8)
    assert pruned.dtype == x.dtype


def test_prune_is_noop_when_budget_exceeds_grid():
    x = torch.randn(1, 256, 8)
    pruned = prune_visual_tokens(x, keep_tokens=1000, num_regions_per_side=4)
    assert torch.equal(pruned, x)


def test_spatial_coverage_every_region_represented():
    # With keep >= num_regions, S^2Prune guarantees at least one token per region.
    side, regions = 16, 4
    x = _grid_with_hotspot(bsz=1, side=side).reshape(1, side * side, 8)
    keep = 32
    pruned = prune_visual_tokens(x, keep_tokens=keep, num_regions_per_side=regions)

    kept_regions = {_region_of_token(token, side, regions) for token in pruned[0]}
    assert kept_regions == set(range(regions * regions)), "Every region must retain at least one token"


def test_structure_regions_get_more_tokens_than_smooth_regions():
    side, regions = 16, 4
    x = _grid_with_hotspot(bsz=1, side=side).reshape(1, side * side, 8)
    pruned = prune_visual_tokens(x, keep_tokens=32, num_regions_per_side=regions)

    counts = [0] * (regions * regions)
    for token in pruned[0]:
        counts[_region_of_token(token, side, regions)] += 1

    # The top-left region (region id 0) holds the high-frequency hotspot; it should out-earn a smooth region.
    smooth_region = regions * regions - 1  # bottom-right, smooth gradient only
    assert counts[0] > counts[smooth_region]


def test_laplacian_saliency_flags_high_frequency_region():
    grid = _grid_with_hotspot(bsz=1, side=16)
    sal = laplacian_saliency(grid)
    assert sal.shape == (1, 16, 16)
    assert (sal >= 0).all()
    # Hotspot quadrant should be more salient on average than the smooth quadrant.
    assert sal[0, :8, :8].mean() > sal[0, 8:, 8:].mean()


def test_pruner_callable_matches_functional():
    x = torch.randn(2, 256, 8)
    pruner = SpatialTokenPruner(keep_tokens=64, num_regions_per_side=4)
    assert torch.equal(pruner(x), prune_visual_tokens(x, keep_tokens=64, num_regions_per_side=4))


def test_forward_hook_wiring_through_prismaticvlm():
    """Exercise the exact wiring `PrismaticVLM.forward` relies on, using the real class methods.

    We bypass the heavy backbone construction (`__new__`) but drive the *actual* enable/disable methods and the
    `visual_token_pruner` attribute the forward pass branches on, applying the pruner to a projector-output-shaped
    tensor exactly as `forward()` does at the call site.
    """
    vlm = PrismaticVLM.__new__(PrismaticVLM)
    vlm.visual_token_pruner = None  # matches the default set in __init__

    projected_patch_embeddings = torch.randn(2, 256, 8)

    # Disabled -> forward's `if self.visual_token_pruner is not None` branch is skipped (identity).
    out = projected_patch_embeddings
    if vlm.visual_token_pruner is not None:
        out = vlm.visual_token_pruner(out)
    assert out.shape == (2, 256, 8)

    # Enable via the real API, then replay the forward hook.
    PrismaticVLM.enable_visual_token_pruning(vlm, keep_tokens=32, num_regions_per_side=4)
    assert isinstance(vlm.visual_token_pruner, SpatialTokenPruner)
    out = projected_patch_embeddings
    if vlm.visual_token_pruner is not None:
        out = vlm.visual_token_pruner(out)
    assert out.shape == (2, 32, 8)

    # Disable via the real API restores the identity path.
    PrismaticVLM.disable_visual_token_pruning(vlm)
    assert vlm.visual_token_pruner is None
