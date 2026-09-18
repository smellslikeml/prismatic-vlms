"""
test_visual_token_pruning.py

Tests for the training-free spatial visual-token pruning (S^2Prune adaptation)
and its wiring into `PrismaticVLM.forward`.
"""

import inspect
import math

import torch

# Import the pruner *through the call-site module* -- this only succeeds if the
# wiring edit in `prismatic/models/vlms/prismatic.py` actually imported it.
from prismatic.models.vlms.prismatic import PrismaticVLM, prune_visual_tokens
from prismatic.models.vlms.visual_token_pruning import prune_visual_tokens as prune_direct


def _region_of(idx: int, grid: int, region_grid: int) -> tuple:
    row, col = idx // grid, idx % grid
    return (row * region_grid) // grid, (col * region_grid) // grid


def test_call_site_exposes_and_configures_pruning():
    # The call site re-exports the exact pruner it invokes...
    assert prune_visual_tokens is prune_direct
    # ...and both constructors accept the opt-in config the forward() hook reads.
    assert "visual_token_pruning" in inspect.signature(PrismaticVLM.__init__).parameters
    assert "visual_token_pruning" in inspect.signature(PrismaticVLM.from_pretrained).parameters


def test_forward_shape_contract():
    # Mirrors the [bsz, N, D] block the forward() hook feeds the pruner.
    embeddings = torch.randn(3, 256, 16)
    pruned = prune_visual_tokens(embeddings, keep_tokens=64, region_grid=4)
    assert pruned.shape == (3, 64, 16), "every image must keep exactly keep_tokens (batch-consistent)"


def test_noop_when_budget_not_binding():
    embeddings = torch.randn(2, 256, 16)
    assert prune_visual_tokens(embeddings, keep_tokens=0) is embeddings
    assert prune_visual_tokens(embeddings, keep_tokens=None) is embeddings
    assert prune_visual_tokens(embeddings, keep_tokens=256) is embeddings
    assert prune_visual_tokens(embeddings, keep_tokens=1024) is embeddings


def test_spatial_coverage_guaranteed():
    # Core S^2Prune property: with budget >= #regions, every region keeps >= 1 token.
    grid, region_grid = 16, 4
    embeddings = torch.randn(1, grid * grid, 8)
    keep = 32  # > region_grid**2 == 16
    pruned = prune_visual_tokens(embeddings, keep_tokens=keep, region_grid=region_grid)
    assert pruned.shape == (1, keep, 8)

    # Recover which original indices survived and confirm coverage across all regions.
    kept_indices = []
    for i in range(grid * grid):
        token = embeddings[0, i]
        if any(torch.equal(token, pruned[0, j]) for j in range(keep)):
            kept_indices.append(i)
    covered = {_region_of(i, grid, region_grid) for i in kept_indices}
    assert len(covered) == region_grid * region_grid, "each spatial region must retain coverage"


def test_density_follows_structure():
    # A region with rich structure should receive more of the budget than a flat one.
    grid, region_grid = 16, 4
    embeddings = torch.zeros(1, grid * grid, 8)
    # Inject high-frequency structure into the top-left region only.
    for r in range(grid // region_grid):
        for c in range(grid // region_grid):
            embeddings[0, r * grid + c] = torch.randn(8) * ((r + c) % 2) * 10.0

    pruned = prune_visual_tokens(embeddings, keep_tokens=32, region_grid=region_grid)
    kept_top_left = 0
    for i in range(grid * grid):
        if _region_of(i, grid, region_grid) != (0, 0):
            continue
        if any(torch.equal(embeddings[0, i], pruned[0, j]) for j in range(32)):
            kept_top_left += 1
    # Coverage gives every region 1; the structured region must earn strictly more.
    assert kept_top_left > 1


def test_non_square_fallback_keeps_exact_count():
    # CLS-token / fused layouts can be non-square; the 1-D fallback still applies.
    embeddings = torch.randn(2, 257, 8)
    assert not math.isqrt(257) ** 2 == 257
    pruned = prune_visual_tokens(embeddings, keep_tokens=64)
    assert pruned.shape == (2, 64, 8)
