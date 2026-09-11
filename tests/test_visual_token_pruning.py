"""
test_visual_token_pruning.py

Tests for the training-free spatial visual-token pruner and its wiring into `PrismaticVLM.forward`.
"""

import torch

from prismatic.util.visual_token_pruning import SpatialTokenPruner, prune_visual_tokens


def test_prunes_to_budget_and_keeps_valid_indices():
    tokens = torch.randn(3, 256, 8)  # 16x16 grid
    pruned, kept = prune_visual_tokens(tokens, budget=64, region_grid=(2, 2))

    assert pruned.shape == (3, 64, 8)
    assert kept.shape == (3, 64)
    # Indices are in-range, unique, and sorted (original patch order preserved).
    for b in range(tokens.shape[0]):
        idx = kept[b]
        assert idx.min() >= 0 and idx.max() < 256
        assert torch.unique(idx).numel() == idx.numel()
        assert torch.equal(idx, torch.sort(idx)[0])
        # Pruned rows actually correspond to the reported indices.
        assert torch.allclose(pruned[b], tokens[b, idx])


def test_spatial_coverage_every_region_represented():
    # Each 8x8 quadrant of a 16x16 grid must keep >= 1 token even with a tight budget.
    tokens = torch.randn(1, 256, 4)
    _, kept = prune_visual_tokens(tokens, budget=4, region_grid=(2, 2))
    rows, cols = kept[0] // 16, kept[0] % 16
    quadrants = {(int(r >= 8), int(c >= 8)) for r, c in zip(rows, cols)}
    assert quadrants == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_richer_region_gets_more_tokens():
    # One quadrant carries strong structure (high-variance features), the rest are flat.
    grid = torch.zeros(1, 16, 16, 4)
    grid[:, :8, :8, :] = torch.randn(1, 8, 8, 4) * 10.0
    tokens = grid.reshape(1, 256, 4)
    _, kept = prune_visual_tokens(tokens, budget=40, region_grid=(2, 2))
    rows, cols = kept[0] // 16, kept[0] % 16
    rich = int(((rows < 8) & (cols < 8)).sum())
    # The structured quadrant should receive more than its uniform 1/4 share.
    assert rich > 40 // 4


def test_identity_when_not_square_or_within_budget():
    within = torch.randn(2, 32, 8)
    pruned, kept = prune_visual_tokens(within, budget=64)
    assert torch.equal(pruned, within) and kept.shape == (2, 32)

    non_square = torch.randn(2, 257, 8)  # 256 + CLS => not a perfect square
    pruned2, _ = prune_visual_tokens(non_square, budget=64)
    assert torch.equal(pruned2, non_square)


def test_forward_wiring_shrinks_patch_sequence(monkeypatch):
    # Exercise the real call-site edit in prismatic.py without constructing a full VLM: the forward
    # method builds `multimodal_embeddings` from `projected_patch_embeddings.shape[1]`, so enabling
    # the pruner must reduce the number of visual tokens threaded through the concat machinery.
    from prismatic.models.vlms.prismatic import PrismaticVLM

    vlm = PrismaticVLM.__new__(PrismaticVLM)  # bypass heavy backbone construction
    vlm.visual_token_pruner = None

    projected = torch.randn(2, 256, 8)
    assert vlm.visual_token_pruner is None

    PrismaticVLM.configure_token_pruning(vlm, budget=64, region_grid=(2, 2))
    assert isinstance(vlm.visual_token_pruner, SpatialTokenPruner)

    pruned, _ = vlm.visual_token_pruner(projected)
    assert pruned.shape[1] == 64 < projected.shape[1]

    PrismaticVLM.configure_token_pruning(vlm, budget=None)
    assert vlm.visual_token_pruner is None
