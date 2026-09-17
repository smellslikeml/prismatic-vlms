"""
test_token_compression.py

Unit tests for the Semantic Connected Components (SCC) visual-token compression (LLaVA-Scissor,
arXiv:2506.21862) and its wiring into `PrismaticVLM` at the projector boundary.
"""

import pytest
import torch

# Import the wiring surface from the existing (non-new) VLM module to prove integration ...
from prismatic.models.vlms.prismatic import PrismaticVLM

# ... alongside the new capability module for focused algorithm checks.
from prismatic.models.vlms.token_compression import (
    compress_tokens_scc,
    compress_visual_tokens,
    semantic_connected_components,
)


def _two_cluster_tokens() -> torch.Tensor:
    """Six tokens forming two clearly-separated semantic regions (three per region)."""
    region_a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    region_b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    return torch.stack([region_a, region_a, region_a, region_b, region_b, region_b])


def test_scc_groups_tokens_into_semantic_regions() -> None:
    labels = semantic_connected_components(_two_cluster_tokens(), similarity_threshold=0.9)

    # Two components, contiguous ids ordered by smallest member index.
    assert int(labels.max().item()) + 1 == 2
    assert labels.tolist() == [0, 0, 0, 1, 1, 1]


def test_compress_pools_each_region_to_one_representative() -> None:
    compressed = compress_tokens_scc(_two_cluster_tokens(), similarity_threshold=0.9)

    assert compressed.shape == (2, 4)
    # Mean-pooled representatives recover each region's prototype.
    assert torch.allclose(compressed[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(compressed[1], torch.tensor([0.0, 1.0, 0.0, 0.0]))


def test_high_threshold_keeps_dissimilar_tokens() -> None:
    tokens = torch.eye(5)  # mutually orthogonal -> no edges survive a strict cutoff
    compressed = compress_tokens_scc(tokens, similarity_threshold=0.99)

    assert compressed.shape == (5, 5)


def test_prismatic_wiring_compresses_single_sequence() -> None:
    # Exercise the exact hook `PrismaticVLM.forward` calls, without instantiating the full model.
    projected = _two_cluster_tokens().unsqueeze(0)  # [1, 6, 4]
    compressed = PrismaticVLM._compress_visual_tokens(projected, similarity_threshold=0.9)

    assert compressed.shape == (1, 2, 4)
    assert compressed.shape[1] < projected.shape[1]


def test_prismatic_wiring_rejects_batched_input() -> None:
    batched = _two_cluster_tokens().unsqueeze(0).repeat(2, 1, 1)  # [2, 6, 4]
    with pytest.raises(ValueError):
        PrismaticVLM._compress_visual_tokens(batched, similarity_threshold=0.9)


def test_compress_visual_tokens_preserves_dtype() -> None:
    projected = _two_cluster_tokens().unsqueeze(0).to(torch.float16)
    compressed = compress_visual_tokens(projected, similarity_threshold=0.9)

    assert compressed.dtype == torch.float16
    assert compressed.shape == (1, 2, 4)
