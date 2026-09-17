"""
token_compression.py

Training-free visual-token compression via Semantic Connected Components (SCC).

Adapted from "LLaVA-Scissor: Token Compression with Semantic Connected Components for Video LLMs"
(Sun et al., 2025; arXiv:2506.21862). Rather than pruning tokens by attention score -- which tends to
miss semantic regions and leave redundancy -- LLaVA-Scissor groups tokens into semantic regions
(connected components of a cosine-similarity graph over token features) and replaces each region with a
single pooled representative, yielding comprehensive semantic coverage with far fewer tokens.

This module ports the core SCC mechanism at full fidelity for the single token-set case exposed at
Prismatic's projector boundary (`[B, N, D] -> [B, N', D]` just after `self.projector`). The paper's
two-stage spatial-then-temporal video pipeline and its separate video benchmark suite are intentionally
out of scope: Prismatic is an image VLM, so a single SCC pass over the projected patch tokens is the
target-native equivalent of the paper's per-frame spatial compression step.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def semantic_connected_components(features: torch.Tensor, similarity_threshold: float) -> torch.Tensor:
    """Assign each token to a semantic connected component.

    Builds an undirected graph whose nodes are tokens and whose edges connect token pairs with cosine
    similarity `>= similarity_threshold`, then labels the connected components with a union-find pass.

    Args:
        features: `[N, D]` token features.
        similarity_threshold: cosine-similarity cutoff in `[-1, 1]`; higher keeps more (smaller) regions.

    Returns:
        `[N]` long tensor of component ids in `[0, num_components)`, ordered by each region's smallest
        member index (so spatially-adjacent tokens tend to stay adjacent after pooling).
    """
    num_tokens = features.shape[0]
    if num_tokens == 0:
        return torch.zeros(0, dtype=torch.long, device=features.device)

    normed = F.normalize(features.float(), dim=-1)
    similarity = normed @ normed.T

    # Upper-triangle adjacency (exclude self-loops) at/above the similarity cutoff.
    adjacency = torch.triu(similarity >= similarity_threshold, diagonal=1)

    parent = list(range(num_tokens))

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for i, j in adjacency.nonzero(as_tuple=False).tolist():
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            # Union toward the smaller root so component ids track smallest member index.
            parent[max(root_i, root_j)] = min(root_i, root_j)

    roots = torch.tensor([find(i) for i in range(num_tokens)], device=features.device)

    # Renumber sparse roots into a dense `[0, num_components)` range (sorted -> smallest index first).
    return torch.unique(roots, sorted=True, return_inverse=True)[1]


def compress_tokens_scc(features: torch.Tensor, similarity_threshold: float) -> torch.Tensor:
    """Compress a token set by mean-pooling each semantic connected component.

    Args:
        features: `[N, D]` token features.
        similarity_threshold: cosine-similarity cutoff passed to `semantic_connected_components`.

    Returns:
        `[N', D]` with one pooled representative per component (`N' <= N`), preserving the input dtype.
    """
    if features.ndim != 2:
        raise ValueError(f"`features` must be `[N, D]`, got shape {tuple(features.shape)}")

    num_tokens = features.shape[0]
    if num_tokens <= 1:
        return features

    labels = semantic_connected_components(features, similarity_threshold)
    num_components = int(labels.max().item()) + 1

    summed = torch.zeros(num_components, features.shape[1], dtype=features.dtype, device=features.device)
    summed.index_add_(0, labels, features)
    counts = torch.zeros(num_components, dtype=features.dtype, device=features.device)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=features.dtype))

    return summed / counts.unsqueeze(-1)


def compress_visual_tokens(projected_patch_embeddings: torch.Tensor, similarity_threshold: float) -> torch.Tensor:
    """Apply SCC compression to a `[B, N, D]` block of projected visual tokens.

    Requires `B == 1`: components are data-dependent, so per-sample counts differ and cannot be stacked
    back into a rectangular batch. This matches LLaVA-Scissor's training-free, single-sequence inference
    setting; batched training/inference should leave compression disabled.

    Args:
        projected_patch_embeddings: `[1, N, D]` projector output.
        similarity_threshold: cosine-similarity cutoff passed to `compress_tokens_scc`.

    Returns:
        `[1, N', D]` compressed visual tokens (`N' <= N`).
    """
    if projected_patch_embeddings.ndim != 3:
        shape = tuple(projected_patch_embeddings.shape)
        raise ValueError(f"`projected_patch_embeddings` must be `[B, N, D]`, got {shape}")
    if projected_patch_embeddings.shape[0] != 1:
        raise ValueError("SCC visual-token compression is only defined for batch size 1 (single sequence).")

    compressed = compress_tokens_scc(projected_patch_embeddings[0], similarity_threshold)
    return compressed.unsqueeze(0)
