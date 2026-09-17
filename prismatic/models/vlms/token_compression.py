"""
token_compression.py

Training-free visual-token compression via Semantic Connected Components (SCC).

Adapted from "LLaVA-Scissor: Token Compression with Semantic Connected Components for Video LLMs"
(Sun et al., 2025; arXiv:2506.21862). Rather than pruning tokens by attention score -- which tends to
miss semantic regions and leave redundancy -- LLaVA-Scissor groups tokens into semantic regions
(connected components of a cosine-similarity graph over token features) and replaces each region with a
single pooled representative, yielding comprehensive semantic coverage with far fewer tokens.

This module ports the reference SCC mechanism for the single token-set case exposed at Prismatic's
projector boundary (`[B, N, D] -> [B, N', D]` just after `self.projector`). Concretely it mirrors two
reference stages: (1) semantic connected components -- computed with the reference's epsilon-sampled,
Boruvka-style *approximate* labeling (tuned for video scale, exact for token counts below the sampling
threshold) -- mean-pooled to one representative per region; then (2) the reference's global "Token Merge
for all tokens", which re-assigns every original token to its nearest representative by cosine argmax and
averages it back in. The paper's two-stage spatial-then-temporal video pipeline and its separate video
benchmark suite are intentionally out of scope: Prismatic is an image VLM, so a single SCC pass over the
projected patch tokens is the target-native equivalent of the paper's per-frame spatial compression step;
the global Token Merge re-aggregation is retained because the reference always runs it as a distinct final
step regardless of how many compression stages precede it.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


class _UnionFind:
    """Union-find with iterative path-halving and union-by-rank (ported from the reference)."""

    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int32)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def batch_union(self, x_arr: np.ndarray, y_arr: np.ndarray) -> None:
        for x, y in zip(x_arr, y_arr):
            x_root = self.find(x)
            y_root = self.find(y)
            if x_root == y_root:
                continue
            if self.rank[x_root] < self.rank[y_root]:
                self.parent[x_root] = y_root
            else:
                self.parent[y_root] = x_root
                if self.rank[x_root] == self.rank[y_root]:
                    self.rank[x_root] += 1


def approximate_components(adj_matrix: np.ndarray, epsilon: float = 0.05) -> list[list[int]]:
    """Approximate connected components via epsilon-sampled Boruvka-style labeling.

    Samples `ceil(log(n) / epsilon**2)` nodes (capped at `n`), unions only sampled-node -> neighbor
    edges, and leaves every un-sampled, un-neighbored node as its own singleton. When the sample size
    saturates at `n` (small graphs, e.g. per-image token counts) this reduces to exact connected
    components; the sampling only takes effect at the video scale the reference targets.

    Args:
        adj_matrix: `[n, n]` boolean adjacency (self-loops on the diagonal are permitted and counted
            toward node degree, matching the reference).
        epsilon: error tolerance controlling the sample size; smaller keeps more nodes (more exact).

    Returns:
        A list of components (each a list of node indices) ordered by their highest-degree member,
        tie-broken toward the smallest index.
    """
    n = adj_matrix.shape[0]
    if n == 0:
        return []

    all_nodes = np.ones(n)
    all_indices = np.arange(0, n)

    sample_size = min(n, int(np.ceil(np.log(n) / epsilon**2)))
    sampled_nodes = np.random.choice(n, size=sample_size, replace=False)
    all_nodes[sampled_nodes] = 0

    # Sparse adjacency over sampled nodes; mark every touched node as covered.
    neighbor_dict = defaultdict(list)
    for i in sampled_nodes:
        neighbors = np.nonzero(adj_matrix[i])[0]
        valid_neighbors = np.intersect1d(neighbors, all_indices, assume_unique=True)
        neighbor_dict[i] = valid_neighbors
        all_nodes[neighbors] = 0

    # Nodes neither sampled nor a neighbor of any sampled node stay singletons.
    remain_nodes = np.nonzero(all_nodes)[0]
    remain_nodes = [[element] for element in remain_nodes]

    uf = _UnionFind(n)
    all_x, all_y = [], []
    for i in sampled_nodes:
        for j in neighbor_dict[i]:
            all_x.append(i)
            all_y.append(j)
    uf.batch_union(np.array(all_x), np.array(all_y))

    sampled_roots = np.array([uf.find(i) for i in sampled_nodes])
    unique_roots = np.unique(sampled_roots)

    components = []
    for root in unique_roots:
        cluster = np.where(uf.parent == root)[0].tolist()
        if len(cluster) > 0:
            components.append(cluster)
    components.extend(remain_nodes)

    degrees = np.count_nonzero(adj_matrix, axis=1)  # per-node degree (diagonal included)

    def get_sort_key(cluster):
        max_degree = -1
        min_node = float("inf")
        for node in cluster:
            current_degree = degrees[node]
            if (current_degree > max_degree) or (current_degree == max_degree and node < min_node):
                max_degree = current_degree
                min_node = node
        return min_node

    components.sort(key=get_sort_key)

    return components


def semantic_connected_components(
    features: torch.Tensor, similarity_threshold: float, epsilon: float = 0.05
) -> torch.Tensor:
    """Assign each token to a semantic connected component.

    Builds a binary cosine-similarity adjacency (`similarity > similarity_threshold`, self-loops
    included) and labels its components with the reference's epsilon-sampled approximate algorithm.

    Args:
        features: `[N, D]` token features.
        similarity_threshold: cosine-similarity cutoff in `[-1, 1]`; higher keeps more (smaller) regions.
        epsilon: error tolerance passed to `approximate_components`.

    Returns:
        `[N]` long tensor of component ids in `[0, num_components)`, ordered by each region's
        highest-degree member (tie-break smallest index), matching the reference output ordering.
    """
    num_tokens = features.shape[0]
    if num_tokens == 0:
        return torch.zeros(0, dtype=torch.long, device=features.device)

    normed = F.normalize(features.float(), dim=-1)
    similarity = normed @ normed.T
    adjacency = (similarity > similarity_threshold).cpu().numpy()

    components = approximate_components(adjacency, epsilon)

    labels = torch.full((num_tokens,), -1, dtype=torch.long, device=features.device)
    for comp_id, cluster in enumerate(components):
        labels[torch.tensor(cluster, dtype=torch.long, device=features.device)] = comp_id

    # Guarantee a total labeling: any node the approximate partition left uncovered becomes its own
    # trailing component. This cannot trigger while the sample size saturates at `N` (per-image scale).
    uncovered = (labels < 0).nonzero(as_tuple=False).flatten()
    for offset, node in enumerate(uncovered.tolist()):
        labels[node] = len(components) + offset

    return labels


def compress_tokens_scc(
    features: torch.Tensor, similarity_threshold: float, epsilon: float = 0.05
) -> torch.Tensor:
    """Compress a token set with SCC mean-pooling followed by the global Token Merge re-aggregation.

    Stage 1 mean-pools each semantic connected component into one representative. Stage 2 -- the
    reference's "Token Merge for all tokens" -- re-assigns every original token to its nearest
    representative by cosine argmax, scatter-adds them onto the representatives, and averages with
    `counts + 1` (blending the component mean with a nearest-representative re-clustering of all tokens).

    Args:
        features: `[N, D]` token features.
        similarity_threshold: cosine-similarity cutoff passed to `semantic_connected_components`.
        epsilon: error tolerance passed to `semantic_connected_components`.

    Returns:
        `[N', D]` with one representative per component (`N' <= N`), preserving the input dtype.
    """
    if features.ndim != 2:
        raise ValueError(f"`features` must be `[N, D]`, got shape {tuple(features.shape)}")

    num_tokens = features.shape[0]
    if num_tokens <= 1:
        return features

    labels = semantic_connected_components(features, similarity_threshold, epsilon)
    num_components = int(labels.max().item()) + 1

    # Stage 1: mean-pool each semantic connected component.
    summed = torch.zeros(num_components, features.shape[1], dtype=features.dtype, device=features.device)
    summed.index_add_(0, labels, features)
    counts = torch.zeros(num_components, dtype=features.dtype, device=features.device)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=features.dtype))
    selected_tokens = summed / counts.unsqueeze(-1)

    # Stage 2: global "Token Merge for all tokens" -- re-assign every original token to its nearest
    # representative by cosine argmax, scatter-add, then average (counts + 1).
    normed_reps = F.normalize(selected_tokens.float(), dim=-1)
    normed_all = F.normalize(features.float(), dim=-1)
    closest_indices = (normed_all @ normed_reps.T).argmax(dim=1)

    merged = torch.zeros_like(selected_tokens)
    merged.index_add_(0, closest_indices, features)
    merge_counts = torch.bincount(closest_indices, minlength=num_components).to(features.dtype) + 1

    return (selected_tokens + merged) / merge_counts.unsqueeze(-1)


def compress_visual_tokens(
    projected_patch_embeddings: torch.Tensor, similarity_threshold: float, epsilon: float = 0.05
) -> torch.Tensor:
    """Apply SCC compression to a `[B, N, D]` block of projected visual tokens.

    Requires `B == 1`: components are data-dependent, so per-sample counts differ and cannot be stacked
    back into a rectangular batch. This matches LLaVA-Scissor's training-free, single-sequence inference
    setting; batched training/inference should leave compression disabled.

    Args:
        projected_patch_embeddings: `[1, N, D]` projector output.
        similarity_threshold: cosine-similarity cutoff passed to `compress_tokens_scc`.
        epsilon: error tolerance passed to `compress_tokens_scc`.

    Returns:
        `[1, N', D]` compressed visual tokens (`N' <= N`).
    """
    if projected_patch_embeddings.ndim != 3:
        shape = tuple(projected_patch_embeddings.shape)
        raise ValueError(f"`projected_patch_embeddings` must be `[B, N, D]`, got {shape}")
    if projected_patch_embeddings.shape[0] != 1:
        raise ValueError("SCC visual-token compression is only defined for batch size 1 (single sequence).")

    compressed = compress_tokens_scc(projected_patch_embeddings[0], similarity_threshold, epsilon)
    return compressed.unsqueeze(0)
