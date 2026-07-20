"""
visual_token_pruning.py

Training-free, query-conditioned visual-token pruning for LLaVA-style VLMs.

Adapted from "AnchorPrune: Relevance-Anchored Contextual Expansion for Visual Token
Pruning" (https://arxiv.org/abs/2607.07033). The paper's core insight is an *ordered*
two-stage design that avoids the failure mode of jointly optimizing relevance and
diversity: relevance-driven selection over-concentrates the budget on correlated local
evidence, while diversity-driven selection can drop indispensable query cues. AnchorPrune
instead (1) builds a *protected* relevance anchor of compact, query-critical evidence
whose size is adapted from the novelty profile of relevance-ranked tokens, then
(2) spends the remaining budget on *importance-weighted novelty* so contextual expansion
never displaces the anchor.

Fidelity notes (this is an adapted port, not a direct port):
  - The ordered anchor -> importance-weighted-novelty expansion mechanism is implemented
    at full fidelity, including the adaptive, novelty-gated anchor size.
  - The paper's architecture-aware relevance signal (derived from the vision encoder's
    internal attention) is substituted with a parameter-free, target-native proxy: cosine
    similarity between the projected visual tokens and the LLM-embedded query tokens.
    Prismatic exposes exactly these two tensors at the projector call site, so no model
    modification or extra parameters are required.

The public entry point `prune_visual_tokens` takes projected visual tokens
[bsz, num_patches, llm_embed_dim] plus the query token embeddings and returns a pruned
tensor [bsz, budget, llm_embed_dim] with the original per-image token dtype/order preserved.
"""

from __future__ import annotations

from typing import List, Optional

import torch


def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def query_relevance(
    visual_tokens: torch.Tensor, query_tokens: torch.Tensor, query_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Parameter-free relevance proxy: mean cosine similarity of each visual token to the query tokens.

    Args:
        visual_tokens: [bsz, num_patches, d] projected visual tokens (LLM embedding space).
        query_tokens: [bsz, seq_len, d] LLM-embedded query tokens.
        query_mask: optional [bsz, seq_len] boolean/float mask (True/1 == real token).

    Returns:
        [bsz, num_patches] relevance scores in [-1, 1].
    """
    v = _l2_normalize(visual_tokens.float())
    q = _l2_normalize(query_tokens.float())
    sim = torch.bmm(v, q.transpose(1, 2))  # [bsz, num_patches, seq_len]
    if query_mask is not None:
        mask = query_mask.to(sim.dtype).unsqueeze(1)  # [bsz, 1, seq_len]
        denom = mask.sum(dim=-1).clamp_min(1.0)  # [bsz, 1]
        return (sim * mask).sum(dim=-1) / denom
    return sim.mean(dim=-1)


def _select_indices(
    relevance: torch.Tensor,
    similarity: torch.Tensor,
    budget: int,
    redundancy_threshold: float,
    importance_weight: float,
    max_anchor_fraction: float,
) -> List[int]:
    """Run relevance-anchored contextual expansion for a single image; return the kept token indices.

    `relevance` is [N]; `similarity` is the [N, N] visual-visual cosine-similarity matrix.
    """
    num_tokens = relevance.shape[0]
    order = torch.argsort(relevance, descending=True).tolist()

    # `running_max_sim[i]` == max cosine similarity of token i to any already-selected token.
    running_max_sim = relevance.new_full((num_tokens,), -1.0)
    selected_mask = torch.zeros(num_tokens, dtype=torch.bool, device=relevance.device)

    def accept(idx: int) -> None:
        selected_mask[idx] = True
        torch.maximum(running_max_sim, similarity[idx], out=running_max_sim)

    # === Stage 1: adaptive, protected relevance anchor ===
    # Always protect the single most query-relevant token, then walk down the relevance
    # ranking adding a token only when it contributes novelty above threshold relative to
    # the current anchor. This deduplicates correlated local evidence so the anchor stays a
    # compact set of *distinct* query-critical cues; its size adapts to the novelty profile.
    max_anchor = max(1, min(round(max_anchor_fraction * budget), budget))
    accept(order[0])
    anchor_size = 1
    for idx in order[1:]:
        if anchor_size >= max_anchor:
            break
        novelty = 1.0 - running_max_sim[idx].item()
        if novelty >= redundancy_threshold:
            accept(idx)
            anchor_size += 1

    # === Stage 2: contextual expansion via importance-weighted novelty ===
    # Greedily fill the remaining budget by maximizing relevance^w * novelty relative to the
    # already-selected set. Because the anchor is selected first, expansion recovers
    # informative, non-redundant context without ever displacing the protected anchor.
    relevance_weight = relevance.clamp_min(0.0).pow(importance_weight)
    while int(selected_mask.sum().item()) < budget:
        novelty = 1.0 - running_max_sim
        score = relevance_weight * novelty
        score = score.masked_fill(selected_mask, float("-inf"))
        accept(int(torch.argmax(score).item()))

    # Preserve original (raster) token order among the survivors for positional coherence.
    return selected_mask.nonzero(as_tuple=False).flatten().tolist()


def select_kept_indices(
    visual_tokens: torch.Tensor,
    query_tokens: torch.Tensor,
    budget: int,
    query_mask: Optional[torch.Tensor] = None,
    redundancy_threshold: float = 0.15,
    importance_weight: float = 1.0,
    max_anchor_fraction: float = 0.5,
) -> torch.Tensor:
    """Return the per-image indices AnchorPrune would keep: LongTensor [bsz, budget] (raster-ordered).

    Selection is a discrete (non-differentiable) operation, so the scoring tensors are detached;
    gradients still flow to the surviving tokens via the gather in `prune_visual_tokens`.
    """
    visual_tokens = visual_tokens.detach()
    query_tokens = query_tokens.detach()
    relevance = query_relevance(visual_tokens, query_tokens, query_mask)  # [bsz, num_patches]
    normed = _l2_normalize(visual_tokens.float())
    similarity = torch.bmm(normed, normed.transpose(1, 2))  # [bsz, num_patches, num_patches]

    bsz = visual_tokens.shape[0]
    keep = visual_tokens.new_zeros((bsz, budget), dtype=torch.long)
    for b in range(bsz):
        indices = _select_indices(
            relevance[b],
            similarity[b],
            budget,
            redundancy_threshold=redundancy_threshold,
            importance_weight=importance_weight,
            max_anchor_fraction=max_anchor_fraction,
        )
        keep[b] = torch.tensor(indices, dtype=torch.long, device=visual_tokens.device)
    return keep


def prune_visual_tokens(
    visual_tokens: torch.Tensor,
    query_tokens: torch.Tensor,
    budget: int,
    query_mask: Optional[torch.Tensor] = None,
    redundancy_threshold: float = 0.15,
    importance_weight: float = 1.0,
    max_anchor_fraction: float = 0.5,
) -> torch.Tensor:
    """Prune visual tokens to `budget` via relevance-anchored contextual expansion (AnchorPrune).

    Args:
        visual_tokens: [bsz, num_patches, d] projected visual tokens.
        query_tokens: [bsz, seq_len, d] LLM-embedded query tokens (relevance conditioning).
        budget: number of visual tokens to keep per image.
        query_mask: optional [bsz, seq_len] mask marking real (non-pad) query tokens.
        redundancy_threshold: min novelty for a relevance-ranked token to grow the anchor.
        importance_weight: exponent `w` on relevance in the expansion score relevance^w * novelty.
        max_anchor_fraction: cap on the anchor as a fraction of `budget` (prevents over-concentration).

    Returns:
        [bsz, budget, d] pruned visual tokens (original dtype/device preserved). If `budget` is
        None or >= num_patches, the input is returned unchanged.
    """
    if budget is None:
        return visual_tokens
    _, num_patches, dim = visual_tokens.shape
    budget = int(budget)
    if budget <= 0 or budget >= num_patches:
        return visual_tokens

    keep = select_kept_indices(
        visual_tokens,
        query_tokens,
        budget,
        query_mask=query_mask,
        redundancy_threshold=redundancy_threshold,
        importance_weight=importance_weight,
        max_anchor_fraction=max_anchor_fraction,
    )
    gather_index = keep.unsqueeze(-1).expand(-1, -1, dim)
    return torch.gather(visual_tokens, 1, gather_index)
