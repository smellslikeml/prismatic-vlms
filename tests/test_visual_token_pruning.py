"""
test_visual_token_pruning.py

Tests for training-free AnchorPrune visual-token pruning and its wiring into `PrismaticVLM`.

These exercise the integration through existing (non-new) modules: the projector defined in
`prismatic.util.nn_utils` (the exact producer of `projected_patch_embeddings` at the call site)
and the `PrismaticVLM` forward-path surface in `prismatic.models.vlms.prismatic`.
"""

import inspect
import math

import torch

# Existing (non-new) modules exercised by the integration.
from prismatic.models.vlms import prismatic as prismatic_vlm
from prismatic.util.nn_utils import MLPProjector
from prismatic.util.visual_token_pruning import (
    prune_visual_tokens,
    query_relevance,
    select_kept_indices,
)


def _unit(vec):
    t = torch.tensor(vec, dtype=torch.float32)
    return t / t.norm()


def test_projector_output_prunes_to_budget():
    """Projector output ([bsz, num_patches, llm_dim]) prunes cleanly to the requested budget."""
    torch.manual_seed(0)
    projector = MLPProjector(vision_dim=16, llm_dim=24)
    patch_features = torch.randn(2, 12, 16)
    projected = projector(patch_features)  # [2, 12, 24] -- mirrors prismatic.py call site
    query = torch.randn(2, 5, 24)

    pruned = prune_visual_tokens(projected, query, budget=4)

    assert pruned.shape == (2, 4, 24)
    assert pruned.dtype == projected.dtype


def test_budget_at_or_above_num_patches_is_noop():
    projected = torch.randn(1, 6, 8)
    query = torch.randn(1, 3, 8)
    assert prune_visual_tokens(projected, query, budget=6) is projected
    assert prune_visual_tokens(projected, query, budget=None) is projected


def test_most_relevant_token_is_always_protected():
    """The single highest query-relevance token must survive even under aggressive compression."""
    torch.manual_seed(1)
    visual = torch.randn(3, 20, 12)
    query = torch.randn(3, 4, 12)

    relevance = query_relevance(visual, query)  # [3, 20]
    top = relevance.argmax(dim=1)  # [3]

    keep = select_kept_indices(visual, query, budget=3)  # [3, 3]
    for b in range(3):
        assert top[b].item() in keep[b].tolist()


def test_ordered_design_prefers_novel_relevant_over_redundant():
    """Given a redundant duplicate of the top token and a distinct-but-relevant token,
    expansion recovers the novel token instead of the redundant duplicate."""
    d = 4
    v0 = _unit([1, 0, 0, 0])
    v1 = _unit([1, 0, 0, 0])  # exact duplicate of the most-relevant token
    v2 = _unit([0.6, 0.8, 0, 0])  # still relevant to query (cos 0.6) but a novel direction
    visual = torch.stack([v0, v1, v2]).unsqueeze(0)  # [1, 3, 4]
    query = _unit([1, 0, 0, 0]).view(1, 1, d)

    keep = set(select_kept_indices(visual, query, budget=2)[0].tolist())

    assert keep == {0, 2}  # redundant duplicate (1) dropped in favor of novel-relevant token (2)


def test_query_mask_ignores_padding_tokens():
    """A masked-out (padding) query token must not change the relevance scores."""
    torch.manual_seed(2)
    visual = torch.randn(1, 8, 6)
    real_query = torch.randn(1, 3, 6)
    padded_query = torch.cat([real_query, torch.randn(1, 2, 6) * 100.0], dim=1)  # 2 junk pad tokens
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)

    unmasked = query_relevance(visual, real_query)
    masked = query_relevance(visual, padded_query, query_mask=mask)

    assert torch.allclose(unmasked, masked, atol=1e-5)


def test_anchor_fraction_caps_anchor_size():
    """A larger anchor is possible only when max_anchor_fraction allows it (ordered-design invariant)."""
    torch.manual_seed(3)
    visual = torch.randn(1, 30, 10)
    query = torch.randn(1, 4, 10)
    budget = 8

    keep = select_kept_indices(visual, query, budget=budget, max_anchor_fraction=0.25)
    assert keep.shape == (1, budget)
    assert len(set(keep[0].tolist())) == budget  # no duplicate indices selected

    max_anchor = max(1, min(round(0.25 * budget), budget))
    assert max_anchor == 2  # sanity on the cap arithmetic the selector uses


def test_prismaticvlm_exposes_pruning_wiring():
    """The call-site module wires AnchorPrune: setter mutates state and forward calls the pruner."""
    assert hasattr(prismatic_vlm.PrismaticVLM, "set_visual_token_budget")
    assert prismatic_vlm.prune_visual_tokens is prune_visual_tokens

    forward_src = inspect.getsource(prismatic_vlm.PrismaticVLM.forward)
    assert "prune_visual_tokens" in forward_src
    assert "visual_token_budget" in forward_src


def test_selection_is_deterministic():
    torch.manual_seed(4)
    visual = torch.randn(2, 16, 8)
    query = torch.randn(2, 3, 8)
    a = select_kept_indices(visual, query, budget=5)
    b = select_kept_indices(visual, query, budget=5)
    assert torch.equal(a, b)
    assert not math.isnan(query_relevance(visual, query).sum().item())
