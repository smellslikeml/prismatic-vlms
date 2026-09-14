"""
test_spatial_token_pruning.py

Unit tests for the training-free spatial visual-token pruner and its wiring into `PrismaticVLM.forward`.
"""

import torch

from prismatic.models.vlms.prismatic import IGNORE_INDEX, PrismaticVLM
from prismatic.models.vlms.spatial_token_pruning import SpatialTokenPruner, spatially_structured_prune


def test_prune_reduces_token_count_and_preserves_batch_dim():
    embeddings = torch.randn(3, 256, 8)  # 16 x 16 patch grid
    pruned = spatially_structured_prune(embeddings, keep_tokens=32, region_grid=4)

    assert pruned.shape == (3, 32, 8)
    assert pruned.dtype == embeddings.dtype


def test_prune_noop_when_budget_exceeds_tokens():
    embeddings = torch.randn(2, 49, 4)
    pruned = spatially_structured_prune(embeddings, keep_tokens=100)
    assert pruned is embeddings


def test_prune_preserves_spatial_coverage():
    # Structure concentrated in one corner; coverage guarantee must still keep tokens from every region.
    embeddings = torch.zeros(1, 64, 4)  # 8 x 8 grid
    embeddings[0, 0] = 10.0  # top-left patch is the only "rich" region
    region_grid = 2

    pruned = spatially_structured_prune(embeddings, keep_tokens=8, region_grid=region_grid)
    assert pruned.shape == (1, 8, 4)

    # With >=1 token per region guaranteed, no region can be fully starved (4 regions -> all represented).
    # Re-derive which flat indices survived by matching rows back to the source grid.
    kept_rows = pruned[0]
    matched_regions = set()
    for token in kept_rows:
        # Find the source index whose embedding matches this kept token.
        match = (embeddings[0] == token).all(dim=-1).nonzero(as_tuple=True)[0]
        for src in match.tolist():
            row, col = src // 8, src % 8
            matched_regions.add((row * region_grid // 8, col * region_grid // 8))
    assert len(matched_regions) == region_grid * region_grid


def test_prune_retains_leading_cls_token():
    embeddings = torch.randn(1, 257, 4)  # 256 patches + leading CLS
    cls = embeddings[:, :1, :].clone()
    pruned = spatially_structured_prune(embeddings, keep_tokens=17, has_cls_token=True)

    assert pruned.shape == (1, 17, 4)
    assert torch.equal(pruned[:, :1, :], cls)


def test_prune_fallback_on_non_square_grid():
    embeddings = torch.randn(2, 30, 4)  # neither N nor N-1 is a perfect square
    pruned = spatially_structured_prune(embeddings, keep_tokens=10)
    assert pruned.shape == (2, 10, 4)


class _FakeLLMBackbone:
    """Minimal stand-in that records the fused inputs handed to the LLM by `PrismaticVLM.forward`."""

    def __init__(self, embed_dim: int) -> None:
        self.embed_dim = embed_dim
        self.captured = {}

    def embed_input_ids(self, input_ids):
        return torch.randn(input_ids.shape[0], input_ids.shape[1], self.embed_dim)

    def __call__(self, **kwargs):
        self.captured = kwargs
        return kwargs


def _make_stub_vlm(pruner, embed_dim=8):
    """Build a PrismaticVLM without running __init__, wiring only the attributes `forward` touches."""
    vlm = PrismaticVLM.__new__(PrismaticVLM)
    vlm.vision_backbone_requires_grad = False
    vlm.visual_token_pruner = pruner
    vlm.vision_backbone = lambda pixel_values: torch.randn(pixel_values.shape[0], 256, embed_dim)
    vlm.projector = lambda patch_features: patch_features  # identity (embed_dim == projected dim)
    vlm.llm_backbone = _FakeLLMBackbone(embed_dim)
    return vlm


def test_forward_hook_shrinks_visual_sequence():
    embed_dim, seq_len, keep = 8, 5, 32
    vlm = _make_stub_vlm(SpatialTokenPruner(keep_tokens=keep, region_grid=4), embed_dim=embed_dim)

    input_ids = torch.randint(0, 100, (2, seq_len))
    attention_mask = torch.ones(2, seq_len, dtype=torch.bool)
    labels = torch.full((2, seq_len), IGNORE_INDEX)
    pixel_values = torch.randn(2, 3, 16, 16)

    PrismaticVLM.forward(
        vlm, input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels
    )

    fused = vlm.llm_backbone.captured["inputs_embeds"]
    # 1 (<BOS>) + keep visual tokens + (seq_len - 1) remaining text tokens.
    assert fused.shape[1] == 1 + keep + (seq_len - 1)
    assert vlm.llm_backbone.captured["attention_mask"].shape[1] == fused.shape[1]
    assert vlm.llm_backbone.captured["labels"].shape[1] == fused.shape[1]


def test_forward_without_pruner_keeps_all_visual_tokens():
    embed_dim, seq_len = 8, 5
    vlm = _make_stub_vlm(pruner=None, embed_dim=embed_dim)

    input_ids = torch.randint(0, 100, (1, seq_len))
    attention_mask = torch.ones(1, seq_len, dtype=torch.bool)
    labels = torch.full((1, seq_len), IGNORE_INDEX)
    pixel_values = torch.randn(1, 3, 16, 16)

    PrismaticVLM.forward(
        vlm, input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels
    )

    fused = vlm.llm_backbone.captured["inputs_embeds"]
    assert fused.shape[1] == 1 + 256 + (seq_len - 1)
