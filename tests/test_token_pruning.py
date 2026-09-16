"""
test_token_pruning.py

Tests for the training-free visual-token pruning utility and its wiring into `PrismaticVLM.forward`.
"""

import torch
import torch.nn as nn

from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.util.token_pruning import spatially_structured_prune


def test_prune_reduces_to_budget_and_preserves_dim():
    tokens = torch.randn(3, 256, 8)  # 16 x 16 grid
    pruned = spatially_structured_prune(tokens, budget=32, num_regions_hw=(4, 4))
    assert pruned.shape == (3, 32, 8)


def test_prune_is_noop_when_budget_exceeds_tokens():
    tokens = torch.randn(2, 64, 4)
    pruned = spatially_structured_prune(tokens, budget=128)
    assert pruned.shape == tokens.shape
    assert torch.equal(pruned, tokens)


def test_prune_skips_non_square_layout():
    tokens = torch.randn(2, 257, 4)  # e.g. 256 patches + CLS --> not a clean grid
    pruned = spatially_structured_prune(tokens, budget=64)
    assert torch.equal(pruned, tokens)


def test_prune_preserves_spatial_coverage():
    # Every region must contribute at least one token when budget >= number of regions.
    tokens = torch.randn(1, 64, 4)  # 8 x 8 grid, 2x2 regions => region size 4x4 = 16 tokens each
    budget = 8
    pruned = spatially_structured_prune(tokens, budget=budget, grid_hw=(8, 8), num_regions_hw=(2, 2))
    assert pruned.shape == (1, budget, 4)

    # Recover which flattened indices survived, then map to their 2x2 region and confirm all four appear.
    kept_rows = pruned[0]
    regions = set()
    for row in kept_rows:
        matches = (tokens[0] == row).all(dim=-1).nonzero(as_tuple=True)[0]
        idx = int(matches[0])
        r, c = divmod(idx, 8)
        regions.add((r // 4) * 2 + (c // 4))
    assert regions == {0, 1, 2, 3}


def test_prune_favors_higher_structure_regions():
    # Smooth low-variation ramp everywhere (unique per token), with strong local structure in region 0.
    grid = torch.arange(8 * 8 * 4, dtype=torch.float32).reshape(1, 8, 8, 4) * 1e-3
    grid[0, :4, :4] += torch.randn(4, 4, 4) * 10.0  # rich structure in region 0
    tokens = grid.reshape(1, 64, 4)

    pruned = spatially_structured_prune(tokens, budget=8, grid_hw=(8, 8), num_regions_hw=(2, 2))

    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for row in pruned[0]:
        matches = (tokens[0] == row).all(dim=-1).nonzero(as_tuple=True)[0]
        idx = int(matches[0])
        r, c = divmod(idx, 8)
        counts[(r // 4) * 2 + (c // 4)] += 1

    # Coverage floor keeps >=1 per region; the structured region should draw the extra budget.
    assert all(v >= 1 for v in counts.values())
    assert counts[0] == max(counts.values())


class _StubLLMBackbone:
    """Minimal stand-in for an `LLMBackbone` exercising only the surface `PrismaticVLM.forward` touches."""

    def __init__(self, embed_dim: int) -> None:
        self.embed_dim = embed_dim
        self.captured = None

    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return torch.zeros(input_ids.shape[0], input_ids.shape[1], self.embed_dim)

    def __call__(self, **kwargs):
        self.captured = kwargs
        return kwargs


def _make_wired_vlm(embed_dim: int) -> PrismaticVLM:
    """Construct a bare `PrismaticVLM` wired with lightweight stubs, bypassing heavy backbone loading."""
    def _identity_vision_backbone(pixel_values):
        return pixel_values

    vlm = PrismaticVLM.__new__(PrismaticVLM)
    nn.Module.__init__(vlm)
    vlm.projector = nn.Identity()
    vlm.vision_backbone = _identity_vision_backbone
    vlm.llm_backbone = _StubLLMBackbone(embed_dim)
    vlm.vision_backbone_requires_grad = False
    vlm.visual_token_budget = None
    vlm.visual_token_prune_regions = (2, 2)
    return vlm


def _run_forward(vlm: PrismaticVLM, num_patches: int, embed_dim: int, seq_len: int = 5):
    pixel_values = torch.randn(2, num_patches, embed_dim)
    input_ids = torch.randint(1, 100, (2, seq_len))
    attention_mask = torch.ones(2, seq_len, dtype=torch.long)
    labels = torch.full((2, seq_len), -100, dtype=torch.long)
    vlm.forward(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)
    return vlm.llm_backbone.captured


def test_forward_without_pruning_keeps_all_visual_tokens():
    embed_dim, num_patches, seq_len = 8, 64, 5
    vlm = _make_wired_vlm(embed_dim)
    captured = _run_forward(vlm, num_patches, embed_dim, seq_len)
    assert captured["inputs_embeds"].shape[1] == seq_len + num_patches


def test_forward_with_pruning_reduces_fused_sequence_length():
    embed_dim, num_patches, seq_len, budget = 8, 64, 5, 16
    vlm = _make_wired_vlm(embed_dim)
    vlm.configure_visual_token_pruning(budget=budget, num_regions_hw=(2, 2))
    captured = _run_forward(vlm, num_patches, embed_dim, seq_len)

    # The forward pass must now carry `budget` visual tokens instead of `num_patches`, with masks/labels in sync.
    assert captured["inputs_embeds"].shape[1] == seq_len + budget
    assert captured["attention_mask"].shape[1] == seq_len + budget
    assert captured["labels"].shape[1] == seq_len + budget
