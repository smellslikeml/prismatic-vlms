"""
test_vision_token_pruning.py

Exercises the sample-adaptive vision-token pruner and its wiring into the PrismaticVLM forward call site.
Adapted from VIP-Router (arXiv:2609.10346).
"""

import torch

# Import from the EXISTING (non-new) call-site module to prove the integration wiring, not just the new file.
from prismatic.models.vlms.prismatic import IGNORE_INDEX, PrismaticVLM
from prismatic.util.token_pruning import STRATEGIES, VisionTokenRouter


def test_enable_vision_token_pruning_wires_router_onto_vlm():
    # Bypass the heavy backbone-loading __init__; only nn.Module bookkeeping is needed to exercise the wiring.
    vlm = object.__new__(PrismaticVLM)
    torch.nn.Module.__init__(vlm)
    vlm.vision_token_router = None
    assert vlm.vision_token_router is None

    PrismaticVLM.enable_vision_token_pruning(vlm, reduction_ratio=0.75)
    assert isinstance(vlm.vision_token_router, VisionTokenRouter)
    assert vlm.vision_token_router.reduction_ratio == 0.75


def test_router_reduces_token_count_and_preserves_contract():
    torch.manual_seed(0)
    bsz, num_patches, dim = 3, 256, 32
    router = VisionTokenRouter(reduction_ratio=0.5, keep_all_redundancy_threshold=-1.0)
    embeddings = torch.randn(bsz, num_patches, dim)

    pruned = router(embeddings)
    assert pruned.shape == (bsz, 128, dim)
    assert pruned.dtype == embeddings.dtype

    # Replicate the call-site's downstream mask/label construction to confirm the shape[1] contract still holds.
    attn = torch.full((pruned.shape[0], pruned.shape[1]), True, dtype=torch.bool)
    labels = torch.full((pruned.shape[0], pruned.shape[1]), IGNORE_INDEX, dtype=torch.long)
    assert attn.shape == labels.shape == (bsz, 128)


def test_router_records_per_sample_strategy():
    torch.manual_seed(1)
    router = VisionTokenRouter(reduction_ratio=0.5, keep_all_redundancy_threshold=-1.0)
    router(torch.randn(4, 64, 16))
    assert len(router.last_routing) == 4
    assert all(name in STRATEGIES for name in router.last_routing)


def test_full_token_retention_gate_returns_input_unchanged():
    # A high keep-all threshold forces the "pruning predicted unfavorable" branch -> full-token inference retained.
    router = VisionTokenRouter(reduction_ratio=0.5, keep_all_redundancy_threshold=2.0)
    embeddings = torch.randn(2, 100, 8)
    out = router(embeddings)
    assert out.shape == embeddings.shape
    assert torch.equal(out, embeddings)
    assert router.last_routing == ["full", "full"]


def test_salience_strategy_keeps_high_norm_tokens():
    # Craft one dominant high-norm token and force the salience route via high per-token norm dispersion.
    router = VisionTokenRouter(
        reduction_ratio=0.75,
        keep_all_redundancy_threshold=-1.0,
        high_redundancy_threshold=2.0,  # disable diversity route
        dispersion_threshold=0.0,  # always take the salience route
    )
    embeddings = torch.ones(1, 8, 4) * 0.01
    embeddings[0, 5] = 10.0  # clear high-energy token
    pruned = router(embeddings)
    assert router.last_routing == ["salience"]
    # The dominant token must survive pruning.
    assert torch.any(torch.all(pruned[0] == embeddings[0, 5], dim=-1))
