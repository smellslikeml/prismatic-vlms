"""
test_visual_token_reduction.py

Tests for the PACE-adapted visual-token reduction module and its wiring into
`PrismaticVLM.forward`. Exercises the new reducer against the output of an
existing (non-new) projector from `prismatic.util.nn_utils`, and checks the
call-site plumbing on `prismatic.models.vlms.prismatic.PrismaticVLM`.
"""

import math

import torch

from prismatic.util.nn_utils import LinearProjector
from prismatic.util.visual_token_reduction import (
    VisualTokenReducer,
    build_visual_token_reducer,
    token_salience,
)


def test_reduces_projected_token_stream_to_budget():
    # Existing projector produces the exact [bsz, N, llm_dim] tensor the reducer consumes at the call site.
    bsz, num_patches, vision_dim, llm_dim = 2, 20, 32, 64
    projector = LinearProjector(vision_dim, llm_dim)
    projected = projector(torch.randn(bsz, num_patches, vision_dim))

    reducer = VisualTokenReducer(retention_ratio=0.5, keep_context=True)
    reduced = reducer(projected)

    expected_k = math.ceil(0.5 * num_patches)
    assert reduced.shape == (bsz, expected_k, llm_dim)
    assert reduced.dtype == projected.dtype


def test_downstream_mask_shape_follows_reduced_count():
    # Mirrors PrismaticVLM.forward: the visual attention mask is built from `.shape[1]`.
    projected = torch.randn(3, 50, 16)
    reducer = VisualTokenReducer(retention_ratio=0.1, keep_context=True)
    reduced = reducer(projected)

    mask = torch.full((reduced.shape[0], reduced.shape[1]), True, dtype=torch.bool)
    assert mask.shape[1] == reduced.shape[1] < projected.shape[1]


def test_context_token_is_mean_pooled():
    projected = torch.randn(1, 12, 8)
    reducer = VisualTokenReducer(retention_ratio=0.5, keep_context=True)
    reduced = reducer(projected)
    # First retained slot is the holistic mean-pooled context token.
    assert torch.allclose(reduced[:, 0, :], projected.mean(dim=1), atol=1e-5)


def test_retains_most_salient_tokens():
    # Build a stream where a few tokens sit far from the mean; they must survive reduction.
    projected = torch.zeros(1, 10, 4)
    projected[0, 3] = 5.0
    projected[0, 7] = -5.0

    reducer = VisualTokenReducer(retention_ratio=0.3, keep_context=False)
    reduced = reducer(projected)

    salience = token_salience(projected)
    assert salience[0, 3] > salience[0, 0]
    # The two outlier tokens should both appear among the retained detail tokens.
    kept = {tuple(row.tolist()) for row in reduced[0]}
    assert (5.0, 5.0, 5.0, 5.0) in kept
    assert (-5.0, -5.0, -5.0, -5.0) in kept


def test_passthrough_and_builder_noop():
    projected = torch.randn(2, 8, 4)
    # retention_ratio == 1.0 is a pass-through.
    assert torch.equal(VisualTokenReducer(retention_ratio=1.0)(projected), projected)
    # Builder returns None for unset / no-op ratios (disabled by default).
    assert build_visual_token_reducer(None) is None
    assert build_visual_token_reducer(1.0) is None
    assert isinstance(build_visual_token_reducer(0.25), VisualTokenReducer)


def test_invalid_ratio_rejected():
    for bad in (0.0, -0.1, 1.5):
        try:
            VisualTokenReducer(retention_ratio=bad)
            raise AssertionError(f"expected ValueError for retention_ratio={bad}")
        except ValueError:
            pass


def test_prismatic_vlm_wiring_surface():
    # The call-site module (non-new) exposes the enable/disable hooks and defaults to disabled.
    from prismatic.models.vlms.prismatic import PrismaticVLM

    assert hasattr(PrismaticVLM, "enable_visual_token_reduction")
    assert hasattr(PrismaticVLM, "disable_visual_token_reduction")
