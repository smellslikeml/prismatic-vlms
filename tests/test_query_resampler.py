"""
test_query_resampler.py

Exercises the `resampler` arch_specifier wiring in `PrismaticVLM` (the projector call site) plus the standalone
`QueryResampler` connector adapted from the Q-Former of https://arxiv.org/abs/2410.09489.

We build lightweight stand-ins for the vision / LLM backbones so we can construct a `PrismaticVLM` without downloading
any real weights -- the only backbone attributes the constructor touches are `embed_dim` and `llm.generation_config`.
"""

from types import SimpleNamespace

import torch

from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.util.nn_utils import LinearProjector, MLPProjector
from prismatic.util.resampler import QueryResampler

VISION_DIM, LLM_DIM = 32, 64


class _FakeTokenizer:
    """Encodes each trigger string to a single dummy token id (the constructor asserts len == 1)."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list:
        return [0]


def _make_vlm(arch_specifier: str) -> PrismaticVLM:
    vision_backbone = SimpleNamespace(embed_dim=VISION_DIM)
    llm_backbone = SimpleNamespace(
        embed_dim=LLM_DIM,
        llm=SimpleNamespace(generation_config=SimpleNamespace()),
        tokenizer=_FakeTokenizer(),
    )
    return PrismaticVLM("test-vlm", vision_backbone, llm_backbone, arch_specifier=arch_specifier)


def test_arch_specifier_dispatches_to_query_resampler() -> None:
    # New branch selects the resampler, and still works behind the existing `no-align+` prefix convention
    assert isinstance(_make_vlm("resampler").projector, QueryResampler)
    assert isinstance(_make_vlm("no-align+resampler").projector, QueryResampler)

    # Existing branches remain intact
    assert isinstance(_make_vlm("linear").projector, LinearProjector)
    assert isinstance(_make_vlm("gelu-mlp").projector, MLPProjector)


def test_resampler_projector_forward_shape_through_vlm() -> None:
    vlm = _make_vlm("resampler")
    num_queries = vlm.projector.num_queries

    # Simulate patch features coming out of the vision backbone: [bsz, num_patches, vision_dim]
    patch_features = torch.randn(2, 137, VISION_DIM)
    projected = vlm.projector(patch_features)

    # Resampler compresses an arbitrary number of patches to a fixed query count at LLM width
    assert projected.shape == (2, num_queries, LLM_DIM)


def test_resampler_registered_in_fsdp_wrapping_policy() -> None:
    # The wiring edit must also let FSDP wrap the new connector as its own unit
    import inspect

    source = inspect.getsource(PrismaticVLM.get_fsdp_wrapping_policy)
    assert "QueryResampler" in source


def test_resampler_is_length_invariant() -> None:
    resampler = QueryResampler(VISION_DIM, LLM_DIM, num_queries=16)
    out_short = resampler(torch.randn(1, 49, VISION_DIM))
    out_long = resampler(torch.randn(1, 400, VISION_DIM))
    assert out_short.shape == out_long.shape == (1, 16, LLM_DIM)
