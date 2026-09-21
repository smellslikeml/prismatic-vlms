"""
test_fusion_projector_integration.py

Integration tests for the LEO-style post-adaptation fusion projector, exercised through the real `PrismaticVLM`
constructor (the call site) rather than in isolation. These assert that the `arch_specifier` dispatch selects the new
projector, that the projector is registered for FSDP wrapping, and that the wired-up module produces the interleaved
sequence-level fusion the LLM boundary expects.
"""

from functools import partial

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import _module_wrap_policy

from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.util.fusion_projectors import PostAdaptationFusionProjector

DINO_DIM, SIGLIP_DIM, LLM_DIM, NUM_PATCHES = 8, 12, 16, 5


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        # Every trigger string PrismaticVLM probes must map to exactly one token.
        del text, add_special_tokens
        return [1]


class _FakeLLM:
    generation_config = None


class _FakeLLMBackbone(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.llm = _FakeLLM()
        self.tokenizer = _FakeTokenizer()

    def get_fsdp_wrapping_policy(self):
        return partial(_module_wrap_policy, module_classes=set())


class _FakeDinoSigLIPBackbone(nn.Module):
    """Stand-in for `DinoSigLIPViTBackbone` exposing the same `embed_dim` / `encoder_dims` contract."""

    def __init__(self, dino_dim: int, siglip_dim: int) -> None:
        super().__init__()
        self._dino_dim, self._siglip_dim = dino_dim, siglip_dim

    @property
    def embed_dim(self) -> int:
        return self._dino_dim + self._siglip_dim

    @property
    def encoder_dims(self):
        return self._dino_dim, self._siglip_dim

    def get_fsdp_wrapping_policy(self):
        return partial(_module_wrap_policy, module_classes=set())


def _build_vlm(arch_specifier: str) -> PrismaticVLM:
    return PrismaticVLM(
        model_id="test-fusion",
        vision_backbone=_FakeDinoSigLIPBackbone(DINO_DIM, SIGLIP_DIM),
        llm_backbone=_FakeLLMBackbone(LLM_DIM),
        arch_specifier=arch_specifier,
    )


def test_arch_specifier_selects_fusion_projector() -> None:
    vlm = _build_vlm("no-align+fused-interleave-gelu-mlp")
    assert isinstance(vlm.projector, PostAdaptationFusionProjector)
    # One independent adapter per encoder (post-adaptation fusion, not a single shared channel projection).
    assert len(vlm.projector.projectors) == 2
    assert list(vlm.projector.encoder_dims) == [DINO_DIM, SIGLIP_DIM]


def test_fused_gelu_specifier_still_uses_default_projector() -> None:
    # The new branch must not shadow the pre-existing fused-gelu-mlp path.
    vlm = _build_vlm("no-align+fused-gelu-mlp")
    assert not isinstance(vlm.projector, PostAdaptationFusionProjector)


def test_projector_registered_for_fsdp_wrapping() -> None:
    vlm = _build_vlm("no-align+fused-interleave-gelu-mlp")
    # The FSDP policy is an `_or_policy` partial; its prismatic sub-policy must cover the new projector class.
    policies = vlm.get_fsdp_wrapping_policy().keywords["policies"]
    covered = set()
    for policy in policies:
        covered |= set(policy.keywords.get("module_classes", set()))
    assert PostAdaptationFusionProjector in covered


def test_forward_interleaves_tokens_along_sequence() -> None:
    vlm = _build_vlm("no-align+fused-interleave-gelu-mlp")
    # Mimic the channel-concatenated features produced by DinoSigLIPViTBackbone.forward.
    fused_patches = torch.randn(2, NUM_PATCHES, DINO_DIM + SIGLIP_DIM)
    out = vlm.projector(fused_patches)

    # Post-adaptation fusion => sequence length doubles (one token per encoder per patch); channels are the LLM dim.
    assert out.shape == (2, NUM_PATCHES * 2, LLM_DIM)

    # Interleaving contract: even positions come from encoder 0 (DINO), odd from encoder 1 (SigLIP).
    dino_proj = vlm.projector.projectors[0](fused_patches[..., :DINO_DIM])
    siglip_proj = vlm.projector.projectors[1](fused_patches[..., DINO_DIM:])
    assert torch.allclose(out[:, 0::2], dino_proj)
    assert torch.allclose(out[:, 1::2], siglip_proj)
