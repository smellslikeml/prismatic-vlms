"""
test_tipsv2_backbone.py

Integration tests for the TIPSv2 vision backbone: verifies registry wiring through the existing
`prismatic.models.materialize` factory and the patch-feature forward contract (shape, dtype-agnostic).

Tests that require downloading `google/tipsv2-*` checkpoints skip automatically when HF Hub is unreachable;
the whole module skips when torch is not installed.
"""

import importlib.util

import pytest

# Skip entire module if torch is unavailable (e.g., lightweight CI environments)
pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")


def _hf_hub_reachable() -> bool:
    """Probe HF Hub connectivity (checkpoint-downloading tests skip when offline)."""
    import urllib.request

    try:
        with urllib.request.urlopen("https://huggingface.co/google/tipsv2-b14/resolve/main/config.json", timeout=5):
            return True
    except Exception:
        return False


def test_tipsv2_registry_wiring() -> None:
    """TIPSv2 backbones are dispatched through the existing vision backbone registry in `materialize.py`."""
    from prismatic.models.backbones.vision import TIPSv2ViTBackbone
    from prismatic.models.materialize import VISION_BACKBONES

    for backbone_id in ("tipsv2-vit-b14", "tipsv2-vit-l14"):
        assert backbone_id in VISION_BACKBONES, f"`{backbone_id}` missing from VISION_BACKBONES!"
        assert VISION_BACKBONES[backbone_id]["cls"] is TIPSv2ViTBackbone
        assert VISION_BACKBONES[backbone_id]["kwargs"]["default_image_size"] == 448


def test_tipsv2_supported_ids_and_dims() -> None:
    """Registry IDs map to valid HF Hub paths with the expected vision embedding dimensions."""
    from prismatic.models.backbones.vision.tipsv2_vit import TIPSV2_EMBED_DIMS, TIPSV2_VISION_BACKBONES

    assert TIPSV2_VISION_BACKBONES == {
        "tipsv2-vit-b14": "google/tipsv2-b14",
        "tipsv2-vit-l14": "google/tipsv2-l14",
    }
    assert TIPSV2_EMBED_DIMS == {"tipsv2-vit-b14": 768, "tipsv2-vit-l14": 1024}
    assert set(TIPSV2_VISION_BACKBONES) == set(TIPSV2_EMBED_DIMS)


@pytest.mark.skipif(not _hf_hub_reachable(), reason="HF Hub unreachable; skipping checkpoint-dependent test")
def test_tipsv2_backbone_forward_patch_shape() -> None:
    """Instantiate `tipsv2-vit-b14` from its registry ID and check the (batch, num_patches, embed_dim) contract."""
    import torch
    from PIL import Image

    from prismatic.models.materialize import get_vision_backbone_and_transform

    vision_backbone, image_transform = get_vision_backbone_and_transform("tipsv2-vit-b14", "resize-naive")

    # Interface contract (no forward pass required)
    assert vision_backbone.default_image_resolution == (3, 448, 448)
    assert vision_backbone.embed_dim == 768
    assert vision_backbone.num_patches == (448 // 14) ** 2
    assert vision_backbone.half_precision_dtype == torch.bfloat16

    # Image transform =>> [0, 1] range tensor at the default resolution
    transformed = image_transform(Image.new("RGB", (256, 128)))
    assert transformed.shape == (3, 448, 448)
    assert 0.0 <= transformed.min() <= transformed.max() <= 1.0

    # Forward pass on a fake input returns patch features with CLS/register tokens stripped
    with torch.no_grad():
        patch_features = vision_backbone(torch.randn(1, 3, 448, 448))
    assert patch_features.shape == (1, vision_backbone.num_patches, vision_backbone.embed_dim)
