"""
test_convnext_backbone.py

Integration tests for the hierarchical (ConvNeXt) vision backbone, exercised through the *existing*
`prismatic.models.materialize` factory + registry (the wiring call site). `timm` model construction is monkeypatched so
the test runs without downloading pretrained weights.
"""

import torch
from torchvision.transforms import Compose, InterpolationMode, Resize, ToTensor

# NOTE: import through the existing (non-new) call-site module to prove the wiring, not just the new module.
from prismatic.models.backbones.vision import ConvNeXtBackbone
from prismatic.models.materialize import VISION_BACKBONES, get_vision_backbone_and_transform

FAKE_EMBED_DIM = 1024
SPATIAL_REDUCTION = 32


class _FakeConvNeXt(torch.nn.Module):
    """Minimal stand-in for a TIMM ConvNeXt: exposes `num_features` and a hierarchical `forward_features`."""

    num_features = FAKE_EMBED_DIM

    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        b, _, h, w = pixel_values.shape
        return torch.zeros(b, self.num_features, h // SPATIAL_REDUCTION, w // SPATIAL_REDUCTION)


def _patch_timm(monkeypatch):
    import timm

    monkeypatch.setattr(timm, "create_model", lambda *args, **kwargs: _FakeConvNeXt())
    monkeypatch.setattr(
        timm.data,
        "resolve_model_data_config",
        lambda _model: {
            "input_size": (3, 256, 256),
            "interpolation": "bicubic",
            "mean": (0.5, 0.5, 0.5),
            "std": (0.5, 0.5, 0.5),
            "crop_pct": 1.0,
            "crop_mode": "center",
        },
    )
    monkeypatch.setattr(
        timm.data,
        "create_transform",
        lambda **_kwargs: Compose([Resize(224, interpolation=InterpolationMode.BICUBIC), ToTensor()]),
    )


def test_convnext_registered_in_materialize():
    # The wiring edit must expose the new ids through the existing registry, bound to ConvNeXtBackbone.
    for backbone_id in ("convnext-clip-b-256px", "convnext-clip-l-512px"):
        assert backbone_id in VISION_BACKBONES
        assert VISION_BACKBONES[backbone_id]["cls"] is ConvNeXtBackbone


def test_factory_builds_convnext_backbone(monkeypatch):
    _patch_timm(monkeypatch)

    backbone, image_transform = get_vision_backbone_and_transform("convnext-clip-b-256px", "resize-naive")

    assert isinstance(backbone, ConvNeXtBackbone)
    assert backbone.embed_dim == FAKE_EMBED_DIM
    # 256px input, 32x hierarchical reduction => an 8x8 grid => 64 visual tokens.
    assert backbone.num_patches == (256 // SPATIAL_REDUCTION) ** 2 == 64
    assert callable(image_transform)


def test_forward_returns_patch_feature_sequence(monkeypatch):
    _patch_timm(monkeypatch)

    backbone, _ = get_vision_backbone_and_transform("convnext-clip-b-256px", "resize-naive")
    pixel_values = torch.zeros(2, 3, backbone.default_image_size, backbone.default_image_size)

    patch_features = backbone(pixel_values)

    # Downstream projector expects [bsz, num_patches, embed_dim] -- exactly what the ViT backbones emit.
    assert patch_features.shape == (2, backbone.num_patches, backbone.embed_dim)


def test_convnext_yields_fewer_tokens_than_vit16(monkeypatch):
    """Core ConvLLaVA insight: a hierarchical backbone produces far fewer high-resolution visual tokens."""
    _patch_timm(monkeypatch)

    resolution = 512
    backbone, _ = get_vision_backbone_and_transform("convnext-clip-l-512px", "resize-naive")

    vit16_tokens = (resolution // 16) ** 2  # a patch16 ViT at the same resolution
    assert backbone.default_image_size == resolution
    assert backbone.num_patches == (resolution // SPATIAL_REDUCTION) ** 2
    assert backbone.num_patches * 4 == vit16_tokens  # 4x token reduction at equal resolution
