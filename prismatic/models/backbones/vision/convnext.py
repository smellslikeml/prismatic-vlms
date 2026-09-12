"""
convnext.py

Hierarchical (ConvNeXt) Vision Backbone for Prismatic VLMs.

Unlike the ViT backbones (`base_vision.TimmViTBackbone`), a ConvNeXt featurizer is *hierarchical*: its stem + three
downsampling stages reduce spatial resolution by a total factor of 32 (vs. 16 for a `patch16` ViT). At an identical
input resolution this yields ~4x fewer visual tokens, and the token count grows sub-quadratically with resolution --
exactly the property that lets high-resolution inputs stay affordable for the downstream projector + LLM.

Adapted from ConvLLaVA: Hierarchical Backbones as Visual Encoder for Large Multimodal Models
(Ge et al., 2024; https://arxiv.org/abs/2405.15738). We keep the paper's core mechanism -- a hierarchical CNN visual
encoder producing compact high-resolution features that drop into the standard `VisionBackbone` interface with no
change to the projector or LLM path. We intentionally do *not* reproduce (a) the paper's additional,
separately-pretrained 5th downsampling stage (32x -> 64x), or (b) its two-stage vision-encoder training recipe -- both
require a pretraining loop and released ConvLLaVA weights that live downstream of this repo's backbone abstraction.
"""

from functools import partial
from typing import Callable, Tuple

import timm
import torch
from timm.models.convnext import ConvNeXt, ConvNeXtStage
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from torchvision.transforms import Compose, Resize

from prismatic.models.backbones.vision.base_vision import LetterboxPad, VisionBackbone

# Registry =>> Supported ConvNeXt Vision Backbones (from TIMM), using CLIP-pretrained weights (as in ConvLLaVA)
CONVNEXT_VISION_BACKBONES = {
    "convnext-clip-b-256px": "convnext_base.clip_laion2b",
    "convnext-clip-l-512px": "convnext_large_mlp.clip_laion2b_augreg",
}

# ConvNeXt reduces spatial resolution by 4x at the stem and 2x at each of the following three stages (4 * 2^3 = 32).
CONVNEXT_SPATIAL_REDUCTION = 32


class ConvNeXtBackbone(VisionBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 256) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)
        self.timm_path_or_url = CONVNEXT_VISION_BACKBONES[vision_backbone_id]
        self.dtype = torch.bfloat16

        # Initialize Featurizer (ConvNeXt) by downloading from HF / TIMM Hub if necessary. ConvNeXt is fully
        #   convolutional and size-agnostic, so (unlike the ViT path) we do *not* pass `img_size` here.
        self.featurizer: ConvNeXt = timm.create_model(self.timm_path_or_url, pretrained=True, num_classes=0)
        self.featurizer.eval()

        # Get Config =>> Note :: Override default image size to ensure correct image transform
        self.data_cfg = timm.data.resolve_model_data_config(self.featurizer)
        self.data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        # Initialize Default Image Transform --> Modified by `self.image_resize_strategy`
        default_image_transform = timm.data.create_transform(**self.data_cfg, is_training=False)

        # Switch on `image_resize_strategy` (mirrors `base_vision.TimmViTBackbone`)
        if self.image_resize_strategy == "resize-naive":
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert isinstance(default_image_transform.transforms[0], Resize)

            target_size = (self.default_image_size, self.default_image_size)
            self.image_transform = Compose(
                [
                    Resize(target_size, interpolation=default_image_transform.transforms[0].interpolation),
                    *default_image_transform.transforms[1:],
                ]
            )

        elif self.image_resize_strategy == "resize-crop":
            self.image_transform = default_image_transform

        elif self.image_resize_strategy == "letterbox":
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert "mean" in self.data_cfg, "TIMM `data_cfg` missing image normalization mean!"

            # Compute Padding Fill Value (rescaled normalization mean if applicable)
            fill = tuple([int(x * 255) for x in self.data_cfg["mean"]])

            # Build New Transform
            self.image_transform = Compose([LetterboxPad(fill), *default_image_transform.transforms])

        else:
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP policy that wraps each ConvNeXt stage and then the _entire_ featurizer."""
        convnext_wrap_policy = partial(_module_wrap_policy, module_classes={ConvNeXt})
        stage_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={ConvNeXtStage})
        return partial(_or_policy, policies=[convnext_wrap_policy, stage_wrap_policy])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run pixels through ConvNeXt, flattening the final [B, C, H, W] feature map to [B, num_patches, embed_dim]."""
        feature_map = self.featurizer.forward_features(pixel_values)
        return feature_map.flatten(2).transpose(1, 2)

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return self.data_cfg["input_size"]

    @property
    def embed_dim(self) -> int:
        return self.featurizer.num_features

    @property
    def num_patches(self) -> int:
        grid = self.default_image_size // CONVNEXT_SPATIAL_REDUCTION
        return grid * grid

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return self.dtype
