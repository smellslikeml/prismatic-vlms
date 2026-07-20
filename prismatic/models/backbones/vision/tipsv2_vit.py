"""
tipsv2_vit.py

Vision Backbone wrapper for the TIPSv2 family of image-text encoders (Google DeepMind), adapted from
"TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment" (arXiv:2604.12012).

TIPSv2's contribution consumed here is its *pretrained* dense patch representation: patch-level distillation and
the iBOT++ objective yield patch features with substantially stronger dense patch-text alignment than SigLIP-style
encoders, making them a drop-in visual featurizer for VLMs targeting grounding/dense tasks.

Deviations from the TIMM-based backbone convention (deliberate; TIPSv2 is not in `timm==0.9.10`):
    - Loads via Hugging Face `AutoModel.from_pretrained(..., trust_remote_code=True)` instead of `timm.create_model`.
    - Image transform is hand-built per the HF model card: resize + ToTensor in [0, 1] range, *no* mean/std
      normalization (unlike the ImageNet/SigLIP normalization resolved from TIMM data configs).
    - Returns the encoder's final-layer `patch_tokens` (CLS/register tokens already stripped) rather than the
      second-to-last layer features used by `TimmViTBackbone`.
"""

from functools import partial
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import _module_wrap_policy
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize, ToTensor
from transformers import AutoModel

from prismatic.models.backbones.vision.base_vision import ImageTransform, LetterboxPad, VisionBackbone

# Registry =>> Supported TIPSv2 Vision Backbones (from Hugging Face Hub) =>> Note:: Patch = 14, native 448px
TIPSV2_VISION_BACKBONES: Dict[str, str] = {
    "tipsv2-vit-b14": "google/tipsv2-b14",
    "tipsv2-vit-l14": "google/tipsv2-l14",
}

# Vision tower embedding dimension per backbone ID (per HF configs: ViT-B/14 =>> 768, ViT-L/14 =>> 1024)
TIPSV2_EMBED_DIMS: Dict[str, int] = {
    "tipsv2-vit-b14": 768,
    "tipsv2-vit-l14": 1024,
}

TIPSV2_PATCH_SIZE: int = 14


class TIPSv2ViTBackbone(VisionBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 448) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)
        self.hf_path: str = TIPSV2_VISION_BACKBONES[vision_backbone_id]
        self.dtype = torch.bfloat16

        # Initialize Featurizer (ViT) by downloading from HF Hub if necessary; TIPSv2 ships as Hub remote code
        self.featurizer: nn.Module = AutoModel.from_pretrained(self.hf_path, trust_remote_code=True)
        self.featurizer.eval()

        # Build Image Transform =>> [0, 1] range, *no* normalization (per HF model card)
        resize = Resize((self.default_image_size, self.default_image_size), interpolation=InterpolationMode.BICUBIC)
        if self.image_resize_strategy == "resize-naive":
            self.image_transform = Compose([resize, ToTensor()])

        elif self.image_resize_strategy == "resize-crop":
            self.image_transform = Compose(
                [
                    Resize(self.default_image_size, interpolation=InterpolationMode.BICUBIC),
                    CenterCrop(self.default_image_size),
                    ToTensor(),
                ]
            )

        elif self.image_resize_strategy == "letterbox":
            # No normalization =>> padding fill value is 0 (rescaled normalization mean convention from TIMM)
            self.image_transform = Compose([LetterboxPad((0, 0, 0)), resize, ToTensor()])

        else:
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a simple FSDP policy that wraps the _entire_ featurizer (remote-code block types not known here)."""
        return partial(_module_wrap_policy, module_classes={self.featurizer.__class__})

    def get_image_transform(self) -> ImageTransform:
        return self.image_transform

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Runs transformed image/pixel tensor through vision backbone, returning _all_ patch features."""
        image_features = self.featurizer.encode_image(pixel_values)
        patch_tokens = getattr(image_features, "patch_tokens", None)
        if patch_tokens is None:
            raise RuntimeError(
                f"TIPSv2 `encode_image` output for `{self.hf_path}` is missing `patch_tokens`; "
                "check the HF model revision for API changes!"
            )
        return patch_tokens

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return (3, self.default_image_size, self.default_image_size)

    @property
    def embed_dim(self) -> int:
        return TIPSV2_EMBED_DIMS[self.identifier]

    @property
    def num_patches(self) -> int:
        return (self.default_image_size // TIPSV2_PATCH_SIZE) ** 2

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return self.dtype
