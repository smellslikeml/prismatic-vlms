"""
fusion_projectors.py

Projection module for hybrid (mixture-of-encoders) vision backbones that performs *post-adaptation* fusion of visual
tokens: each vision encoder's patch features are adapted to the LLM embedding space by an **independent** MLP, and the
resulting per-encoder token sequences are fused along the *sequence* dimension (interleaved) rather than concatenated
along the *channel* dimension before a single shared projector.

This is the core fusion mechanism from LEO (Azadani et al., "LEO: Boosting Mixture of Vision Encoders for Multimodal
Large Language Models", https://arxiv.org/abs/2501.06986). Prismatic's default `FusedMLPProjector` implements the
opposite ("pre-adaptation") strategy for the DINO+SigLIP path: it channel-concatenates the two encoders' features and
maps them through one MLP. LEO shows that giving each encoder its own adapter and fusing afterwards lets the LLM attend
to encoder-specific tokens directly.

We keep LEO's fusion mechanism at full fidelity; we intentionally do *not* port LEO's auxiliary components (pixel
un-shuffle token compression and dynamic high-resolution image tiling), which are orthogonal to the fusion question and
can be layered on independently.
"""

from typing import Sequence

import torch
import torch.nn as nn


class PostAdaptationFusionProjector(nn.Module):
    """Independent per-encoder GELU-MLP adapters + sequence-level (interleaved) fusion.

    Expects the *channel-concatenated* features produced by a mixture-of-encoders vision backbone (e.g. the
    ``torch.cat([dino_patches, siglip_patches], dim=2)`` returned by ``DinoSigLIPViTBackbone``). The input is split back
    into its per-encoder components along the channel dimension, each component is projected to ``llm_dim`` by its own
    MLP, and the projected token sequences are interleaved into a single ``[bsz, num_encoders * num_patches, llm_dim]``
    sequence — the boundary the downstream LLM already consumes.
    """

    def __init__(self, encoder_dims: Sequence[int], llm_dim: int, mlp_type: str = "gelu-mlp") -> None:
        super().__init__()
        if mlp_type != "gelu-mlp":
            raise ValueError(f"Fusion Projector with `{mlp_type = }` is not supported!")
        if len(encoder_dims) < 2:
            raise ValueError(
                f"`PostAdaptationFusionProjector` expects >= 2 encoders for post-adaptation fusion, got {encoder_dims}!"
            )

        self.encoder_dims = list(encoder_dims)
        self.llm_dim = llm_dim

        # One independent adapter per vision encoder (post-adaptation fusion => *no* shared channel projection).
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(encoder_dim, llm_dim, bias=True),
                    nn.GELU(),
                    nn.Linear(llm_dim, llm_dim, bias=True),
                )
                for encoder_dim in self.encoder_dims
            ]
        )

    def forward(self, fused_img_patches: torch.Tensor) -> torch.Tensor:
        # Split the channel-concatenated features back into their per-encoder components.
        if fused_img_patches.shape[-1] != sum(self.encoder_dims):
            raise ValueError(
                f"Expected fused feature dim {sum(self.encoder_dims)} (= sum of {self.encoder_dims}), "
                f"got {fused_img_patches.shape[-1]}!"
            )
        per_encoder_patches = torch.split(fused_img_patches, self.encoder_dims, dim=-1)

        # Adapt each encoder independently to the LLM space => list of [bsz, num_patches, llm_dim].
        projected = [projector(patches) for projector, patches in zip(self.projectors, per_encoder_patches)]

        # Post-adaptation fusion: interleave the per-encoder token sequences along the sequence dimension so that
        # spatially-aligned tokens from each encoder are adjacent => [bsz, num_encoders * num_patches, llm_dim].
        stacked = torch.stack(projected, dim=2)
        bsz, num_patches, num_encoders, llm_dim = stacked.shape
        return stacked.reshape(bsz, num_patches * num_encoders, llm_dim)
