"""
visual_token_reduction.py

Training-free visual-token reduction for the post-projector token stream.

Adapted from the *Extract* stage of PACE ("PACE: A Unified Condense-and-Extract
Paradigm for Fast VLM Inference", https://github.com/jjL357/PACE). PACE observes
that under a strict visual-token budget, a good selector must *jointly* preserve
(a) the holistic visual context and (b) the fine-grained, task-critical detail --
dropping either one degrades downstream reasoning.

This module implements that core Extract-stage mechanism at the `[bsz, N, dim]`
token stream leaving `self.projector` (just before the LLM concat):

  * fine-grained detail  -> keep the top-k most *salient* tokens, where salience
    is a parameter-free, encoder-internal proxy (each token's L2 distance from the
    per-image mean token; distinctive tokens sit far from the mean).
  * holistic context     -> optionally reserve one slot for a mean-pooled context
    token that summarizes every input token, so global layout survives even when
    the retained set is small.

Two auxiliary components of PACE are intentionally *not* reproduced here, since
they live on different repo surfaces than this call site:

  * The DDAE's fusion with *LLM* semantic-attention signals is replaced by the
    encoder-internal salience proxy above (no second forward pass / attention
    hooks required -- this stays training-free and single-pass).
  * The *Condense* stage (Adaptive Pixel Compressor, pixel-space downsampling
    ahead of the vision encoder) is out of scope; it belongs on the vision
    backbone / image-transform surface, not the projected-token stream.

The reducer is disabled by default and holds no learned parameters, so it never
affects training runs unless explicitly enabled for inference.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def token_salience(patch_embeddings: torch.Tensor) -> torch.Tensor:
    """Parameter-free per-token salience :: [bsz, N, dim] --> [bsz, N].

    Salience is the L2 distance of each token from its per-image mean token. Tokens
    that deviate strongly from the mean carry distinctive (fine-grained) information,
    whereas near-mean tokens are redundant with the pooled context representation.
    """
    mean_token = patch_embeddings.mean(dim=1, keepdim=True)
    return torch.linalg.vector_norm(patch_embeddings - mean_token, dim=-1)


class VisualTokenReducer(nn.Module):
    """Retain a budgeted subset of projected visual tokens (PACE Extract stage).

    Args:
        retention_ratio: Fraction of the incoming tokens to keep, in ``(0, 1]``.
            ``1.0`` is a pass-through. The kept count is ``ceil(retention_ratio * N)``.
        keep_context: If ``True``, reserve one of the retained slots for a
            mean-pooled context token summarizing all input tokens (holistic
            context); the remaining slots hold the most salient tokens.
        min_tokens: Floor on the number of retained tokens, so aggressive ratios on
            small token streams still leave a usable set.
    """

    def __init__(self, retention_ratio: float, keep_context: bool = True, min_tokens: int = 1) -> None:
        super().__init__()
        if not 0.0 < retention_ratio <= 1.0:
            raise ValueError(f"`retention_ratio` must be in (0, 1], got {retention_ratio}")
        if min_tokens < 1:
            raise ValueError(f"`min_tokens` must be >= 1, got {min_tokens}")
        self.retention_ratio = retention_ratio
        self.keep_context = keep_context
        self.min_tokens = min_tokens

    def budget(self, num_tokens: int) -> int:
        """Number of tokens to retain for an input stream of length ``num_tokens``."""
        keep = int(math.ceil(self.retention_ratio * num_tokens))
        return max(self.min_tokens, min(num_tokens, keep))

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """Reduce ``[bsz, N, dim] -> [bsz, k, dim]`` (k == ``budget(N)``).

        Downstream code in ``PrismaticVLM.forward`` derives the visual attention
        mask and label padding purely from ``patch_embeddings.shape[1]``, so shrinking
        the token axis here transparently propagates to those tensors.
        """
        bsz, num_tokens, _ = patch_embeddings.shape
        keep = self.budget(num_tokens)
        if keep >= num_tokens:
            return patch_embeddings

        n_context = 1 if self.keep_context else 0
        n_detail = keep - n_context

        salience = token_salience(patch_embeddings)
        # Top-`n_detail` salient token indices, then restored to ascending positional order
        # so the retained tokens keep their relative image layout.
        topk_idx = salience.topk(n_detail, dim=1).indices
        topk_idx, _ = topk_idx.sort(dim=1)
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, patch_embeddings.shape[-1])
        detail_tokens = torch.gather(patch_embeddings, dim=1, index=gather_idx)

        if n_context == 0:
            return detail_tokens

        context_token = patch_embeddings.mean(dim=1, keepdim=True)
        return torch.cat([context_token, detail_tokens], dim=1)


def build_visual_token_reducer(
    retention_ratio: Optional[float], keep_context: bool = True
) -> Optional[VisualTokenReducer]:
    """Construct a reducer, or ``None`` when ``retention_ratio`` is unset / a no-op."""
    if retention_ratio is None or retention_ratio >= 1.0:
        return None
    return VisualTokenReducer(retention_ratio, keep_context=keep_context)
