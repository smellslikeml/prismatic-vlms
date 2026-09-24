"""
token_pruning.py

Sample-adaptive vision-token pruning for Prismatic VLMs.

Adapted from "Beyond One-Size-Fits-All: Sample-Adaptive Strategy Routing for Vision Token Pruning in MLLMs"
(VIP-Router, arXiv:2609.10346). The paper's core insight is that no single fixed pruning strategy is best for
every image: alternative strategies are superior on a substantial fraction of individual samples, so a lightweight
router should *select the pruning strategy per input* at a specified reduction level, while retaining full-token
inference as an option when pruning is predicted to be unfavorable.

This module keeps that mechanism at full fidelity:
    * a small pool of training-free candidate pruning strategies operating on the projected patch tensor
      ([bsz, num_patches, d] -> [bsz, keep_k, d]),
    * per-sample routing among those strategies conditioned on low-cost visual features, and
    * a full-token retention gate that skips pruning when the batch looks information-dense.

What is intentionally substituted for this target (Mode 2, adapted port):
    * The paper's *learned* router (an MLP adding ~0.017% trainable params, conditioned on visual **and** textual
      features) is replaced by a parameter-free proxy that routes on visual token statistics (redundancy vs. the
      mean token and per-token norm dispersion). This keeps the integration training-free and plug-and-play — no
      new weights to checkpoint — at the cost of the learned router's fitted decision boundary.
    * The paper's VTC-Bench evaluation suite is out of scope here; evaluation belongs in a downstream PR.
The forward-path token-selection contract ([bsz, N, d] -> [bsz, K, d]) is faithful to the paper.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Candidate strategy names, ordered to match the router's integer decisions.
STRATEGIES = ("uniform", "salience", "diversity")


def _uniform_indices(num_patches: int, keep_k: int, batch_size: int, device: torch.device) -> torch.Tensor:
    """Evenly spaced (stride) token indices — preserves spatial coverage. Shared across the batch."""
    idx = (torch.arange(keep_k, device=device) * num_patches) // keep_k
    return idx.unsqueeze(0).expand(batch_size, -1)


def _salience_indices(norms: torch.Tensor, keep_k: int) -> torch.Tensor:
    """Keep the `keep_k` highest-L2-norm tokens per sample — retains high-energy (salient) patches."""
    idx = norms.topk(keep_k, dim=1).indices
    return idx.sort(dim=1).values


def _diversity_indices(sim_to_mean: torch.Tensor, keep_k: int) -> torch.Tensor:
    """Keep the `keep_k` tokens *least* similar to the sample's mean token — de-duplicates redundant background."""
    idx = (-sim_to_mean).topk(keep_k, dim=1).indices
    return idx.sort(dim=1).values


class VisionTokenRouter(nn.Module):
    """Parameter-free, per-sample router that prunes projected vision tokens with an adaptively selected strategy.

    Contract: ``forward(patch_embeddings: [bsz, num_patches, d]) -> [bsz, keep_k, d]`` where ``keep_k`` is fixed for
    the whole batch at a given reduction level (so downstream mask/label/padding paths that key on
    ``embeddings.shape[1]`` need no changes). When the batch is predicted to be information-dense, the router returns
    the input unchanged (full-token retention).
    """

    def __init__(
        self,
        reduction_ratio: float = 0.5,
        min_tokens: int = 1,
        keep_all_redundancy_threshold: float = 0.15,
        high_redundancy_threshold: float = 0.6,
        dispersion_threshold: float = 0.5,
        strategies: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= reduction_ratio < 1.0:
            raise ValueError(f"`reduction_ratio` must be in [0, 1); got {reduction_ratio}")
        self.reduction_ratio = reduction_ratio
        self.min_tokens = min_tokens
        self.keep_all_redundancy_threshold = keep_all_redundancy_threshold
        self.high_redundancy_threshold = high_redundancy_threshold
        self.dispersion_threshold = dispersion_threshold
        self.strategies = tuple(strategies) if strategies is not None else STRATEGIES

        # Inspection hook: names of the strategy chosen per sample on the last forward (or "full" if retained).
        self.last_routing: List[str] = []

    def _keep_count(self, num_patches: int) -> int:
        keep_k = int(round(num_patches * (1.0 - self.reduction_ratio)))
        return max(self.min_tokens, min(num_patches, keep_k))

    def route(self, redundancy: torch.Tensor, dispersion: torch.Tensor) -> torch.Tensor:
        """Map low-cost per-sample visual features to a strategy index in ``[0, len(strategies))``.

        High norm dispersion => a few tokens carry most energy => keep the peaks (``salience``).
        Otherwise, very redundant samples => de-duplicate (``diversity``); the rest fall back to ``uniform``.
        """
        choice = torch.zeros_like(redundancy, dtype=torch.long)  # default: uniform (index 0)
        choice = torch.where(redundancy >= self.high_redundancy_threshold, torch.full_like(choice, 2), choice)
        choice = torch.where(dispersion >= self.dispersion_threshold, torch.full_like(choice, 1), choice)
        return choice.clamp_(max=len(self.strategies) - 1)

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        bsz, num_patches, dim = patch_embeddings.shape
        keep_k = self._keep_count(num_patches)
        if keep_k >= num_patches:
            self.last_routing = ["full"] * bsz
            return patch_embeddings

        # Low-cost visual features (fp32 for stable statistics regardless of autocast dtype).
        feats = patch_embeddings.detach().float()
        mean_token = feats.mean(dim=1, keepdim=True)
        sim_to_mean = F.cosine_similarity(feats, mean_token, dim=-1)  # [bsz, num_patches]
        redundancy = sim_to_mean.mean(dim=1)  # [bsz]
        norms = feats.norm(dim=-1)  # [bsz, num_patches]
        dispersion = norms.std(dim=1) / (norms.mean(dim=1) + 1e-6)  # [bsz]

        # Full-token retention gate: information-dense batch (low redundancy) => pruning predicted unfavorable.
        if redundancy.mean() < self.keep_all_redundancy_threshold:
            self.last_routing = ["full"] * bsz
            return patch_embeddings

        # Per-sample strategy routing => gather each sample's chosen keep-indices.
        choice = self.route(redundancy, dispersion)
        candidate_indices = torch.stack(
            [
                _uniform_indices(num_patches, keep_k, bsz, patch_embeddings.device),
                _salience_indices(norms, keep_k),
                _diversity_indices(sim_to_mean, keep_k),
            ],
            dim=0,
        )  # [num_strategies, bsz, keep_k]
        keep_indices = candidate_indices[choice, torch.arange(bsz, device=patch_embeddings.device)]  # [bsz, keep_k]

        self.last_routing = [self.strategies[int(c)] for c in choice]
        gather_index = keep_indices.unsqueeze(-1).expand(-1, -1, dim)
        return torch.gather(patch_embeddings, 1, gather_index)
