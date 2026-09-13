"""
resampler.py

Query-based visual resampler connector for Prismatic VLMs. Compresses a variable-length sequence of vision patch
features into a small, *fixed* set of LLM-dimension tokens via cross-attention with a bank of learnable query
embeddings. Slots into the existing projector call site (an alternative to `LinearProjector` / `MLPProjector`), keeping
the same signature :: [..., num_patches, vision_dim] --> [..., num_queries, llm_dim].

Adapted from "Towards Efficient Visual-Language Alignment of the Q-Former for Visual Reasoning Tasks"
(Kim et al., 2024; https://arxiv.org/abs/2410.09489). We keep the Q-Former's *core* mechanism at full fidelity --
learnable query tokens that cross-attend to visual patch features, refined by query self-attention and a feed-forward
block. We deliberately drop the paper's auxiliary components that do not fit Prismatic's vision-only projector contract
and align-stage training loop:
    - the BERT text encoder / text-conditioned queries (the projector here only sees patch features), and
    - the ITC / ITM / ITG contrastive pretraining objectives (this connector is trained end-to-end with the
      language-modeling loss, exactly like the existing MLP projectors).

The efficiency angle the paper motivates is preserved: a few hundred patch tokens are resampled down to `num_queries`
(default 64), shortening the sequence handed to the LLM and keeping the connector compact (default `depth=2`).
"""

import torch
import torch.nn as nn


class ResamplerBlock(nn.Module):
    """One Q-Former block: query self-attention -> query->patch cross-attention -> feed-forward (all pre-norm)."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.cross_attn_q_norm = nn.LayerNorm(embed_dim)
        self.cross_attn_kv_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.ffn_norm = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim))

    def forward(self, queries: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        # Query self-attention =>> lets the query bank coordinate before reading the image
        normed = self.self_attn_norm(queries)
        queries = queries + self.self_attn(normed, normed, normed, need_weights=False)[0]

        # Query -> patch cross-attention =>> the resampling step (queries pull information from patch features)
        q = self.cross_attn_q_norm(queries)
        kv = self.cross_attn_kv_norm(patches)
        queries = queries + self.cross_attn(q, kv, kv, need_weights=False)[0]

        # Position-wise feed-forward
        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries


class QueryResampler(nn.Module):
    """Q-Former-style resampler projector: [..., num_patches, vision_dim] --> [..., num_queries, llm_dim]."""

    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        num_queries: int = 64,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if llm_dim % num_heads != 0:
            raise ValueError(f"QueryResampler requires `llm_dim` ({llm_dim}) divisible by `num_heads` ({num_heads})!")

        self.num_queries = num_queries

        # Project patch features into the resampler working dim (== llm_dim) so cross-attention shares one width
        self.patch_proj = nn.Linear(vision_dim, llm_dim, bias=True)

        # Learnable query bank =>> the fixed-size output token set
        self.queries = nn.Parameter(torch.empty(num_queries, llm_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.blocks = nn.ModuleList(
            [ResamplerBlock(llm_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)]
        )
        self.out_norm = nn.LayerNorm(llm_dim)

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        patches = self.patch_proj(img_patches)
        queries = self.queries.unsqueeze(0).expand(patches.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, patches)
        return self.out_norm(queries)
