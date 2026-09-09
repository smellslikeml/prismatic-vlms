"""
optimization.py

Optimizer construction utilities for Prismatic training strategies.

Alongside the default full-rank AdamW optimizer, this module provides a memory-efficient
`GaLoreAdamW` optimizer implementing Gradient Low-Rank Projection (GaLore), from:

    GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection
    (Zhao et al., 2024 -- https://arxiv.org/abs/2403.03507)

Unlike LoRA (which constrains the *weights* to a low-rank adapter), GaLore trains full-rank weights
while storing Adam's first/second-moment statistics in a low-rank subspace of each 2D weight's
gradient. The subspace is refreshed every `update_proj_gap` steps via a truncated SVD of the current
gradient, shrinking optimizer-state memory from O(m*n) to O(r*n) per matrix (r << min(m, n)).

Only 2D parameters (linear / attention projection weights) whose smaller dimension exceeds the
projection rank are projected; biases, norms, and other <=1D tensors fall back to standard AdamW.
Under fully-sharded (FSDP) training, parameters are flattened to 1D shards, so projection naturally
falls back to AdamW there -- GaLore's memory savings are realized under DDP / unsharded parameters.
Sharding-aware projection is intentionally left to a downstream change.
"""

from typing import Iterable, Tuple, Union

import torch
from torch.optim import AdamW, Optimizer

# Param groups accepted by the optimizers: either a flat iterable of tensors or a list of group dicts
ParamsT = Union[Iterable[torch.Tensor], Iterable[dict]]

OPTIMIZER_TYPES = ("adamw", "galore-adamw")


class GaLoreProjector:
    """Maintains the low-rank projection subspace for a single 2D parameter's gradient.

    Follows the "std" projection variant of the reference GaLore implementation: the projector picks
    the orientation that yields the smaller low-rank gradient, computes an orthonormal basis via
    truncated SVD, and refreshes it every `update_proj_gap` optimizer steps.
    """

    def __init__(self, rank: int, update_proj_gap: int, scale: float) -> None:
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None

    def project(self, full_rank_grad: torch.Tensor, step: int) -> torch.Tensor:
        if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
            if self.ortho_matrix is None or step % self.update_proj_gap == 0:
                self.ortho_matrix = self._orthogonal_basis(full_rank_grad, side="right")
            return torch.matmul(full_rank_grad, self.ortho_matrix.t())

        if self.ortho_matrix is None or step % self.update_proj_gap == 0:
            self.ortho_matrix = self._orthogonal_basis(full_rank_grad, side="left")
        return torch.matmul(self.ortho_matrix.t(), full_rank_grad)

    def project_back(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
        if self.ortho_matrix.shape[0] == low_rank_grad.shape[1]:
            # `right` projector: ortho is [rank, n]; low-rank grad is [m, rank]
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        else:
            # `left` projector: ortho is [m, rank]; low-rank grad is [rank, n]
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        return full_rank_grad * self.scale

    def _orthogonal_basis(self, grad: torch.Tensor, side: str) -> torch.Tensor:
        # SVD is only supported in float32; cast/restore around the decomposition as needed
        original_dtype = grad.dtype
        matrix = grad if original_dtype == torch.float32 else grad.float()
        u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
        basis = vh[: self.rank, :] if side == "right" else u[:, : self.rank]
        return basis.to(dtype=original_dtype, device=grad.device)


class GaLoreAdamW(Optimizer):
    """AdamW with Gradient Low-Rank Projection for eligible 2D parameters.

    A drop-in replacement for `torch.optim.AdamW` (same `step()` contract). Eligible 2D weights have
    their Adam moments maintained in an `r`-dimensional subspace; every other parameter uses the
    standard decoupled-weight-decay AdamW update, so behavior on biases/norms is unchanged.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        rank: int = 128,
        update_proj_gap: int = 200,
        scale: float = 0.25,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if rank < 1:
            raise ValueError(f"Invalid GaLore rank: {rank}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "rank": rank,
            "update_proj_gap": update_proj_gap,
            "scale": scale,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            rank = group["rank"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("GaLoreAdamW does not support sparse gradients")

                state = self.state[p]
                use_galore = grad.dim() == 2 and min(grad.shape) > rank
                if len(state) == 0:
                    state["step"] = 0
                    if use_galore:
                        state["projector"] = GaLoreProjector(rank, group["update_proj_gap"], group["scale"])

                if use_galore:
                    grad = state["projector"].project(grad, state["step"])

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                step_size = group["lr"] * (bias_correction2**0.5) / bias_correction1

                update = exp_avg / denom
                if use_galore:
                    update = state["projector"].project_back(update)

                p.add_(update, alpha=-step_size)

                # Decoupled (AdamW-style) weight decay applied to the full-rank parameter
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])

        return loss


def build_optimizer(
    params: ParamsT,
    optimizer_type: str = "adamw",
    *,
    lr: float,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    galore_rank: int = 128,
    galore_update_proj_gap: int = 200,
    galore_scale: float = 0.25,
) -> Optimizer:
    """Construct the training optimizer for a strategy call site.

    `optimizer_type="adamw"` (default) returns a stock `torch.optim.AdamW`, preserving existing
    behavior exactly. `optimizer_type="galore-adamw"` returns the memory-efficient `GaLoreAdamW`.
    Accepts either a flat parameter iterable or pre-built parameter-group dicts (weight-decay groups
    are respected in both branches).
    """
    optimizer_type = optimizer_type.lower()
    if optimizer_type == "adamw":
        return AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if optimizer_type == "galore-adamw":
        return GaLoreAdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rank=galore_rank,
            update_proj_gap=galore_update_proj_gap,
            scale=galore_scale,
        )
    raise ValueError(f"Optimizer type `{optimizer_type}` is not supported (expected one of {OPTIMIZER_TYPES})!")
