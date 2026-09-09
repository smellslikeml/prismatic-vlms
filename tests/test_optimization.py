"""
test_optimization.py

Tests for the GaLore (Gradient Low-Rank Projection) optimizer path and its integration into the
Prismatic training strategies (`prismatic/training/strategies/{ddp,fsdp}.py`).

Reference: GaLore -- https://arxiv.org/abs/2403.03507
"""

import inspect

import torch
from torch.optim import AdamW

# `materialize` / `strategies.*` are existing (non-new) call-site modules -- importing them here
# exercises the wiring edits made to them; `optimization` is the new capability module under test.
from prismatic.training.materialize import get_train_strategy
from prismatic.training.optimization import GaLoreAdamW, build_optimizer
from prismatic.training.strategies import base_strategy, ddp, fsdp


def test_call_sites_use_build_optimizer():
    """The DDP/FSDP strategies should route optimizer construction through `build_optimizer`."""
    assert ddp.build_optimizer is build_optimizer
    assert fsdp.build_optimizer is build_optimizer

    # `AdamW` should no longer be constructed directly at either call site.
    assert "build_optimizer(" in inspect.getsource(ddp.DDPStrategy.run_setup)
    assert "build_optimizer(" in inspect.getsource(fsdp.FSDPStrategy.run_setup)


def test_optimizer_type_is_plumbed_through_strategies():
    """`optimizer_type` must be a settable knob on the base strategy and the public factory."""
    assert "optimizer_type" in inspect.signature(base_strategy.TrainingStrategy.__init__).parameters
    assert "optimizer_type" in inspect.signature(fsdp.FSDPStrategy.__init__).parameters
    assert "optimizer_type" in inspect.signature(get_train_strategy).parameters


def test_build_optimizer_default_is_plain_adamw():
    """Default path must preserve existing behavior exactly (stock AdamW)."""
    param = torch.nn.Parameter(torch.randn(8, 8))
    opt = build_optimizer([param], lr=1e-3, weight_decay=0.01)
    assert isinstance(opt, AdamW) and not isinstance(opt, GaLoreAdamW)


def test_build_optimizer_galore_selection_and_validation():
    param = torch.nn.Parameter(torch.randn(8, 8))
    opt = build_optimizer([param], "galore-adamw", lr=1e-3, galore_rank=4)
    assert isinstance(opt, GaLoreAdamW)

    try:
        build_optimizer([param], "nonexistent", lr=1e-3)
    except ValueError:
        pass
    else:
        raise AssertionError("build_optimizer should reject unknown optimizer types")


def test_galore_reduces_optimizer_state_memory():
    """Core paper claim: Adam moments for a large 2D weight live in a low-rank subspace."""
    rank = 8
    weight = torch.nn.Parameter(torch.randn(128, 128))
    opt = build_optimizer([weight], "galore-adamw", lr=1e-3, galore_rank=rank)

    weight.grad = torch.randn_like(weight)
    opt.step()

    state = opt.state[weight]
    full_rank_numel = weight.numel()
    projected_numel = state["exp_avg"].numel() + state["exp_avg_sq"].numel()
    # Two low-rank moment buffers (r x n each) must be smaller than two full-rank buffers.
    assert projected_numel < 2 * full_rank_numel
    assert min(state["exp_avg"].shape) == rank


def test_galore_falls_back_to_adamw_for_small_and_1d_params():
    """Biases / norms / sub-rank matrices should not be projected (plain AdamW moments)."""
    bias = torch.nn.Parameter(torch.randn(32))
    small = torch.nn.Parameter(torch.randn(4, 4))
    opt = build_optimizer([bias, small], "galore-adamw", lr=1e-3, galore_rank=8)

    bias.grad = torch.randn_like(bias)
    small.grad = torch.randn_like(small)
    opt.step()

    assert opt.state[bias]["exp_avg"].shape == bias.shape
    assert opt.state[small]["exp_avg"].shape == small.shape
    assert "projector" not in opt.state[bias]
    assert "projector" not in opt.state[small]


def test_galore_optimizes_a_toy_regression():
    """Sanity: the projected update actually drives a full-rank weight toward a target."""
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.zeros(64, 64))
    target = torch.randn(64, 64)
    opt = build_optimizer([weight], "galore-adamw", lr=1e-2, galore_rank=16, galore_update_proj_gap=5)

    def loss_fn():
        return ((weight - target) ** 2).mean()

    initial = loss_fn().item()
    for _ in range(50):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()

    assert loss_fn().item() < initial
