from __future__ import annotations

import math

import torch
from torch import nn


class _ATanSpikeFn(torch.autograd.Function):
    """Binary spike function with a fixed arctangent surrogate backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: torch.Tensor):
        """Return thresholded spikes and save tensors for gradient computation."""
        if x.requires_grad or alpha.requires_grad:
            ctx.save_for_backward(x, alpha)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Propagate gradients through the arctangent surrogate."""
        x, alpha = ctx.saved_tensors
        alpha = alpha.clamp_min(1e-4)
        z = (math.pi * alpha * x) / 2.0
        grad_x = grad_output * (alpha / (2.0 * (1.0 + z.pow(2))))
        grad_alpha = None
        if ctx.needs_input_grad[1]:
            grad_alpha = torch.zeros_like(alpha)
        return grad_x, grad_alpha


class _FastSigmoidSpikeFn(torch.autograd.Function):
    """Binary spike function with a fast-sigmoid surrogate gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: torch.Tensor):
        """Return thresholded spikes and save tensors for gradient computation."""
        if x.requires_grad or alpha.requires_grad:
            ctx.save_for_backward(x, alpha)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Propagate gradients through the fast-sigmoid surrogate."""
        x, alpha = ctx.saved_tensors
        alpha = alpha.clamp_min(1e-4)
        denom = 1.0 + alpha * x.abs()
        grad_x = grad_output / denom.pow(2)
        grad_alpha = None
        if ctx.needs_input_grad[1]:
            grad_alpha = (grad_output * (-2.0 * x.abs()) / denom.pow(3)).sum().reshape_as(alpha)
        return grad_x, grad_alpha


class SurrogateSpike(nn.Module):
    """Dispatch between supported spike surrogate functions and alpha policies."""

    def __init__(
        self,
        surrogate_type: str = "atan",
        surrogate_alpha: float = 2.0,
        surrogate_alpha_min: float = 0.5,
        surrogate_alpha_max: float = 10.0,
    ):
        """Store the configured surrogate type and its alpha parameterization."""
        super().__init__()
        self.surrogate_type = surrogate_type
        self.surrogate_alpha_min = surrogate_alpha_min
        self.surrogate_alpha_max = surrogate_alpha_max

        if surrogate_type == "learnable_fast_sigmoid":
            init_alpha = min(max(surrogate_alpha, surrogate_alpha_min), surrogate_alpha_max)
            ratio = (init_alpha - surrogate_alpha_min) / max(surrogate_alpha_max - surrogate_alpha_min, 1e-6)
            ratio = min(max(ratio, 1e-4), 1.0 - 1e-4)
            self.alpha_logit = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32))
            self.register_buffer("fixed_alpha", torch.tensor(float(init_alpha), dtype=torch.float32))
        else:
            self.register_buffer("fixed_alpha", torch.tensor(float(surrogate_alpha), dtype=torch.float32))
            self.alpha_logit = None

    @property
    def alpha(self) -> torch.Tensor:
        """Return the effective surrogate sharpness parameter."""
        if self.alpha_logit is None:
            return self.fixed_alpha
        scale = self.surrogate_alpha_max - self.surrogate_alpha_min
        return self.surrogate_alpha_min + scale * torch.sigmoid(self.alpha_logit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the configured surrogate spike function to the input tensor."""
        alpha = self.alpha.to(dtype=x.dtype, device=x.device)
        if self.surrogate_type == "atan":
            return _ATanSpikeFn.apply(x, alpha)
        if self.surrogate_type in {"fast_sigmoid", "learnable_fast_sigmoid"}:
            return _FastSigmoidSpikeFn.apply(x, alpha)
        raise ValueError(f"Unsupported surrogate_type={self.surrogate_type!r}")

    def extra_repr(self) -> str:
        """Render the surrogate type and current alpha for module reprs."""
        alpha = float(self.alpha.detach().cpu())
        return f"type={self.surrogate_type}, alpha={alpha:.4f}"
