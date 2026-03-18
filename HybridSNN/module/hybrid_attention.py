"""
Addition-Only Hybrid Attention (AOHA) module for HybridSNN.

Avoids the SSA bug (calling utils.reset on non-existent attributes) by
building a fresh attention module that never references snntorch utils.reset.
"""

import math
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate as snn_surrogate


# Inline ATan surrogate to avoid circular imports through SeqSNN.network.__init__
import torch as _torch
import math as _math


@_torch.jit.script
def _atan_backward(grad_output: _torch.Tensor, x: _torch.Tensor, alpha: float):
    return alpha / 2 / (1 + (_math.pi / 2 * alpha * x).pow_(2)) * grad_output, None


class ATan(_torch.autograd.Function):
    """ATan surrogate gradient (mirrors SeqSNN.network.snn.surrogate.atan)."""

    @staticmethod
    def forward(ctx, x, alpha=2.0):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return (x >= 0).to(x)

    @staticmethod
    def backward(ctx, grad_output):
        return _atan_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class TernaryNode(nn.Module):
    """Ternary spiking neuron outputting {-1, 0, +1}.

    Uses ATan surrogate gradient. No hardcoded dimensions.
    forward(v) computes:
        pos = H(v - threshold)
        neg = H(-v - threshold)
        out = pos - neg
    """

    def __init__(self, threshold: float = 1.0, alpha: float = 2.0):
        super().__init__()
        self.threshold = threshold
        self.alpha = alpha

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        pos = ATan.apply(v - self.threshold, self.alpha)
        neg = ATan.apply(-v - self.threshold, self.alpha)
        return pos - neg


def _bn1d_on_last(bn: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
    """Apply BatchNorm1d over the last dimension of an arbitrary-rank tensor.

    BatchNorm1d expects (N, C) or (N, C, L). We flatten all leading dims,
    transpose so feature dim is second, apply BN, then restore shape.
    """
    *leading, D = x.shape
    x_flat = x.reshape(-1, D)          # (N, D)
    x_bn = bn(x_flat)                   # (N, D)
    return x_bn.reshape(*leading, D)


class AOHA(nn.Module):
    """Addition-Only Hybrid Attention.

    Q: binary {0, 1}  via snntorch.Leaky
    K: continuous (high precision) via ReLU
    V: ternary {-1, 0, +1} via TernaryNode

    Input/output shape: (B, T, L, D)
    """

    def __init__(self, dim: int, heads: int = 8, qk_scale: float = None, qkv_bias: bool = False):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qk_scale = qk_scale or math.sqrt(self.head_dim) ** -1

        # Q path: Linear -> BN -> Leaky (binary spikes)
        self.q_m = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = snn.Leaky(
            beta=0.5,
            spike_grad=snn_surrogate.atan(alpha=2.0),
            init_hidden=True,
            output=True,
        )

        # K path: Linear -> BN -> ReLU (continuous)
        self.k_m = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_relu = nn.ReLU()

        # V path: Linear -> BN -> TernaryNode ({-1, 0, +1})
        self.v_m = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_ternary = TernaryNode()

        # Output path: Linear -> BN -> Leaky
        self.out_m = nn.Linear(dim, dim)
        self.out_bn = nn.BatchNorm1d(dim)
        self.out_lif = snn.Leaky(
            beta=0.5,
            spike_grad=snn_surrogate.atan(alpha=2.0),
            init_hidden=True,
            output=True,
        )

        # Track firing rate of Q spikes for spike-rate regularization
        self.firing_rate: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, L, D)
        Returns:
            out: (B, T, L, D)
        """
        B, T, L, D = x.shape
        x_flat = x.flatten(0, 1)  # (BT, L, D)

        # --- Q path ---
        q = self.q_m(x_flat)                        # (BT, L, D)
        q = _bn1d_on_last(self.q_bn, q)             # (BT, L, D)
        q = q.reshape(B, T, L, D)
        # snntorch Leaky processes time-unrolled input; we iterate over T
        q_spk = []
        for t in range(T):
            spk, _ = self.q_lif(q[:, t, :, :].reshape(B * L, D))
            q_spk.append(spk.reshape(B, L, D))
        q = torch.stack(q_spk, dim=1)              # (B, T, L, D)
        self.firing_rate = q.mean().detach()

        # --- K path ---
        k = self.k_m(x_flat)                        # (BT, L, D)
        k = _bn1d_on_last(self.k_bn, k)
        k = self.k_relu(k)
        k = k.reshape(B, T, L, D)

        # --- V path ---
        v = self.v_m(x_flat)                        # (BT, L, D)
        v = _bn1d_on_last(self.v_bn, v)
        v = v.reshape(B, T, L, D)
        v = self.v_ternary(v)

        # --- Multi-head attention ---
        # Reshape to (B, T, heads, L, head_dim)
        q = q.reshape(B, T, L, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = k.reshape(B, T, L, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, T, L, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)

        # Addition-only: Q@K^T (integer/binary ops dominate)
        attn = (q @ k.transpose(-2, -1)) * self.qk_scale  # (B, T, heads, L, L)
        out = attn @ v                                      # (B, T, heads, L, head_dim)

        # Merge heads
        out = out.permute(0, 1, 3, 2, 4).reshape(B, T, L, D)  # (B, T, L, D)
        out_flat = out.flatten(0, 1)                             # (BT, L, D)

        # --- Output path ---
        out_flat = self.out_m(out_flat)
        out_flat = _bn1d_on_last(self.out_bn, out_flat)
        out_flat = out_flat.reshape(B, T, L, D)

        out_spk = []
        for t in range(T):
            spk, _ = self.out_lif(out_flat[:, t, :, :].reshape(B * L, D))
            out_spk.append(spk.reshape(B, L, D))
        out = torch.stack(out_spk, dim=1)  # (B, T, L, D)

        return out


class HybridBlock(nn.Module):
    """Single transformer block: AOHA residual + spiking MLP residual.

    Input/output shape: (B, T, L, D)
    """

    def __init__(self, dim: int, d_ff: int, heads: int = 8, qk_scale: float = None):
        super().__init__()
        self.attn = AOHA(dim=dim, heads=heads, qk_scale=qk_scale)

        # Spiking MLP: Linear -> BN -> Leaky -> Linear -> BN -> Leaky
        self.mlp_fc1 = nn.Linear(dim, d_ff)
        self.mlp_bn1 = nn.BatchNorm1d(d_ff)
        self.mlp_lif1 = snn.Leaky(
            beta=0.5,
            spike_grad=snn_surrogate.atan(alpha=2.0),
            init_hidden=True,
            output=True,
        )
        self.mlp_fc2 = nn.Linear(d_ff, dim)
        self.mlp_bn2 = nn.BatchNorm1d(dim)
        self.mlp_lif2 = snn.Leaky(
            beta=0.5,
            spike_grad=snn_surrogate.atan(alpha=2.0),
            init_hidden=True,
            output=True,
        )

    def _mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, L, D = x.shape
        x_flat = x.flatten(0, 1)           # (BT, L, D)

        h = self.mlp_fc1(x_flat)           # (BT, L, d_ff)
        h = _bn1d_on_last(self.mlp_bn1, h)
        h = h.reshape(B, T, L, -1)

        h_spk = []
        for t in range(T):
            spk, _ = self.mlp_lif1(h[:, t, :, :].reshape(B * L, -1))
            h_spk.append(spk.reshape(B, L, -1))
        h = torch.stack(h_spk, dim=1)     # (B, T, L, d_ff)

        h_flat = h.flatten(0, 1)
        h = self.mlp_fc2(h_flat)          # (BT, L, D)
        h = _bn1d_on_last(self.mlp_bn2, h)
        h = h.reshape(B, T, L, D)

        h_spk = []
        for t in range(T):
            spk, _ = self.mlp_lif2(h[:, t, :, :].reshape(B * L, D))
            h_spk.append(spk.reshape(B, L, D))
        h = torch.stack(h_spk, dim=1)     # (B, T, L, D)

        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self._mlp_forward(x)
        return x

    @property
    def firing_rate(self) -> torch.Tensor:
        return self.attn.firing_rate
