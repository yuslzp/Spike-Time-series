from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from HybridSNN.module.neuron import build_spike_neuron


class ATan(torch.autograd.Function):
    """Binary spike function with an arctangent surrogate gradient."""

    @staticmethod
    def forward(ctx, x, alpha=2.0):
        """Threshold the input while saving tensors for the backward pass."""
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """Return the arctangent surrogate gradient for the saved activations."""
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad = alpha / (2.0 * (1.0 + ((math.pi * alpha * x) / 2.0).pow(2)))
        return grad_output * grad, None


class TernaryNode(nn.Module):
    """Emit ternary activations using positive and negative spike thresholds."""

    def __init__(self, threshold: float = 1.0, alpha: float = 2.0):
        """Store the threshold and surrogate sharpness for ternary spikes."""
        super().__init__()
        self.threshold = threshold
        self.alpha = alpha

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Convert membrane values into {-1, 0, 1} ternary activations."""
        pos = ATan.apply(v - self.threshold, self.alpha)
        neg = ATan.apply(-v - self.threshold, self.alpha)
        return pos - neg


class BinaryNode(nn.Module):
    """Emit binary activations using a single spike threshold."""

    def __init__(self, threshold: float = 0.0, alpha: float = 2.0):
        """Store the threshold and surrogate sharpness for binary spikes."""
        super().__init__()
        self.threshold = threshold
        self.alpha = alpha

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Convert membrane values into {0, 1} binary activations."""
        return ATan.apply(v - self.threshold, self.alpha)


def _bn1d_on_last(bn: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
    """Apply a `BatchNorm1d` layer over the last dimension of an arbitrary tensor."""
    *leading, dim = x.shape
    x_flat = x.reshape(-1, dim)
    x_bn = bn(x_flat)
    return x_bn.reshape(*leading, dim)


def _state_tensor_for_plot(state: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    """Select a representative state tensor for visualization."""
    for key in ("vs", "mem", "vd", "smix"):
        if key in state:
            return state[key]
    return None


def _run_spike_neuron(
    neuron: nn.Module, x: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Run a spike neuron over time-major inputs and stack its states."""
    batch, steps, length, dim = x.shape
    spikes = []
    states = []
    for t in range(steps):
        spk, state = neuron(x[:, t].reshape(batch * length, dim))
        spikes.append(spk.reshape(batch, length, dim))
        states.append({k: v.reshape(batch, length, dim) for k, v in state.items() if v.dim() > 0})
    stacked_spikes = torch.stack(spikes, dim=1)
    stacked_states: Dict[str, torch.Tensor] = {}
    if states:
        for key in states[0]:
            stacked_states[key] = torch.stack([state[key] for state in states], dim=1)
    return stacked_spikes, stacked_states


class AOHA(nn.Module):
    """Apply spiking attention with binary queries and ternary values."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        qk_scale: float = None,
        qkv_bias: bool = False,
        attention_mode: str = "aoha",
        value_mode: str = "ternary",
        neuron_type: str = "tslif",
        neuron_kwargs: Optional[dict] = None,
    ):
        """Initialize the spike-aware attention projections and readout path."""
        super().__init__()
        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"
        if attention_mode not in {"aoha", "softmax"}:
            raise ValueError(
                f"Unknown attention_mode={attention_mode!r}. Expected 'aoha' or 'softmax'."
            )
        if value_mode not in {"ternary", "binary"}:
            raise ValueError(
                f"Unknown value_mode={value_mode!r}. Expected 'ternary' or 'binary'."
            )
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qk_scale = qk_scale or math.sqrt(self.head_dim) ** -1
        self.attention_mode = attention_mode
        self.value_mode = value_mode
        self.record_mode = False
        self._viz_data: Dict[str, torch.Tensor] = {}

        self.q_m = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_neuron = build_spike_neuron(neuron_type, **(neuron_kwargs or {}))

        self.k_m = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_relu = nn.ReLU()

        self.v_m = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_ternary = TernaryNode()
        self.v_binary = BinaryNode()

        self.out_m = nn.Linear(dim, dim)
        self.out_bn = nn.BatchNorm1d(dim)
        self.out_neuron = build_spike_neuron(neuron_type, **(neuron_kwargs or {}))

        self.firing_rate = torch.tensor(0.0)
        self._firing_rate_tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multi-head attention over each simulation step and token."""
        batch, steps, length, dim = x.shape
        self._viz_data = {}
        x_flat = x.flatten(0, 1)

        q = _bn1d_on_last(self.q_bn, self.q_m(x_flat)).reshape(batch, steps, length, dim)
        q_spk, q_states = _run_spike_neuron(self.q_neuron, q)
        self._firing_rate_tensor = q_spk.mean()
        self.firing_rate = self._firing_rate_tensor.detach()

        k = self.k_relu(_bn1d_on_last(self.k_bn, self.k_m(x_flat))).reshape(batch, steps, length, dim)
        v_in = _bn1d_on_last(self.v_bn, self.v_m(x_flat)).reshape(batch, steps, length, dim)
        if self.value_mode == "ternary":
            v = self.v_ternary(v_in)
        else:
            v = self.v_binary(v_in)

        q_heads = q_spk.reshape(batch, steps, length, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k_heads = k.reshape(batch, steps, length, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v_heads = v.reshape(batch, steps, length, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)

        attn_scores = (q_heads @ k_heads.transpose(-2, -1)) * self.qk_scale
        if self.attention_mode == "softmax":
            attn = torch.softmax(attn_scores, dim=-1)
        else:
            attn = attn_scores
        out = attn @ v_heads
        out = out.permute(0, 1, 3, 2, 4).reshape(batch, steps, length, dim)
        out = _bn1d_on_last(self.out_bn, self.out_m(out.flatten(0, 1))).reshape(batch, steps, length, dim)
        out_spk, out_states = _run_spike_neuron(self.out_neuron, out)

        if self.record_mode:
            self._viz_data["q_spk"] = q_spk.detach().cpu()
            self._viz_data["attention_mode"] = self.attention_mode
            self._viz_data["value_mode"] = self.value_mode
            self._viz_data["attn_scores"] = attn_scores.detach().cpu()
            self._viz_data["attn_weights"] = attn.detach().cpu()
            q_mem = _state_tensor_for_plot(q_states)
            out_mem = _state_tensor_for_plot(out_states)
            if q_mem is not None:
                self._viz_data["q_mem"] = q_mem.detach().cpu()
            if out_mem is not None:
                self._viz_data["out_mem"] = out_mem.detach().cpu()
            for key in ("vd", "vs", "sd", "ss"):
                if key in q_states:
                    self._viz_data[f"q_{key}"] = q_states[key].detach().cpu()
                if key in out_states:
                    self._viz_data[f"out_{key}"] = out_states[key].detach().cpu()

        return out_spk


class HybridBlock(nn.Module):
    """Compose spike attention and a spike MLP with residual connections."""

    def __init__(
        self,
        dim: int,
        d_ff: int,
        heads: int = 8,
        qk_scale: float = None,
        attention_mode: str = "aoha",
        value_mode: str = "ternary",
        neuron_type: str = "tslif",
        neuron_kwargs: Optional[dict] = None,
    ):
        """Build one residual HybridSNN backbone block."""
        super().__init__()
        self.record_mode = False
        self._viz_data: Dict[str, torch.Tensor] = {}

        self.attn = AOHA(
            dim=dim,
            heads=heads,
            qk_scale=qk_scale,
            attention_mode=attention_mode,
            value_mode=value_mode,
            neuron_type=neuron_type,
            neuron_kwargs=neuron_kwargs,
        )

        self.mlp_fc1 = nn.Linear(dim, d_ff)
        self.mlp_bn1 = nn.BatchNorm1d(d_ff)
        self.mlp_neuron1 = build_spike_neuron(neuron_type, **(neuron_kwargs or {}))
        self.mlp_fc2 = nn.Linear(d_ff, dim)
        self.mlp_bn2 = nn.BatchNorm1d(dim)
        self.mlp_neuron2 = build_spike_neuron(neuron_type, **(neuron_kwargs or {}))

    def _mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the feed-forward spike MLP branch and collect diagnostics."""
        batch, steps, length, dim = x.shape
        h1 = _bn1d_on_last(self.mlp_bn1, self.mlp_fc1(x.flatten(0, 1))).reshape(batch, steps, length, -1)
        h1_spk, h1_states = _run_spike_neuron(self.mlp_neuron1, h1)

        h2 = _bn1d_on_last(self.mlp_bn2, self.mlp_fc2(h1_spk.flatten(0, 1))).reshape(batch, steps, length, dim)
        h2_spk, h2_states = _run_spike_neuron(self.mlp_neuron2, h2)

        if self.record_mode:
            h1_mem = _state_tensor_for_plot(h1_states)
            h2_mem = _state_tensor_for_plot(h2_states)
            if h1_mem is not None:
                self._viz_data["mlp_lif1_mem"] = h1_mem.detach().cpu()
            if h2_mem is not None:
                self._viz_data["mlp_lif2_mem"] = h2_mem.detach().cpu()
            for key in ("vd", "vs", "sd", "ss"):
                if key in h1_states:
                    self._viz_data[f"mlp1_{key}"] = h1_states[key].detach().cpu()
                if key in h2_states:
                    self._viz_data[f"mlp2_{key}"] = h2_states[key].detach().cpu()

        return h2_spk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the attention block followed by the spike MLP residual branch."""
        self._viz_data = {}
        self.attn.record_mode = self.record_mode
        x = x + self.attn(x)
        if self.record_mode:
            self._viz_data.update(self.attn._viz_data)
        x = x + self._mlp_forward(x)
        return x

    @property
    def firing_rate(self) -> torch.Tensor:
        """Return the attention query firing rate tracked for this block."""
        return self.attn.firing_rate
