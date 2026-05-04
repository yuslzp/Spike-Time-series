from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from HybridSNN.module.neuron import build_spike_neuron


class RepeatEncoder(nn.Module):
    """Repeat each input sequence across simulation steps and spike it."""

    def __init__(
        self,
        output_size: int,
        neuron_type: str = "tslif",
        neuron_kwargs: Optional[dict] = None,
    ):
        """Create a repeated-input encoder backed by a spike neuron module."""
        super().__init__()
        self.out_size = output_size
        self.neuron = build_spike_neuron(neuron_type, **(neuron_kwargs or {}))

    def forward(self, inputs: torch.Tensor):
        """Encode a sequence by replaying it for each configured time step."""
        inputs = inputs.unsqueeze(1).repeat(1, self.out_size, 1, 1)
        outputs = []
        for t in range(self.out_size):
            spk, _ = self.neuron(inputs[:, t])
            outputs.append(spk)
        return torch.stack(outputs, dim=1)


class ConvEncoder(nn.Module):
    """Project the input sequence with a temporal convolution before spiking."""

    def __init__(
        self,
        output_size: int,
        kernel_size: int = 3,
        neuron_type: str = "tslif",
        neuron_kwargs: Optional[dict] = None,
        encoder_dropout: float = 0.0,
    ):
        """Build the convolutional encoder and its spike neuron backend."""
        super().__init__()
        self.output_size = output_size
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=output_size,
                kernel_size=(1, kernel_size),
                stride=1,
                padding=(0, kernel_size // 2),
            ),
            nn.BatchNorm2d(output_size),
        )
        self.encoder_dropout = nn.Dropout(encoder_dropout)
        self.neuron = build_spike_neuron(neuron_type, **(neuron_kwargs or {}))

    def forward(
        self, inputs: torch.Tensor, return_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Encode inputs into spike tensors and optionally stacked neuron states."""
        inputs = inputs.permute(0, 2, 1).unsqueeze(1)
        enc = self.encoder(inputs)
        enc = self.encoder_dropout(enc)
        spikes = []
        mems = []
        for t in range(self.output_size):
            spk, state = self.neuron(enc[:, t])
            spikes.append(spk)
            mems.append(state)
        outputs = torch.stack(spikes, dim=1)
        if not return_states:
            return outputs, None
        return outputs, _stack_state_dicts(mems)


class DeltaEncoder(nn.Module):
    """Encode normalized first-order temporal differences into spikes."""

    def __init__(
        self,
        output_size: int,
        neuron_type: str = "tslif",
        neuron_kwargs: Optional[dict] = None,
        delta_scale_init: float = 5.0,
        delta_bias_init: float = 0.1,
        encoder_dropout: float = 0.0,
    ):
        """Initialize the delta-based encoder branch."""
        super().__init__()
        self.output_size = output_size
        self.norm = nn.BatchNorm2d(1)
        self.enc = nn.Linear(1, output_size)
        self.delta_scale = nn.Parameter(torch.tensor(float(delta_scale_init)))
        self.delta_bias = nn.Parameter(torch.tensor(float(delta_bias_init)))
        self.encoder_dropout = nn.Dropout(encoder_dropout)
        self.neuron = build_spike_neuron(neuron_type, **(neuron_kwargs or {}))

    def forward(
        self, inputs: torch.Tensor, return_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Convert sequence deltas into spike activations for each step."""
        delta = torch.zeros_like(inputs)
        delta[:, 1:] = inputs[:, 1:, :] - inputs[:, :-1, :]
        delta_std = delta.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
        delta = delta / delta_std
        delta = F.softplus(self.delta_scale) * delta + self.delta_bias
        delta = delta.unsqueeze(1).permute(0, 1, 3, 2)
        delta = self.norm(delta)
        delta = delta.permute(0, 2, 3, 1)
        enc = self.enc(delta).permute(0, 3, 1, 2)
        enc = self.encoder_dropout(enc)

        spikes = []
        mems = []
        for t in range(self.output_size):
            spk, state = self.neuron(enc[:, t])
            spikes.append(spk)
            mems.append(state)
        outputs = torch.stack(spikes, dim=1)
        if not return_states:
            return outputs, None
        return outputs, _stack_state_dicts(mems)


class DeltaConvEncoder(nn.Module):
    """Fuse delta and convolutional spike encoders with branch balancing."""

    def __init__(
        self,
        num_steps: int,
        kernel_size: int = 3,
        neuron_type: str = "tslif",
        neuron_kwargs: Optional[dict] = None,
        fusion_mode: str = "weighted_sum",
        branch_balance_strength: float = 1.0,
        branch_balance_clamp: float = 2.5,
        branch_balance_eps: float = 1e-4,
        delta_scale_init: float = 5.0,
        delta_bias_init: float = 0.1,
        encoder_dropout: float = 0.0,
    ):
        """Construct the paired encoder branches and fusion parameters."""
        super().__init__()
        self.num_steps = num_steps
        self.fusion_mode = fusion_mode
        self.branch_balance_strength = branch_balance_strength
        self.branch_balance_clamp = branch_balance_clamp
        self.branch_balance_eps = branch_balance_eps
        self.delta_encoder = DeltaEncoder(
            output_size=num_steps,
            neuron_type=neuron_type,
            neuron_kwargs=neuron_kwargs,
            delta_scale_init=delta_scale_init,
            delta_bias_init=delta_bias_init,
            encoder_dropout=encoder_dropout,
        )
        self.conv_encoder = ConvEncoder(
            output_size=num_steps,
            kernel_size=kernel_size,
            neuron_type=neuron_type,
            neuron_kwargs=neuron_kwargs,
            encoder_dropout=encoder_dropout,
        )
        self.branch_logits = nn.Parameter(torch.zeros(2))
        self.branch_log_scales = nn.Parameter(torch.zeros(2))

    def _balance_branch_outputs(
        self, delta_out: torch.Tensor, conv_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scale branch outputs toward matched firing rates and return balance loss."""
        rates = torch.stack([delta_out.float().mean(), conv_out.float().mean()])
        rates = torch.nan_to_num(rates, nan=0.0, posinf=1.0, neginf=0.0)
        target_rate = rates.mean().detach()
        balance_factors = (target_rate / (rates.detach() + self.branch_balance_eps)).clamp(
            min=1.0 / self.branch_balance_clamp,
            max=self.branch_balance_clamp,
        )
        balance_factors = torch.nan_to_num(balance_factors, nan=1.0, posinf=self.branch_balance_clamp, neginf=1.0)
        if self.branch_balance_strength <= 0:
            balance_factors = torch.ones_like(balance_factors)
        else:
            balance_factors = 1.0 + self.branch_balance_strength * (balance_factors - 1.0)

        learned_scales = torch.exp(self.branch_log_scales.clamp(min=-2.0, max=2.0)).to(delta_out.dtype)
        delta_scale = balance_factors[0] * learned_scales[0]
        conv_scale = balance_factors[1] * learned_scales[1]
        delta_scale = torch.nan_to_num(delta_scale, nan=1.0, posinf=float(self.branch_balance_clamp), neginf=1.0)
        conv_scale = torch.nan_to_num(conv_scale, nan=1.0, posinf=float(self.branch_balance_clamp), neginf=1.0)
        delta_balanced = torch.nan_to_num(delta_out * delta_scale, nan=0.0, posinf=0.0, neginf=0.0)
        conv_balanced = torch.nan_to_num(conv_out * conv_scale, nan=0.0, posinf=0.0, neginf=0.0)
        balance_loss = (rates[0] - rates[1]).abs() / (target_rate + self.branch_balance_eps)
        balance_loss = torch.nan_to_num(balance_loss, nan=0.0, posinf=float(self.branch_balance_clamp), neginf=0.0)
        return delta_balanced, conv_balanced, balance_factors, balance_loss, rates

    def forward(
        self, inputs: torch.Tensor, return_components: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Encode inputs with both branches and optionally return diagnostics."""
        delta_out, delta_states = self.delta_encoder(inputs, return_states=return_components)
        conv_out, conv_states = self.conv_encoder(inputs, return_states=return_components)
        delta_balanced, conv_balanced, balance_factors, balance_loss, rates = self._balance_branch_outputs(
            delta_out, conv_out
        )

        if self.fusion_mode == "sum":
            fused = delta_balanced + conv_balanced
            weights = torch.tensor([1.0, 1.0], device=fused.device, dtype=fused.dtype)
        else:
            weights = torch.softmax(self.branch_logits, dim=0)
            fused = weights[0] * delta_balanced + weights[1] * conv_balanced

        if not return_components:
            return fused, None

        return fused, {
            "delta_spk": delta_out,
            "conv_spk": conv_out,
            "delta_balanced": delta_balanced,
            "conv_balanced": conv_balanced,
            "fusion_weights": weights.detach(),
            "branch_balance_factors": balance_factors.detach(),
            "branch_balance_loss": balance_loss,
            "delta_rate": rates[0],
            "conv_rate": rates[1],
            "delta_states": delta_states,
            "conv_states": conv_states,
        }


def _stack_state_dicts(states: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack per-step neuron state dictionaries into batched tensors."""
    stacked: Dict[str, torch.Tensor] = {}
    for key in states[0]:
        values = [state[key] for state in states if key in state]
        if not values:
            continue
        if values[0].dim() == 0:
            stacked[key] = torch.stack(values)
        else:
            stacked[key] = torch.stack(values, dim=1)
    return stacked
