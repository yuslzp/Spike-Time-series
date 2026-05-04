from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tslif import TemporalSegmentLIF
from .surrogates import SurrogateSpike


def reset_module_state(module: nn.Module) -> None:
    """Reset every child module that exposes a `reset()` method."""
    for child in module.modules():
        if hasattr(child, "reset"):
            child.reset()


class LIFSpikeNeuron(nn.Module):
    """Leaky integrate-and-fire neuron with learnable input scaling."""

    def __init__(
        self,
        beta: float = 0.5,
        threshold: float = 1.0,
        current_gain_init: float = 1.0,
        input_bias_init: float = 0.0,
        surrogate_type: str = "atan",
        surrogate_alpha: float = 2.0,
        surrogate_alpha_min: float = 0.5,
        surrogate_alpha_max: float = 10.0,
    ):
        """Initialize a single-compartment spike neuron and its surrogate."""
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.surrogate = SurrogateSpike(
            surrogate_type=surrogate_type,
            surrogate_alpha=surrogate_alpha,
            surrogate_alpha_min=surrogate_alpha_min,
            surrogate_alpha_max=surrogate_alpha_max,
        )
        self.current_gain_raw = nn.Parameter(
            torch.tensor(math.log(math.exp(max(current_gain_init, 1e-4)) - 1.0), dtype=torch.float32)
        )
        self.input_bias = nn.Parameter(torch.tensor(input_bias_init, dtype=torch.float32))
        self._mem: torch.Tensor | None = None

    def _ensure_state(self, x: torch.Tensor) -> None:
        """Allocate or refresh the membrane buffer to match the current input."""
        if (
            self._mem is None
            or self._mem.shape != x.shape
            or self._mem.device != x.device
            or self._mem.dtype != x.dtype
        ):
            self._mem = torch.zeros_like(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Advance the neuron one step and return spikes plus diagnostic state."""
        self._ensure_state(x)
        assert self._mem is not None
        current = F.softplus(self.current_gain_raw) * x + self.input_bias
        mem = self.beta * self._mem + current
        spk = self.surrogate(mem - self.threshold)
        self._mem = mem - self.threshold * spk.detach()
        return spk, {"mem": mem, "smix": spk, "surrogate_alpha": self.surrogate.alpha.detach()}

    def reset(self) -> None:
        """Clear the cached membrane state between independent sequences."""
        self._mem = None


class TSLIFSpikeNeuron(nn.Module):
    """Thin wrapper around the two-compartment TS-LIF implementation."""

    def __init__(
        self,
        threshold: float = 1.0,
        current_gain_init: float = 1.0,
        input_bias_init: float = 0.0,
        surrogate_type: str = "atan",
        surrogate_alpha: float = 2.0,
        surrogate_alpha_min: float = 0.5,
        surrogate_alpha_max: float = 10.0,
        alpha_d_init: float = 0.9,
        alpha_s_init: float = 0.2,
        beta_d_init: float = 0.1,
        beta_s_init: float = 0.25,
        gamma_d_init: float = 0.4,
        gamma_s_init: float = 1.0,
        kappa_init: float = 0.5,
    ):
        """Construct a TS-LIF neuron with the provided initialization values."""
        super().__init__()
        self.neuron = TemporalSegmentLIF(
            threshold=threshold,
            surrogate_type=surrogate_type,
            surrogate_alpha=surrogate_alpha,
            surrogate_alpha_min=surrogate_alpha_min,
            surrogate_alpha_max=surrogate_alpha_max,
            alpha_d_init=alpha_d_init,
            alpha_s_init=alpha_s_init,
            beta_d_init=beta_d_init,
            beta_s_init=beta_s_init,
            gamma_d_init=gamma_d_init,
            gamma_s_init=gamma_s_init,
            kappa_init=kappa_init,
            current_gain_init=current_gain_init,
            input_bias_init=input_bias_init,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Delegate one forward step to the underlying TS-LIF neuron."""
        return self.neuron(x)

    def reset(self) -> None:
        """Clear the wrapped TS-LIF neuron state."""
        self.neuron.reset()


def build_spike_neuron(
    neuron_type: str = "tslif",
    beta: float = 0.5,
    threshold: float = 0.75,
    current_gain_init: float = 1.0,
    input_bias_init: float = 0.05,
    surrogate_type: str = "atan",
    surrogate_alpha: float = 2.0,
    surrogate_alpha_min: float = 0.5,
    surrogate_alpha_max: float = 10.0,
    alpha_d_init: float = 0.9,
    alpha_s_init: float = 0.2,
    beta_d_init: float = 0.1,
    beta_s_init: float = 0.25,
    gamma_d_init: float = 0.4,
    gamma_s_init: float = 1.0,
    kappa_init: float = 0.5,
) -> nn.Module:
    """Instantiate the requested spike neuron backend from config values."""
    if neuron_type == "lif":
        return LIFSpikeNeuron(
            beta=beta,
            threshold=threshold,
            current_gain_init=current_gain_init,
            input_bias_init=input_bias_init,
            surrogate_type=surrogate_type,
            surrogate_alpha=surrogate_alpha,
            surrogate_alpha_min=surrogate_alpha_min,
            surrogate_alpha_max=surrogate_alpha_max,
        )
    if neuron_type == "tslif":
        return TSLIFSpikeNeuron(
            threshold=threshold,
            current_gain_init=current_gain_init,
            input_bias_init=input_bias_init,
            surrogate_type=surrogate_type,
            surrogate_alpha=surrogate_alpha,
            surrogate_alpha_min=surrogate_alpha_min,
            surrogate_alpha_max=surrogate_alpha_max,
            alpha_d_init=alpha_d_init,
            alpha_s_init=alpha_s_init,
            beta_d_init=beta_d_init,
            beta_s_init=beta_s_init,
            gamma_d_init=gamma_d_init,
            gamma_s_init=gamma_s_init,
            kappa_init=kappa_init,
        )
    raise ValueError(f"Unsupported neuron_type={neuron_type!r}")
