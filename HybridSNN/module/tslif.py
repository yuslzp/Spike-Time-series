"""Reference TS-LIF neuron implementation from Section 4.1 of TS-LIF.pdf."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .surrogates import SurrogateSpike


def _logit(value: float) -> float:
    """Convert a probability-like scalar into logit space with clamping."""
    value = min(max(value, 1e-4), 1.0 - 1e-4)
    return math.log(value / (1.0 - value))


def _inv_softplus(value: float) -> float:
    """Approximate the inverse softplus transform for positive scalars."""
    value = max(value, 1e-4)
    return math.log(math.exp(value) - 1.0)


class TemporalSegmentLIF(nn.Module):
    """Dual-compartment TS-LIF neuron with direct soma current injection.

    The update follows Equation (5) from Feng et al.

        vd[t] = alpha_d * vd[t-1] + beta_d * vs[t-1] + (1-alpha_d) * x[t] - gamma_d * sd[t-1]
        vs[t] = alpha_s * vs[t-1] + beta_s * vd[t]   + (1-alpha_s) * x[t] - gamma_s * ss[t-1]
        sd[t] = H(vd[t] - v_th)
        ss[t] = H(vs[t] - v_th)
        s[t]  = kappa * sd[t] + (1-kappa) * ss[t]
    """

    def __init__(
        self,
        threshold: float = 1.0,
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
        current_gain_init: float = 1.0,
        input_bias_init: float = 0.0,
    ):
        """Initialize learnable TS-LIF dynamics and recurrent state buffers."""
        super().__init__()
        self.threshold = threshold
        self.surrogate = SurrogateSpike(
            surrogate_type=surrogate_type,
            surrogate_alpha=surrogate_alpha,
            surrogate_alpha_min=surrogate_alpha_min,
            surrogate_alpha_max=surrogate_alpha_max,
        )

        self.alpha_d_logit = nn.Parameter(torch.tensor(_logit(alpha_d_init), dtype=torch.float32))
        self.alpha_s_logit = nn.Parameter(torch.tensor(_logit(alpha_s_init), dtype=torch.float32))
        self.beta_d_raw = nn.Parameter(torch.tensor(beta_d_init, dtype=torch.float32))
        self.beta_s_raw = nn.Parameter(torch.tensor(beta_s_init, dtype=torch.float32))
        self.gamma_d_raw = nn.Parameter(torch.tensor(_inv_softplus(gamma_d_init), dtype=torch.float32))
        self.gamma_s_raw = nn.Parameter(torch.tensor(_inv_softplus(gamma_s_init), dtype=torch.float32))
        self.kappa_logit = nn.Parameter(torch.tensor(_logit(kappa_init), dtype=torch.float32))
        self.current_gain_raw = nn.Parameter(
            torch.tensor(_inv_softplus(current_gain_init), dtype=torch.float32)
        )
        self.input_bias = nn.Parameter(torch.tensor(input_bias_init, dtype=torch.float32))

        self._vd: Optional[torch.Tensor] = None
        self._vs: Optional[torch.Tensor] = None
        self._sd_prev: Optional[torch.Tensor] = None
        self._ss_prev: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Clear all recurrent membrane and spike history buffers."""
        self._vd = None
        self._vs = None
        self._sd_prev = None
        self._ss_prev = None

    def _ensure_state(self, x: torch.Tensor) -> None:
        """Allocate recurrent buffers that match the current input tensor."""
        if (
            self._vd is None
            or self._vd.shape != x.shape
            or self._vd.device != x.device
            or self._vd.dtype != x.dtype
        ):
            zeros = torch.zeros_like(x)
            self._vd = zeros
            self._vs = zeros.clone()
            self._sd_prev = zeros.clone()
            self._ss_prev = zeros.clone()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Advance the TS-LIF dynamics by one step and expose internal states."""
        self._ensure_state(x)
        assert self._vd is not None
        assert self._vs is not None
        assert self._sd_prev is not None
        assert self._ss_prev is not None

        alpha_d = torch.sigmoid(self.alpha_d_logit)
        alpha_s = torch.sigmoid(self.alpha_s_logit)
        beta_d = torch.tanh(self.beta_d_raw)
        beta_s = torch.tanh(self.beta_s_raw)
        gamma_d = F.softplus(self.gamma_d_raw)
        gamma_s = F.softplus(self.gamma_s_raw)
        kappa = torch.sigmoid(self.kappa_logit)
        current_gain = F.softplus(self.current_gain_raw)

        current = current_gain * x + self.input_bias
        vd = alpha_d * self._vd + beta_d * self._vs + (1.0 - alpha_d) * current - gamma_d * self._sd_prev
        vs = alpha_s * self._vs + beta_s * vd + (1.0 - alpha_s) * current - gamma_s * self._ss_prev

        sd = self.surrogate(vd - self.threshold)
        ss = self.surrogate(vs - self.threshold)
        smix = kappa * sd + (1.0 - kappa) * ss

        self._vd = vd
        self._vs = vs
        self._sd_prev = sd
        self._ss_prev = ss

        state = {
            "vd": vd,
            "vs": vs,
            "sd": sd,
            "ss": ss,
            "smix": smix,
            "alpha_d": alpha_d.detach(),
            "alpha_s": alpha_s.detach(),
            "kappa": kappa.detach(),
            "surrogate_alpha": self.surrogate.alpha.detach(),
        }
        return smix, state

    def extra_repr(self) -> str:
        """Summarize threshold and surrogate settings for module reprs."""
        alpha = float(self.surrogate.alpha.detach().cpu())
        return f"threshold={self.threshold}, surrogate={self.surrogate.surrogate_type}, surrogate_alpha={alpha:.4f}"
