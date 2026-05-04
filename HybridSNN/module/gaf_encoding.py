"""
Gramian Angular Field (GAF) spike encoder.

Converts a multivariate time series (B, L, C) into a spike image tensor
(B, num_steps, C, L', L') via GASF + Bernoulli rate-coding.
"""

import torch
import torch.nn as nn


class GAFEncoder(nn.Module):
    """Encode a multivariate time series as Gramian Angular Summation Field spikes.

    Pipeline (per batch, per variable):
        1. Subsample: L -> L' = L // subsample_rate
        2. Min-max normalize each variable to [-1, 1]
        3. arccos to get angles phi in [0, pi]
        4. GASF: cos(phi_i + phi_j)  -> (B, C, L', L')
        5. Shift to [0, 1] for Bernoulli rate-coding
        6. Sample num_steps binary frames -> (B, num_steps, C, L', L')

    Args:
        num_steps: number of SNN simulation steps (= output time dimension T)
        subsample_rate: stride for subsampling before GAF to control L' size
    """

    def __init__(self, num_steps: int = 4, subsample_rate: int = 7):
        """Configure the number of sampled frames and temporal subsampling rate."""
        super().__init__()
        self.num_steps = num_steps
        self.subsample_rate = subsample_rate
        self.record_mode: bool = False
        self._last_gasf: "torch.Tensor | None" = None

    def compute_probabilities(
        self,
        inputs: torch.Tensor,
        record_mode: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            inputs: (B, L, C)  float tensor
        Returns:
            prob: (B, C, L', L')  Bernoulli probabilities
        """
        self._last_gasf = None

        # 1. Subsample
        x = inputs[:, ::self.subsample_rate, :]  # (B, L', C)
        B, Lp, C = x.shape

        # 2. Per-variable min-max normalize to [-1, 1]
        x_min = x.min(dim=1, keepdim=True).values   # (B, 1, C)
        x_max = x.max(dim=1, keepdim=True).values   # (B, 1, C)
        denom = (x_max - x_min).clamp(min=1e-8)
        x_norm = 2.0 * (x - x_min) / denom - 1.0   # (B, L', C) in [-1, 1]
        x_norm = x_norm.clamp(-1.0, 1.0)            # numerical safety

        # 3. arccos -> angles in [0, pi]
        phi = torch.acos(x_norm)  # (B, L', C)

        # 4. GASF: cos(phi_i + phi_j)
        # phi: (B, L', C) -> permute to (B, C, L')
        phi = phi.permute(0, 2, 1)  # (B, C, L')
        # Broadcast: (B, C, L', 1) + (B, C, 1, L') -> (B, C, L', L')
        gasf = torch.cos(phi.unsqueeze(-1) + phi.unsqueeze(-2))  # (B, C, L', L')
        if record_mode:
            self._last_gasf = gasf.detach().cpu()

        # 5. Shift from [-1, 1] to [0, 1] for Bernoulli rate-coding
        return (gasf + 1.0) / 2.0  # (B, C, L', L') in [0, 1]

    def sample_spike_frame(self, prob: torch.Tensor) -> torch.Tensor:
        """Sample a single spike frame from GAF Bernoulli probabilities."""
        return torch.bernoulli(prob)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, L, C)  float tensor
        Returns:
            spikes: (B, num_steps, C, L', L')  binary float tensor
        """
        prob = self.compute_probabilities(inputs, record_mode=self.record_mode)
        B, C, Lp, _ = prob.shape

        # 6. Bernoulli rate-coding over num_steps
        spikes = [self.sample_spike_frame(prob) for _ in range(self.num_steps)]
        spikes = torch.stack(spikes, dim=1)  # (B, num_steps, C, L', L')

        return spikes
