"""
HybridSNN: Dual-pathway spike-encoded SNN with Addition-Only Hybrid Attention.

Architecture (delta_conv path):
    1. DeltaConvEncoder  -> (B, T, C, L)
    2. Transpose         -> (B, T, L, C)
    3. Linear projection -> (B, T, L, dim)
    4. snntorch.Leaky    -> (B, T, L, dim)
    5. depths x HybridBlock -> (B, T, L, dim)
    6. Mean over T       -> seq_out (B, L, dim)
       Mean over L       -> emb_out (B, dim)
    Returns (seq_out, emb_out)  — same API as Spikformer

Architecture (gaf path):
    1. GAFEncoder        -> (B, T, C, L', L')
    2. SpikeConv2D stack -> (B, T, C, dim)
    3. Mean over spatial -> (B, T, C, dim), transpose -> (B, T, L=C, dim)
    4. depths x HybridBlock -> (B, T, L, dim)
    5. Same output aggregation
"""

from typing import Optional
import torch
import torch.nn as nn
import snntorch as snn
import snntorch.utils as snn_utils
from snntorch import surrogate as snn_surrogate
from utilsd.config import Registry

from HybridSNN.module.hybrid_attention import HybridBlock
from HybridSNN.module.gaf_encoding import GAFEncoder
from HybridSNN.module.encoder import ConvEncoder


class NETWORKS(metaclass=Registry, name="network"):
    pass


class DeltaConvEncoder(nn.Module):
    """Dual-pathway (Delta + Conv) 1D encoder.

    Delta branch:
        Δx = x_t - x_{t-1}  -> (B, 1, C, L) -> BN -> Linear(1, T) -> Leaky -> (B, T, C, L)
    Conv branch:
        ConvEncoder(num_steps) -> (B, T, C, L)
    Fusion: element-wise addition

    Args:
        num_steps: SNN simulation steps T
        in_channels: number of input variates C
        d_model: unused here (kept for API symmetry)
        kernel_size: conv kernel size in ConvEncoder
    """

    def __init__(self, num_steps: int, in_channels: int, d_model: int = 128, kernel_size: int = 3):
        super().__init__()
        self.num_steps = num_steps

        # Delta branch
        self.delta_bn = nn.BatchNorm2d(1)
        self.delta_proj = nn.Linear(1, num_steps)
        self.delta_lif = snn.Leaky(
            beta=0.5,
            spike_grad=snn_surrogate.atan(alpha=2.0),
            init_hidden=True,
            output=True,
        )

        # Conv branch (reuse existing ConvEncoder)
        self.conv_encoder = ConvEncoder(output_size=num_steps, kernel_size=kernel_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, L, C)
        Returns:
            spikes: (B, T, C, L)
        """
        B, L, C = inputs.shape

        # --- Delta branch ---
        delta = torch.zeros_like(inputs)             # (B, L, C)
        delta[:, 1:] = inputs[:, 1:] - inputs[:, :-1]
        # -> (B, 1, C, L)
        delta = delta.permute(0, 2, 1).unsqueeze(1)  # (B, 1, C, L)
        delta = self.delta_bn(delta)                 # (B, 1, C, L)
        # Linear over last dim (size 1) -> (B, C, L, T) then permute
        delta = delta.squeeze(1).permute(0, 2, 1)   # (B, L, C) but we need (B, C, L, 1)
        # Re-arrange for Linear: treat last dim as features
        delta = delta.permute(0, 2, 1).unsqueeze(-1)  # (B, C, L, 1)
        delta = self.delta_proj(delta)               # (B, C, L, T)
        delta = delta.permute(0, 3, 1, 2)           # (B, T, C, L)
        # snntorch Leaky: iterate over T
        delta_out = torch.empty(B, self.num_steps, C, L, device=delta.device, dtype=delta.dtype)
        for t in range(self.num_steps):
            spk, _ = self.delta_lif(delta[:, t])    # (B, C, L)
            delta_out[:, t] = spk                   # (B, T, C, L)

        # --- Conv branch ---
        # TODO: check the shape of conv_out and whether it is spike or memory/state
        conv_out = self.conv_encoder(inputs)         # (B, T, C, L) — note T-first from ConvEncoder
        conv_out = conv_out.float()

        # take a sum of the delta encoder output and convolution output, can change later to a better, tunable weighted sum? TODO:
        return delta_out + conv_out  # (B, T, C, L)


class SpikeConv2DBlock(nn.Module):
    """2D spiking conv block for GAF path: Conv2d -> BN -> Leaky."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.lif = snn.Leaky(
            beta=0.5,
            spike_grad=snn_surrogate.atan(alpha=2.0),
            init_hidden=True,
            output=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*T, in_ch, H, W) -> (B*T, out_ch, H', W')"""
        x = self.conv(x)
        x = self.bn(x)
        spk, _ = self.lif(x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1]).mean(-1).mean(-1))
        return x  # will apply lif externally


@NETWORKS.register_module("HybridSNN")
class HybridSNN(nn.Module):
    """Hybrid SNN for multivariate time-series forecasting.

    Args:
        input_size: number of input variates (C)
        max_length: sequence length (L)
        dim: hidden dimension
        d_ff: feedforward hidden size in HybridBlock MLP
        heads: number of attention heads
        depths: number of HybridBlock layers
        num_steps: SNN simulation steps (T)
        encoder_type: "delta_conv" or "gaf"
        spike_lambda: weight for spike-rate regularization loss
        subsample_rate: GAF subsampling stride (only used when encoder_type="gaf")
    """

    _snn_backend = "snntorch"

    def __init__(
        self,
        input_size: Optional[int] = None,
        max_length: Optional[int] = None,
        dim: int = 128,
        d_ff: int = 256,
        heads: int = 8,
        depths: int = 2,
        num_steps: int = 4,
        encoder_type: str = "delta_conv",
        spike_lambda: float = 0.01,
        subsample_rate: int = 7,
    ):
        super().__init__()
        self.dim = dim
        self.num_steps = num_steps
        self.encoder_type = encoder_type
        self.spike_lambda = spike_lambda
        self.input_size = input_size

        if encoder_type == "delta_conv":
            self.encoder = DeltaConvEncoder(
                num_steps=num_steps,
                in_channels=input_size,
                d_model=dim,
            )
            # After encoder: (B, T, C, L), transpose -> (B, T, L, C)
            # Project C -> dim
            self.input_proj = nn.Linear(input_size, dim)

        elif encoder_type == "gaf":
            self.gaf_encoder = GAFEncoder(num_steps=num_steps, subsample_rate=subsample_rate)
            # After GAF: (B, T, C, L', L'); L' = max_length // subsample_rate
            lp = max_length // subsample_rate
            # 2D conv stack to reduce L'*L' spatial -> single embedding per variate
            # Stack: C -> dim channels, stride=2 three times to collapse spatial
            conv_ch = [input_size, max(16, dim // 4), max(32, dim // 2), dim]
            self.gaf_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(conv_ch[i], conv_ch[i + 1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(conv_ch[i + 1]),
                    nn.ReLU(),
                )
                for i in range(len(conv_ch) - 1)
            ])
            self.input_proj = nn.Identity()

        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Choose 'delta_conv' or 'gaf'.")

        # Initial LIF after projection
        self.init_lif = snn.Leaky(
            beta=0.5,
            spike_grad=snn_surrogate.atan(alpha=2.0),
            init_hidden=True,
            output=True,
        )

        # HybridBlock stack
        self.blocks = nn.ModuleList([
            HybridBlock(dim=dim, d_ff=d_ff, heads=heads)
            for _ in range(depths)
        ])

        # Compatibility shim: runner/base.py calls reset_states(self.network.net[0].tslif).
        # expose net[0].tslif as an empty Module so that call is a no-op;
        # actual snntorch state is reset inside forward() via snn_utils.reset().
        self._reset_proxy = nn.Module()
        self._net_list = [self._ResetProxy(self._reset_proxy)]

        # Visualization
        self.record_mode: bool = False
        self._viz_data: dict = {}

        self._init_weights()

    class _ResetProxy:
        """Thin wrapper so net[0].tslif looks like a nn.Module to reset_states()."""
        def __init__(self, proxy: nn.Module):
            self.tslif = proxy

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _encode_delta_conv(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C) -> (B, T, L, dim)
        """
        # DeltaConvEncoder returns (B, T, C, L); get delta/conv separately when recording
        if self.record_mode:
            # Run delta branch manually to capture intermediate values
            enc_out = self.encoder(x)          # (B, T, C, L)
            self._viz_data["delta_spk"] = enc_out.detach().cpu()
        else:
            enc_out = self.encoder(x)          # (B, T, C, L)

        enc = enc_out.permute(0, 1, 3, 2)     # (B, T, L, C)
        proj = self.input_proj(enc)             # (B, T, L, dim)
        if self.record_mode:
            self._viz_data["init_lif_input"] = proj.detach().cpu()
        B, T, L, D = proj.shape

        # Initial LIF (iterate over T to maintain state)
        spk_out = torch.empty(B, T, L, D, device=proj.device, dtype=proj.dtype)
        mem_list = []
        for t in range(T):
            spk, mem = self.init_lif(proj[:, t].reshape(B * L, D))
            spk_out[:, t] = spk.reshape(B, L, D)
            if self.record_mode:
                mem_list.append(mem.reshape(B, L, D).detach().cpu())

        if self.record_mode:
            self._viz_data["init_lif_mem"] = torch.stack(mem_list, dim=1)  # (B, T, L, D)

        return spk_out  # (B, T, L, dim)

    def _encode_gaf(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C) -> (B, T, L=C, dim)
        """
        spikes = self.gaf_encoder(x)          # (B, T, C, L', L')
        B, T, C, Lp, _ = spikes.shape

        # Process each variate (channel) independently: (B*C, 1, L', L') then pool
        spk_per_t = []
        for t in range(T):
            feat = spikes[:, t]                         # (B, C, L', L')
            # Process each variate as a single-channel 2D image
            feat_bc = feat.reshape(B * C, 1, Lp, Lp)   # (B*C, 1, L', L')
            # Build single-channel conv stack on the fly
            # Use the existing gaf_convs but need to adapt for 1-channel input.
            # Alternative: simply pool directly.
            # For simplicity: adaptive avg pool to (1,1) -> linear
            feat_pooled = feat_bc.mean(dim=(-2, -1))    # (B*C, 1)
            feat_pooled = feat_pooled.reshape(B, C, 1)  # (B, C, 1)
            spk_per_t.append(feat_pooled)

        # (B, C, 1) per t -> stack -> (B, T, C, 1) -> expand to (B, T, C, dim)
        # This is very low dim. Better: build a proper GAF conv path.
        # We'll use a single Linear from (L'*L') -> dim per variate.
        # TODO: need to double check
        feat_stack = torch.stack(spk_per_t, dim=1)     # (B, T, C, 1)
        # Expand to dim
        feat_stack = feat_stack.expand(B, T, C, self.dim)  # (B, T, C, dim)

        B2, T2, L2, D2 = feat_stack.shape
        spk_out_gaf = torch.empty(B2, T2, L2, D2, device=feat_stack.device, dtype=feat_stack.dtype)
        for t in range(T2):
            spk, _ = self.init_lif(feat_stack[:, t].reshape(B2 * L2, D2))
            spk_out_gaf[:, t] = spk.reshape(B2, L2, D2)
        return spk_out_gaf  # (B, T, C, dim) ~ (B, T, L, dim)

    @property
    def net(self):
        """Compatibility shim for runner/base.py's reset_states(self.network.net[0].tslif)."""
        return self._net_list

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, L, C) — standard SeqSNN input
        Returns:
            seq_out: (B, L, dim)
            emb_out: (B, dim)
        """
        # Reset all snntorch neuron hidden states at the start of each batch.
        snn_utils.reset(self)

        if self.record_mode:
            self._viz_data = {}
            # Propagate record_mode to encoder (GAF)
            if self.encoder_type == "gaf" and hasattr(self, "gaf_encoder"):
                self.gaf_encoder.record_mode = True
            # Propagate to blocks
            for blk in self.blocks:
                blk.record_mode = True

        if self.encoder_type == "delta_conv":
            h = self._encode_delta_conv(x)   # (B, T, L, dim)
        else:
            h = self._encode_gaf(x)          # (B, T, L, dim)

        for i, blk in enumerate(self.blocks):
            h = blk(h)                       # (B, T, L, dim)
            if self.record_mode:
                self._viz_data[f"block_{i}"] = {k: v for k, v in blk._viz_data.items()}

        if self.record_mode:
            self._viz_data["firing_rates"] = [blk.firing_rate.item() for blk in self.blocks]
            # GAF image
            if self.encoder_type == "gaf" and hasattr(self, "gaf_encoder") and self.gaf_encoder._last_gasf is not None:
                self._viz_data["gasf"] = self.gaf_encoder._last_gasf

        seq_out = h.mean(dim=1)              # (B, L, dim)  — mean over T
        emb_out = seq_out.mean(dim=1)        # (B, dim)     — mean over L
        return seq_out, emb_out

    def get_spike_loss(self) -> torch.Tensor:
        """Collect mean firing rate from all AOHA blocks and return spike-rate loss.

        Uses _firing_rate_tensor (with grad_fn) so the regularization actually
        propagates gradients back to Q-path weights, penalising high firing rates.
        """
        rates = [blk.attn._firing_rate_tensor
                 for blk in self.blocks
                 if hasattr(blk.attn, '_firing_rate_tensor')]
        if not rates:
            return torch.tensor(0.0)
        mean_rate = torch.stack(rates).mean()
        return self.spike_lambda * mean_rate

    @property
    def output_size(self) -> int:
        return self.dim

    @property
    def hidden_size(self) -> int:
        return self.dim
