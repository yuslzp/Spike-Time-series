"""
HybridSNN model with configurable encoders, TS-LIF support, and visualization hooks.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from utilsd.config import Registry

from HybridSNN.module.encoder import DeltaConvEncoder
from HybridSNN.module.gaf_encoding import GAFEncoder
from HybridSNN.module.hybrid_attention import HybridBlock
from HybridSNN.module.neuron import build_spike_neuron, reset_module_state


class NETWORKS(metaclass=Registry, name="network"):
    """Registry for available HybridSNN network backbones."""

    pass

def _primary_membrane(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the most informative membrane-like tensor from a state dict."""
    if "vs" in state:
        return state["vs"]
    if "mem" in state:
        return state["mem"]
    if "vd" in state:
        return state["vd"]
    raise KeyError(f"Unsupported state keys: {list(state.keys())}")


class CausalLowRankChannelMixer(nn.Module):
    """Mix channels with depthwise temporal convolution and low-rank gating."""

    def __init__(
        self,
        input_size: int,
        rank: int = 16,
        kernel_size: int = 3,
        dropout: float = 0.0,
        residual_gate: bool = True,
    ):
        """Initialize the channel mixer and optional residual gate."""
        super().__init__()
        self.input_size = input_size
        self.kernel_size = max(1, kernel_size)
        self.residual_gate = residual_gate
        hidden_rank = max(1, rank)
        self.temporal = nn.Conv1d(
            input_size,
            input_size,
            kernel_size=self.kernel_size,
            padding=self.kernel_size - 1,
            groups=input_size,
            bias=False,
        )
        self.norm = nn.LayerNorm(input_size)
        self.in_proj = nn.Linear(input_size, hidden_rank)
        self.out_proj = nn.Linear(hidden_rank, input_size)
        self.dropout = nn.Dropout(dropout)
        self.gate_proj = nn.Linear(input_size * 2, input_size) if residual_gate else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Mix input channels causally and optionally return the residual gate."""
        mixed = self.temporal(x.transpose(1, 2))
        mixed = mixed[:, :, : x.shape[1]].transpose(1, 2)
        mixed = self.out_proj(F.gelu(self.in_proj(self.norm(mixed))))
        mixed = self.dropout(mixed)
        if self.gate_proj is None:
            return x + mixed, None
        gate = torch.sigmoid(self.gate_proj(torch.cat([x, mixed], dim=-1)))
        return x + gate * mixed, gate


class GAFTokenEncoder(nn.Module):
    """Convert GAF spike images into per-step token embeddings."""

    def __init__(
        self,
        num_steps: int,
        input_size: int,
        max_length: int,
        dim: int,
        subsample_rate: int = 7,
        channel_chunk_size: int = 32,
        backbone_chunk_size: int = 1024,
        checkpoint_backbone: bool = True,
    ):
        """Build the GAF image backbone and channel-to-time projection."""
        super().__init__()
        self.max_length = max_length
        self.dim = dim
        self.channel_chunk_size = max(1, channel_chunk_size)
        self.backbone_chunk_size = max(1, backbone_chunk_size)
        self.checkpoint_backbone = checkpoint_backbone
        self.gaf_encoder = GAFEncoder(num_steps=num_steps, subsample_rate=subsample_rate)
        hidden_1 = max(16, dim // 4)
        hidden_2 = max(32, dim // 2)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, hidden_1, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_1),
            nn.GELU(),
            nn.Conv2d(hidden_1, hidden_2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_2),
            nn.GELU(),
            nn.Conv2d(hidden_2, dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.channel_to_time = nn.Linear(input_size, max_length)

    def _encode_backbone_chunk(self, feat: torch.Tensor) -> torch.Tensor:
        """Encode a chunk of GAF images with optional activation checkpointing."""
        if self.training and self.checkpoint_backbone and feat.requires_grad:
            return checkpoint(self.backbone, feat, use_reentrant=False)
        return self.backbone(feat)

    def forward(self, x: torch.Tensor, record_mode: bool = False):
        """Generate token embeddings and optional GAF diagnostics from inputs."""
        prob = self.gaf_encoder.compute_probabilities(x, record_mode=record_mode)
        bsz, channels, side, _ = prob.shape
        tokens_per_step = []
        recorded_spikes = [] if record_mode else None
        step_rates = []
        for _ in range(self.gaf_encoder.num_steps):
            step_spikes = self.gaf_encoder.sample_spike_frame(prob)
            step_rates.append(step_spikes.mean())
            chunk_embeddings = []
            for start in range(0, channels, self.channel_chunk_size):
                end = min(start + self.channel_chunk_size, channels)
                feat = step_spikes[:, start:end].reshape(bsz * (end - start), 1, side, side)
                flat_embeddings = []
                for feat_start in range(0, feat.shape[0], self.backbone_chunk_size):
                    feat_end = min(feat_start + self.backbone_chunk_size, feat.shape[0])
                    flat_embeddings.append(self._encode_backbone_chunk(feat[feat_start:feat_end]))
                emb = torch.cat(flat_embeddings, dim=0).reshape(bsz, end - start, self.dim)
                chunk_embeddings.append(emb)
            step_embeddings = torch.cat(chunk_embeddings, dim=1)
            step_tokens = self.channel_to_time(step_embeddings.permute(0, 2, 1)).permute(0, 2, 1)
            tokens_per_step.append(step_tokens)
            if recorded_spikes is not None:
                recorded_spikes.append(step_spikes.detach().cpu())
        tokens = torch.stack(tokens_per_step, dim=1)
        aux = {"gaf_spike_rate": torch.stack(step_rates).mean()}
        if record_mode and self.gaf_encoder._last_gasf is not None:
            aux["gasf"] = self.gaf_encoder._last_gasf
            aux["gaf_spikes"] = torch.stack(recorded_spikes, dim=1)
        return tokens, aux


@NETWORKS.register_module("HybridSNN")
class HybridSNN(nn.Module):
    """Hybrid spiking backbone with configurable encoders and readout states."""

    _snn_backend = "hybrid"

    def __init__(
        self,
        input_size: int = 0,
        max_length: int = 0,
        dim: int = 128,
        d_ff: int = 256,
        heads: int = 8,
        depths: int = 2,
        num_steps: int = 8,
        encoder_type: str = "delta_conv",
        neuron_type: str = "tslif",
        spike_lambda: float = 0.01,
        spike_target: float = 0.25,
        spike_target_margin: float = 0.05,
        spike_loss_mode: str = "target",
        branch_balance_lambda: float = 0.02,
        fusion_balance_lambda: float = 0.01,
        merged_fusion_hidden: int = 128,
        attention_mode: str = "aoha",
        value_mode: str = "ternary",
        channel_mixer_type: str = "none",
        channel_mixer_rank: int = 16,
        channel_mixer_kernel_size: int = 3,
        channel_mixer_dropout: float = 0.0,
        channel_mixer_residual_gate: bool = True,
        subsample_rate: int = 7,
        gaf_channel_chunk_size: int = 32,
        gaf_backbone_chunk_size: int = 1024,
        gaf_checkpoint_backbone: bool = True,
        kernel_size: int = 3,
        encoder_dropout: float = 0.0,
        analog_delta_residual: bool = True,
        analog_delta_residual_weight: float = 0.15,
        analog_delta_residual_dropout: float = 0.0,
        init_input_dropout: float = 0.0,
        delta_spike_target: float = 0.0,
        conv_spike_target: float = 0.0,
        delta_spike_lambda: float = 0.0,
        conv_spike_lambda: float = 0.0,
        neuron_threshold: float = 0.75,
        threshold: float = -1.0,
        current_gain_init: float = 1.0,
        input_bias_init: float = 0.05,
        delta_scale_init: float = 5.0,
        delta_bias_init: float = 0.1,
        surrogate_type: str = "atan",
        surrogate_alpha: float = 2.0,
        surrogate_alpha_min: float = 0.5,
        surrogate_alpha_max: float = 10.0,
        spike_warmup_epochs: int = 10,
        alpha_d_init: float = 0.9,
        alpha_s_init: float = 0.2,
        beta_d_init: float = 0.1,
        beta_s_init: float = 0.25,
        gamma_d_init: float = 0.4,
        gamma_s_init: float = 1.0,
        kappa_init: float = 0.5,
    ):
        """Construct the HybridSNN encoder, backbone blocks, and spike losses."""
        super().__init__()
        if input_size <= 0 or max_length <= 0:
            raise ValueError("HybridSNN requires input_size and max_length at build time.")

        self.dim = dim
        self.num_steps = num_steps
        self.encoder_type = encoder_type
        self.neuron_type = neuron_type
        self.spike_lambda = spike_lambda
        self.spike_target = spike_target
        self.spike_target_margin = spike_target_margin
        self.spike_loss_mode = spike_loss_mode
        self.branch_balance_lambda = branch_balance_lambda
        self.fusion_balance_lambda = fusion_balance_lambda
        self.input_size = input_size
        self.max_length = max_length
        self.channel_mixer_type = channel_mixer_type
        self.attention_mode = attention_mode
        self.value_mode = value_mode
        self.record_mode = False
        self._viz_data: Dict[str, torch.Tensor | List[float]] = {}
        self._rate_tensors: List[torch.Tensor] = []
        self._aux_loss_terms: List[torch.Tensor] = []
        self._warmup_aux_loss_terms: List[torch.Tensor] = []
        self.spike_warmup_epochs = max(0, spike_warmup_epochs)
        self.analog_delta_residual = analog_delta_residual
        self.analog_delta_residual_weight = analog_delta_residual_weight
        self.delta_spike_target = max(0.0, delta_spike_target)
        self.conv_spike_target = max(0.0, conv_spike_target)
        self.delta_spike_lambda = max(0.0, delta_spike_lambda)
        self.conv_spike_lambda = max(0.0, conv_spike_lambda)
        effective_threshold = neuron_threshold if threshold < 0 else threshold

        neuron_kwargs = {
            "beta": 0.5,
            "threshold": effective_threshold,
            "current_gain_init": current_gain_init,
            "input_bias_init": input_bias_init,
            "surrogate_type": surrogate_type,
            "surrogate_alpha": surrogate_alpha,
            "surrogate_alpha_min": surrogate_alpha_min,
            "surrogate_alpha_max": surrogate_alpha_max,
            "alpha_d_init": alpha_d_init,
            "alpha_s_init": alpha_s_init,
            "beta_d_init": beta_d_init,
            "beta_s_init": beta_s_init,
            "gamma_d_init": gamma_d_init,
            "gamma_s_init": gamma_s_init,
            "kappa_init": kappa_init,
        }
        self._neuron_kwargs = neuron_kwargs

        self.delta_conv_encoder = DeltaConvEncoder(
            num_steps=num_steps,
            kernel_size=kernel_size,
            neuron_type=neuron_type,
            neuron_kwargs=neuron_kwargs,
            branch_balance_strength=1.0,
            delta_scale_init=delta_scale_init,
            delta_bias_init=delta_bias_init,
            encoder_dropout=encoder_dropout,
        )
        self.input_proj = nn.Linear(input_size, dim)
        self.analog_delta_norm = nn.LayerNorm(input_size)
        self.analog_delta_proj = nn.Linear(input_size, dim)
        self.analog_delta_dropout = nn.Dropout(analog_delta_residual_dropout)
        self.init_input_dropout = nn.Dropout(init_input_dropout)
        if channel_mixer_type == "none":
            self.channel_mixer = None
        elif channel_mixer_type == "causal_low_rank":
            self.channel_mixer = CausalLowRankChannelMixer(
                input_size=input_size,
                rank=channel_mixer_rank,
                kernel_size=channel_mixer_kernel_size,
                dropout=channel_mixer_dropout,
                residual_gate=channel_mixer_residual_gate,
            )
        else:
            raise ValueError(
                f"Unknown channel_mixer_type={channel_mixer_type!r}. Expected 'none' or 'causal_low_rank'."
            )

        self.gaf_encoder = GAFTokenEncoder(
            num_steps=num_steps,
            input_size=input_size,
            max_length=max_length,
            dim=dim,
            subsample_rate=subsample_rate,
            channel_chunk_size=gaf_channel_chunk_size,
            backbone_chunk_size=gaf_backbone_chunk_size,
            checkpoint_backbone=gaf_checkpoint_backbone,
        )
        self.encoder_mix_logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        fusion_hidden = max(32, merged_fusion_hidden)
        self.encoder_fusion_gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, fusion_hidden),
            nn.GELU(),
            nn.Linear(fusion_hidden, 1),
        )

        self.init_neuron = build_spike_neuron(neuron_type=neuron_type, **neuron_kwargs)
        self.blocks = nn.ModuleList(
            [
                HybridBlock(
                    dim=dim,
                    d_ff=d_ff,
                    heads=heads,
                    attention_mode=attention_mode,
                    value_mode=value_mode,
                    neuron_type=neuron_type,
                    neuron_kwargs=neuron_kwargs,
                )
                for _ in range(depths)
            ]
        )

        self._reset_proxy = nn.Module()
        self._net_list = [self._ResetProxy(self._reset_proxy)]
        self._init_weights()

    class _ResetProxy:
        """Expose a resettable proxy so legacy code can traverse the network."""

        def __init__(self, proxy: nn.Module):
            """Store the wrapped proxy module."""
            self.tslif = proxy

    def _init_weights(self) -> None:
        """Initialize linear layers with the project's default weight scheme."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _run_init_neuron(self, encoded: torch.Tensor, prefix: str = "init"):
        """Run the input spike neuron over encoded tokens and store diagnostics."""
        bsz, steps, seq_len, d_model = encoded.shape
        spikes = []
        states = []
        for step in range(steps):
            spk, state = self.init_neuron(encoded[:, step].reshape(bsz * seq_len, d_model))
            spikes.append(spk.reshape(bsz, seq_len, d_model))
            states.append({k: v.reshape(bsz, seq_len, -1) for k, v in state.items() if v.ndim >= 2})
        spike_tensor = torch.stack(spikes, dim=1)
        self._rate_tensors.append(spike_tensor.mean())
        if self.record_mode:
            self._viz_data[f"{prefix}_spk"] = spike_tensor.detach().cpu()
            self._viz_data[f"{prefix}_mem"] = torch.stack(
                [_primary_membrane(state) for state in states], dim=1
            ).detach().cpu()
            if "vd" in states[0]:
                self._viz_data[f"{prefix}_dend_mem"] = torch.stack(
                    [state["vd"] for state in states], dim=1
                ).detach().cpu()
                self._viz_data[f"{prefix}_soma_mem"] = torch.stack(
                    [state["vs"] for state in states], dim=1
                ).detach().cpu()
        return spike_tensor

    def _apply_channel_mixer(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the optional channel mixer and record its diagnostics."""
        if self.channel_mixer is None:
            return x
        mixed, gate = self.channel_mixer(x)
        if self.record_mode:
            self._viz_data["channel_mixer_output"] = mixed.detach().cpu()
            if gate is not None:
                self._viz_data["channel_mixer_gate_mean"] = gate.mean(dim=(0, 1)).detach().cpu()
        return mixed

    def _compute_analog_delta_residual(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Project normalized analog deltas into an additive encoder residual."""
        if not self.analog_delta_residual or self.analog_delta_residual_weight <= 0:
            return None
        delta = torch.zeros_like(x)
        delta[:, 1:] = x[:, 1:] - x[:, :-1]
        delta_std = delta.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        delta = torch.nan_to_num(delta / delta_std, nan=0.0, posinf=0.0, neginf=0.0)
        residual = self.analog_delta_proj(self.analog_delta_norm(delta))
        residual = self.analog_delta_dropout(residual)
        residual = self.analog_delta_residual_weight * residual
        residual = residual.unsqueeze(1).expand(-1, self.num_steps, -1, -1)
        if self.record_mode:
            self._viz_data["analog_delta_residual"] = residual.detach().cpu()
        return residual

    def _append_encoder_spike_losses(self, aux: Dict[str, torch.Tensor]) -> None:
        """Apply direct floor penalties to encoder branch firing rates."""
        branch_rates = {
            "delta": aux["delta_rate"],
            "conv": aux["conv_rate"],
        }
        branch_specs = (
            ("delta", self.delta_spike_target, self.delta_spike_lambda),
            ("conv", self.conv_spike_target, self.conv_spike_lambda),
        )
        for branch_name, target, weight in branch_specs:
            if target <= 0 or weight <= 0:
                continue
            self._warmup_aux_loss_terms.append(weight * F.relu(target - branch_rates[branch_name]))
        if self.record_mode:
            self._viz_data["encoder_branch_rates"] = {
                name: float(rate.detach().cpu()) for name, rate in branch_rates.items()
            }

    def _encode_delta_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Encode inputs with the delta-conv branch and input spike neuron."""
        enc, aux = self.delta_conv_encoder(x, return_components=True)
        self._rate_tensors.extend([aux["delta_spk"].mean(), aux["conv_spk"].mean()])
        self._aux_loss_terms.append(self.branch_balance_lambda * aux["branch_balance_loss"])
        self._append_encoder_spike_losses(aux)
        proj = self.input_proj(enc.permute(0, 1, 3, 2))
        analog_delta_residual = self._compute_analog_delta_residual(x)
        if analog_delta_residual is not None:
            proj = proj + analog_delta_residual
        proj = self.init_input_dropout(proj)
        if self.record_mode:
            self._viz_data["delta_spk"] = aux["delta_spk"].detach().cpu()
            self._viz_data["conv_spk"] = aux["conv_spk"].detach().cpu()
            self._viz_data["encoder_branch_weights"] = aux["fusion_weights"].detach().cpu()
            self._viz_data["encoder_branch_balance"] = aux["branch_balance_factors"].detach().cpu()
            self._viz_data["init_input"] = proj.detach().cpu()
        return self._run_init_neuron(proj, prefix="init")

    def _encode_gaf(self, x: torch.Tensor) -> torch.Tensor:
        """Encode inputs with the GAF branch and input spike neuron."""
        tokens, aux = self.gaf_encoder(x, record_mode=self.record_mode)
        self._rate_tensors.append(aux["gaf_spike_rate"])
        tokens = self.init_input_dropout(tokens)
        if self.record_mode:
            self._viz_data["init_input"] = tokens.detach().cpu()
            for key, value in aux.items():
                self._viz_data[key] = value
        return self._run_init_neuron(tokens, prefix="init")

    def _encode_merged(self, x: torch.Tensor) -> torch.Tensor:
        """Blend delta-conv and GAF encodings with a learned fusion gate."""
        delta_enc, aux = self.delta_conv_encoder(x, return_components=True)
        delta_proj = self.input_proj(delta_enc.permute(0, 1, 3, 2))
        analog_delta_residual = self._compute_analog_delta_residual(x)
        if analog_delta_residual is not None:
            delta_proj = delta_proj + analog_delta_residual
        gaf_tokens, gaf_aux = self.gaf_encoder(x, record_mode=self.record_mode)
        fusion_input = torch.cat([delta_proj, gaf_tokens], dim=-1)
        prior_logit = (self.encoder_mix_logits[0] - self.encoder_mix_logits[1]).view(1, 1, 1, 1)
        delta_gate = torch.sigmoid(self.encoder_fusion_gate(fusion_input) + prior_logit)
        encoder_weights = torch.cat([delta_gate, 1.0 - delta_gate], dim=-1)
        merged = delta_gate * delta_proj + (1.0 - delta_gate) * gaf_tokens
        mean_mix = encoder_weights.mean(dim=(0, 1, 2))
        fusion_balance_loss = ((mean_mix - 0.5) ** 2).mean()

        self._rate_tensors.extend([aux["delta_spk"].mean(), aux["conv_spk"].mean(), gaf_aux["gaf_spike_rate"]])
        self._aux_loss_terms.append(self.branch_balance_lambda * aux["branch_balance_loss"])
        self._aux_loss_terms.append(self.fusion_balance_lambda * fusion_balance_loss)
        self._append_encoder_spike_losses(aux)
        merged = self.init_input_dropout(merged)
        if self.record_mode:
            self._viz_data["delta_spk"] = aux["delta_spk"].detach().cpu()
            self._viz_data["conv_spk"] = aux["conv_spk"].detach().cpu()
            self._viz_data["encoder_branch_weights"] = aux["fusion_weights"].detach().cpu()
            self._viz_data["encoder_branch_balance"] = aux["branch_balance_factors"].detach().cpu()
            self._viz_data["encoder_mix_weights"] = mean_mix.detach().cpu()
            self._viz_data["init_input"] = merged.detach().cpu()
            for key, value in gaf_aux.items():
                self._viz_data[key] = value
        return self._run_init_neuron(merged, prefix="init")

    @property
    def net(self):
        """Expose a legacy-compatible network list used by existing tooling."""
        return self._net_list

    def forward(self, x: torch.Tensor):
        """Encode the sequence, run HybridBlocks, and return sequence embeddings."""
        reset_module_state(self)
        self._viz_data = {}
        self._rate_tensors = []
        self._aux_loss_terms = []
        self._warmup_aux_loss_terms = []
        x = self._apply_channel_mixer(x)

        if self.record_mode:
            self.gaf_encoder.gaf_encoder.record_mode = True
        else:
            self.gaf_encoder.gaf_encoder.record_mode = False

        if self.encoder_type == "delta_conv":
            h = self._encode_delta_conv(x)
        elif self.encoder_type == "gaf":
            h = self._encode_gaf(x)
        elif self.encoder_type == "merged":
            h = self._encode_merged(x)
        else:
            raise ValueError(
                f"Unknown encoder_type={self.encoder_type!r}. Expected delta_conv, gaf, or merged."
            )

        firing_rates = []
        for idx, block in enumerate(self.blocks):
            block.record_mode = self.record_mode
            h = block(h)
            self._rate_tensors.append(block.attn._firing_rate_tensor)
            firing_rates.append(float(block.firing_rate))
            if self.record_mode:
                self._viz_data[f"block_{idx}"] = dict(block._viz_data)

        if self.record_mode:
            self._viz_data["firing_rates"] = [float(rate.detach().cpu()) for rate in self._rate_tensors]

        seq_out = h.mean(dim=1)
        emb_out = seq_out.mean(dim=1)
        return seq_out, emb_out

    def get_spike_loss(self, epoch: int = 0) -> torch.Tensor:
        """Compute spike-rate and auxiliary balancing penalties for training."""
        device = next(self.parameters()).device
        base_loss = torch.zeros((), device=device)
        if self._rate_tensors:
            rates = torch.stack(self._rate_tensors)
            if self.spike_loss_mode == "target":
                lower = self.spike_target - self.spike_target_margin
                upper = self.spike_target + self.spike_target_margin
                base_loss = (F.relu(lower - rates).pow(2) + F.relu(rates - upper).pow(2)).mean()
            elif self.spike_loss_mode == "boost":
                base_loss = F.relu(self.spike_target - rates).mean()
            elif self.spike_loss_mode == "suppress":
                base_loss = rates.mean()
            else:
                raise ValueError(f"Unknown spike_loss_mode={self.spike_loss_mode!r}")
        aux_loss = torch.stack(self._aux_loss_terms).sum() if self._aux_loss_terms else torch.zeros((), device=device)
        warmup_aux_loss = (
            torch.stack(self._warmup_aux_loss_terms).sum()
            if self._warmup_aux_loss_terms
            else torch.zeros((), device=device)
        )
        if self.spike_warmup_epochs <= 0:
            warmup_factor = 1.0
        else:
            warmup_factor = min(1.0, max(0.0, float(epoch)) / float(self.spike_warmup_epochs))
        return self.spike_lambda * warmup_factor * base_loss + aux_loss + warmup_factor * warmup_aux_loss

    @property
    def output_size(self) -> int:
        """Return the embedding width produced by the network."""
        return self.dim

    @property
    def hidden_size(self) -> int:
        """Alias the backbone embedding width for legacy consumers."""
        return self.dim
