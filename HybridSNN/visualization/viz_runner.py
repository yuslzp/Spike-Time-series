"""
Visualization orchestrator for one recorded forward pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch

from HybridSNN.visualization.plots import (
    plot_attention_heatmap,
    plot_branch_firing_rates,
    plot_current_vs_membrane,
    plot_dead_neuron_fractions,
    plot_encoder_comparison,
    plot_firing_rates,
    plot_gasf_image,
    plot_membrane_potential,
    plot_spike_density_heatmap,
    plot_spike_raster,
    plot_tslif_compartments,
)


def _mean_spike_rate(spk: torch.Tensor) -> float:
    """Return the mean firing rate of a spike tensor."""
    return float(spk.detach().float().mean().cpu())


def _dead_neuron_fraction(spk: torch.Tensor, spike_threshold: float = 0.05) -> float:
    """Estimate the fraction of neurons that never exceed the spike threshold."""
    tensor = spk.detach().float().cpu()
    if tensor.ndim < 2:
        return 0.0
    if tensor.ndim == 2:
        flat = tensor
        active = flat.sum(dim=0) > spike_threshold
    else:
        flat = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
        active = flat.sum(dim=(0, 1)) > spike_threshold
    return float((~active).float().mean())


def _branch_name(key: str) -> str:
    """Normalize visualization keys into branch names for summary plots."""
    if key.startswith("block_") and key.endswith("_q"):
        return key.replace("_q", "")
    return key


def _collect_spike_diagnostics(viz: Dict[str, object]) -> tuple[Dict[str, float], Dict[str, float]]:
    """Collect branch firing rates and dead-neuron fractions from viz tensors."""
    branch_rates: Dict[str, float] = {}
    dead_fractions: Dict[str, float] = {}

    for key in ["delta_spk", "conv_spk", "gaf_spikes", "init_spk"]:
        if key in viz and isinstance(viz[key], torch.Tensor):
            name = key.removesuffix("_spk")
            branch_rates[name] = _mean_spike_rate(viz[key])
            dead_fractions[name] = _dead_neuron_fraction(viz[key])

    for key, block_data in sorted(viz.items()):
        if not key.startswith("block_") or not isinstance(block_data, dict):
            continue
        q_spk = block_data.get("q_spk")
        if isinstance(q_spk, torch.Tensor):
            branch_key = _branch_name(f"{key}_q")
            branch_rates[branch_key] = _mean_spike_rate(q_spk)
            dead_fractions[branch_key] = _dead_neuron_fraction(q_spk)

    return branch_rates, dead_fractions


def run_visualization(
    network: torch.nn.Module,
    data_batch: torch.Tensor,
    output_dir: str,
    epoch: int = 0,
    wandb_run=None,
) -> None:
    """Run one recorded forward pass and persist all configured visualizations."""
    save_dir = Path(output_dir) / "viz" / f"epoch_{epoch:04d}"
    save_dir.mkdir(parents=True, exist_ok=True)

    network.record_mode = True
    with torch.no_grad():
        network(data_batch)
    viz = dict(network._viz_data)
    network.record_mode = False

    figures = {}
    branch_rates, dead_fractions = _collect_spike_diagnostics(viz)

    if "firing_rates" in viz:
        figures["firing_rates"] = plot_firing_rates(
            viz["firing_rates"],
            title=f"Firing Rates (epoch {epoch})",
            target_rate=getattr(network, "spike_target", None),
            save_path=str(save_dir / "firing_rates.png"),
        )
    if branch_rates:
        figures["branch_firing_rates"] = plot_branch_firing_rates(
            branch_rates,
            title=f"Branch-Wise Firing Rates (epoch {epoch})",
            target_rate=getattr(network, "spike_target", None),
            save_path=str(save_dir / "branch_firing_rates.png"),
        )
    if dead_fractions:
        figures["dead_neuron_fraction"] = plot_dead_neuron_fractions(
            dead_fractions,
            title=f"Dead Neuron Fraction (epoch {epoch})",
            save_path=str(save_dir / "dead_neuron_fraction.png"),
        )

    if "init_mem" in viz:
        figures["init_mem"] = plot_membrane_potential(
            viz["init_mem"][0],
            title=f"Input Neuron Membrane (epoch {epoch})",
            save_path=str(save_dir / "init_mem.png"),
        )
    if "init_input" in viz and "init_mem" in viz:
        figures["input_vs_mem"] = plot_current_vs_membrane(
            viz["init_input"][0],
            viz["init_mem"][0],
            title=f"Injected Current vs Voltage (epoch {epoch})",
            save_path=str(save_dir / "input_vs_mem.png"),
        )
    if "init_dend_mem" in viz and "init_soma_mem" in viz:
        figures["init_tslif"] = plot_tslif_compartments(
            viz["init_dend_mem"][0],
            viz["init_soma_mem"][0],
            title=f"Input TS-LIF Compartments (epoch {epoch})",
            save_path=str(save_dir / "init_tslif_compartments.png"),
        )

    if "delta_spk" in viz:
        figures["delta_raster"] = plot_spike_raster(
            viz["delta_spk"][0].permute(0, 2, 1),
            title=f"Delta Encoder Raster (epoch {epoch})",
            save_path=str(save_dir / "delta_raster.png"),
        )
        figures["delta_density"] = plot_spike_density_heatmap(
            viz["delta_spk"][0].permute(0, 2, 1),
            title=f"Delta Encoder Spike Density (epoch {epoch})",
            save_path=str(save_dir / "delta_density.png"),
        )
    if "conv_spk" in viz:
        figures["conv_raster"] = plot_spike_raster(
            viz["conv_spk"][0].permute(0, 2, 1),
            title=f"Conv Encoder Raster (epoch {epoch})",
            save_path=str(save_dir / "conv_raster.png"),
        )
        figures["conv_density"] = plot_spike_density_heatmap(
            viz["conv_spk"][0].permute(0, 2, 1),
            title=f"Conv Encoder Spike Density (epoch {epoch})",
            save_path=str(save_dir / "conv_density.png"),
        )
    if "gaf_spikes" in viz:
        figures["gaf_density"] = plot_spike_density_heatmap(
            viz["gaf_spikes"][0].permute(0, 2, 3, 1),
            title=f"GAF Encoder Spike Density (epoch {epoch})",
            save_path=str(save_dir / "gaf_density.png"),
        )
    if "delta_spk" in viz and "conv_spk" in viz:
        figures["encoder_comparison"] = plot_encoder_comparison(
            viz["delta_spk"][0],
            viz["conv_spk"][0],
            title=f"Encoder Branch Comparison (epoch {epoch})",
            save_path=str(save_dir / "encoder_comparison.png"),
        )

    if "gasf" in viz:
        figures["gasf"] = plot_gasf_image(
            viz["gasf"],
            title=f"GASF View (epoch {epoch})",
            save_path=str(save_dir / "gasf.png"),
        )

    for key, block_data in viz.items():
        if not key.startswith("block_") or not isinstance(block_data, dict):
            continue
        if "q_spk" in block_data:
            figures[f"{key}_q_spk"] = plot_spike_raster(
                block_data["q_spk"][0],
                title=f"{key} Q Spikes (epoch {epoch})",
                save_path=str(save_dir / f"{key}_q_spk.png"),
            )
            figures[f"{key}_q_density"] = plot_spike_density_heatmap(
                block_data["q_spk"][0],
                title=f"{key} Q Spike Density (epoch {epoch})",
                save_path=str(save_dir / f"{key}_q_density.png"),
            )
        if "q_mem" in block_data:
            figures[f"{key}_q_mem"] = plot_membrane_potential(
                block_data["q_mem"][0],
                title=f"{key} Q Membrane (epoch {epoch})",
                save_path=str(save_dir / f"{key}_q_mem.png"),
            )
        if "q_vd" in block_data and "q_vs" in block_data:
            figures[f"{key}_q_tslif"] = plot_tslif_compartments(
                block_data["q_vd"][0],
                block_data["q_vs"][0],
                title=f"{key} TS-LIF Q Compartments (epoch {epoch})",
                save_path=str(save_dir / f"{key}_q_tslif.png"),
            )
        if "attn_weights" in block_data:
            figures[f"{key}_attn"] = plot_attention_heatmap(
                block_data["attn_weights"][0],
                title=f"{key} Attention (epoch {epoch})",
                save_path=str(save_dir / f"{key}_attn.png"),
            )
        if "mlp_lif1_mem" in block_data:
            figures[f"{key}_mlp_lif1"] = plot_membrane_potential(
                block_data["mlp_lif1_mem"][0],
                title=f"{key} MLP Layer 1 (epoch {epoch})",
                save_path=str(save_dir / f"{key}_mlp_lif1.png"),
            )

    if wandb_run is not None and figures:
        try:
            import wandb

            wandb_run.log({f"viz/{name}": wandb.Image(fig) for name, fig in figures.items()}, step=epoch)
        except Exception as exc:
            print(f"wandb visualization logging failed: {exc}")

    import matplotlib.pyplot as plt

    for fig in figures.values():
        plt.close(fig)

    spike_summary = {
        "firing_rates": viz.get("firing_rates", []),
        "branch_firing_rates": branch_rates,
        "dead_neuron_fraction": dead_fractions,
    }
    with open(save_dir / "spike_summary.json", "w") as file:
        json.dump(spike_summary, file, indent=2, sort_keys=True)

    print(f"Visualization saved to {save_dir} ({len(figures)} plots)")
