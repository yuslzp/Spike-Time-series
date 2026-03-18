"""
Visualization orchestrator: runs one forward pass in record_mode and generates all plots.
"""

import os
from pathlib import Path
from typing import Optional

import torch

from HybridSNN.visualization.plots import (
    plot_membrane_potential,
    plot_current_vs_membrane,
    plot_spike_raster,
    plot_attention_heatmap,
    plot_firing_rates,
    plot_gasf_image,
)


def run_visualization(
    network: torch.nn.Module,
    data_batch: torch.Tensor,
    output_dir: str,
    epoch: int = 0,
    wandb_run=None,
) -> None:
    """Run one forward pass in record_mode and generate all diagnostic plots.

    Args:
        network: HybridSNN model (or any network with record_mode and _viz_data)
        data_batch: input tensor (B, L, C) — will use first sample for per-sample plots
        output_dir: base output directory; plots saved to output_dir/viz/epoch_{epoch}/
        epoch: current epoch number (used in path and wandb step)
        wandb_run: optional wandb run object; if given, log figures as wandb.Image
    """
    save_dir = Path(output_dir) / "viz" / f"epoch_{epoch:04d}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Forward pass in record mode ---
    network.record_mode = True
    with torch.no_grad():
        network(data_batch)
    viz = network._viz_data
    network.record_mode = False

    figures = {}

    # --- Firing rates ---
    if "firing_rates" in viz:
        fig = plot_firing_rates(
            viz["firing_rates"],
            title=f"Firing Rates (epoch {epoch})",
            save_path=str(save_dir / "firing_rates.png"),
        )
        figures["firing_rates"] = fig

    # --- init_lif membrane potential (first sample) ---
    if "init_lif_mem" in viz:
        mem = viz["init_lif_mem"][0]   # (T, L, D)
        fig = plot_membrane_potential(
            mem,
            title=f"Init LIF Membrane (epoch {epoch})",
            save_path=str(save_dir / "init_lif_mem.png"),
        )
        figures["init_lif_mem"] = fig

    if "init_lif_input" in viz and "init_lif_mem" in viz:
        fig = plot_current_vs_membrane(
            viz["init_lif_input"][0],   # (T, L, D)
            viz["init_lif_mem"][0],
            title=f"Input vs Membrane (epoch {epoch})",
            save_path=str(save_dir / "input_vs_mem.png"),
        )
        figures["input_vs_mem"] = fig

    # --- Delta spike raster (first sample) ---
    if "delta_spk" in viz:
        spk = viz["delta_spk"][0]  # (T, C, L)
        T, C, L = spk.shape
        spk_tld = spk.permute(0, 2, 1)  # (T, L, C)
        fig = plot_spike_raster(
            spk_tld,
            title=f"Delta Encoder Spikes (epoch {epoch})",
            save_path=str(save_dir / "delta_spk_raster.png"),
        )
        figures["delta_spk_raster"] = fig

    # --- GASF image ---
    if "gasf" in viz:
        fig = plot_gasf_image(
            viz["gasf"],
            title=f"GASF (epoch {epoch})",
            save_path=str(save_dir / "gasf.png"),
        )
        figures["gasf"] = fig

    # --- Per-block visualizations ---
    for key, block_data in viz.items():
        if not key.startswith("block_") or not isinstance(block_data, dict):
            continue
        block_name = key  # e.g. "block_0"

        # Q spikes raster
        if "q_spk" in block_data:
            spk = block_data["q_spk"][0]   # (T, L, D)
            fig = plot_spike_raster(
                spk,
                title=f"{block_name} Q Spikes (epoch {epoch})",
                save_path=str(save_dir / f"{block_name}_q_spk.png"),
            )
            figures[f"{block_name}_q_spk"] = fig

        # Q membrane
        if "q_mem" in block_data:
            mem = block_data["q_mem"][0]   # (T, L, D)
            fig = plot_membrane_potential(
                mem,
                title=f"{block_name} Q Membrane (epoch {epoch})",
                save_path=str(save_dir / f"{block_name}_q_mem.png"),
            )
            figures[f"{block_name}_q_mem"] = fig

        # Attention weights
        if "attn_weights" in block_data:
            attn = block_data["attn_weights"][0]   # (T, heads, L, L)
            fig = plot_attention_heatmap(
                attn,
                title=f"{block_name} Attention (epoch {epoch})",
                save_path=str(save_dir / f"{block_name}_attn.png"),
            )
            figures[f"{block_name}_attn"] = fig

        # MLP LIF1 membrane
        if "mlp_lif1_mem" in block_data:
            mem = block_data["mlp_lif1_mem"][0]   # (T, L, d_ff)
            fig = plot_membrane_potential(
                mem,
                title=f"{block_name} MLP LIF1 Mem (epoch {epoch})",
                save_path=str(save_dir / f"{block_name}_mlp_lif1_mem.png"),
            )
            figures[f"{block_name}_mlp_lif1_mem"] = fig

    # --- Log to wandb ---
    if wandb_run is not None:
        try:
            import wandb
            wandb_images = {}
            for name, fig in figures.items():
                wandb_images[f"viz/{name}"] = wandb.Image(fig)
            wandb_run.log(wandb_images, step=epoch)
        except Exception as e:
            print(f"wandb visualization logging failed: {e}")

    # Close all figures to free memory
    import matplotlib.pyplot as plt
    for fig in figures.values():
        plt.close(fig)

    print(f"Visualization saved to {save_dir} ({len(figures)} plots)")
