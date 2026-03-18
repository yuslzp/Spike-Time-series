"""
Visualization functions for HybridSNN neuron dynamics, spike trains, and attention.

All functions return matplotlib.figure.Figure and accept an optional save_path kwarg.
"""

from typing import List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import torch


def _save_and_return(fig: plt.Figure, save_path: Optional[str] = None) -> plt.Figure:
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=100)
    return fig


def plot_membrane_potential(
    mem: torch.Tensor,
    title: str = "Membrane Potential",
    num_neurons: int = 8,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot membrane potential over time for selected neurons.

    Args:
        mem: (T, L, D) single sample membrane potential
        title: plot title
        num_neurons: how many neurons to show (sampled uniformly)
        save_path: if given, save figure here
    Returns:
        matplotlib Figure
    """
    T, L, D = mem.shape
    # Flatten L and D, select num_neurons uniformly
    mem_flat = mem.reshape(T, -1).numpy()  # (T, L*D)
    total = mem_flat.shape[1]
    idxs = np.linspace(0, total - 1, min(num_neurons, total), dtype=int)

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, idx in enumerate(idxs):
        ax.plot(mem_flat[:, idx], label=f"neuron {idx}", alpha=0.8)
    ax.set_xlabel("Time step T")
    ax.set_ylabel("Membrane potential")
    ax.set_title(title)
    ax.legend(fontsize=6, loc="upper right")
    plt.tight_layout()
    return _save_and_return(fig, save_path)


def plot_current_vs_membrane(
    input_tensor: torch.Tensor,
    mem: torch.Tensor,
    title: str = "Input vs Membrane",
    num_neurons: int = 500,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter plot of pre-LIF input vs membrane potential.

    Args:
        input_tensor: (T, L, D) pre-LIF projection
        mem: (T, L, D) membrane potential
        title: plot title
        num_neurons: max points to plot (random subsample)
        save_path: if given, save figure here
    """
    T, L, D = mem.shape
    inp_flat = input_tensor.reshape(-1).numpy()
    mem_flat = mem.reshape(-1).numpy()
    if len(inp_flat) > num_neurons:
        idxs = np.random.choice(len(inp_flat), num_neurons, replace=False)
        inp_flat = inp_flat[idxs]
        mem_flat = mem_flat[idxs]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(inp_flat, mem_flat, alpha=0.3, s=5)
    ax.set_xlabel("Pre-LIF input")
    ax.set_ylabel("Membrane potential")
    ax.set_title(title)
    plt.tight_layout()
    return _save_and_return(fig, save_path)


def plot_spike_raster(
    spk: torch.Tensor,
    title: str = "Spike Raster",
    max_neurons: int = 64,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Raster plot: time on x-axis, neuron on y-axis, dots for spikes.

    Args:
        spk: (T, L, D) binary spike tensor (single sample)
        title: plot title
        max_neurons: max neurons to show (flatten L*D, take first N)
        save_path: if given, save figure here
    """
    T, L, D = spk.shape
    spk_flat = spk.reshape(T, -1).numpy()  # (T, N)
    N = min(max_neurons, spk_flat.shape[1])
    spk_flat = spk_flat[:, :N]

    fig, ax = plt.subplots(figsize=(10, 4))
    for neuron in range(N):
        times = np.where(spk_flat[:, neuron] > 0.5)[0]
        ax.scatter(times, np.full_like(times, neuron), s=3, c="black", marker="|")
    ax.set_xlabel("Time step T")
    ax.set_ylabel("Neuron index")
    ax.set_title(title)
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    plt.tight_layout()
    return _save_and_return(fig, save_path)


def plot_attention_heatmap(
    attn: torch.Tensor,
    title: str = "Attention Weights",
    max_heads: int = 4,
    max_timesteps: int = 2,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of attention weights per head and timestep (single sample).

    Args:
        attn: (T, heads, L, L) attention tensor (single sample, already on CPU)
        title: plot title
        max_heads: how many heads to show
        max_timesteps: how many time steps to show
        save_path: if given, save figure here
    """
    T, H, L, _ = attn.shape
    T_show = min(max_timesteps, T)
    H_show = min(max_heads, H)

    fig, axes = plt.subplots(T_show, H_show, figsize=(3 * H_show, 3 * T_show))
    if T_show == 1 and H_show == 1:
        axes = np.array([[axes]])
    elif T_show == 1:
        axes = axes[np.newaxis, :]
    elif H_show == 1:
        axes = axes[:, np.newaxis]

    for t in range(T_show):
        for h in range(H_show):
            a = attn[t, h].numpy()
            axes[t, h].imshow(a, aspect="auto", cmap="viridis")
            axes[t, h].set_title(f"t={t} h={h}", fontsize=8)
            axes[t, h].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    return _save_and_return(fig, save_path)


def plot_firing_rates(
    rates: List[float],
    names: Optional[List[str]] = None,
    title: str = "Firing Rates per Block",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of per-block firing rates.

    Args:
        rates: list of float firing rates
        names: optional bar labels (defaults to "block_0", "block_1", ...)
        title: plot title
        save_path: if given, save figure here
    """
    if names is None:
        names = [f"block_{i}" for i in range(len(rates))]

    fig, ax = plt.subplots(figsize=(max(4, len(rates) * 1.5), 4))
    ax.bar(names, rates, color="steelblue", edgecolor="black")
    ax.set_xlabel("Block")
    ax.set_ylabel("Mean firing rate")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return _save_and_return(fig, save_path)


def plot_gasf_image(
    gasf: torch.Tensor,
    title: str = "GASF Image",
    variate_idx: int = 0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of a GASF matrix (single sample, single variate).

    Args:
        gasf: (B, C, L', L') or (C, L', L') GASF matrix (CPU)
        title: plot title
        variate_idx: which variate (channel) to visualize
        save_path: if given, save figure here
    """
    if gasf.dim() == 4:
        g = gasf[0, variate_idx].numpy()
    else:
        g = gasf[variate_idx].numpy()

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(g, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_title(f"{title} (variate {variate_idx})")
    plt.tight_layout()
    return _save_and_return(fig, save_path)


def plot_encoder_comparison(
    delta: torch.Tensor,
    conv: torch.Tensor,
    title: str = "Delta vs Conv Encoder",
    max_channels: int = 4,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side raster plots for delta and conv encoder outputs.

    Args:
        delta: (T, C, L) spike tensor (single sample)
        conv: (T, C, L) spike tensor (single sample)
        title: plot title
        max_channels: how many channels to show per panel
        save_path: if given, save figure here
    """
    T, C, L = delta.shape
    C_show = min(max_channels, C)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, spk, label in zip(axes, [delta, conv], ["Delta", "Conv"]):
        spk_flat = spk[:, :C_show, :].reshape(T, -1).numpy()  # (T, C_show*L)
        N = spk_flat.shape[1]
        for neuron in range(N):
            times = np.where(spk_flat[:, neuron] > 0.5)[0]
            ax.scatter(times, np.full_like(times, neuron), s=2, c="black", marker="|")
        ax.set_xlabel("Time step T")
        ax.set_ylabel("Channel × Position")
        ax.set_title(f"{label} encoder")
        ax.set_xlim(-0.5, T - 0.5)

    fig.suptitle(title)
    plt.tight_layout()
    return _save_and_return(fig, save_path)
