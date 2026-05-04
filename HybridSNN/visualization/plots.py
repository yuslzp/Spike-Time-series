"""
Visualization helpers for HybridSNN diagnostics and post-hoc analysis.
"""

from __future__ import annotations

import os
from typing import List, Mapping, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, r2_score

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save_and_return(fig: plt.Figure, save_path: Optional[str] = None) -> plt.Figure:
    """Optionally save a figure before returning it to the caller."""
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a tensor-like object into a NumPy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x)


def _flatten_spike_tensor(spk: torch.Tensor | np.ndarray) -> np.ndarray:
    """Flatten all non-time spike dimensions into a neuron axis."""
    arr = _to_numpy(spk)
    if arr.ndim < 2:
        raise ValueError(f"Expected spike tensor with at least 2 dims, got shape {arr.shape}.")
    return arr.reshape(arr.shape[0], -1)


def _select_active_neurons(spk_flat: np.ndarray, max_neurons: int) -> np.ndarray:
    """Select the most active neurons up to the requested plotting limit."""
    neuron_count = min(max_neurons, spk_flat.shape[1])
    if neuron_count <= 0:
        return spk_flat[:, :0]
    activity = spk_flat.mean(axis=0)
    if spk_flat.shape[1] <= neuron_count:
        indices = np.arange(spk_flat.shape[1])
    else:
        indices = np.argsort(activity)[-neuron_count:]
    indices = indices[np.argsort(activity[indices])[::-1]]
    return spk_flat[:, indices]


def plot_membrane_potential(
    mem: torch.Tensor,
    title: str = "Membrane Potential",
    num_neurons: int = 8,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot membrane traces for a subset of neurons across simulation steps."""
    mem_flat = mem.reshape(mem.shape[0], -1).numpy()
    total = mem_flat.shape[1]
    idxs = np.linspace(0, total - 1, min(num_neurons, total), dtype=int)
    fig, ax = plt.subplots(figsize=(10, 4))
    for idx in idxs:
        ax.plot(mem_flat[:, idx], alpha=0.8)
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Voltage")
    ax.set_title(title)
    plt.tight_layout()
    return _save_and_return(fig, save_path)


def plot_tslif_compartments(
    dend_mem: torch.Tensor,
    soma_mem: torch.Tensor,
    title: str = "TS-LIF Compartments",
    num_neurons: int = 6,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot dendritic and somatic TS-LIF voltages for selected neurons."""
    dend_flat = dend_mem.reshape(dend_mem.shape[0], -1).numpy()
    soma_flat = soma_mem.reshape(soma_mem.shape[0], -1).numpy()
    total = dend_flat.shape[1]
    idxs = np.linspace(0, total - 1, min(num_neurons, total), dtype=int)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for idx in idxs:
        axes[0].plot(dend_flat[:, idx], alpha=0.8)
        axes[1].plot(soma_flat[:, idx], alpha=0.8)
    axes[0].set_ylabel("Dendritic V")
    axes[1].set_ylabel("Somatic V")
    axes[1].set_xlabel("Simulation step")
    axes[0].set_title(title)
    plt.tight_layout()
    return _save_and_return(fig, save_path)


def plot_current_vs_membrane(
    input_tensor: torch.Tensor,
    mem: torch.Tensor,
    title: str = "Input vs Membrane",
    max_points: int = 500,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter injected current against membrane voltage samples."""
    inp_flat = input_tensor.reshape(-1).numpy()
    mem_flat = mem.reshape(-1).numpy()
    if len(inp_flat) > max_points:
        idxs = np.random.choice(len(inp_flat), max_points, replace=False)
        inp_flat = inp_flat[idxs]
        mem_flat = mem_flat[idxs]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(inp_flat, mem_flat, alpha=0.25, s=6)
    ax.set_xlabel("Injected Current")
    ax.set_ylabel("Voltage")
    ax.set_title(title)
    plt.tight_layout()
    return _save_and_return(fig, save_path)


def plot_spike_raster(
    spk: torch.Tensor,
    title: str = "Spike Raster",
    max_neurons: int = 96,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Render a raster plot for the most active neurons in a spike tensor."""
    spk_flat = _select_active_neurons(_flatten_spike_tensor(spk), max_neurons=max_neurons)
    neuron_count = spk_flat.shape[1]
    fig, ax = plt.subplots(figsize=(10, 4))
    for neuron in range(neuron_count):
        times = np.where(spk_flat[:, neuron] > 0.05)[0]
        ax.scatter(times, np.full_like(times, neuron), s=4, c="black", marker="|")
    if neuron_count == 0 or np.count_nonzero(spk_flat > 0.05) == 0:
        ax.text(0.5, 0.5, "No spikes in selected neurons", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Active neuron rank")
    ax.set_title(title)
    ax.set_xlim(-0.5, spk.shape[0] - 0.5)
    plt.tight_layout()
    return _save_and_return(fig, save_path)


def plot_spike_density_heatmap(
    spk: torch.Tensor | np.ndarray,
    title: str = "Spike Density Heatmap",
    max_neurons: int = 128,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Render a heatmap of spike activity over time for selected neurons."""
    spk_flat = _select_active_neurons(_flatten_spike_tensor(spk), max_neurons=max_neurons)
    fig, ax = plt.subplots(figsize=(10, 5))
    if spk_flat.size == 0:
        ax.text(0.5, 0.5, "No spike channels available", ha="center", va="center", transform=ax.transAxes)
    else:
        im = ax.imshow(
            spk_flat.T,
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            vmin=0.0,
            vmax=max(1.0, float(spk_flat.max())),
        )
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Spike activity")
        if np.count_nonzero(spk_flat > 0.05) == 0:
            ax.text(0.5, 0.5, "Selected neurons stayed silent", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Active neuron rank")
    ax.set_title(title)
    plt.tight_layout()
    return _save_and_return(fig, save_path)


def plot_attention_heatmap(
    attn: torch.Tensor,
    title: str = "Attention Weights",
    max_heads: int = 4,
    max_timesteps: int = 2,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize attention matrices for a subset of heads and time steps."""
    time_steps = min(max_timesteps, attn.shape[0])
    num_heads = min(max_heads, attn.shape[1])
    fig, axes = plt.subplots(time_steps, num_heads, figsize=(3 * num_heads, 3 * time_steps))
    if time_steps == 1 and num_heads == 1:
        axes = np.array([[axes]])
    elif time_steps == 1:
        axes = axes[np.newaxis, :]
    elif num_heads == 1:
        axes = axes[:, np.newaxis]
    for step in range(time_steps):
        for head in range(num_heads):
            axes[step, head].imshow(attn[step, head].numpy(), aspect="auto", cmap="viridis")
            axes[step, head].set_title(f"t={step} h={head}", fontsize=8)
            axes[step, head].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    return _save_and_return(fig, save_path)


def plot_firing_rates(
    rates: Sequence[float],
    names: Optional[List[str]] = None,
    title: str = "Firing Rates",
    target_rate: Optional[float] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot average firing rates for named model components."""
    if names is None:
        names = [f"block_{idx}" for idx in range(len(rates))]
    fig, ax = plt.subplots(figsize=(max(4, len(rates) * 1.25), 4))
    ax.bar(names, rates, color="#2c7fb8", edgecolor="black")
    if target_rate is not None:
        ax.axhline(target_rate, color="#d95f0e", linestyle="--", linewidth=1.5, label="target")
        ax.legend()
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean firing rate")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    return _save_and_return(fig, save_path)


def plot_branch_firing_rates(
    branch_rates: Mapping[str, float],
    title: str = "Branch-Wise Firing Rates",
    target_rate: Optional[float] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot branch-specific firing rates from a mapping of component names."""
    names = list(branch_rates.keys())
    rates = [float(branch_rates[name]) for name in names]
    return plot_firing_rates(rates, names=names, title=title, target_rate=target_rate, save_path=save_path)


def plot_dead_neuron_fractions(
    dead_fractions: Mapping[str, float],
    title: str = "Dead Neuron Fraction",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot the fraction of inactive neurons for named components."""
    names = list(dead_fractions.keys())
    values = [float(dead_fractions[name]) for name in names]
    fig, ax = plt.subplots(figsize=(max(4, len(names) * 1.25), 4))
    ax.bar(names, values, color="#dd8452", edgecolor="black")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Dead neuron fraction")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    return _save_and_return(fig, save_path)


def plot_gasf_image(
    gasf: torch.Tensor,
    title: str = "GASF Image",
    variate_idx: int = 0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Render one GASF matrix for a selected input channel."""
    matrix = gasf[0, variate_idx].numpy() if gasf.dim() == 4 else gasf[variate_idx].numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_title(f"{title} (channel {variate_idx})")
    plt.tight_layout()
    return _save_and_return(fig, save_path)


def plot_encoder_comparison(
    delta: torch.Tensor,
    conv: torch.Tensor,
    title: str = "Delta vs Conv Encoder",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Compare delta and convolutional branch spike rasters side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for axis, tensor, label in zip(axes, [delta, conv], ["Delta", "Conv"]):
        flat = _select_active_neurons(_flatten_spike_tensor(tensor), max_neurons=96)
        for neuron in range(flat.shape[1]):
            times = np.where(flat[:, neuron] > 0.05)[0]
            axis.scatter(times, np.full_like(times, neuron), s=4, c="black", marker="|")
        if flat.shape[1] == 0 or np.count_nonzero(flat > 0.05) == 0:
            axis.text(0.5, 0.5, "No spikes", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(f"{label} branch")
        axis.set_xlabel("Simulation step")
        axis.set_ylabel("Active neuron rank")
    fig.suptitle(title)
    plt.tight_layout()
    return _save_and_return(fig, save_path)


def plot_forecast_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    variable_indices: Sequence[int],
    sample_idx: int = 0,
    title: str = "Forecast Trajectories",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot ground-truth and predicted trajectories for selected variables."""
    horizon = y_true.shape[1]
    fig, axes = plt.subplots(len(variable_indices), 1, figsize=(10, 2.5 * len(variable_indices)), sharex=True)
    if len(variable_indices) == 1:
        axes = [axes]
    xs = np.arange(horizon)
    for axis, variable_idx in zip(axes, variable_indices):
        axis.plot(xs, y_true[sample_idx, :, variable_idx], label="truth", linewidth=2)
        axis.plot(xs, y_pred[sample_idx, :, variable_idx], label="pred", linewidth=2)
        axis.set_ylabel(f"var {variable_idx}")
    axes[0].set_title(title)
    axes[-1].set_xlabel("Forecast horizon")
    axes[0].legend()
    plt.tight_layout()
    return _save_and_return(fig, save_path)


def plot_regression_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Prediction Scatter",
    max_points: int = 5000,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter flattened regression targets against predictions."""
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    if y_true_flat.size > max_points:
        idxs = np.random.choice(y_true_flat.size, max_points, replace=False)
        y_true_flat = y_true_flat[idxs]
        y_pred_flat = y_pred_flat[idxs]
    low = float(min(y_true_flat.min(), y_pred_flat.min()))
    high = float(max(y_true_flat.max(), y_pred_flat.max()))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true_flat, y_pred_flat, s=6, alpha=0.2)
    ax.plot([low, high], [low, high], color="#d95f0e", linewidth=2)
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")
    ax.set_title(title)
    plt.tight_layout()
    return _save_and_return(fig, save_path)


def plot_residual_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Histogram",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a histogram of flattened prediction residuals."""
    residuals = (y_pred - y_true).reshape(-1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=60, color="#41ab5d", alpha=0.85)
    ax.set_xlabel("Prediction - Truth")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.tight_layout()
    return _save_and_return(fig, save_path)


def plot_horizon_error_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Horizon-Wise Forecast Errors",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot R^2, RRSE, and MAE for each forecast horizon step."""
    if y_true.ndim != 3 or y_pred.ndim != 3:
        raise ValueError(f"Expected y_true/y_pred with shape [N, H, V], got {y_true.shape} and {y_pred.shape}.")

    horizon = y_true.shape[1]
    r2_values = np.zeros(horizon, dtype=np.float32)
    rrse_values = np.zeros(horizon, dtype=np.float32)
    mae_values = np.zeros(horizon, dtype=np.float32)

    for idx in range(horizon):
        yt = y_true[:, idx, :].reshape(-1)
        yp = y_pred[:, idx, :].reshape(-1)
        denom = np.sqrt(np.square(yt - yt.mean()).sum())
        rrse_values[idx] = float(np.sqrt(np.square(yp - yt).sum()) / denom) if denom > 1e-8 else np.nan
        mae_values[idx] = float(np.mean(np.abs(yp - yt)))
        try:
            r2_values[idx] = float(r2_score(yt, yp))
        except ValueError:
            r2_values[idx] = np.nan

    xs = np.arange(1, horizon + 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(xs, r2_values, marker="o", color="#2c7fb8", linewidth=2)
    axes[0].set_ylabel("R^2")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25)

    axes[1].plot(xs, rrse_values, marker="o", color="#d95f0e", linewidth=2)
    axes[1].set_ylabel("RRSE")
    axes[1].grid(alpha=0.25)

    axes[2].plot(xs, mae_values, marker="o", color="#41ab5d", linewidth=2)
    axes[2].set_ylabel("MAE")
    axes[2].set_xlabel("Forecast horizon")
    axes[2].grid(alpha=0.25)
    plt.tight_layout()
    return _save_and_return(fig, save_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Render a confusion matrix for classification predictions."""
    cm = confusion_matrix(y_true.reshape(-1), y_pred.reshape(-1))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    return _save_and_return(fig, save_path)
