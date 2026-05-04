"""
Quick sanity-check visualizations for GAFEncoder.

Produces two figures:
  1. gasf_fields.png  – GASF matrix for each variable in a single sample
  2. spike_raster.png – spike raster (time-step x flattened-pixel) per variable

Run from the project root:
    python viz_gaf.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── make HybridSNN importable ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from HybridSNN.module.gaf_encoding import GAFEncoder

# ── synthetic signals (B=1, L=168, C=4) ─────────────────────────────────────
torch.manual_seed(0)
L, C = 168, 4
t = torch.linspace(0, 4 * np.pi, L)
signals = torch.stack([
    torch.sin(t),                                          # pure sine
    torch.sin(t) + 0.3 * torch.randn(L),                  # noisy sine
    torch.sin(2 * t) * torch.exp(-t / (4 * np.pi)),       # damped
    torch.cumsum(torch.randn(L) * 0.1, dim=0),            # random walk
], dim=1).unsqueeze(0)  # (1, L, C)

var_names = ["Sine", "Noisy Sine", "Damped", "Random Walk"]

# ── encoder ──────────────────────────────────────────────────────────────────
enc = GAFEncoder(num_steps=8, subsample_rate=7)
enc.record_mode = True
enc.eval()

with torch.no_grad():
    spikes = enc(signals)  # (1, 8, C, L', L')

gasf = enc._last_gasf[0]   # (C, L', L')
Lp = gasf.shape[-1]
print(f"L' = {Lp},  spikes shape = {tuple(spikes.shape)}")

# ── Figure 1: GASF fields ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, C, figsize=(4 * C, 4))
for i, ax in enumerate(axes):
    im = ax.imshow(gasf[i].numpy(), vmin=-1, vmax=1, cmap="RdBu_r", origin="upper")
    ax.set_title(var_names[i])
    ax.set_xlabel("time step j")
    ax.set_ylabel("time step i")
    plt.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle(f"GASF matrices  (L={L}, L'={Lp}, subsample×{enc.subsample_rate})", fontsize=13)
fig.tight_layout()
fig.savefig("gasf_fields.png", dpi=120)
print("Saved gasf_fields.png")

# ── Figure 2: spike rasters ───────────────────────────────────────────────────
# Flatten spatial dims: (T, C, L', L') -> (C, T, L'*L')
sp = spikes[0]  # (T, C, L', L')
T = sp.shape[0]
sp_flat = sp.permute(1, 0, 2, 3).reshape(C, T, -1).numpy()  # (C, T, pixels)

fig2, axes2 = plt.subplots(C, 1, figsize=(10, 2.5 * C), sharex=True)
for i, ax in enumerate(axes2):
    # rows = pixels, cols = time steps
    data = sp_flat[i]  # (T, pixels)
    firing_rate = data.mean()
    # show as a heatmap: pixels on y-axis, time-step on x-axis
    ax.imshow(data.T, aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    ax.set_ylabel(f"{var_names[i]}\n(pixels)")
    ax.set_title(f"Mean firing rate = {firing_rate:.3f}", fontsize=9)

axes2[-1].set_xlabel("SNN time step")
# x-ticks at integer steps
axes2[-1].set_xticks(range(T))
axes2[-1].set_xticklabels([str(s) for s in range(T)])

fig2.suptitle(f"Spike raster  (T={T}, L'={Lp}, {Lp}×{Lp}={Lp**2} pixels per step)",
              fontsize=13)
fig2.tight_layout()
fig2.savefig("spike_raster.png", dpi=120)
print("Saved spike_raster.png")

# ── quick sanity stats ────────────────────────────────────────────────────────
print("\n── Sanity checks ──")
print(f"  spikes dtype      : {spikes.dtype}")
print(f"  all binary (0/1)  : {spikes.unique().tolist() == [0.0, 1.0] or set(spikes.unique().tolist()).issubset({0.0, 1.0})}")
print(f"  GASF range        : [{gasf.min():.4f}, {gasf.max():.4f}]  (expected [-1, 1])")
print(f"  GASF symmetric    : {torch.allclose(gasf, gasf.transpose(-1,-2), atol=1e-5)}")
diag_vals = torch.stack([gasf[c].diagonal() for c in range(C)])
print(f"  GASF diag (=cos(2φ)) range: [{diag_vals.min():.4f}, {diag_vals.max():.4f}]")
mean_rates = sp_flat.mean(axis=(1, 2))
for i, r in enumerate(mean_rates):
    print(f"  Mean firing rate [{var_names[i]:12s}]: {r:.4f}")
