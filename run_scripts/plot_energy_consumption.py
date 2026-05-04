#!/usr/bin/env python
"""Plot theoretical HybridSNN energy/power-proxy consumption comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = {
    "SHD": REPO_ROOT / "outputs/energy_reports/shd_delta_conv_seed40_energy.json",
    "METR-LA h6": REPO_ROOT / "outputs/energy_reports/metr-la_h6_delta_conv_seed40_energy.json",
}
OUTPUT_PATH = REPO_ROOT / "outputs/energy_reports/power_consumption_comparison.png"


def load_energy_rows() -> list[dict[str, float | str]]:
    """Load dense and event-driven energy estimates from report JSON files."""
    rows = []
    for label, path in REPORTS.items():
        report = json.loads(path.read_text())
        energy = report["energy"]
        rows.append(
            {
                "dataset": label,
                "dense_mj": float(energy["dense_baseline_mj"]),
                "event_mj": float(energy["event_estimate_mj"]),
                "reduction_x": float(energy["reduction_x"]),
                "saved_percent": float(energy["saved_percent"]),
            }
        )
    return rows


def plot_energy_comparison(rows: list[dict[str, float | str]], output_path: Path) -> None:
    """Create and save a grouped bar chart comparing dense vs event estimates."""
    datasets = [str(row["dataset"]) for row in rows]
    dense = np.array([float(row["dense_mj"]) for row in rows])
    event = np.array([float(row["event_mj"]) for row in rows])
    reductions = [float(row["reduction_x"]) for row in rows]
    saved = [float(row["saved_percent"]) for row in rows]

    x = np.arange(len(datasets))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=180)
    dense_bars = ax.bar(
        x - width / 2,
        dense,
        width,
        label="Dense baseline (Same architecture, but with ANN)",
        color="#4C566A",
    )
    event_bars = ax.bar(
        x + width / 2,
        event,
        width,
        label="Event-driven SNN estimate (Our SNN model)",
        color="#2E8B57",
    )

    ax.set_title("Theoretical Power Consumption Proxy", pad=34)
    ax.set_ylabel("Energy per sample (mJ)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2)

    ymax = max(dense.max(), event.max()) * 1.46
    ax.set_ylim(0, ymax)

    for bars in (dense_bars, event_bars):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + ymax * 0.025,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    for idx, (reduction, saved_percent) in enumerate(zip(reductions, saved)):
        y = max(dense[idx], event[idx]) + ymax * 0.095
        ax.text(
            x[idx],
            y,
            f"{reduction:.2f}x lower\n{saved_percent:.1f}% saved",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#1F2933",
        )

    fig.text(
        0.01,
        0.01,
        "Operation-level estimate: dense MACs priced at 4.6 pJ, event ACs at 0.9 pJ.",
        fontsize=8,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate the comparison figure."""
    plot_energy_comparison(load_energy_rows(), OUTPUT_PATH)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
