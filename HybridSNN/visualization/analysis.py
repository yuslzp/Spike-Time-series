from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from HybridSNN.common.utils import to_torch
from HybridSNN.visualization.plots import (
    plot_confusion_matrix,
    plot_forecast_samples,
    plot_horizon_error_metrics,
    plot_regression_scatter,
    plot_residual_histogram,
)


def _collect_targets(dataset) -> np.ndarray:
    """Load a dataset split and concatenate all target tensors."""
    dataset.load()
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    targets = []
    for _, label in loader:
        if isinstance(label, torch.Tensor):
            targets.append(label.cpu().numpy())
        else:
            targets.append(np.asarray(label))
    return np.concatenate(targets, axis=0)


def generate_posthoc_analysis(
    runner,
    dataset,
    predictions,
    output_dir: str | Path,
    split_name: str = "test",
) -> None:
    """Create post-hoc forecast or confusion-matrix diagnostics for a split."""
    output_path = Path(output_dir) / "analysis"
    output_path.mkdir(parents=True, exist_ok=True)

    y_true = _collect_targets(dataset)
    y_pred = predictions.to_numpy()
    task = runner.hyper_paras.get("task", "regression")

    if task == "regression":
        horizon = getattr(dataset, "horizon", None)
        if horizon is None:
            return
        num_variables = y_true.shape[1] // horizon
        y_true = y_true.reshape(-1, horizon, num_variables)
        y_pred = y_pred.reshape(-1, horizon, num_variables)
        variable_scores = y_true.var(axis=(0, 1))
        top_variables = np.argsort(variable_scores)[-min(4, num_variables) :][::-1]

        plot_forecast_samples(
            y_true,
            y_pred,
            variable_indices=top_variables,
            title=f"{split_name} forecast traces",
            save_path=str(output_path / f"{split_name}_forecast_traces.png"),
        )
        plot_regression_scatter(
            y_true,
            y_pred,
            title=f"{split_name} prediction scatter",
            save_path=str(output_path / f"{split_name}_scatter.png"),
        )
        plot_residual_histogram(
            y_true,
            y_pred,
            title=f"{split_name} residual histogram",
            save_path=str(output_path / f"{split_name}_residuals.png"),
        )
        plot_horizon_error_metrics(
            y_true,
            y_pred,
            title=f"{split_name} horizon-wise errors",
            save_path=str(output_path / f"{split_name}_horizon_errors.png"),
        )
        return

    if task == "multiclassification":
        pred_labels = y_pred.argmax(axis=1)
        plot_confusion_matrix(
            y_true,
            pred_labels,
            title=f"{split_name} confusion matrix",
            save_path=str(output_path / f"{split_name}_confusion_matrix.png"),
        )
