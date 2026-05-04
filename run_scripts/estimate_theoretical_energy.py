#!/usr/bin/env python
"""Estimate theoretical inference energy for a trained HybridSNN run.

The estimator is intentionally architecture-level rather than wall-power based:
it counts dense multiply-accumulate opportunities (MACs), event-driven
accumulate opportunities (ACs), observed spike/nonzero densities, and converts
them to energy with configurable per-operation constants.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from HybridSNN.dataset import DATASETS
from HybridSNN.network import NETWORKS
from HybridSNN.runner import RUNNERS


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAC_PJ = 4.6
DEFAULT_AC_PJ = 0.9


@dataclass(frozen=True)
class EnergyAssumptions:
    """Per-operation energy constants used by the theoretical estimate."""

    mac_energy_pj: float
    ac_energy_pj: float


@dataclass(frozen=True)
class OperationEstimate:
    """Per-sample operation counts after averaging across evaluated batches."""

    dense_macs: float
    event_estimated_ops: float
    dense_only_macs: float
    event_eligible_macs: float
    event_eligible_acs: float
    aoha_dense_attention_macs: float
    aoha_q_gated_qk_acs: float


class EvaluationResult(NamedTuple):
    """Raw counters collected from sampled inference batches."""

    samples: int
    batch_count: int
    operation_totals: defaultdict[str, float]
    per_module: defaultdict[str, defaultdict[str, float]]
    aoha_totals: defaultdict[str, float]
    firing_rates: list[float]


def _registry_get(registry, name: str):
    """Return a class from the utilsd registry by name."""
    try:
        return registry._module_dict[name]
    except KeyError as exc:
        available = list(registry._module_dict)
        raise KeyError(f"Unknown registry entry {name!r}; available={available}") from exc


def _resolve_path(path_value: str | None) -> str | None:
    """Resolve config paths relative to the repository root."""
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _build_from_config(section: dict[str, Any], registry, **extra_kwargs):
    """Instantiate a registry object from a saved JSON config section."""
    kwargs = dict(section)
    type_name = kwargs.pop("type", None) or _infer_type_name(registry, kwargs)
    cls = _registry_get(registry, type_name)
    kwargs.update(extra_kwargs)
    return cls(**kwargs)


def _infer_type_name(registry, kwargs: dict[str, Any]) -> str:
    """Infer old-style config sections that omit explicit registry type."""
    names = list(registry._module_dict)
    if len(names) == 1:
        return names[0]
    if registry is DATASETS:
        if "num_time_bins" in kwargs or "num_neurons" in kwargs or "test_file" in kwargs:
            return "SHDDataset"
        return "TSMSDataset"
    if registry is NETWORKS:
        return "HybridSNN"
    if registry is RUNNERS:
        return "hybrid_ts" if "spike_lambda" in kwargs or "viz_every" in kwargs else "ts"
    raise ValueError(f"Cannot infer registry type from keys={sorted(kwargs)}")


def _sanitize_runner_config(
    config: dict[str, Any],
    batch_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Disable training-only services and point logs at a temporary folder."""
    runner_cfg = dict(config["runner"])
    runner_cfg["batch_size"] = batch_size
    runner_cfg["wandb_enabled"] = False
    runner_cfg["num_workers"] = 0
    runner_cfg["persistent_workers"] = False
    runner_cfg["model_path"] = None
    runner_cfg["output_dir"] = output_dir
    runner_cfg["checkpoint_dir"] = output_dir / "checkpoints"
    runner_cfg["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    return runner_cfg


def _base_dataset(dataset):
    """Return the concrete dataset when a split is wrapped in Subset."""
    return dataset.dataset if isinstance(dataset, Subset) else dataset


def _prepare_config(run_dir: Path) -> dict[str, Any]:
    """Load and normalize the saved run config."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json under {run_dir}")
    config = json.loads(config_path.read_text())
    for attr in ("file", "test_file"):
        if attr in config["data"] and config["data"][attr] is not None:
            config["data"][attr] = _resolve_path(config["data"][attr])
    return config


def _load_best_weights(runner: nn.Module, run_dir: Path, device: torch.device) -> dict[str, Any]:
    """Load the best full runner checkpoint when available."""
    candidates = [
        run_dir / "checkpoints" / "model_best.pkl",
        run_dir / "checkpoints" / "network_best.pkl",
    ]
    for path in candidates:
        if not path.exists():
            continue
        state = torch.load(path, map_location=device, weights_only=False)
        target = runner if path.name == "model_best.pkl" else runner.network
        try:
            target.load_state_dict(state, strict=True)
            return {"path": str(path), "strict": True, "missing_keys": [], "unexpected_keys": []}
        except RuntimeError:
            loaded = target.load_state_dict(state, strict=False)
            return {
                "path": str(path),
                "strict": False,
                "missing_keys": list(loaded.missing_keys),
                "unexpected_keys": list(loaded.unexpected_keys),
            }
    checkpoint_dir = run_dir / "checkpoints"
    raise FileNotFoundError(f"No model_best.pkl or network_best.pkl found under {checkpoint_dir}")


def _linear_macs(module: nn.Linear, output: torch.Tensor) -> int:
    """Return dense MACs for one Linear forward pass."""
    return int(output.numel() * module.in_features)


def _conv_macs(module: nn.modules.conv._ConvNd, output: torch.Tensor) -> int:
    """Return dense MACs for one Conv forward pass."""
    kernel_ops = (module.in_channels // module.groups) * math.prod(module.kernel_size)
    return int(output.numel() * kernel_ops)


def _batch_size_from_tensor(tensor: torch.Tensor) -> int:
    """Infer the user batch dimension from a module input/output tensor."""
    return int(tensor.shape[0]) if tensor.ndim > 0 else 1


def _nonzero_density(tensor: torch.Tensor) -> float:
    """Compute observed nonzero density for sparse/event-driven estimates."""
    if tensor.numel() == 0:
        return 0.0
    return float((tensor.detach() != 0).float().mean().item())


def _is_dense_only(name: str) -> bool:
    """Classify modules whose current computation should remain dense MACs."""
    dense_tokens = (
        "analog_delta",
        "channel_mixer",
        "encoder_fusion_gate",
        "horizon_decoder",
        "fc_out",
        "sequence_pool",
        "gaf_encoder.backbone",
        "gaf_encoder.channel_to_time",
    )
    return any(token in name for token in dense_tokens)


def _install_hooks(runner: nn.Module):
    """Install operation-count hooks on Linear, Conv, and MultiheadAttention modules."""
    totals = defaultdict(float)
    per_module = defaultdict(lambda: defaultdict(float))
    handles = []
    name_by_module = {module: name for name, module in runner.named_modules()}

    def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any):
        if not inputs or not torch.is_tensor(inputs[0]):
            return
        x = inputs[0]
        y = output[0] if isinstance(output, tuple) and torch.is_tensor(output[0]) else output
        if not torch.is_tensor(y):
            return
        name = name_by_module.get(module, module.__class__.__name__)
        batch_size = max(1, _batch_size_from_tensor(x))
        if isinstance(module, nn.Linear):
            macs = _linear_macs(module, y)
        elif isinstance(module, nn.modules.conv._ConvNd):
            macs = _conv_macs(module, y)
        elif isinstance(module, nn.MultiheadAttention):
            macs = _multihead_attention_macs(module, inputs)
        else:
            return
        density = _nonzero_density(x)
        macs_per_sample = macs / batch_size
        is_dense_only = _is_dense_only(name)
        acs_per_sample = macs_per_sample * density
        totals["dense_macs"] += macs_per_sample
        totals["sparse_acs"] += macs_per_sample if is_dense_only else acs_per_sample
        totals["dense_only_macs"] += macs_per_sample if is_dense_only else 0.0
        totals["event_eligible_macs"] += 0.0 if is_dense_only else macs_per_sample
        per_module[name]["dense_macs"] += macs_per_sample
        per_module[name]["sparse_acs"] += macs_per_sample if is_dense_only else acs_per_sample
        per_module[name]["input_density_sum"] += density
        per_module[name]["calls"] += 1

    for module in runner.modules():
        if isinstance(module, (nn.Linear, nn.modules.conv._ConvNd, nn.MultiheadAttention)):
            handles.append(module.register_forward_hook(hook))
    return totals, per_module, handles


def _multihead_attention_macs(module: nn.MultiheadAttention, inputs: tuple[Any, ...]) -> int:
    """Count projection and attention-product MACs for one attention forward pass."""
    query, key, _value = inputs[:3]
    batch_first = bool(module.batch_first)
    batch = int(query.shape[0] if batch_first else query.shape[1])
    query_length = int(query.shape[1] if batch_first else query.shape[0])
    key_length = int(key.shape[1] if batch_first else key.shape[0])
    embed_dim = int(module.embed_dim)
    heads = int(module.num_heads)
    head_dim = embed_dim // heads

    query_projection_macs = query_length * embed_dim * embed_dim
    key_value_projection_macs = 2 * key_length * embed_dim * embed_dim
    attention_product_macs = 2 * heads * query_length * key_length * head_dim
    return batch * (query_projection_macs + key_value_projection_macs + attention_product_macs)


def _estimate_aoha_attention(network: nn.Module) -> dict[str, float]:
    """Estimate AOHA attention ACs from recorded q spikes and value activations."""
    totals = defaultdict(float)
    for key, value in getattr(network, "_viz_data", {}).items():
        if not key.startswith("block_") or not isinstance(value, dict):
            continue
        q_spk = value.get("q_spk")
        attn_scores = value.get("attn_scores")
        if not torch.is_tensor(q_spk) or not torch.is_tensor(attn_scores):
            continue
        batch, steps, length, dim = q_spk.shape
        heads = attn_scores.shape[2]
        head_dim = dim // heads
        q_active = float((q_spk != 0).sum().item())
        dense_qk = batch * steps * heads * length * length * head_dim
        totals["aoha_dense_attention_macs"] += dense_qk / batch
        totals["aoha_q_gated_qk_acs"] += q_active * length / batch
        totals["aoha_q_rate_sum"] += float((q_spk != 0).float().mean().item())
        totals["aoha_blocks"] += 1
    return dict(totals)


def _build_sampled_dataset(config: dict[str, Any], split: str, max_samples: int | None):
    """Build and optionally cap a dataset split."""
    data_cfg = dict(config["data"])
    data_cfg["dataset_name"] = split
    dataset = _build_from_config(data_cfg, DATASETS)
    dataset.load()
    if max_samples is not None and max_samples > 0 and len(dataset) > max_samples:
        return Subset(dataset, range(max_samples))
    return dataset


def _build_runner(
    config: dict[str, Any],
    dataset,
    batch_size: int,
    output_dir: Path,
) -> nn.Module:
    """Create the network and runner objects needed for inference."""
    source_dataset = _base_dataset(dataset)
    network = _build_from_config(
        config["network"],
        NETWORKS,
        input_size=source_dataset.num_variables,
        max_length=source_dataset.max_seq_len,
    )

    runner_cfg = _sanitize_runner_config(config, batch_size, output_dir)
    out_size = runner_cfg.get("out_size")
    if out_size is None:
        out_size = source_dataset.num_classes
    residual_meta = getattr(source_dataset, "get_forecast_residual_meta", lambda: None)()

    return _build_from_config(
        runner_cfg,
        RUNNERS,
        network=network,
        out_size=out_size,
        forecast_horizon=config["data"].get("horizon"),
        forecast_residual_meta=residual_meta,
    )


def _run_evaluation(
    runner: nn.Module,
    dataset,
    device: torch.device,
    batch_size: int,
    max_batches: int | None,
) -> EvaluationResult:
    """Evaluate sampled batches while hooks collect operation/activity counters."""
    totals, per_module, handles = _install_hooks(runner)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    samples = 0
    evaluated_batches = 0
    firing_rates: list[float] = []
    aoha_totals = defaultdict(float)

    try:
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                data = data.to(device)
                _ = runner(data)
                samples += int(data.shape[0])
                evaluated_batches += 1

                rates = getattr(runner.network, "_rate_tensors", [])
                firing_rates.extend(float(rate.detach().cpu()) for rate in rates)
                for key, value in _estimate_aoha_attention(runner.network).items():
                    aoha_totals[key] += float(value)
    finally:
        for handle in handles:
            handle.remove()

    if samples <= 0:
        raise RuntimeError("No samples were evaluated.")

    return EvaluationResult(
        samples=samples,
        batch_count=max(1, evaluated_batches),
        operation_totals=totals,
        per_module=per_module,
        aoha_totals=aoha_totals,
        firing_rates=firing_rates,
    )


def _average_operations(result: EvaluationResult) -> OperationEstimate:
    """Convert accumulated hook counters into per-sample operation estimates."""
    batch_count = result.batch_count
    totals = result.operation_totals
    aoha_totals = result.aoha_totals

    dense_only_macs = totals["dense_only_macs"] / batch_count
    event_eligible_macs = totals["event_eligible_macs"] / batch_count
    event_eligible_acs = max(0.0, totals["sparse_acs"] / batch_count - dense_only_macs)
    aoha_dense_macs = aoha_totals["aoha_dense_attention_macs"] / batch_count
    aoha_q_gated_acs = aoha_totals["aoha_q_gated_qk_acs"] / batch_count
    dense_macs = totals["dense_macs"] / batch_count + aoha_dense_macs
    event_ops = dense_only_macs + event_eligible_acs + aoha_q_gated_acs

    return OperationEstimate(
        dense_macs=dense_macs,
        event_estimated_ops=event_ops,
        dense_only_macs=dense_only_macs,
        event_eligible_macs=event_eligible_macs,
        event_eligible_acs=event_eligible_acs,
        aoha_dense_attention_macs=aoha_dense_macs,
        aoha_q_gated_qk_acs=aoha_q_gated_acs,
    )


def _estimate_energy(
    operations: OperationEstimate,
    assumptions: EnergyAssumptions,
) -> dict[str, float]:
    """Compute dense and event-driven energy estimates in millijoules."""
    dense_energy_pj = operations.dense_macs * assumptions.mac_energy_pj
    event_ac_ops = operations.event_eligible_acs + operations.aoha_q_gated_qk_acs
    event_energy_pj = operations.dense_only_macs * assumptions.mac_energy_pj
    event_energy_pj += event_ac_ops * assumptions.ac_energy_pj
    reduction_x = dense_energy_pj / event_energy_pj if event_energy_pj > 0 else float("inf")
    saved_percent = 100.0
    if math.isfinite(reduction_x) and reduction_x > 0:
        saved_percent = 100.0 * (1.0 - 1.0 / reduction_x)

    return {
        "dense_baseline_mj": dense_energy_pj / 1e9,
        "event_estimate_mj": event_energy_pj / 1e9,
        "reduction_x": reduction_x,
        "saved_percent": saved_percent,
    }


def _build_module_rows(
    per_module: defaultdict[str, defaultdict[str, float]],
    batch_count: int,
    limit: int,
) -> list[dict[str, float | str]]:
    """Summarize the largest modules by dense MAC count."""
    module_rows = []
    for name, row in per_module.items():
        calls = max(1.0, row["calls"])
        module_rows.append(
            {
                "name": name,
                "dense_macs": row["dense_macs"] / batch_count,
                "event_estimated_ops": row["sparse_acs"] / batch_count,
                "mean_input_density": row["input_density_sum"] / calls,
                "calls_per_forward": calls / batch_count,
            }
        )
    module_rows.sort(key=lambda item: item["dense_macs"], reverse=True)
    return module_rows[:limit]


def _activity_summary(result: EvaluationResult) -> dict[str, float]:
    """Summarize recorded firing-rate and AOHA activity counters."""
    aoha_blocks = result.aoha_totals.get("aoha_blocks", 0.0)
    aoha_query_rate = 0.0
    if aoha_blocks > 0:
        aoha_query_rate = result.aoha_totals["aoha_q_rate_sum"] / max(1.0, aoha_blocks)

    return {
        "mean_recorded_firing_rate": _mean(result.firing_rates),
        "aoha_query_rate": aoha_query_rate,
    }


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean, or zero for an empty series."""
    return sum(values) / len(values) if values else 0.0


def _jsonify(value: Any) -> Any:
    """Convert tensors and Paths to JSON-compatible objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    """Write a compact human-readable energy report."""
    e = report["energy"]
    ops = report["operations_per_sample"]
    text = f"""# HybridSNN Theoretical Energy Estimate

Run: `{report["run_dir"]}`
Loaded weights: `{report["weights"]}`
Strict weight load: `{report["weight_load"]["strict"]}`
Split: `{report["split"]}` over {report["samples"]} samples

## Assumptions

- Dense multiply-accumulate: {report["assumptions"]["mac_energy_pj"]} pJ/op.
- Event-driven accumulate: {report["assumptions"]["ac_energy_pj"]} pJ/op.
- Dense-only components include analog residuals, channel mixer, fusion gate,
  GAF CNN backbone, and readout heads.
- Event-driven estimates scale eligible Linear/Conv layers by observed nonzero input density.
- AOHA q-gated attention is reported separately because it is not represented as a PyTorch module.

## Per-Sample Operation Estimate

| Quantity | Value |
| --- | ---: |
| Dense baseline MACs | {ops["dense_macs"]:.4e} |
| Event-estimated mixed ops | {ops["event_estimated_ops"]:.4e} |
| Dense-only MACs | {ops["dense_only_macs"]:.4e} |
| Event-eligible dense MACs | {ops["event_eligible_macs"]:.4e} |
| Event-eligible ACs after sparsity | {ops["event_eligible_acs"]:.4e} |
| AOHA dense attention MACs | {ops.get("aoha_dense_attention_macs", 0.0):.4e} |
| AOHA q-gated QK ACs | {ops.get("aoha_q_gated_qk_acs", 0.0):.4e} |

## Per-Sample Energy Estimate

| Quantity | Value |
| --- | ---: |
| Dense baseline energy | {e["dense_baseline_mj"]:.6f} mJ |
| Event-driven estimate | {e["event_estimate_mj"]:.6f} mJ |
| Theoretical reduction | {e["reduction_x"]:.3f}x |
| Energy saved | {e["saved_percent"]:.2f}% |

## Observed Activity

Mean recorded firing-rate tensors: {report["activity"]["mean_recorded_firing_rate"]:.4f}
AOHA query firing rate: {report["activity"].get("aoha_query_rate", 0.0):.4f}
"""
    path.write_text(text)


def estimate(args: argparse.Namespace) -> dict[str, Any]:
    """Run the estimator and return a JSON-compatible report."""
    run_dir = Path(args.run_dir).expanduser().resolve()
    config = _prepare_config(run_dir)
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    assumptions = EnergyAssumptions(
        mac_energy_pj=args.mac_energy_pj,
        ac_energy_pj=args.ac_energy_pj,
    )

    dataset = _build_sampled_dataset(config, args.split, args.max_samples)
    with tempfile.TemporaryDirectory(prefix="hybridsnn_energy_") as tmp:
        runner = _build_runner(config, dataset, args.batch_size, Path(tmp))
        weights = _load_best_weights(runner, run_dir, device)
        runner.to(device)
        runner.eval()
        if hasattr(runner.network, "record_mode"):
            runner.network.record_mode = True

        evaluation = _run_evaluation(
            runner,
            dataset,
            device,
            args.batch_size,
            args.max_batches,
        )

    operations = _average_operations(evaluation)
    report = {
        "run_dir": str(run_dir),
        "weights": weights["path"],
        "weight_load": weights,
        "split": args.split,
        "samples": evaluation.samples,
        "assumptions": {
            "mac_energy_pj": assumptions.mac_energy_pj,
            "ac_energy_pj": assumptions.ac_energy_pj,
            "method": "architecture-level theoretical estimate from observed activity",
        },
        "operations_per_sample": {
            "dense_macs": operations.dense_macs,
            "event_estimated_ops": operations.event_estimated_ops,
            "dense_only_macs": operations.dense_only_macs,
            "event_eligible_macs": operations.event_eligible_macs,
            "event_eligible_acs": operations.event_eligible_acs,
            "aoha_dense_attention_macs": operations.aoha_dense_attention_macs,
            "aoha_q_gated_qk_acs": operations.aoha_q_gated_qk_acs,
        },
        "energy": _estimate_energy(operations, assumptions),
        "activity": _activity_summary(evaluation),
        "top_modules": _build_module_rows(
            evaluation.per_module,
            evaluation.batch_count,
            args.top_modules,
        ),
    }
    return _jsonify(report)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Existing output directory containing config.json and checkpoints/",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "valid", "test"],
        help="Dataset split to sample",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--max-samples", type=int, default=64, help="Maximum samples to evaluate")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional maximum number of batches",
    )
    parser.add_argument(
        "--mac-energy-pj",
        type=float,
        default=DEFAULT_MAC_PJ,
        help="Energy per dense MAC in picojoules",
    )
    parser.add_argument(
        "--ac-energy-pj",
        type=float,
        default=DEFAULT_AC_PJ,
        help="Energy per event-driven AC in picojoules",
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--out", type=Path, default=None, help="JSON output path")
    parser.add_argument("--markdown-out", type=Path, default=None, help="Markdown output path")
    parser.add_argument(
        "--top-modules",
        type=int,
        default=20,
        help="Number of largest modules to include",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    report = estimate(args)
    text = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(report, args.markdown_out)
    print(text)


if __name__ == "__main__":
    main()
