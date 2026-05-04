import warnings
from pathlib import Path

from utilsd import get_checkpoint_dir, get_output_dir, setup_experiment
from utilsd.config import PythonConfig, RegistryConfig, RuntimeConfig, configclass
from utilsd.experiment import print_config

from HybridSNN.dataset import DATASETS
from HybridSNN.network import NETWORKS
from HybridSNN.runner import RUNNERS
from HybridSNN.visualization.analysis import generate_posthoc_analysis

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parents[2]


@configclass
class SeqSNNConfig(PythonConfig):
    """Bundle dataset, network, runner, and runtime config registries."""

    data: RegistryConfig[DATASETS]
    network: RegistryConfig[NETWORKS]
    runner: RegistryConfig[RUNNERS]
    runtime: RuntimeConfig = RuntimeConfig()


def _resolve_repo_path(path_str: str | None) -> str | None:
    """Resolve a config path relative to the repository root when needed."""
    if not path_str:
        return path_str
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _resolve_repo_path_obj(path_value) -> Path | None:
    """Resolve a config path and return it as a `Path` object."""
    if path_value is None:
        return None
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _normalize_paths(config) -> None:
    """Normalize dataset and output paths in the experiment config."""
    for attr in ("file", "test_file"):
        value = getattr(config.data, attr, None)
        if value is not None:
            setattr(config.data, attr, _resolve_repo_path(value))

    output_dir = getattr(config.runtime, "output_dir", None)
    if output_dir is not None:
        config.runtime.output_dir = _resolve_repo_path_obj(output_dir)


def run_train(config):
    """Run training, prediction export, and post-hoc analysis for one config."""
    _normalize_paths(config)
    if getattr(config.runtime, "output_dir", None) is not None:
        Path(config.runtime.output_dir).parent.mkdir(parents=True, exist_ok=True)
    setup_experiment(config.runtime)
    print_config(config)

    trainset = config.data.build(dataset_name="train")
    validset = config.data.build(dataset_name="valid")
    testset = config.data.build(dataset_name="test")

    network = config.network.build(input_size=trainset.num_variables, max_length=trainset.max_seq_len)
    runner = config.runner.build(
        network=network,
        output_dir=get_output_dir(),
        checkpoint_dir=get_checkpoint_dir(),
        out_size=config.runner.out_size or trainset.num_classes,
        forecast_horizon=getattr(config.data, "horizon", None),
        forecast_residual_meta=getattr(trainset, "get_forecast_residual_meta", lambda: None)(),
    )

    runner.fit(trainset, validset, testset)
    train_pred = runner.predict(trainset, "train")
    valid_pred = runner.predict(validset, "valid")
    test_pred = runner.predict(testset, "test")

    try:
        generate_posthoc_analysis(runner, validset, valid_pred, get_output_dir(), split_name="valid")
        generate_posthoc_analysis(runner, testset, test_pred, get_output_dir(), split_name="test")
    except Exception as exc:
        print(f"Post-hoc analysis skipped: {exc}")

    return {
        "train": train_pred,
        "valid": valid_pred,
        "test": test_pred,
    }


if __name__ == "__main__":
    run_train(SeqSNNConfig.fromcli())
