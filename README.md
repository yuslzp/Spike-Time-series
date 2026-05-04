# HybridSNN

HybridSNN is a spiking sequence model for multivariate time-series forecasting and spike-based classification. The current method combines a delta-convolutional spike encoder, TS-LIF neurons, and an Addition-Only Hybrid Attention (AOHA) block. The same backbone is used for forecasting datasets such as METR-LA and Electricity, and for SHD classification through a task-specific readout.

The current development priority is `SHD` and `METR-LA`. `electricity` is mainly used as a sanity-check dataset. The preferred encoder direction is `delta_conv`, with `merged` as a secondary path and `gaf` retained mainly for ablations.

## Repository Contents

```text
HybridSNN/
├── HybridSNN/
│   ├── entry/run.py              # CLI entry point: config -> train -> predict -> analysis
│   ├── dataset/                  # TSMSDataset and SHDDataset
│   ├── module/                   # encoders, TS-LIF neuron, AOHA, visualization helpers
│   ├── network/model.py          # HybridSNN backbone
│   ├── runner/                   # training, evaluation, EMA, checkpointing
│   └── visualization/            # post-hoc plots and internal spike diagnostics
├── exp/                          # reproducible YAML configs
└── requirements.txt              # pinned dependency set
```

## Method Summary

```text
Input sequence (B, L, C)
        |
        v
Encoder: delta_conv / merged / gaf
        |
        v
Projection + TS-LIF initialization
        |
        v
HybridBlock stack
  - AOHA: Q binary spikes, K ReLU analog, V ternary spikes
  - spiking MLP
        |
        v
Task readout
  - forecasting: HorizonDecoder + optional residual baseline
  - SHD: attention pooling + classifier
```

Training uses task loss plus spike-rate regularization and encoder branch-balancing terms. The runner also supports warmup-cosine scheduling, gradient clipping, EMA evaluation, periodic visualization, and saved prediction artifacts.

## Reproducible Setup

The recommended environment is Python 3.10 with the pinned packages in [requirements.txt](requirements.txt). The provided requirements file uses the CUDA 12.1 PyTorch wheel.

```bash
conda create -n SeqSNN python=3.10 -y
conda activate SeqSNN

pip install -r requirements.txt
pip install -e . --no-deps
```

For CPU-only machines or a different CUDA version, install the matching PyTorch wheel first from the official PyTorch selector, then install the remaining packages from `requirements.txt` after editing or removing the `torch==2.5.1+cu121` line.

Verify the install:

```bash
python - <<'PY'
import torch
import HybridSNN
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("HybridSNN import ok")
PY
```

## Dataset Setup

Place the datasets in the repository root, or update the `file:` paths in the YAML files under `exp/forecast/dataset/` and `exp/classify/hybrid_snn/`.

| File | Used by | Notes |
| --- | --- | --- |
| `LD2011_2014_processed.txt` | Electricity | 370 clients, 15-minute intervals |
| `METR-LA.h5` | METR-LA | 207 traffic sensors, 5-minute intervals |
| `shd_train.h5` | SHD train/valid | Spiking Heidelberg Digits |
| `shd_test.h5` | SHD test | Dedicated SHD test file |

Download sources:

- SHD: `https://zenkelab.org/datasets/`
- METR-LA: `https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset`
- Electricity: `https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014`

Path examples:

```yaml
# exp/forecast/dataset/electricity-v2.yml
data:
  file: LD2011_2014_processed.txt

# exp/forecast/dataset/metr-la-v2.yml
data:
  file: METR-LA.h5

# exp/classify/hybrid_snn/hybrid_snn_shd_run.yml
data:
  file: shd_train.h5
  test_file: shd_test.h5
```

## Quick Start

Run a short smoke test on Electricity. This uses a fixed seed, disables W&B, and writes to `outputs/smoke_electricity_h24`.

```bash
bash smoke_test.sh --horizon 24
```

Expected output directory:

```text
outputs/smoke_electricity_h24/
├── checkpoints/
├── tb/
├── config.json
└── analysis/
```

## Example Runs

All examples below use explicit output directories and seeds. This makes runs easier to compare and avoids overwriting previous results.

### Electricity Forecasting

```bash
WANDB_MODE=disabled python -m HybridSNN.entry.run \
  exp/forecast/hybrid_snn/hybrid_snn_electricity_run.yml \
  --runtime.use_cuda true \
  --runtime.seed 40 \
  --runtime.output_dir outputs/repro/electricity_h24_delta_conv_seed40 \
  --data.horizon 24 \
  --network.encoder_type delta_conv \
  --runner.wandb_enabled false
```

### METR-LA Forecasting

```bash
WANDB_MODE=disabled python -m HybridSNN.entry.run \
  exp/forecast/hybrid_snn/hybrid_snn_metr-la_run.yml \
  --runtime.use_cuda true \
  --runtime.seed 40 \
  --runtime.output_dir outputs/repro/metr-la_h6_delta_conv_seed40 \
  --data.horizon 6 \
  --network.encoder_type delta_conv \
  --runner.wandb_enabled false
```

### SHD Classification

```bash
WANDB_MODE=disabled python -m HybridSNN.entry.run \
  exp/classify/hybrid_snn/hybrid_snn_shd_run.yml \
  --runtime.use_cuda true \
  --runtime.seed 40 \
  --runtime.output_dir outputs/repro/shd_delta_conv_seed40 \
  --network.encoder_type delta_conv \
  --runner.wandb_enabled false
```

### Matrix Runner

Use the matrix runner for reproducible seed, horizon, and encoder sweeps:

```bash
bash run_experiment_matrix.sh \
  --datasets metr-la,electricity,shd \
  --horizons 6,24 \
  --encoders delta_conv \
  --seeds 40,41,42 \
  --gpus 0,1,2,3 \
  --slots-per-gpu 2 \
  --wandb-enabled false \
  --run-root outputs/repro_matrix \
  --log-dir outputs/repro_matrix_logs
```


## Reproducibility Checklist

For each reported run, record:

- Git commit or working-tree diff.
- Full command line.
- Resolved `config.json` saved inside the output directory.
- `runtime.seed`.
- Dataset file paths and preprocessing settings.
- Output directory containing `checkpoints/res.json`.
- Saved prediction files: `train_pre.pkl`, `valid_pre.pkl`, and `test_pre.pkl`.
- Analysis artifacts under `analysis/` and spike visualizations under `viz/` when enabled.

Important metric caveat: when `ema_eval: true`, verify metrics from `checkpoints/train_pre.pkl`, `valid_pre.pkl`, and `test_pre.pkl` in addition to `checkpoints/res.json` and logs. Some runner paths can underreport forecasting and SHD test metrics in `res.json` or stdout because the saved best-model predictions are the more reliable source.

## Outputs

Each run writes to `runtime.output_dir`:

```text
outputs/my_run/
├── checkpoints/
│   ├── model_best.pkl
│   ├── network_best.pkl
│   ├── res.json
│   ├── train_pre.pkl
│   ├── valid_pre.pkl
│   ├── test_pre.pkl
│   └── resume.pth
├── tb/
├── config.json
├── stdout.log
├── analysis/
└── viz/
```

Forecasting metrics are `loss`, `r2`, and `rrse`. SHD uses classification accuracy. Benchmark targets are summarized in [benchmark.md](benchmark.md).


## Results plots
Experiment result plots are under [final_plots](final_plots). The structure looks like this

- final_plots/README.md: source notes, contents, counts.
- final_plots/summary: metric CSVs and comparison PNGs.
- final_plots/final_method: final-method analysis and selected final-epoch diagnostics.
- final_plots/ablations: ablation analysis plots.
- the sub-directory have the files for each experiments.


## Troubleshooting

- Disable W&B in offline or unauthenticated environments with `WANDB_MODE=disabled` and `--runner.wandb_enabled false`.
- If CUDA memory is unstable, reduce `--slots-per-gpu`, lower `runner.batch_size`, or avoid `gaf` and `merged` runs.
- If a run produces `NaN` weights, do not resume from that checkpoint. Start from a fresh output directory.
- If dataset files are outside the repo root, use absolute paths in the dataset YAMLs or pass CLI overrides such as `--data.file /path/to/METR-LA.h5`.
- For large forecasting splits, logged train/valid metrics may be capped by tracker limits. Use saved prediction files for final reporting.
