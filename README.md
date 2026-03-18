# HybridSNN

A Spiking Neural Network (SNN) for multivariate time-series forecasting combining a dual-pathway spike encoder with an Addition-Only Hybrid Attention (AOHA) mechanism.

## Architecture overview

```
Input (B, L, C)
      │
      ▼
┌─────────────────────────────┐
│  DeltaConvEncoder            │   delta branch (Δx → BN → Linear → Leaky)
│  or GAFEncoder               │ + conv branch  (ConvEncoder)
└─────────────────────────────┘
      │  (B, T, C, L)
      ▼
  Linear projection → init LIF   → (B, T, L, dim)
      │
      ▼  × depths
┌─────────────────────────────┐
│  HybridBlock                 │
│    AOHA  (Q binary, K ReLU, V ternary)
│    Spiking MLP               │
└─────────────────────────────┘
      │  (B, T, L, dim)
      ▼
  mean over T → seq_out (B, L, dim)
  mean over L → emb_out (B, dim)
      │
      ▼
  MLP head → forecast (B, horizon × C)
```

**Loss:** MSE + `spike_lambda × mean_firing_rate` (spike-rate regularization)

---

## Project structure

```
HybridSNN/
├── entry/
│   └── run.py                 # entry point; parses YAML → trains → predicts
├── dataset/
│   ├── tsforecast.py          # TSMSDataset — sliding-window loader for .txt/.h5/.csv
│   └── shd.py                 # SHDDataset  — Spiking Heidelberg Digits (classification)
├── network/
│   └── model.py               # HybridSNN model (registered as "HybridSNN")
├── module/
│   ├── hybrid_attention.py    # HybridBlock, AOHA, TernaryNode
│   ├── gaf_encoding.py        # GAFEncoder — Gramian Angular Field spike encoder
│   └── encoder.py             # ConvEncoder — conv-based spike encoder
├── runner/
│   ├── base.py                # BaseRunner: training loop, AMP, early stopping
│   ├── runner.py              # TS runner + HybridTS runner (spike-rate loss, viz)
│   └── utils.py               # reset_states() for snntorch hidden state
├── common/
│   ├── utils.py
│   └── function.py
└── visualization/
    ├── plots.py
    └── viz_runner.py
```

Config files live outside the package:

```
exp/forecast/
├── hybrid_snn/
│   ├── hybrid_snn_electricity_run.yml   # electricity experiment
│   └── hybrid_snn_metr-la_run.yml       # METR-LA experiment
├── model/
│   └── hybrid_snn.yml                   # model hyperparameters base
└── dataset/
    ├── electricity-v2.yml               # electricity data path + settings
    └── metr-la-v2.yml                   # METR-LA data path + settings
```

---

## Setup

### 1. Create the conda environment

```bash
conda create -n SeqSNN python=3.10
conda activate SeqSNN
```

### 2. Install dependencies

From the project root:

```bash
pip install -e .
```

This installs `HybridSNN` as an editable package along with all required dependencies:
`torch`, `snntorch`, `spikingjelly`, `utilsd`, `pandas==2.1`, `scikit_learn==1.3`, `tensorboard`, `wandb`, `h5py`, `numba`, `tables`, and others.

### 3. Verify the datasets are in place

| File | Dataset |
|------|---------|
| `LD2011_2014_processed.txt` | Electricity (370 clients, 15-min intervals) |
| `METR-LA.h5` | Traffic speed (207 sensors, 5-min intervals) |
| `shd_train.h5` + `shd_test.h5` | Spiking Heidelberg Digits (classification) |


You can download the SHD datasets from https://zenkelab.org/datasets/, METR-LA.h5: https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset, and the electricity dataset here: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014.

Update the `file:` path in the dataset YAMLs if files are stored elsewhere:

```bash
# exp/forecast/dataset/electricity-v2.yml
data:
  file: /path/to/LD2011_2014_processed.txt

# exp/forecast/dataset/metr-la-v2.yml
data:
  file: /path/to/METR-LA.h5
```

---

## Running experiments

### Smoke test (3 epochs, no wandb, worst-case h=96)

```bash
bash smoke_test.sh              # default horizon 96
bash smoke_test.sh --horizon 24
```

### Single experiment

```bash
python -m HybridSNN.entry.run \
    exp/forecast/hybrid_snn/hybrid_snn_electricity_run.yml \
    --runtime.use_cuda true \
    --runtime.seed 42 \
    --data.horizon 24 \
    --runtime.output_dir outputs/my_run
```

### Multi-seed / multi-GPU sweep

```bash
# 3 seeds × 2 datasets across available GPUs
bash run_experiments_24.sh --wandb-project HybridSNN --horizon 24

# 8-GPU layout (electricity + metr-la + shd, 3 seeds)
bash run_experiments_8gpu.sh --wandb-project HybridSNN --horizon 24
```

Logs are written to `outputs/logs_h<N>/`.

---

## Key hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dim` | 128 | Hidden dimension |
| `d_ff` | 256 | MLP feedforward size |
| `heads` | 8 | Attention heads |
| `depths` | 2 | Number of HybridBlock layers |
| `num_steps` | 4 | SNN simulation steps (T) |
| `encoder_type` | `delta_conv` | `delta_conv` or `gaf` |
| `spike_lambda` | 0.005 | Weight for spike-rate regularization loss |
| `lr` | 1e-4 | Learning rate |
| `batch_size` | 48 | Batch size (reduce if OOM) |
| `window` | 168 | Input sequence length (look-back) |
| `horizon` | 24 | Forecast horizon |

Override any parameter on the command line with `--section.key value`, e.g.:

```bash
python -m HybridSNN.entry.run exp/forecast/hybrid_snn/hybrid_snn_electricity_run.yml \
    --network.depths 4 \
    --network.dim 256 \
    --runner.lr 5e-5 \
    --data.horizon 96
```

---

## Outputs

Each run writes to `runtime.output_dir`:

```
outputs/my_run/
├── checkpoints/
│   ├── model_best.pkl      # best model weights
│   ├── network_best.pkl    # best network weights
│   ├── res.json            # final metrics (train/valid/test)
│   ├── train_pre.pkl       # train set predictions
│   ├── valid_pre.pkl       # validation set predictions
│   ├── test_pre.pkl        # test set predictions
│   └── resume.pth          # checkpoint for resuming
├── tb/                     # TensorBoard event files
├── config.json             # resolved config snapshot
└── stdout.log              # training log
```

Metrics reported: `loss` (MSE), `r2` (R²), `rrse` (Relative Root Square Error).


## Visualizations

You can checkout the visualizations for the milestone report at `./plots`, the structure looks like this

```
plots
├── electricity_h24/        # Electricity Dataset
├── metrla_h24/             # METR-LA (traffic) Dataset
├── shd/                    # Spiking Heidelberg Digits
```

Some plots for each dataset would have different timestep because we are running with EarlyStop with 30 limited patience. We are investigating why electricity_h24 with seed 40 performs strictly worse than the other two, and for the report we only includede seed 41 and 42 result for reporting.

---

## Datasets

### Electricity (`TSMSDataset`)
- 140,256 timesteps × 370 clients, 15-minute intervals (2011–2014)
- Preprocessing: global z-score normalization (`normalize: 3`); zeros left as-is
- Split: 60% train / 20% valid / 20% test

### METR-LA (`TSMSDataset`)
- 34,272 timesteps × 207 sensors, 5-minute intervals
- Values of 0 indicate missing / sensor error
- Split: 70% train / 20% valid / 10% test

### SHD (`SHDDataset`)
- 8,156 training samples, 20 classes (English + German spoken digits)
- Each sample: ragged spike arrays → binned into dense `(num_time_bins, num_neurons)` tensor
- Default: 100 time bins, 700 neurons
- Requires `shd_train.h5` (and optionally `shd_test.h5` for the dedicated test split)
