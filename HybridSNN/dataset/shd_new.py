"""
SHD (Spiking Heidelberg Digits) Dataset loader.

Converts ragged spike-time data from shd_train.h5 into dense binary tensors
suitable for the SeqSNN pipeline.

Each sample is a dense tensor of shape (num_time_bins, num_neurons) where
entries are 1 if any spike fired in that bin, 0 otherwise.

Labels: 0-9 = English digits, 10-19 = German digits (20 classes total).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .tsforecast import DATASETS


@DATASETS.register_module()
class SHDDataset(Dataset):
    """Spiking Heidelberg Digits dataset.

    Loads spikes from shd_train.h5, bins spike times into fixed-width bins,
    and creates dense binary tensors.

    Args:
        file: path to shd_train.h5
        num_time_bins: number of temporal bins (= sequence length L)
        num_neurons: number of neuron IDs (= num_variables / C)
        train_ratio: fraction of samples for training
        test_ratio: fraction of samples for testing (validation = 1 - train - test)
        dataset_name: one of "train", "valid", "test"
        max_time: maximum spike time in seconds (SHD uses ~0.7s recordings)
    """

    def __init__(
        self,
        file: str,
        num_time_bins: int = 100,
        num_neurons: int = 700,
        train_ratio: float = 0.8,
        test_ratio: float = 0.1,
        dataset_name: str = "train",
        max_time: float = 1.4,
        adaptive_binning: bool = False,
        spike_count: bool = False,
        test_file: str = None,
    ):
        self.file = file
        self.num_time_bins = num_time_bins
        self.num_neurons = num_neurons
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.dataset_name = dataset_name
        self.max_time = max_time
        self.adaptive_binning = adaptive_binning
        self.spike_count = spike_count
        # When test_file is set, use it as the dedicated test split, since we do some tuning here
        self._active_file = test_file if (test_file and dataset_name == "test") else file

        self._data = None   # dense tensors, loaded in load()
        self._labels = None

        # Load indices during __init__ to expose num_variables etc.
        self._load_index()

    def _load_index(self):
        """Load only labels and compute split indices (fast)."""
        import h5py
        with h5py.File(self._active_file, "r") as f:
            labels = f["labels"][:]
        self._all_labels = labels.astype(np.int64)
        n_total = len(self._all_labels)

        # If using a dedicated test file, use all its samples
        if self._active_file != self.file:
            self._indices = np.arange(n_total)
            return

        n_train = int(n_total * self.train_ratio)
        n_test = int(n_total * self.test_ratio)
        n_valid = n_total - n_train - n_test

        if self.dataset_name == "train":
            self._indices = np.arange(0, n_train)
        elif self.dataset_name == "valid":
            self._indices = np.arange(n_train, n_train + n_valid)
        elif self.dataset_name == "test":
            self._indices = np.arange(n_train + n_valid, n_total)
        else:
            raise ValueError(f"dataset_name must be 'train', 'valid', or 'test', got {self.dataset_name!r}")

    def load(self):
        """Load and preprocess spike data into dense tensors."""
        if self._data is not None:
            return  # already loaded

        import h5py
        with h5py.File(self._active_file, "r") as f:
            times_ds = f["spikes/times"]
            units_ds = f["spikes/units"]

            dense_list = []
            for idx in self._indices:
                spike_times = times_ds[idx].astype(np.float32)
                spike_units = units_ds[idx].astype(np.int64)

                # Determine max time for this sample (adaptive) or fixed
                if self.adaptive_binning and len(spike_times) > 0:
                    sample_max_time = float(spike_times.max()) + 1e-6
                else:
                    sample_max_time = self.max_time

                # Bin spike times
                bin_edges = np.linspace(0.0, sample_max_time, self.num_time_bins + 1)
                bin_ids = np.digitize(spike_times, bin_edges[1:-1])  # shape: (num_spikes,)
                bin_ids = np.clip(bin_ids, 0, self.num_time_bins - 1)

                # Clip neuron IDs
                valid_mask = spike_units < self.num_neurons
                bin_ids = bin_ids[valid_mask]
                spike_units_clipped = spike_units[valid_mask]

                # Build dense matrix (num_time_bins, num_neurons)
                dense = np.zeros((self.num_time_bins, self.num_neurons), dtype=np.float32)
                if self.spike_count:
                    # Float spike counts per bin (multi-spike collisions preserved)
                    np.add.at(dense, (bin_ids, spike_units_clipped), 1.0)
                else:
                    # Binary: 1 if any spike in bin
                    dense[bin_ids, spike_units_clipped] = 1.0
                dense_list.append(dense)

        self._data = np.stack(dense_list, axis=0)         # (N_split, T, C)
        self._labels = self._all_labels[self._indices]     # (N_split,)

    def freeup(self):
        """Release loaded data from memory."""
        self._data = None
        self._labels = None

    def get_index(self):
        return np.arange(len(self._indices))

    @property
    def num_variables(self) -> int:
        """Number of input features (= num_neurons = C)."""
        return self.num_neurons

    @property
    def max_seq_len(self) -> int:
        """Sequence length (= num_time_bins = L)."""
        return self.num_time_bins

    @property
    def num_classes(self) -> int:
        """Number of output classes."""
        return 20

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int):
        """
        Returns:
            x: float32 tensor of shape (num_time_bins, num_neurons) = (L, C)
            y: int64 label scalar tensor
        """
        if self._data is None:
            raise RuntimeError("Call SHDDataset.load() before iterating.")
        x = torch.from_numpy(self._data[index])       # (L, C) float32
        y = torch.tensor(self._labels[index], dtype=torch.long)
        return x, y
