from typing import Any, Optional, Union, Iterable
from numbers import Number
import time
import datetime
from collections import namedtuple
import copy

import numpy as np
import torch


# Environment Classes
Obs = namedtuple(
    "Obs",
    (
        "date",
        "t",
        "target",
        "position",
        "sell",
        "p",  # vwap price at t-1
        "v",  # market volume at t-1
    ),
)


def pprint(*args):
    """Print a UTC+8 timestamp prefix followed by the provided arguments."""
    time = (
        "[" + str(datetime.datetime.utcnow() + datetime.timedelta(hours=8))[:19] + "] -"
    )
    print(time, *args, flush=True)


def to_torch(
    x: Any,
    dtype: Optional[torch.dtype] = None,
    device: Union[str, int, torch.device] = "cpu",
) -> Optional[Union[torch.Tensor, Iterable]]:
    """Return an object without np.ndarray."""
    if isinstance(x, np.ndarray) and issubclass(x.dtype.type, (np.bool_, np.number)):
        x = torch.from_numpy(x).to(device, non_blocking=True)  # type: ignore
        if dtype is not None:
            x = x.type(dtype)
        return x
    elif isinstance(x, torch.Tensor):  # second often case
        if dtype is not None:
            x = x.type(dtype)
        return x.to(device, non_blocking=True)  # type: ignore
    elif isinstance(x, (np.number, np.bool_, Number)):
        return to_torch(np.asanyarray(x), dtype, device)
    elif isinstance(x, dict):
        return {k: to_torch(v, dtype, device) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return (to_torch(i, dtype, device) for i in x)
    else:  # fallback
        raise TypeError(f"object {x} cannot be converted to torch.")


class Timer():
    """Measure elapsed wall-clock time within a context manager."""

    def __init__(self, message=None):
        """Store an optional message printed when the timer starts."""
        self.message = message

    def __enter__(self):
        """Start timing and optionally print the configured message."""
        if self.message is not None:
            print(self.message, end="\t")
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Print the elapsed wall-clock time when leaving the context."""
        print(time.time() - self.start_time)


# Evaluation Metrics


class MovingAverage():
    """Track an exponential moving average while ignoring NaN entries."""

    def __init__(self, decay, init_val=0, shape=None):
        """Initialize the decay factor and starting average value."""
        self.decay = decay
        if type(init_val) == int:
            self.value = init_val * np.ones(shape)
        else:
            self.value = init_val

    def add(self, val):
        """Update the moving average for all finite entries in `val`."""
        mask = np.isnan(val)
        self.value[~mask] = (1 - self.decay) * self.value[~mask] + self.decay * val[
            ~mask
        ]


class AverageMeter():
    """Accumulate scalar statistics such as running loss averages."""

    def __init__(self):
        """Initialize the accumulator and reset its counters."""
        self.reset()

    def reset(self):
        """Reset the tracked value, sum, average, and count to zero."""
        self.val = 0.0
        self.sum = 0.0
        self.avg = 0.0
        self.count = 0

    def update(self, val, n=1):
        """Add a new scalar value weighted by `n` observations."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = 1.0 * self.sum / self.count

    def performance(self, care="avg"):
        """Return one tracked statistic such as `avg` or `sum`."""
        return getattr(self, care)

    def status(self):
        """Return the default string representation of the tracked statistic."""
        return str(self.performance())


class GlobalMeter():
    """Store full prediction and target arrays for a single metric function."""

    def __init__(self, f=lambda x, y: 0):
        """Initialize the metric callback and empty storage buffers."""
        self.reset()
        self.f = f

    def reset(self):
        """Clear all cached predictions and targets."""
        self.ys = []  # np.array([], dtype=np.int) # ground truths
        self.preds = []  # np.array([], dtype=np.float) # predictions

    def update(self, ys, preds):
        """Append a batch of targets and predictions to the buffers."""
        if isinstance(ys, torch.Tensor):
            ys = ys.detach().squeeze(-1).cpu().numpy()
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().squeeze(-1).float().cpu().numpy()
        assert isinstance(ys, np.ndarray) and isinstance(
            preds, np.ndarray
        ), "Please input as type of ndarray."
        self.ys.append(ys)
        self.preds.append(preds)

    def concat(self):
        """Concatenate buffered batches into full target and prediction arrays."""
        if isinstance(self.ys, list) and isinstance(self.preds, list):
            self.ys = [
                np.expand_dims(ys, 0) if len(ys.shape) == 0 else ys for ys in self.ys
            ]
            self.preds = [
                np.expand_dims(preds, 0) if len(preds.shape) == 0 else preds
                for preds in self.preds
            ]
            self.ys = np.concatenate(self.ys, axis=0)
            self.preds = np.concatenate(self.preds, axis=0)

    def get_ys(self):
        """Return all cached targets as one concatenated array."""
        # deprecated
        return np.concatenate(self.ys, axis=0)

    def get_preds(self):
        """Return all cached predictions as one concatenated array."""
        # deprecated
        return np.concatenate(self.preds, axis=0)

    def performance(self):
        """Compute the metric over all cached targets and predictions."""
        return self.f(self.ys, self.preds)

    def status(self):
        """Return the metric value as a string."""
        return str(self.performance())


class AverageTracker():
    """Manage a fixed set of `AverageMeter` instances keyed by metric name."""

    def __init__(self, metrics):
        """Create one average tracker per metric name."""
        self.metrics = metrics  # isolated metric list to guarantee metric order
        self.trackers = {}
        self.ss = {}  # snapshot status
        for m in self.metrics:
            self.trackers[m] = AverageMeter()

    def update(self, metric, val, n=1):
        """Update one named metric with a new value."""
        try:
            meter = self.trackers[metric]
        except Exception:
            raise KeyError("Metric has not been found. %s" % metric)
        meter.upate(val, n)

    def get(self, metric, care="avg"):
        """cared_value"""
        assert metric in self.metrics, "Metric %s not found." % metric
        return getattr(self.trackers[metric], care)

    def performance(self, metric="all", care="avg"):
        """{metric: cared_value}"""
        stat = {}
        if isinstance(metric, str) and isinstance(care, str):
            assert (metric == "all") or (metric in self.metrics), (
                "Not support %s metric." % metric
            )
            assert care in ["val", "avg", "sum", "count"], (
                "Not support %s in performance meter." % care
            )
            if metric == "all":
                for m in self.metrics:
                    stat[m] = getattr(self.trackers[m], care)
            else:
                stat[metric] = getattr(self.trackers[metric], care)
        else:
            # TODO metrics=[m1, m2, ...] care=[c1, c2, ...]
            # TODO metrics==[m1, m2, ...] care=care
            raise NotImplementedError("TODO")
        return stat

    def _snapshot(self):
        """Refresh the performance"""
        stat = self.performance()
        self.ss = stat
        return self.ss

    def snapshot_metric(self, metric):
        """Return the latest performance of the given metric without refresh"""
        assert metric in self.metrics, "Metric %s not found." % metric
        if len(self.ss) == 0:
            self._snapshot()
        return self.ss[metric]

    def snapshot(self):
        """Return the cached average metrics, refreshing them if needed."""
        # assert len(self.snapshot) > 0, "Please update Tracker.performance() first!"
        if len(self.ss) == 0:
            return self._snapshot()
        return self.ss

    def status(self):
        """Refresh and return all the performance"""
        self.snapshot()
        return "\t".join([str(self.ss[m]) for m in self.metrics])

    def __str__(self):
        return self.status()


class GlobalTracker(GlobalMeter):
    """Track full-history metrics plus streaming regressions/classification stats."""

    def __init__(self, metrics, metric_fn):
        """Initialize named metrics and their callable implementations."""
        self.metrics = metrics
        self.metric_fn = metric_fn
        self.ss = {}
        self.reset()

    def reset(self):
        """Clear cached outputs and all streaming metric accumulators."""
        self.ys = []
        self.preds = []
        self.ss = {}
        self._sum_sq_error = 0.0
        self._sum_y = 0.0
        self._sum_y_sq = 0.0
        self._count_values = 0
        self._sum_y_dim = None
        self._sum_y_sq_dim = None
        self._count_rows = 0
        self._correct = 0
        self._total = 0

    @staticmethod
    def _to_numpy(arr):
        """Convert tensors and array-like inputs to NumPy arrays."""
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    def update(self, ys, preds):
        """Update streaming statistics and optionally store full outputs."""
        ys = self._to_numpy(ys)
        preds = self._to_numpy(preds)

        tracked = False
        if any(metric in {"r2", "rrse"} for metric in self.metrics):
            ys_reg = ys.reshape(ys.shape[0], -1) if ys.ndim > 1 else ys.reshape(-1, 1)
            preds_reg = preds.reshape(ys_reg.shape[0], -1)
            diff = preds_reg - ys_reg
            self._sum_sq_error += float(np.square(diff).sum())
            self._sum_y += float(ys_reg.sum())
            self._sum_y_sq += float(np.square(ys_reg).sum())
            if self._sum_y_dim is None:
                self._sum_y_dim = np.zeros(ys_reg.shape[1], dtype=np.float64)
                self._sum_y_sq_dim = np.zeros(ys_reg.shape[1], dtype=np.float64)
            self._sum_y_dim += ys_reg.sum(axis=0, dtype=np.float64)
            self._sum_y_sq_dim += np.square(ys_reg).sum(axis=0, dtype=np.float64)
            self._count_values += ys_reg.size
            self._count_rows += ys_reg.shape[0]
            tracked = True

        if "accuracy" in self.metrics:
            ys_cls = ys.reshape(-1)
            preds_cls = preds.reshape(len(ys_cls), -1)
            if preds_cls.shape[-1] == 1:
                pred_labels = (preds_cls.squeeze(-1) > 0).astype(np.int64)
            else:
                pred_labels = np.argmax(preds_cls, axis=1)
            self._correct += int((pred_labels == ys_cls).sum())
            self._total += len(ys_cls)
            tracked = True

        if not tracked:
            super().update(ys, preds)

    def concat(self):
        """Concatenate stored batches when full-history metrics require them."""
        if isinstance(self.ys, list) and self.ys:
            self.ys = [
                np.expand_dims(ys, 0) if len(ys.shape) == 0 else ys for ys in self.ys
            ]
            self.preds = [
                np.expand_dims(preds, 0) if len(preds.shape) == 0 else preds
                for preds in self.preds
            ]
            self.ys = np.concatenate(self.ys, axis=0)
            self.preds = np.concatenate(self.preds, axis=0)

    def _performance_from_accumulators(self, metric: str):
        """Compute accuracy, R^2, or RRSE from streaming accumulators."""
        eps = np.finfo(np.float64).eps
        if metric == "accuracy":
            return 0.0 if self._total == 0 else self._correct / self._total
        if metric == "r2":
            if self._count_values == 0:
                return 0.0
            total = self._sum_y_sq - (self._sum_y ** 2) / max(1, self._count_values)
            total = max(float(total), eps)
            return 1.0 - float(self._sum_sq_error) / total
        if metric == "rrse":
            if self._count_rows == 0 or self._sum_y_dim is None or self._sum_y_sq_dim is None:
                return 0.0
            denom = self._sum_y_sq_dim - (self._sum_y_dim ** 2) / max(1, self._count_rows)
            denom = max(float(denom.sum()), eps)
            return float(np.sqrt(self._sum_sq_error) / np.sqrt(denom))
        raise KeyError(metric)

    def performance(self, metric="all"):
        """Compute one named metric or the full tracked metric dictionary."""
        stat = {}
        if isinstance(metric, str):
            assert (metric == "all") or (metric in self.metrics), (
                "Not support %s metric." % metric
            )
            if metric == "all":
                for m in self.metrics:
                    if m in {"accuracy", "r2", "rrse"}:
                        res = self._performance_from_accumulators(m)
                    else:
                        res = self.metric_fn[m](self.ys, self.preds)
                    if hasattr(res, "item"):
                        res = res.item()
                    stat[m] = res
                    self.ss[m] = stat[m]
            else:
                if metric in {"accuracy", "r2", "rrse"}:
                    res = self._performance_from_accumulators(metric)
                else:
                    res = self.metric_fn[metric](self.ys, self.preds)
                if hasattr(res, "item"):
                    res = res.item()
                stat[metric] = res
                self.ss[metric] = stat[metric]
        else:
            raise NotImplementedError("TODO")
        return stat

    def snapshot(self, metric="all"):
        """Return the last computed metric snapshot without recomputation."""
        stat = {}
        if isinstance(metric, str):
            assert (metric == "all") or (metric in self.metrics), (
                "Not support %s metric." % metric
            )
            if metric == "all":
                for m in self.metrics:
                    try:
                        stat[m] = self.metric_fn[m]
                    except Exception:
                        raise KeyError("Run performance first")
            else:
                try:
                    stat[metric] = self.ss[metric]
                except Exception:
                    raise KeyError("Run performance first")
        else:
            raise NotImplementedError("TODO")
        return stat


def __deepcopy__(self, memo={}):
    """Create a best-effort deepcopy for tracker-like objects."""
    cls = self.__class__
    copyobj = cls.__new__(cls)
    memo[id(self)] = copyobj
    for attr, value in self.__dict__.items():
        try:
            setattr(copyobj, attr, copy.deepcopy(value, memo))
        except Exception:
            pass
    return copyobj
