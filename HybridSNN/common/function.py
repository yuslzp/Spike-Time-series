import re
import time
from numba import njit, prange

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
    r2_score,
    precision_recall_curve,
)
from sklearn.metrics import auc as area_under_curve


EPS = 1e-5


def printt(s=None):
    """Print a value followed by a tab, or a blank line when `s` is `None`."""
    if s is None:
        print()
    else:
        print(str(s), end="\t")


def format_time(t):
    """Format a Unix timestamp as `MMDDHHMMSS`."""
    return time.strftime("%m%d%H%M%S", time.localtime(t))

# calculated weighted average, replace nan with 0.
def nan_weighted_avg(vals, weights, axis=None):
    """Compute a weighted average while ignoring NaN values in either array."""
    assert vals.shape == weights.shape
    vals = vals.copy()
    weights = weights.copy()
    is_valid = np.logical_and(~np.isnan(vals), ~np.isnan(weights))
    # if some of the value are still nan, return nan directly
    if not np.any(is_valid):
        return np.nan
    weights[~is_valid] = 0
    vals[~is_valid] = 0
    return (vals * weights).sum(axis=axis) / weights.sum(axis=axis)

# calculate the z score for the `mask` indexes of the `ser` array
def z_score_mask(ser, mask):
    """Compute z-scores for the masked entries of an array."""
    ser = ser.copy()
    mean = ser[mask].mean()
    std = ser[mask].std()
    return (ser[mask] - mean) / std


# loss and metric functions


class K:
    """backend kernel"""

    @staticmethod
    def sum(x, axis=0, keepdims=True):
        """Sum an array or tensor along the requested axis."""
        if isinstance(x, np.ndarray):
            return x.sum(axis=axis, keepdims=keepdims)
        if isinstance(x, torch.Tensor):
            return x.sum(dim=axis, keepdim=keepdims)
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def clip(x, min_val, max_val):
        """Clip an array or tensor to the inclusive `[min_val, max_val]` range."""
        if isinstance(x, np.ndarray):
            return np.clip(x, min_val, max_val)
        if isinstance(x, torch.Tensor):
            return torch.clamp(x, min_val, max_val)
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def mean(x, axis=0, keepdims=True):
        """Compute the mean of an array or tensor along one axis."""
        # print(x.max())
        if isinstance(x, np.ndarray):
            return x.mean(axis=axis, keepdims=keepdims)
        if isinstance(x, torch.Tensor):
            return x.mean(dim=axis, keepdim=keepdims)
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def seq_mean(x, keepdims=True):
        """Compute the scalar mean of a full array or tensor."""
        if isinstance(x, torch.Tensor):
            return x.mean()
        if isinstance(x, np.ndarray):
            return x.mean()
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def std(x, axis=0, keepdims=True):
        """Compute the standard deviation of an array or tensor."""
        if isinstance(x, np.ndarray):
            return x.std(axis=axis, keepdims=keepdims)
        if isinstance(x, torch.Tensor):
            return x.std(dim=axis, unbiased=False, keepdim=keepdims)
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def median(x, axis=0, keepdims=True):
        """Compute the median of an array or tensor along one axis."""
        # NOTE: numpy will average when size is even,
        # but tensorflow and pytorch don't average
        if isinstance(x, np.ndarray):
            return np.median(x, axis=axis, keepdims=keepdims)
        if isinstance(x, torch.Tensor):
            return torch.median(x, dim=axis, keepdim=keepdims)[0]
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def shape(x):
        """Return the shape of an array or tensor."""
        if isinstance(x, np.ndarray):
            return x.shape
        if isinstance(x, torch.Tensor):
            return list(x.shape)
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def cast(x, dtype="float"):
        """Cast an array or tensor to the requested dtype."""
        if isinstance(x, np.ndarray):
            return x.astype(dtype)
        if isinstance(x, torch.Tensor):
            return x.type(getattr(torch, dtype))
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def maximum(x, y):
        """Return an elementwise maximum-like clamp across array/tensor inputs."""
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            return np.minimum(x, y)
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            return torch.max(x, y)
        if isinstance(x, torch.Tensor):
            return torch.clamp(x, max=y)
        if isinstance(y, torch.Tensor):
            return torch.clamp(y, max=x)
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def auc(y, p):
        """Compute ROC AUC for binary labels and prediction scores."""
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(p, torch.Tensor):
            p = p.detach().cpu().numpy()
        if isinstance(y, np.ndarray) and isinstance(p, np.ndarray):
            y, p = y.reshape(-1), p.reshape(-1)
            assert len(y) == len(p), "Shapes of labels and predictions not match."
            return roc_auc_score(y, p)
        if isinstance(y, list) and isinstance(p, list):
            y, p = np.array(y), np.array(p)
            assert len(y) == len(p), "Shapes of labels and predictions not match."
            return roc_auc_score(y, p)
        raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))

    @staticmethod
    def accuracy(y, p):
        """Compute classification accuracy from labels and predictions."""
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(p, torch.Tensor):
            p = p.detach().cpu().numpy()
        if isinstance(y, np.ndarray) and isinstance(p, np.ndarray):
            y = y.reshape(-1)
            p = p.reshape(len(y), -1)
            if p.shape[-1] == 1:
                p = p.squeeze(-1)
            if len(p.shape) == 2:
                p = np.argmax(p, axis=1)
            assert len(y) == len(p), "Shapes of labels and predictions not match."
            return accuracy_score(y, p)
        if isinstance(y, list) or isinstance(p, list):
            assert len(y) == len(p), "Shapes of labels and predictions not match."
            y, p = np.array(y), np.array(p)
            return accuracy(y, p)
        raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))

    @staticmethod
    def bce(y, p, reduce=True):
        """Compute binary cross-entropy for tensors or NumPy arrays."""
        assert type(y) == type(p), "Type of label and prediction not match."
        if isinstance(y, torch.Tensor):
            y = y.reshape(-1).to(torch.float)
            p = p.reshape(-1)
            assert len(p) == len(y)
            loss = torch.nn.BCELoss(reduction="mean" if reduce else "none")
            return loss(p, y)
        if isinstance(y, np.ndarray):
            y = y.reshape(-1)
            p = p.reshape(len(y), -1)
            if p.shape[-1] == 1:
                p = p.squeeze(-1)
            return log_loss(y, p)
        raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))

    @staticmethod
    def cross_entropy(y, p, reduce=True):
        """Compute multiclass cross-entropy for tensors or NumPy arrays."""
        assert type(y) == type(p), "Type of label and prediction not match."
        if isinstance(y, torch.Tensor):
            y = y.reshape(-1).to(torch.long)
            p = p.float().reshape(len(y), -1).squeeze(-1)
            assert len(p) == len(y)
            loss = torch.nn.NLLLoss(reduction="mean" if reduce else "none")
            return loss(p, y)
        if isinstance(y, np.ndarray):
            y = y.reshape(-1)
            p = p.reshape(len(y), -1)
            if p.shape[-1] == 1:
                p = p.squeeze(-1)
            p = np.exp(p)
            return log_loss(y, p)
        raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))

    @staticmethod
    def nll(y, p, reduce=True):
        """Compute negative log-likelihood loss for log-probability inputs."""
        assert type(y) == type(p), "Type of label and prediction not match."
        if isinstance(y, torch.Tensor):
            loss = torch.nn.NLLLoss(reduction="mean" if reduce else "none")
            return loss(p, y)
        p = np.exp(p)
        p = np.transpose(p, (0, 2, 1))
        p = p.reshape(-1, p.shape[-1])
        y = y.reshape(-1)
        return log_loss(y, p)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def mauc(y, p):
        """Average per-dimension ROC AUC scores across a multivariate target."""
        aucs = 0
        ratios = 0
        _, m = y.shape
        for t in prange(m):
            auc, ratio = fast_auc(y[:, t], p[:, t])
            aucs += auc
            ratios += ratio
        # print(ratios)
        return aucs / ratios

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def dauc(y, p):
        """Average per-sample ROC AUC scores across a multivariate target."""
        aucs = 0
        ratios = 0
        n, _ = y.shape
        for i in prange(n):
            auc, ratio = fast_auc(y[i, :], p[i, :])
            aucs += auc
            ratios += ratio
        print(ratios)
        return aucs / ratios

    @staticmethod
    def mauprc(y, p):
        """Average per-dimension AUPRC scores across a multivariate target."""
        aucs = 0
        ratios = 0
        _, m = y.shape
        for t in prange(m):
            auc, ratio = fast_auprc(y[:, t], p[:, t])
            aucs += auc
            ratios += ratio
        # print(ratios)
        return aucs / ratios

    @staticmethod
    def r2_score(y, p):
        """Compute the regression R^2 score for labels and predictions."""
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(p, torch.Tensor):
            p = p.detach().cpu().numpy()
        if isinstance(y, np.ndarray) and isinstance(p, np.ndarray):
            y = y.reshape(-1)
            p = p.reshape(len(y), -1)
            if p.shape[-1] == 1:
                p = p.squeeze(-1)
            return r2_score(y, p)
        if isinstance(y, list) and isinstance(p, list):
            y, p = np.array(y), np.array(p)
            return K.r2_score(y, p)
        raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))

    @staticmethod
    def ap(y, p):
        """Compute average precision for binary labels and scores."""
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(p, torch.Tensor):
            p = p.detach().cpu().numpy()
        if isinstance(y, np.ndarray) and isinstance(p, np.ndarray):
            y, p = y.reshape(-1), p.reshape(-1)
            assert len(y) == len(p), "Shapes of labels and predictions not match."
            return average_precision_score(y, p)
        if isinstance(y, list) and isinstance(p, list):
            assert len(y) == len(p), "Shapes of labels and predictions not match."
            return average_precision_score(y, p)
        raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))

    @staticmethod
    def auprc(y, p):
        """Compute area under the precision-recall curve."""
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(p, torch.Tensor):
            p = p.detach().cpu().numpy()
        if isinstance(y, np.ndarray) and isinstance(p, np.ndarray):
            y, p = y.reshape(-1), p.reshape(-1)
            assert len(y) == len(p), "Shapes of labels and predictions not match."
            precision, recall, _ = precision_recall_curve(y, p)
            return area_under_curve(recall, precision)
        if isinstance(y, list) and isinstance(p, list):
            assert len(y) == len(p), "Shapes of labels and predictions not match."
            precision, recall, _ = precision_recall_curve(y, p)
            return area_under_curve(recall, precision)
        raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))


# Add Static Methods
def generic_ops(method):
    """Wrap a NumPy/Torch unary op so it works on either backend."""
    def wrapper(x, *args):
        """Dispatch the wrapped unary op to NumPy or Torch."""
        if isinstance(x, np.ndarray):
            return getattr(np, method)(x, *args)
        if isinstance(x, torch.Tensor):
            return getattr(torch, method)(x, *args)
        raise NotImplementedError("unsupported data type %s" % type(x))

    return wrapper


for method in [
    "abs",
    "log",
    "sqrt",
    "exp",
    "log1p",
    "tanh",
    "cosh",
    "squeeze",
    "reshape",
    "zeros_like",
]:
    setattr(K, method, staticmethod(generic_ops(method)))

# Functions


def zscore(x, axis=0):
    """Standardize values with the mean and standard deviation."""
    mean = K.mean(x, axis=axis)
    std = K.std(x, axis=axis)
    return (x - mean) / (std + EPS)


def robust_zscore(x, axis=0):
    """Standardize values with the median and MAD, then clip extremes."""
    med = K.median(x, axis=axis)
    mad = K.median(K.abs(x - med), axis=axis)
    x = (x - med) / (mad * 1.4826 + EPS)
    return K.clip(x, -3, 3)


def batch_corr(x, y, axis=0, keepdims=True):
    """Estimate correlation by multiplying z-scored batches."""
    x = zscore(x, axis=axis)
    y = zscore(y, axis=axis)
    return (x * y).mean()


def robust_batch_corr(x, y, axis=0, keepdims=True):
    """Estimate correlation after robust z-scoring of both inputs."""
    x = robust_zscore(x, axis=axis)
    y = robust_zscore(y, axis=axis)
    return batch_corr(x, y)


@njit
def fast_auc(y_true, y_prob):
    """Compute ROC AUC quickly while ignoring NaN labels."""
    mask = np.logical_not(np.isnan(y_true))
    ratio = np.sum(mask) / mask.size
    y_true = np.extract(mask, y_true)
    y_prob = np.extract(mask, y_prob)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += 1 - y_i
        auc += y_i * nfalse
    auc /= nfalse * (n - nfalse)
    return auc * ratio, ratio


def fast_auprc(y, p):
    """Compute average precision quickly while accounting for NaN coverage."""
    mask = np.logical_not(np.isnan(y))
    ratio = np.sum(mask) / mask.size
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    if isinstance(p, torch.Tensor):
        p = p.detach().cpu().numpy()
    if isinstance(y, np.ndarray) and isinstance(p, np.ndarray):
        y, p = y.reshape(-1), p.reshape(-1)
        assert len(y) == len(p), "Shapes of labels and predictions not match."
        return average_precision_score(y, p) * ratio, ratio
    if isinstance(y, list) and isinstance(p, list):
        assert len(y) == len(p), "Shapes of labels and predictions not match."
        return average_precision_score(y, p) * ratio, ratio
    raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))


def auc(y, preds):
    """Delegate binary ROC AUC computation to `K.auc`."""
    return K.auc(y, preds)


def mauc(y, preds):
    """Delegate multivariate ROC AUC computation to `K.mauc`."""
    return K.mauc(y, preds)


def dauc(y, preds):
    """Delegate per-sample ROC AUC computation to `K.dauc`."""
    return K.dauc(y, preds)


def auprc(y, preds):
    """Delegate AUPRC computation to `K.auprc`."""
    return K.auprc(y, preds)


def ap(y, preds):
    """Delegate average precision computation to `K.ap`."""
    return K.ap(y, preds)


def mauprc(y, preds):
    """Delegate multivariate AUPRC computation to `K.mauprc`."""
    return K.mauprc(y, preds)


def r2(y, preds):
    """Delegate regression R^2 computation to `K.r2_score`."""
    return K.r2_score(y, preds)


def sequence_mse(y_true, y_pred):
    """Compute mean squared error over all sequence elements."""
    loss = (y_true - y_pred) ** 2
    # return torch.mean(loss)

    return K.seq_mean(loss, keepdims=False)


def sequence_mae(y_true, y_pred):
    """Compute mean absolute error over all sequence elements."""
    loss = torch.abs(y_true - y_pred)
    # return torch.mean(loss)
    return K.seq_mean(loss, keepdims=False)


def sequence_mase(y_true, y_pred):
    """Compute the repo's combined squared-plus-absolute sequence error."""
    if isinstance(y_true, np.ndarray):
        loss = (y_true - y_pred) ** 2 + np.abs(y_true - y_pred)
    else:
        loss = (y_true - y_pred) ** 2 + (y_true - y_pred).abs()
    return K.seq_mean(loss, keepdims=False)


def single_mase(y_true, y_pred):
    """Compute the repo's combined squared-plus-absolute error per sample."""
    if isinstance(y_true, np.ndarray):
        loss = (y_true - y_pred) ** 2 + np.abs(y_true - y_pred)
    else:
        loss = (y_true - y_pred) ** 2 + (y_true - y_pred).abs()
    return K.mean(loss, keepdims=False)


def huber_loss(y_true, y_pred, delta=1.0):
    """Compute Huber loss while ignoring NaNs in tensor targets."""
    if isinstance(y_true, np.ndarray):
        residual = y_true - y_pred
        abs_residual = np.abs(residual)
        loss = np.where(
            abs_residual < delta,
            0.5 * residual**2,
            delta * (abs_residual - 0.5 * delta),
        )
        return np.nanmean(loss)
    y_true = y_true.float()
    y_pred = y_pred.float()
    if torch.isnan(y_true).any():
        mask = ~torch.isnan(y_true)
        y_pred = torch.masked_select(y_pred, mask)
        y_true = torch.masked_select(y_true, mask)
    return F.huber_loss(y_pred, y_true, delta=delta, reduction="mean")


def single_mae(y_true, y_pred):
    """Compute mean absolute error for array or tensor inputs."""
    if isinstance(y_true, np.ndarray):
        loss = np.abs(y_true - y_pred)
        return np.nanmean(loss)
    loss = (y_true - y_pred).abs()
    return loss.mean()


def rrse(y_true, y_pred):
    """Compute root relative squared error for regression outputs."""
    if isinstance(y_true, np.ndarray):
        y_bar = y_true.mean(axis=0)
        loss = np.sqrt(((y_pred - y_true) ** 2).sum()) / np.sqrt(
            ((y_true - y_bar) ** 2).sum()
        )
        return np.nanmean(loss)
    y_bar = y_true.mean(dim=0)
    loss = torch.sqrt(((y_pred - y_true) ** 2).sum()) / torch.sqrt(
        ((y_true - y_bar) ** 2).sum()
    )
    return loss.mean()


def mape(y_true, y_pred, log=False):
    """Compute mean absolute percentage error, optionally in exp space."""
    if isinstance(y_true, np.ndarray):
        if log:
            y_true = np.exp(y_true)
            y_pred = np.exp(y_pred)
        loss = np.abs(y_true - y_pred) / y_true
    else:
        if log:
            y_true = y_true.exp()
            y_pred = y_pred.exp()
        loss = (y_true - y_pred).abs()
    return loss.mean()


def mape_log(y_true, y_pred, log=True):
    """Compute mean absolute percentage error with log-space decoding enabled."""
    if isinstance(y_true, np.ndarray):
        if log:
            y_true = np.exp(y_true)
            y_pred = np.exp(y_pred)
        loss = np.abs(y_true - y_pred) / y_true
    else:
        if log:
            y_true = torch.exp(y_true)
            y_pred = torch.exp(y_pred)
        loss = (y_true - y_pred).abs()
    return K.mean(loss, keepdims=False)


def single_mse(y_true, y_pred):
    """Compute mean squared error while ignoring NaNs in tensor targets."""
    if isinstance(y_true, np.ndarray):
        loss = (y_true - y_pred) ** 2
        return np.nanmean(loss)
    y_true = y_true.float()
    y_pred = y_pred.float()
    if torch.isnan(y_true).any():
        mask = ~torch.isnan(y_true)
        y_pred = torch.masked_select(y_pred, mask)
        y_true = torch.masked_select(y_true, mask)
    loss = (y_true - y_pred) ** 2
    loss = loss.mean()

    return loss


def bce(y_true, y_pred, reduce=True):
    """Delegate binary cross-entropy to `K.bce`."""
    return K.bce(y_true, y_pred, reduce)


def cross_entropy(y_true, y_pred, reduce=True):
    """Delegate cross-entropy to `K.cross_entropy`."""
    return K.cross_entropy(y_true, y_pred, reduce)


def cross_entropy_smooth(y_true, y_pred, smoothing=0.1, reduce=True):
    """Compute label-smoothed cross-entropy for arrays or tensors."""
    assert type(y_true) == type(y_pred), "Type of label and prediction not match."
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.reshape(-1).to(torch.long)
        log_probs = y_pred.float().reshape(len(y_true), -1)
        n_classes = log_probs.shape[-1]
        y_true_onehot = F.one_hot(y_true, n_classes).float()
        y_true_smooth = (1.0 - smoothing) * y_true_onehot + smoothing / n_classes
        loss = -(y_true_smooth * log_probs).sum(dim=-1)
        return loss.mean() if reduce else loss
    if isinstance(y_true, np.ndarray):
        y_true = y_true.reshape(-1)
        log_probs = y_pred.reshape(len(y_true), -1)
        n_classes = log_probs.shape[-1]
        onehot = np.eye(n_classes, dtype=np.float32)[y_true.astype(np.int64)]
        smoothed = (1.0 - smoothing) * onehot + smoothing / n_classes
        loss = -(smoothed * log_probs).sum(axis=-1)
        return loss.mean() if reduce else loss
    raise NotImplementedError("unsupported data type %s or %s" % (type(y_true), type(y_pred)))


def accuracy(y_true, y_pred):
    """Delegate accuracy computation to `K.accuracy`."""
    return K.accuracy(y_true, y_pred)


def nll(y_true, y_pred, reduce=True):
    """Delegate negative log-likelihood to `K.nll`."""
    return K.nll(y_true, y_pred, reduce)


def outside_cross_entropy(y_true, y_pred, reduce=True):
    """Compute numerically stable sigmoid cross-entropy from logits."""
    # https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/ops/nn_impl.py#L142
    y = K.cast(y_true > 0, "float")
    p = y_pred
    loss = K.maximum(p, 0) - p * y + K.log1p(K.exp(-K.abs(p)))
    if reduce:
        return K.mean(loss, keepdims=False)
    return loss


def neg_wrapper(func):
    """Wrap a metric or loss so that larger values become smaller and vice versa."""
    def wrapper(*args, **kwargs):
        """Return the negated value of the wrapped callable."""
        return -1 * func(*args, **kwargs)

    return wrapper


def get_loss_fn(loss_fn):
    """Resolve a configured loss name into a callable."""
    # reflection: legacy name
    if loss_fn == "mse":
        return single_mse
    if loss_fn == "single_mse":
        return single_mse
    if loss_fn == "outside_bce":
        return outside_cross_entropy
    if loss_fn == "mase":
        return sequence_mase
    if loss_fn == "mae":
        return single_mae
    if loss_fn == "huber":
        return huber_loss
    if loss_fn.startswith("label"):
        return single_mse
    if loss_fn == "cross_entropy":
        return cross_entropy
    if loss_fn == "cross_entropy_smooth":
        return cross_entropy_smooth
    # if loss_fn == 'mape_log':
    #     return partial(mape, log=True)

    # return function by name
    try:
        return eval(loss_fn)  # dangerous eval
    except Exception:
        pass
    # return negative function by name
    try:
        return neg_wrapper(eval(re.sub("^neg_", "", loss_fn)))
    except Exception:
        raise NotImplementedError("loss function %s is not implemented" % loss_fn)


def get_metric_fn(eval_metric):
    """Resolve a configured metric name into a callable."""
    # reflection: legacy name
    if eval_metric == "corr":
        return neg_wrapper(robust_batch_corr)  # more stable
    if eval_metric == "mse":
        return single_mse
    if eval_metric == "mae":
        return single_mae
    if eval_metric in ["rse", "rrse"]:
        return rrse
    # return function by name
    # if eval_metric == 'mape_log':
    #     return partial(mape, log=True)
    try:
        return eval(eval_metric)  # dangerous eval
    except Exception:
        pass
    # return negative function by name
    try:
        return neg_wrapper(eval(re.sub("^neg_", "", eval_metric)))
    except Exception:
        raise NotImplementedError("metric function %s is not implemented" % eval_metric)


def test():
    """Placeholder manual test entrypoint for this utility module."""
    pass


if __name__ == "__main__":
    test()
