from typing import Optional, List

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utilsd.config import Registry

from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class DATASETS(metaclass=Registry, name="dataset"):
    pass


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, timeenc=1, freq="h") -> np.ndarray:
    """
    > `time_features` takes in a `dates` dataframe with a 'dates' column and extracts the date down to `freq` where freq can be any of the following if `timeenc` is 0:
    > * m - [month]
    > * w - [month]
    > * d - [month, day, weekday]
    > * b - [month, day, weekday]
    > * h - [month, day, weekday, hour]
    > * t - [month, day, weekday, hour, *minute]
    >
    > If `timeenc` is 1, a similar, but different list of `freq` values are supported (all encoded between [-0.5 and 0.5]):
    > * Q - [month]
    > * M - [month]
    > * W - [Day of month, week of year]
    > * D - [Day of week, day of month, day of year]
    > * B - [Day of week, day of month, day of year]
    > * H - [Hour of day, day of week, day of month, day of year]
    > * T - [Minute of hour*, hour of day, day of week, day of month, day of year]
    > * S - [Second of minute, minute of hour, hour of day, day of week, day of month, day of year]

    *minute returns a number from 0-3 corresponding to the 15 minute period it falls into.
    """
    if timeenc == 0:
        dates["month"] = dates.date.apply(lambda row: row.month, 1)
        dates["day"] = dates.date.apply(lambda row: row.day, 1)
        dates["weekday"] = dates.date.apply(lambda row: row.weekday(), 1)
        dates["hour"] = dates.date.apply(lambda row: row.hour, 1)
        dates["minute"] = dates.date.apply(lambda row: row.minute, 1)
        dates["minute"] = dates.minute.map(lambda x: x // 15)
        freq_map = {
            "y": [],
            "m": ["month"],
            "w": ["month"],
            "d": ["month", "day", "weekday"],
            "b": ["month", "day", "weekday"],
            "h": ["month", "day", "weekday", "hour"],
            "t": ["month", "day", "weekday", "hour", "minute"],
        }
        return dates[freq_map[freq.lower()]].values
    else:
        dates = pd.to_datetime(dates.date.values)
        return np.vstack(
            [feat(dates) for feat in time_features_from_frequency_str(freq)]
        ).transpose(1, 0)


@DATASETS.register_module()
class TSMSDataset(Dataset):
    def __init__(
        self,
        file: str,
        window: int,
        horizon: int,
        train_ratio: float = 0.8,
        test_ratio: float = 0.2,
        normalize: int = 2,
        last_label: bool = False,
        raw_label: bool = True,
        dataset_name: Optional[str] = None,
        missing_value_strategy: Optional[str] = None,
    ):
        self.window = window
        self.horizon = horizon
        if file.endswith(".txt"):
            with open(file) as f:
                self.raw_data = np.loadtxt(f, delimiter=",").astype(np.float32)
        elif file.endswith(".csv"):
            with open(file) as f:
                self.raw_data = np.loadtxt(
                    f, delimiter=",", skiprows=1, dtype=object
                )[:, 1:].astype(np.float32)
                self.dates = pd.DataFrame(
                    np.loadtxt(f, delimiter=",", skiprows=1, dtype=object)[:, 0],
                    columns=["date"],
                )
            self.dates["date"] = self.dates["date"].map(pd.Timestamp)
            self.dates = time_features(self.dates, freq="t")
        elif file.endswith(".h5"):
            self.raw_data = (
                pd.read_hdf(file).reset_index().values[:, 1:].astype(np.float32)
            )
            self.dates = pd.DataFrame(pd.read_hdf(file).reset_index()["index"]).rename(
                columns={"index": "date"}
            )
            self.dates = time_features(self.dates, freq="t")
        # Apply missing value strategy before normalization
        if missing_value_strategy == "zero_to_nan":
            self.raw_data = self.raw_data.copy()
            self.raw_data[self.raw_data == 0] = np.nan
        elif missing_value_strategy == "interpolate":
            df = pd.DataFrame(self.raw_data)
            df.replace(0, np.nan, inplace=True)
            df.interpolate(method="linear", axis=0, inplace=True)
            df.ffill(inplace=True)
            df.bfill(inplace=True)
            self.raw_data = df.values.astype(np.float32)

        self.dat = np.zeros(self.raw_data.shape, dtype=np.float32)
        self.n, self.m = self.dat.shape
        if (train_ratio + test_ratio) == 1 and dataset_name == "valid":
            dataset_name = "test"
        self.dataset_name = dataset_name
        self.last_label = last_label
        self.raw_label = raw_label
        self._normalized(normalize)
        self._split(train_ratio, test_ratio, self.dataset_name)

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if normalize == 0:
            self.dat = self.raw_data

        if normalize == 1:
            self.dat = self.raw_data / np.max(self.raw_data)

        # normlized by the maximum value of each row(sensor).
        if normalize == 2:
            for i in range(self.m):
                self.dat[:, i] = self.raw_data[:, i] / np.max(
                    np.abs(self.raw_data[:, i])
                )

        if normalize == 3:
            self.dat = (self.raw_data - np.mean(self.raw_data)) / (
                np.std(self.raw_data) + np.finfo(float).eps
            )

        # normalize == 4: per-column z-score
        if normalize == 4:
            self._col_mean = np.nanmean(self.raw_data, axis=0, keepdims=True)  # (1, m)
            self._col_std = np.nanstd(self.raw_data, axis=0, keepdims=True)    # (1, m)
            self._col_std = np.where(self._col_std < np.finfo(float).eps, 1.0, self._col_std)
            self.dat = (self.raw_data - self._col_mean) / self._col_std

        # normalize == 5: per-column robust scaling (median / IQR)
        if normalize == 5:
            self._col_median = np.nanmedian(self.raw_data, axis=0, keepdims=True)  # (1, m)
            q75 = np.nanpercentile(self.raw_data, 75, axis=0, keepdims=True)
            q25 = np.nanpercentile(self.raw_data, 25, axis=0, keepdims=True)
            self._col_iqr = q75 - q25
            self._col_iqr = np.where(self._col_iqr < np.finfo(float).eps, 1.0, self._col_iqr)
            self.dat = (self.raw_data - self._col_median) / self._col_iqr

    def _split(self, train_ratio, test_ratio, dataset_name):
        total_size = self.n - self.window - self.horizon + 1
        train_size = int(total_size * train_ratio)
        test_size = int(total_size * test_ratio)
        valid_size = total_size - test_size - train_size
        if dataset_name == "train":
            self.length = train_size
            self.start_idx = 0
        elif dataset_name == "valid":
            self.length = valid_size
            self.start_idx = train_size
        elif dataset_name == "test":
            self.length = test_size
            self.start_idx = train_size + valid_size
        else:
            raise ValueError

    def freeup(self):
        pass

    def load(self):
        pass

    @property
    def num_variables(self):
        if hasattr(self, "dates"):
            return self.dates.shape[1] + self.raw_data.shape[1]
        else:
            return self.raw_data.shape[1]

    def __len__(self):
        return self.length

    @property
    def max_seq_len(self):
        return self.window

    @property
    def num_classes(self):
        if self.last_label:
            return self.horizon
        else:
            return self.raw_data.shape[1] * self.horizon

    def get_index(self):
        return np.arange(self.length)

    def __getitem__(self, index):
        index = index + self.start_idx
        X = self.dat[index : index + self.window, :]
        # add time features
        if hasattr(self, "dates"):
            X = np.concatenate([X, self.dates[index : index + self.window]], axis=1)
        if self.raw_label:
            label_data = self.raw_data
        else:
            label_data = self.dat
        if self.last_label:
            y = label_data[index + self.window : index + self.window + self.horizon, -1]
        else:
            y = label_data[
                index + self.window : index + self.window + self.horizon, :
            ].reshape(-1)
        assert len(y) == self.num_classes, (len(y), self.num_classes)
        return X.astype(np.float32), y.astype(np.float32)
