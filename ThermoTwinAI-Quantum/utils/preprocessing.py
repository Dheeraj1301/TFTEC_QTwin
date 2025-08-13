# utils/preprocessing.py
import numpy as np
import torch
import torch.nn as nn
from utils.data_augmentation import augment_time_series


class SensorFusion(nn.Module):
    """Learnable sensor weighting module.

    A weight is learned for each sensor/feature and normalised with a softmax
    so that the weights form a convex combination (sum to one). The module can
    be prepended to any model expecting inputs of shape ``(batch, seq, feat)``
    to softly emphasise or de-emphasise particular sensors before further
    processing.
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        w = torch.softmax(self.weights, dim=0)
        return x * w


def load_and_split_data(
    path: str,
    window_size: int = 15,
    use_augmentation: bool = False,
    seed: int | None = 42,
):
    """Load CSV data and create windowed training and test splits.

    Parameters
    ----------
    path:
        Path to the CSV file containing the time series. The first column is
        ignored (assumed to be an index) and the remaining columns are scaled
        to ``[0, 1]``.
    window_size:
        Length of each input sequence.
    use_augmentation:
        If ``True`` the training portion of the data is augmented prior to
        windowing.
    seed:
        Optional seed forwarded to :func:`augment_time_series` for
        deterministic behaviour.
    """

    raw = np.genfromtxt(path, delimiter=",", skip_header=1)

    # Exclude the week column and clamp outliers using the IQR method before
    # scaling. This reduces the impact of extreme sensor spikes which could
    # otherwise distort the min–max normalisation.
    data = raw[:, 1:]
    q1 = np.percentile(data, 25, axis=0)
    q3 = np.percentile(data, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    data = np.clip(data, lower, upper)

    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    data = (data - min_vals) / (max_vals - min_vals + 1e-8)

    split = int(len(data) * 0.8)
    train_data = data[:split]
    test_data = data[split:]

    if use_augmentation:
        train_data = augment_time_series(train_data, seed=seed)

    def create_windows(series: np.ndarray):
        X, y = [], []
        for i in range(window_size, len(series)):
            X.append(series[i - window_size : i, :])
            y.append(series[i, 0])  # predict CoP (first feature)
        return np.array(X), np.array(y)

    X_train, y_train = create_windows(train_data)
    X_test, y_test = create_windows(test_data)

    return X_train, y_train, X_test, y_test
