# utils/preprocessing.py
import numpy as np
from utils.data_augmentation import augment_time_series


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

    # Exclude the week column; perform min-max scaling manually
    data = raw[:, 1:]
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    data = (data - min_vals) / (max_vals - min_vals + 1e-8)

    split = int(len(data) * 0.8)
    train_data = data[:split]
    test_data = data[split:]

    if use_augmentation:
        train_data = augment_time_series(train_data, seed=seed)
        # Augmentation may slightly drift features outside the original
        # min-max range; clip them back to preserve a stable distribution
        train_data = np.clip(train_data, 0.0, 1.0)

    def create_windows(series: np.ndarray):
        X, y = [], []
        for i in range(window_size, len(series)):
            X.append(series[i - window_size : i, :])
            y.append(series[i, 0])  # predict CoP (first feature)
        return np.array(X), np.array(y)

    X_train, y_train = create_windows(train_data)
    X_test, y_test = create_windows(test_data)

    return X_train, y_train, X_test, y_test
