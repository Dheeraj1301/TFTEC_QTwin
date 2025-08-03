# utils/preprocessing.py
import numpy as np


def load_and_split_data(path: str, window_size: int = 15):
    """Load CSV data and create windowed training and test splits.

    This function avoids heavy dependencies such as pandas or scikit-learn and
    instead relies purely on ``numpy``. The CSV file is expected to have a
    header row where the first column is ``week`` and the remaining columns are
    the features with ``CoP`` being the first feature after ``week``.
    """

    raw = np.genfromtxt(path, delimiter=",", skip_header=1)

    # Exclude the week column; perform min-max scaling manually
    data = raw[:, 1:]
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    data = (data - min_vals) / (max_vals - min_vals + 1e-8)

    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size : i, :])
        y.append(data[i, 0])  # predict CoP (first feature)

    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    return X[:split], y[:split], X[split:], y[split:]
