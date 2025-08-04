import csv
import os
import time
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


def detect_drift(series: pd.Series, window: int = 10, threshold: float = 2.0):
    """Flag drifts in a time series via rolling z-scores.

    A rolling mean and standard deviation are computed over ``window`` samples.
    Samples whose absolute z-score exceeds ``threshold`` are marked as drift
    (1) while the rest are 0. NaNs from the initial window are safely
    initialised to 0.
    """

    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    z_scores = (series - rolling_mean) / (rolling_std + 1e-6)
    drift_flags = (z_scores.abs() > threshold).astype(int)
    return drift_flags.fillna(0).astype(int)


def apply_drift_mask(X, y, drift_flags, window: int):
    """Remove samples whose end timestep is flagged as drift.

    Parameters
    ----------
    X, y : arrays
        Windowed training data produced by ``load_and_split_data``.
    drift_flags : Series or array
        Binary flags for each original timestep in the dataset.
    window : int
        Window size used to create ``X`` and ``y``. Required for alignment.
    """

    flags = np.asarray(drift_flags)[window:]
    flags = flags[: len(X)]
    mask = flags == 0
    return X[mask], y[mask]


def adjust_learning_rate(optimizer, severity: Optional[float], base_lr: float, min_lr: float = 1e-5):
    """Scale the learning rate based on drift severity.

    ``severity`` can be any positive number (e.g. max z-score). Larger values
    yield a smaller learning rate down to ``min_lr``.
    """

    if optimizer is None or severity is None:
        return
    factor = 1.0 / (1.0 + float(severity))
    new_lr = max(base_lr * factor, min_lr)
    for group in optimizer.param_groups:
        group["lr"] = new_lr


class DriftDetector:
    """Simple drift detector based on sliding window MAE comparison.

    The detector stores recent prediction errors and compares the mean error of
    the most recent window with the preceding window. A relative increase beyond
    ``threshold`` triggers a drift flag.
    """

    def __init__(
        self,
        window_size: int = 20,
        threshold: float = 0.3,
        log_dir: str = "drift_logs",
    ) -> None:
        self.window_size = window_size
        self.threshold = threshold
        self.errors: List[float] = []

        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "detected_drifts.csv")
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "model", "epoch", "prev_mae", "curr_mae"])

    def update(self, error: float) -> Tuple[bool, Optional[float], Optional[float]]:
        """Update the detector with a new error value.

        Returns a tuple ``(is_drift, prev_mean, curr_mean)`` where ``is_drift``
        indicates if drift was detected.
        """

        self.errors.append(error)
        if len(self.errors) >= 2 * self.window_size:
            recent = self.errors[-self.window_size :]
            past = self.errors[-2 * self.window_size : -self.window_size]
            mean_recent = float(np.mean(recent))
            mean_past = float(np.mean(past))
            if mean_past > 0 and (mean_recent - mean_past) / mean_past > self.threshold:
                return True, mean_past, mean_recent
        return False, None, None

    def log(self, model: str, epoch: int, prev: float, curr: float) -> None:
        """Persist a drift event to the CSV log."""
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), model, epoch, prev, curr])

    def reset(self) -> None:
        self.errors.clear()
