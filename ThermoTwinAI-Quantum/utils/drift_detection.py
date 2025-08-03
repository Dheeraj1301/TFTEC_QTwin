import csv
import os
import time
from typing import List, Tuple, Optional

import numpy as np


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
