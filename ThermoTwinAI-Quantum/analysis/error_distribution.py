"""Plot error distributions before and after optimisation."""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_error_distribution(errors_before: np.ndarray, errors_after: np.ndarray, save_path: str = "plots/error_hist.png") -> None:
    """Plot histograms of prediction errors before and after optimisation."""

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure()
    plt.hist(errors_before, bins=30, alpha=0.5, label="Before")
    plt.hist(errors_after, bins=30, alpha=0.5, label="After")
    plt.xlabel("Prediction error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
