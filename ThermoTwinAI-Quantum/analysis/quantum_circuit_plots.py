"""Utilities to visualise quantum circuit parameters over time."""

from __future__ import annotations

import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def plot_qubit_angles(angles: Iterable[Iterable[float]], save_path: str = "plots/qubit_angles.png") -> None:
    """Plot qubit rotation angles over time.

    Parameters
    ----------
    angles:
        2D iterable where each row corresponds to a timestep and each column to a
        qubit's rotation angle.
    save_path:
        Location to save the resulting plot.
    """

    angles = np.array(list(angles))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure()
    for q in range(angles.shape[1]):
        plt.plot(angles[:, q], label=f"Qubit {q}")
    plt.xlabel("Timestep")
    plt.ylabel("Rotation angle")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
