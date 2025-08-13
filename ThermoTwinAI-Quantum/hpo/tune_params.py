"""Hyperparameter tuning utilities for quantum models."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

import numpy as np

from evaluation.evaluate_models import evaluate_model
from models.quantum_lstm import train_quantum_lstm
from models.quantum_prophet import train_quantum_prophet
from utils.preprocessing import load_and_split_data


@dataclass
class TrialResult:
    params: dict
    mae: float


def grid_search(model: str, param_grid: dict, data_path: str) -> TrialResult:
    """Exhaustively evaluate all combinations in ``param_grid``."""

    X_train, y_train, X_test, y_test = load_and_split_data(data_path)
    best: TrialResult | None = None

    keys = list(param_grid.keys())
    for values in itertools.product(*param_grid.values()):
        params = dict(zip(keys, values))
        if model == "q_lstm":
            mdl, preds = train_quantum_lstm(
                X_train,
                y_train,
                X_test,
                epochs=20,
                lr=params["lr"],
                hidden_size=params["hidden_size"],
                q_depth=params["q_depth"],
                dropout=params["dropout"],
            )
        else:
            mdl, preds = train_quantum_prophet(
                X_train,
                y_train,
                X_test,
                epochs=20,
                lr=params["lr"],
                hidden_dim=params["hidden_size"],
                q_depth=params["q_depth"],
                dropout=params["dropout"],
            )
        metrics = evaluate_model(y_test, preds, name="tune", plot=False)
        trial = TrialResult(params, metrics["MAE"])
        if best is None or trial.mae < best.mae:
            best = trial
    assert best is not None
    return best


def random_search(model: str, param_grid: dict, data_path: str, n_iter: int = 10) -> TrialResult:
    """Sample ``n_iter`` random combinations from ``param_grid``."""

    X_train, y_train, X_test, y_test = load_and_split_data(data_path)
    best: TrialResult | None = None

    keys = list(param_grid.keys())
    for _ in range(n_iter):
        params = {k: random.choice(v) for k, v in param_grid.items()}
        if model == "q_lstm":
            mdl, preds = train_quantum_lstm(
                X_train,
                y_train,
                X_test,
                epochs=20,
                lr=params["lr"],
                hidden_size=params["hidden_size"],
                q_depth=params["q_depth"],
                dropout=params["dropout"],
            )
        else:
            mdl, preds = train_quantum_prophet(
                X_train,
                y_train,
                X_test,
                epochs=20,
                lr=params["lr"],
                hidden_dim=params["hidden_size"],
                q_depth=params["q_depth"],
                dropout=params["dropout"],
            )
        metrics = evaluate_model(y_test, preds, name="tune", plot=False)
        trial = TrialResult(params, metrics["MAE"])
        if best is None or trial.mae < best.mae:
            best = trial
    assert best is not None
    return best


if __name__ == "__main__":  # pragma: no cover - manual use
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter tuning")
    parser.add_argument("--model", choices=["q_lstm", "q_prophet"], default="q_lstm")
    parser.add_argument("--data", default="data/synthetic_tftec_cop.csv")
    parser.add_argument("--random", action="store_true", help="Use random search")
    args = parser.parse_args()

    grid = {
        "lr": [0.001, 0.005],
        "hidden_size": [16, 32],
        "q_depth": [1, 2],
        "dropout": [0.1, 0.25],
    }

    if args.random:
        result = random_search(args.model, grid, args.data)
    else:
        result = grid_search(args.model, grid, args.data)

    print("Best params:", result.params, "MAE=", result.mae)
