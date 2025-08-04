"""Benchmark classical models against quantum counterparts."""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from evaluation.evaluate_models import evaluate_model
from models.quantum_lstm import train_quantum_lstm
from models.quantum_prophet import train_quantum_prophet
from utils.preprocessing import load_and_split_data


class VanillaLSTM(nn.Module):
    """Minimal LSTM baseline without any quantum components."""

    def __init__(self, input_size: int, hidden: int = 32, dropout: float = 0.25) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


def train_vanilla_lstm(X_train, y_train, X_test, epochs: int = 50, lr: float = 0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VanillaLSTM(X_train.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train[:, None], dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(X_train)
        loss = loss_fn(out, y_train)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test).cpu().numpy().flatten()
    return model, preds


def train_prophet_baseline(y_train, periods: int):  # pragma: no cover - heavy dep
    from neuralprophet import NeuralProphet
    import pandas as pd

    df = pd.DataFrame({"y": y_train, "ds": pd.date_range("2000", periods=len(y_train))})
    m = NeuralProphet(n_lags=0, n_forecasts=periods)
    m.fit(df, freq="D", progress_bar=False)
    future = m.make_future_dataframe(df, periods=periods, n_historic_predictions=False)
    forecast = m.predict(future)["yhat1"].values
    return forecast


def run_benchmarks(data_path: str = "data/synthetic_tftec_cop.csv", results_path: str = "results/benchmark_results.csv") -> None:
    X_train, y_train, X_test, y_test = load_and_split_data(data_path)

    rows = []

    # Vanilla LSTM
    lstm_model, lstm_preds = train_vanilla_lstm(X_train, y_train, X_test)
    metrics = evaluate_model(y_test, lstm_preds, name="Vanilla LSTM", plot=False)
    rows.append(("Vanilla LSTM", metrics))

    # Prophet/NeuralProphet baseline if available
    try:  # pragma: no cover - optional dependency
        prophet_preds = train_prophet_baseline(np.concatenate([y_train, y_test[:-1]]), len(y_test))
        prophet_metrics = evaluate_model(y_test, prophet_preds, name="NeuralProphet", plot=False)
        rows.append(("NeuralProphet", prophet_metrics))
    except Exception:
        pass

    # Quantum models
    qlstm_model, qlstm_preds = train_quantum_lstm(X_train, y_train, X_test, epochs=30, dropout=0.25)
    qlstm_metrics = evaluate_model(y_test, qlstm_preds, name="Quantum LSTM", plot=False)
    rows.append(("Quantum LSTM", qlstm_metrics))

    qprophet_model, qprophet_preds = train_quantum_prophet(X_train, y_train, X_test, epochs=30, dropout=0.25)
    qprophet_metrics = evaluate_model(y_test, qprophet_preds, name="Quantum Prophet", plot=False)
    rows.append(("Quantum Prophet", qprophet_metrics))

    # Write results
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Model", "MAE", "RMSE", "MAPE", "R2", "Corr"]
        writer.writerow(header)
        for name, m in rows:
            writer.writerow([name, m["MAE"], m["RMSE"], m["MAPE"], m["R2"], m["Corr"]])


if __name__ == "__main__":  # pragma: no cover - manual use
    run_benchmarks()
