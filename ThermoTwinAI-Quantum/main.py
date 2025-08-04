# main.py
from utils.preprocessing import load_and_split_data
from models.quantum_lstm import train_quantum_lstm
from models.quantum_prophet import train_quantum_prophet
from utils.drift_detection import DriftDetector, detect_drift, apply_drift_mask
from evaluation.evaluate_models import evaluate_model
import numpy as np
import csv
import os
import argparse
import pandas as pd

def generate_tftec_cop_data(n_weeks: int = 156):
    """Create a synthetic multivariate dataset with simple degradation effects."""

    base_cop = 1.5
    time = np.arange(n_weeks)

    # Additional synthetic sensor signals
    ambient_temp = 25 + 5 * np.sin(2 * np.pi * time / 52) + np.random.normal(0, 0.5, n_weeks)
    module_current = 5 + 0.5 * np.sin(2 * np.pi * time / 26) - 0.01 * time + np.random.normal(0, 0.1, n_weeks)
    humidity = 40 + 10 * np.sin(2 * np.pi * time / 52) + np.random.normal(0, 1.0, n_weeks)

    # Degradation + noise + anomalies
    degradation = 0.0015 * time
    anomalies = np.zeros(n_weeks)
    for idx in np.random.choice(range(20, n_weeks - 10), size=4, replace=False):
        anomalies[idx : idx + 5] -= np.linspace(0.05, 0.15, 5)
    noise = np.random.normal(0, 0.01, n_weeks)

    # Sensor influence on CoP
    cop = base_cop - degradation + anomalies + noise
    cop += 0.002 * (ambient_temp - 25) - 0.001 * (module_current - 5) + 0.0005 * (humidity - 40)
    cop = np.clip(cop, 0, None)

    os.makedirs("data", exist_ok=True)
    with open("data/synthetic_tftec_cop.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["week", "CoP", "ambient_temp", "module_current", "humidity"])
        for i in range(n_weeks):
            writer.writerow([time[i], cop[i], ambient_temp[i], module_current[i], humidity[i]])

    print("✅ Synthetic multivariate data saved to data/synthetic_tftec_cop.csv")

def main():
    parser = argparse.ArgumentParser(description="ThermoTwinAI-Quantum pipeline")
    parser.add_argument("--window", type=int, default=15, help="Sliding window size")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default 0.005, or 0.001 when --use_drift is set)",
    )
    parser.add_argument(
        "--use_drift",
        action="store_true",
        help="Enable z-score drift masking for training data",
    )
    args = parser.parse_args()

    if args.lr is None:
        args.lr = 0.001 if args.use_drift else 0.005

    print("🚀 ThermoTwinAI-Quantum Forecasting Pipeline")

    generate_tftec_cop_data()

    df = pd.read_csv("data/synthetic_tftec_cop.csv")

    X_train, y_train, X_test, y_test = load_and_split_data(
        "data/synthetic_tftec_cop.csv", window_size=args.window
    )

    if args.use_drift:
        drift_flags = detect_drift(df["CoP"])
        X_train, y_train = apply_drift_mask(
            X_train, y_train, drift_flags, window=args.window
        )

    drift_detector = DriftDetector(window_size=5, threshold=0.2)

    print("\n🔮 Training Quantum LSTM...")
    qlstm_preds = train_quantum_lstm(
        X_train, y_train, X_test, epochs=args.epochs, lr=args.lr, drift_detector=drift_detector
    )

    drift_detector.reset()

    print("\n📈 Training Quantum NeuralProphet...")
    qprophet_preds = train_quantum_prophet(
        X_train, y_train, X_test, epochs=args.epochs, lr=args.lr, drift_detector=drift_detector
    )

    print("\n📊 Evaluation:")
    evaluate_model(y_test, qlstm_preds, name="Quantum LSTM", plot=True)
    evaluate_model(y_test, qprophet_preds, name="Quantum NeuralProphet", plot=True)

if __name__ == "__main__":
    main()
