# main.py
from utils.preprocessing import load_and_split_data
from models.quantum_lstm import train_quantum_lstm
from models.quantum_prophet import train_quantum_prophet
from evaluation.evaluate_models import evaluate_model
import numpy as np
import pandas as pd
import os

def generate_tftec_cop_data(n_weeks=156):
    base_cop = 1.5
    time = np.arange(n_weeks)

    # Additional synthetic sensor signals
    ambient_temp = 25 + 5 * np.sin(2 * np.pi * time / 52) + np.random.normal(0, 0.5, n_weeks)
    current = 5 + 0.5 * np.sin(2 * np.pi * time / 26) - 0.01 * time + np.random.normal(0, 0.1, n_weeks)
    zt = 1.0 + 0.1 * np.cos(2 * np.pi * time / 52) - 0.005 * time + np.random.normal(0, 0.01, n_weeks)

    # Degradation + noise + anomalies
    degradation = 0.0015 * time
    anomalies = np.zeros(n_weeks)
    for idx in np.random.choice(range(20, n_weeks - 10), size=4, replace=False):
        anomalies[idx:idx+5] -= np.linspace(0.05, 0.15, 5)
    noise = np.random.normal(0, 0.01, n_weeks)

    # Sensor influence on CoP
    cop = base_cop - degradation + anomalies + noise
    cop += 0.002 * (ambient_temp - 25) - 0.001 * (current - 5) + 0.05 * (zt - 1)
    cop = np.clip(cop, 0, None)

    df = pd.DataFrame(
        {
            "week": time,
            "CoP": cop,
            "ambient_temp": ambient_temp,
            "current": current,
            "ZT": zt,
        }
    )

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/synthetic_tftec_cop.csv", index=False)
    print("✅ Synthetic multivariate data saved to data/synthetic_tftec_cop.csv")

def main():
    print("🚀 ThermoTwinAI-Quantum Forecasting Pipeline")

    generate_tftec_cop_data()

    X_train, y_train, X_test, y_test = load_and_split_data("data/synthetic_tftec_cop.csv", window_size=15)

    print("\n🔮 Training Quantum LSTM...")
    qlstm_preds = train_quantum_lstm(X_train, y_train, X_test, epochs=50, lr=0.005)

    print("\n📈 Training Quantum NeuralProphet...")
    qprophet_preds = train_quantum_prophet(X_train, y_train, X_test, epochs=50, lr=0.005)

    print("\n📊 Evaluation:")
    evaluate_model(y_test, qlstm_preds, name="Quantum LSTM", plot=True)
    evaluate_model(y_test, qprophet_preds, name="Quantum NeuralProphet", plot=True)

if __name__ == "__main__":
    main()
