# Architecture — ThermoTwinAI-Quantum

This document describes the **research implementation reflected in the repository**. It intentionally distinguishes the implemented modules from unverified industrial claims.

## Data and training stages

```mermaid
flowchart TD
    A["main.py: generate 156 weeks of synthetic CoP + sensor values"] --> B[("data/synthetic_tftec_cop.csv")]
    B --> C["utils/preprocessing.py: CoP difference, IQR clip, min-max scale"]
    C --> D["Sliding windows (default 15 steps)"]
    D --> E["80/20 chronological split"]
    E --> F{"Optional CLI flags"}
    F -->|"--use_augmentation"| G["Time-series augmentation of training data"]
    F -->|"--use_drift"| H["Drift detection / optional mask"]
    F -->|"neither"| I["Base training windows"]
    G --> J["QLSTM and/or QProphet training"]
    H --> J
    I --> J
    J --> K["evaluate_models.py: MAE, RMSE, MAPE, sMAPE, R², correlation"]
    J --> L["Optional MC-dropout predictive intervals"]
    K --> M[("results/metrics.json + plots/")]
    L --> M
```

## Model paths

```mermaid
flowchart LR
    X["Input window: time × sensor features"] --> SF["Learned sensor fusion"]
    SF --> A["QLSTM: LSTM + attention / last-step fusion"]
    SF --> B["QProphet-style: 1D CNN + pooled features"]
    A --> C["Four-qubit variational PennyLane circuit"]
    B --> C
    C --> D["Adaptive causal graph attention and feature fusion"]
    D --> E["PyTorch regression head"]
    E --> F["CoP prediction"]
```

The two models integrate the attention/fusion paths in **different orders**; the diagram summarizes components shared conceptually rather than describing every tensor operation.

## Experiment surfaces

| Location | Purpose |
| --- | --- |
| `main.py` | CLI switches for architecture, window length, learning rate, drift, augmentation and predictive uncertainty |
| `models/quantum_lstm.py` | Temporal model with LSTM, attention and quantum layer |
| `models/quantum_prophet.py` | Custom CNN / quantum / regression architecture |
| `utils/quantum_layers.py` | Four-qubit PennyLane simulator wrapped as a PyTorch module |
| `utils/causal_graph_attention.py` | Sensor relation module |
| `hpo/tune_params.py` | Experimental grid or random search by MAE |
| `benchmarks/classical_models.py` | Vanilla LSTM and optional NeuralProphet comparison |
| `evaluation/evaluate_models.py` | Metrics, optional intervals, JSON output and plots |

## Scientific interpretation

The default dataset is generated, not observed. `results/metrics.json` records example results that can vary between runs; the code does not fix all random seeds at the main entry point. The existing tuning helper evaluates candidate configurations on its test partition, which should be treated as a **validation-design limitation** rather than external benchmark evidence. The reported R² and correlation are transformed by the current evaluator. A robust experimental paper should use fixed train/validation/test time ranges, multiple seeds and matching classical baselines.
