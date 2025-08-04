"""Quantum-augmented NeuralProphet-style model.

The file lazily imports ``torch`` and ``pennylane`` so that the training helper
remains importable even when the heavy dependencies are absent. When available,
the model optionally smooths inputs with a light CNN, normalises them, passes
them through a two-layer entangling quantum circuit and applies a small MLP
head. Dropout layers around the quantum circuit help stabilise training.
"""

from typing import Any

from utils.drift_detection import DriftDetector, adjust_learning_rate

try:  # pragma: no cover - executed only when deps are available
    import random
    import numpy as np
    import torch
    import torch.nn as nn

    from utils.quantum_layers import QuantumLayer, n_qubits
except Exception as exc:  # pragma: no cover - used for graceful degradation
    torch = None
    nn = None
    QuantumLayer = None  # type: ignore
    n_qubits = 0  # type: ignore
    _IMPORT_ERROR = exc
else:  # pragma: no cover - executed only when deps are available
    _IMPORT_ERROR = None


if torch is not None:  # pragma: no cover - executed only when deps are available

    class QProphetModel(nn.Module):
        """1D CNN features, quantum layer and a compact classical head."""

        def __init__(
            self,
            num_features: int,
            hidden_dim: int = 32,
            q_depth: int = 2,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()

            # 1D CNN → GELU smooths local patterns before the quantum circuit
            self.pre_q = nn.Sequential(
                nn.Conv1d(num_features, n_qubits, kernel_size=3, padding=1),
                nn.GELU(),
            )

            # Normalise to stabilise variance before entering the quantum circuit
            # Explicitly use four features as required by the QNode
            self.norm = nn.LayerNorm(4)

            # Quantum layer: depth=2 entangling layers
            self.q_layer = QuantumLayer(n_layers=2)
            self.q_dropout = nn.Dropout(0.1)

            # Post-QNode MLP: Linear(4→16) → GELU → Dropout(0.2) → Linear(16→1)
            self.classical_head = nn.Sequential(
                nn.Linear(n_qubits, 16),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(16, 1),
            )

            # ``dropout`` argument retained for API compatibility though unused.

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Input comes as (batch, seq, features); rearrange for Conv1d
            x = x.permute(0, 2, 1)
            x = self.pre_q(x)  # conv + GELU -> (batch, n_qubits, seq)

            # Mean pooling over timesteps then normalise before quantum layer
            x = x.mean(dim=2)
            x = self.norm(x)
            x = self.q_dropout(self.q_layer(x))
            out = self.classical_head(x)
            return torch.clamp(out, -3, 3)


def train_quantum_prophet(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    epochs: int = 50,
    lr: float = 0.001,
    hidden_dim: int = 32,
    q_depth: int = 2,
    drift_detector: DriftDetector | None = None,
):
    """Train ``QProphetModel`` and return predictions for ``X_test``.

    If the optional dependencies (``torch`` and ``pennylane``) are not
    installed the function will raise a clear and immediate ``ImportError``
    explaining which package is missing.
    """

    if torch is None or QuantumLayer is None:
        raise ImportError(
            "train_quantum_prophet requires `torch` and `pennylane` to be installed"
        ) from _IMPORT_ERROR

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_features = X_train.shape[2]
    model = QProphetModel(num_features, hidden_dim=hidden_dim, q_depth=q_depth).to(
        device
    )
    # Optimise the mean absolute error directly to promote stable,
    # directionally consistent predictions
    criterion = nn.L1Loss()
    # AdamW with weight decay, AMSGrad and a plateau scheduler provide
    # "safe" optimisation
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, min_lr=1e-5
    )

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train[:, None], dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Early stopping to curb overfitting based on MAE
    best_mae = float("inf")
    epochs_no_improve = 0
    patience = 10

    def adapt_model(severity: float | None = None) -> None:
        if drift_detector is None:
            return
        window = drift_detector.window_size
        x_recent = X_train[-window:]
        y_recent = y_train[-window:]
        for name, param in model.named_parameters():
            param.requires_grad = name in [
                "classical_head.0.weight",
                "classical_head.0.bias",
                "classical_head.3.weight",
                "classical_head.3.bias",
            ]
        adapt_opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=1e-4,
            amsgrad=True,
        )
        adjust_learning_rate(adapt_opt, severity, lr)
        model.train()
        adapt_opt.zero_grad()
        out = model(x_recent)
        adapt_loss = criterion(out, y_recent)
        adapt_loss.backward()
        adapt_opt.step()
        for param in model.parameters():
            param.requires_grad = True

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # Step the scheduler on the MAE to align with the early-stopping metric
        mae = loss.item()
        scheduler.step(mae)
        if drift_detector is not None:
            drift, prev, curr = drift_detector.update(mae)
            if drift:
                severity = (curr - prev) / prev if prev else None
                adjust_learning_rate(optimizer, severity, lr)
                drift_detector.log("QProphet", epoch + 1, prev, curr)
                print(f"[QProphet] Drift detected at epoch {epoch + 1}. Adapting...")
                adapt_model(severity)

        if mae < best_mae - 1e-6:
            best_mae = mae
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[QProphet] Early stopping at epoch {epoch + 1}")
                break

        print(f"[QProphet] Epoch {epoch + 1}/{epochs} - MAE: {mae:.6f}")

    # Evaluate correlation and error on training data for diagnostics
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train).cpu()
        preds = model(X_test).cpu().numpy().flatten()

    corr = float(
        torch.corrcoef(torch.stack((train_preds.squeeze(), y_train.squeeze())))[0, 1]
    )
    mse = nn.functional.mse_loss(train_preds, y_train).item()
    print(f"[QProphet] Train Corr: {corr:.3f} - MSE: {mse:.6f}")

    return preds
