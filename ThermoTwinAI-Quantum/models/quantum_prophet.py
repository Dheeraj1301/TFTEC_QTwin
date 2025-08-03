"""Quantum-augmented NeuralProphet-style model.

This module originally relied on heavy third-party libraries such as
``torch`` and ``pennylane``.  Importing the file without those dependencies
installed would raise an ``ImportError`` before the training helper function
could be accessed, which in turn triggered the
``ImportError: cannot import name 'train_quantum_prophet'`` message observed
by users.  To make the failure mode clearer we attempt the imports lazily and
provide a stub implementation when the dependencies are missing.
"""

from typing import Any

try:  # pragma: no cover - executed only when deps are available
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
        """Linear projection into a quantum layer with a small classical head."""

        def __init__(
            self,
            num_features: int,
            hidden_dim: int = 32,
            q_depth: int = 2,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()

            # 1D convolution followed by ReLU prior to the quantum circuit
            self.pre_q = nn.Sequential(
                nn.Conv1d(num_features, n_qubits, kernel_size=3, padding=1),
                nn.ReLU(),
            )

            # Normalise to stabilise variance before entering the quantum circuit
            self.norm = nn.LayerNorm(4)

            # Quantum layer configured with depth=1
            self.q_layer = QuantumLayer(n_layers=1)

            # Post-quantum head: Linear(4→32) → BatchNorm1d → ReLU → Dropout(0.2) → Linear(32→1)
            # ``dropout`` argument retained for backward compatibility but fixed
            # to 0.2 as per the updated design.
            self.classical_head = nn.Sequential(
                nn.Linear(n_qubits, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
            )

            # Previous residual and sigmoid heads removed to reduce bias toward
            # inverted trends; output is linear for downstream processing.

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Input comes as (batch, seq, features); rearrange for Conv1d
            x = x.permute(0, 2, 1)
            x = self.pre_q(x)  # conv + ReLU -> (batch, n_qubits, seq)

            # Use only the last timestep then normalise before quantum layer
            x = x[:, :, -1]
            x = self.norm(x)
            x = self.q_layer(x)
            out = self.classical_head(x)
            return out


def train_quantum_prophet(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    epochs: int = 50,
    lr: float = 0.005,
    hidden_dim: int = 32,
    q_depth: int = 2,
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_features = X_train.shape[2]
    model = QProphetModel(num_features, hidden_dim=hidden_dim, q_depth=q_depth).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train[:, None], dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        print(f"[QProphet] Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.6f}")

    # Evaluate correlation and error on training data for diagnostics
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train).cpu()
        preds = model(X_test).cpu().numpy().flatten()

    corr = float(torch.corrcoef(
        torch.stack((train_preds.squeeze(), y_train.squeeze()))
    )[0, 1])
    mse = nn.functional.mse_loss(train_preds, y_train).item()
    print(f"[QProphet] Train Corr: {corr:.3f} - MSE: {mse:.6f}")

    return preds
