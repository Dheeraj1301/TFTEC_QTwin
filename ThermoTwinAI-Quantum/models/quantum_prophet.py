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

            # Project the input sequence to four features using a 1×1 convolution
            self.input_proj = nn.Conv1d(num_features, n_qubits, kernel_size=1)

            # Normalise to stabilise variance before entering the quantum circuit
            self.norm = nn.LayerNorm(4)

            # Residual linear path to model direct classical trends and
            # mitigate sign inversion observed in predictions
            self.residual_head = nn.Linear(n_qubits, 1)

            # Quantum layer with fixed shallow depth (exactly two layers) to
            # maintain stability.  ``AngleEmbedding`` is implicitly implemented
            # via rotations inside :class:`QuantumLayer`.  ``q_depth`` remains in
            # the signature for backward compatibility but is not used.
            self.q_layer = QuantumLayer(n_layers=2)

            # Post-quantum head: Linear(4→32) → BatchNorm1d → GELU → Dropout → Linear(32→1)
            # ``hidden_dim`` is kept in the signature for compatibility but the
            # architecture is fixed to 32 units as per the research setup.
            self.classical_head = nn.Sequential(
                nn.Linear(n_qubits, 32),
                nn.BatchNorm1d(32),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )

            # Bound the final prediction to [0, 1] (the range of the normalised
            # target) which also helps to avoid inverted trends.
            self.output_activation = nn.Sigmoid()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Input comes as (batch, seq, features); rearrange for Conv1d
            x = x.permute(0, 2, 1)
            x = self.input_proj(x)  # (batch, n_qubits, seq)

            # Use the most recent time step rather than averaging the entire
            # window.  Averaging tended to blur the downward CoP trend and could
            # result in predictions with inverted slope.
            x = x[:, :, -1]
            x = self.norm(x)  # normalise features prior to quantum layer

            trend = self.residual_head(x)  # learn linear trend directly
            x = self.q_layer(x)
            out = self.classical_head(x) + trend  # combine quantum and classical paths
            return self.output_activation(out)


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

    model.eval()
    with torch.no_grad():
        preds = model(X_test).cpu().numpy().flatten()
    return preds
