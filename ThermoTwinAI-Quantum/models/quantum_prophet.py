"""Quantum-augmented NeuralProphet-style model.

This version removes recent experimental additions and reverts to the
minimal design that previously achieved the best scores.  A simple
projection reduces the input to four features which are normalised and fed
to a shallow quantum circuit.  A lightweight classical head then predicts the
target value.
"""

import torch
import torch.nn as nn

from utils.quantum_layers import QuantumLayer, n_qubits


class QProphetModel(nn.Module):
    """Linear projection into a quantum layer with a small classical head."""

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 16,
        q_depth: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # Project the input sequence to four features using a 1×1 convolution
        self.input_proj = nn.Conv1d(num_features, n_qubits, kernel_size=1)

        # Normalize features before entering the quantum layer
        self.norm = nn.LayerNorm(n_qubits)

        # Shallow quantum circuit
        self.q_layer = QuantumLayer(n_layers=q_depth)

        # Classical head: Linear -> BatchNorm1d -> ReLU
        self.classical_head = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input comes as (batch, seq, features); rearrange for Conv1d
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)  # (batch, n_qubits, seq)
        x = x.mean(dim=-1)  # aggregate over time -> (batch, n_qubits)
        x = self.norm(x)
        x = self.q_layer(x)
        x = self.classical_head(x)
        x = self.dropout(x)
        return self.out(x)


def train_quantum_prophet(
    X_train,
    y_train,
    X_test,
    epochs: int = 50,
    lr: float = 0.005,
    hidden_dim: int = 16,
    q_depth: int = 2,
):
    """Train ``QProphetModel`` and return predictions for ``X_test``."""

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
