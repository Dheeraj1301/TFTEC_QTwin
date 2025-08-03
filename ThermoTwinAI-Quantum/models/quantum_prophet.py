"""Quantum-augmented NeuralProphet-style model.

This version removes recent experimental additions and reverts to the
minimal design that previously achieved the best scores.  A simple
projection reduces the input to four features which are normalised and fed
to a shallow quantum circuit.  A lightweight classical head then predicts the
target value.
"""

import torch
import torch.nn as nn

from utils.quantum_layers import QuantumLayer


class QProphetModel(nn.Module):
    """Linear projection into a quantum layer with a small classical head."""

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 16,
        q_depth: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Project the input sequence to four features using a 1×1 convolution
        self.input_proj = nn.Conv1d(num_features, 4, kernel_size=1)

        # Normalize features before entering the quantum layer
        self.norm = nn.LayerNorm(4)

        # Shallow quantum circuit
        self.q_layer = QuantumLayer(n_layers=q_depth)

        # Classical head: Linear -> BatchNorm1d -> ReLU -> Dropout -> Linear
        self.classical_head = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input comes as (batch, seq, features); rearrange for Conv1d
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)  # (batch, 4, seq)
        x = x.mean(dim=-1)  # aggregate over time -> (batch, 4)
        x = self.norm(x)
        x = self.q_layer(x)
        return self.classical_head(x)


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
