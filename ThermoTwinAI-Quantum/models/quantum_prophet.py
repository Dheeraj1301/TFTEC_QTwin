"""Quantum-augmented NeuralProphet-style model with CNN preprocessing.

The original implementation flattened the entire input window and employed a
deep quantum circuit.  This revision introduces a lightweight 1D CNN to extract
temporal features and restricts the quantum component to a single layer for
improved stability.
"""

import torch
import torch.nn as nn

from utils.quantum_layers import QuantumLayer


class QProphetModel(nn.Module):
    """1D CNN feature extractor followed by a quantum-enhanced regressor."""

    def __init__(
        self, num_features: int, hidden_dim: int = 16, q_depth: int = 4
    ) -> None:
        super().__init__()

        # CNN over the time dimension. Input shape: (batch, seq, features)
        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Projection to four quantum inputs
        self.feature_proj = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

        # Quantum layer with increased entangling depth
        self.q_layer = QuantumLayer(n_layers=q_depth)

        # Normalize quantum outputs before the classical head
        self.norm = nn.LayerNorm(4)

        # Deeper classical post-quantum head
        self.classical_head = nn.Sequential(
            nn.BatchNorm1d(4),
            nn.Dropout(0.2),
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rearrange to (batch, channels, seq) for Conv1d
        x = x.permute(0, 2, 1)
        cnn_out = self.cnn(x).squeeze(-1)  # (batch, 8)

        x = self.feature_proj(cnn_out)
        x = self.q_layer(x)
        x = self.norm(x)
        return self.classical_head(x)


def train_quantum_prophet(
    X_train,
    y_train,
    X_test,
    epochs: int = 50,
    lr: float = 0.005,
    hidden_dim: int = 16,
    q_depth: int = 4,
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
