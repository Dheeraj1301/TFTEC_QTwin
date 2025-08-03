"""Quantum-enhanced LSTM model.

This file restores the simple architecture that previously yielded the best
results.  The model feeds the final LSTM timestep directly into a shallow
quantum layer.  A small classical head then produces the regression output.
Stability is improved by normalising the quantum inputs while keeping the
quantum circuit depth low.
"""

import torch
import torch.nn as nn

from utils.quantum_layers import QuantumLayer


class QLSTMModel(nn.Module):
    """Bidirectional LSTM followed by a shallow quantum readout."""

    def __init__(self, input_size: int, hidden_size: int = 16, q_depth: int = 2) -> None:
        super().__init__()

        # Bidirectional LSTM to capture global temporal context
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        # LayerNorm stabilizes features before entering the quantum layer
        self.norm = nn.LayerNorm(4)

        # Shallow quantum circuit with entanglement depth of two
        self.q_layer = QuantumLayer(n_layers=q_depth)

        # Lightweight classical head: Linear -> ReLU -> Linear
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM produces representations for each timestep
        lstm_out, _ = self.lstm(x)
        final = lstm_out[:, -1, :]  # output of the final timestep

        # Normalize and pass only first four features to the quantum layer
        q_input = self.norm(final[:, :4])
        q_out = self.q_layer(q_input)

        # Lightweight output head to avoid overfitting
        out = torch.relu(self.fc1(q_out))
        return self.fc2(out)


def train_quantum_lstm(
    X_train,
    y_train,
    X_test,
    epochs: int = 50,
    lr: float = 0.005,
    hidden_size: int = 16,
    q_depth: int = 2,
):
    """Train ``QLSTMModel`` and return predictions for ``X_test``."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = X_train.shape[2]
    model = QLSTMModel(num_features, hidden_size=hidden_size, q_depth=q_depth).to(device)

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
        print(f"[QLSTM] Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test).cpu().numpy().flatten()
    return preds
