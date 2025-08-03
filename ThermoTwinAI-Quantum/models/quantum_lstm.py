"""Quantum-enhanced LSTM model.

This module implements a *stable* hybrid model composed of a bidirectional
LSTM followed by a single quantum layer.  Previous experiments introduced
deep/staked recurrent layers and attention mechanisms which degraded the
forecasting performance.  Those components have been removed and the quantum
circuit depth is constrained to a single layer to maintain simulation
stability.
"""

import torch
import torch.nn as nn

from utils.quantum_layers import QuantumLayer


class QLSTMModel(nn.Module):
    """Minimal bidirectional LSTM with a quantum readout."""

    def __init__(
        self, input_size: int, hidden_size: int = 16, q_depth: int = 1
    ) -> None:
        super().__init__()

        # FINAL_FIX: optional temporal conv to capture short-range dependencies
        self.temporal_conv = nn.Conv1d(
            input_size, input_size, kernel_size=3, padding=1
        )

        # Bidirectional LSTM encodes temporal context
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        # Soft attention over LSTM outputs to preserve temporal signals
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Normalize features passed to the quantum layer
        self.norm = nn.LayerNorm(4)

        # FINAL_FIX: single shallow quantum layer for stability
        self.q_layer = QuantumLayer(n_layers=q_depth)

        # Dropout regularization before classical readout
        self.dropout = nn.Dropout(0.1)

        # Classical readout after quantum layer
        self.fc = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FINAL_FIX: temporal convolution over features
        x = x.transpose(1, 2)
        x = torch.relu(self.temporal_conv(x))
        x = x.transpose(1, 2)

        # ``lstm_out`` has shape (batch, seq, hidden*2)
        lstm_out, _ = self.lstm(x)

        # Compute soft attention weights across the sequence dimension
        attn_scores = self.attention(lstm_out)  # (batch, seq, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        # Slice first four features and normalize before quantum processing
        q_input = self.norm(context[:, :4])
        q_out = self.q_layer(q_input)

        return self.fc(self.dropout(q_out))


def train_quantum_lstm(
    X_train,
    y_train,
    X_test,
    epochs: int = 50,
    lr: float = 0.005,
    hidden_size: int = 16,
    q_depth: int = 1,
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
