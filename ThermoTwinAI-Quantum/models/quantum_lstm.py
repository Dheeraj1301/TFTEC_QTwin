"""Quantum-enhanced LSTM model.

This module implements a *stable* hybrid model composed of a bidirectional
LSTM followed by a single quantum layer.  Previous experiments introduced
deep/staked recurrent layers and attention mechanisms which degraded the
forecasting performance.  Those components have been removed and the quantum
circuit depth is constrained to a single layer to maintain simulation
stability.
"""

import math
import torch
import torch.nn as nn

from utils.quantum_layers import QuantumLayer


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to the input sequence."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:seq_len]


class QLSTMModel(nn.Module):
    """Minimal bidirectional LSTM with a quantum readout."""

    def __init__(
        self, input_size: int, hidden_size: int = 16, q_depth: int = 1
    ) -> None:
        super().__init__()

        # Positional encoding to provide explicit time information
        self.pos_encoder = PositionalEncoding(input_size)

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

        # Projection for residual connection from input to LSTM output
        self.input_proj = nn.Linear(input_size, hidden_size * 2)

        # Gated attention over LSTM outputs to preserve temporal signals
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.attn_gate = nn.Linear(hidden_size * 2, 1)

        # Normalize features passed to the quantum layer
        self.norm = nn.LayerNorm(4)

        # FINAL_FIX: single shallow quantum layer for stability
        self.q_layer = QuantumLayer(n_layers=q_depth)

        # Dropout regularization before classical readout
        self.dropout = nn.Dropout(0.1)

        # Classical readout after quantum layer
        self.fc = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add positional information and temporal convolution over features
        x = self.pos_encoder(x)
        x = x.transpose(1, 2)
        x = torch.relu(self.temporal_conv(x))
        x = x.transpose(1, 2)

        # Residual connection: project input to match LSTM output dimensions
        residual = self.input_proj(x)

        # ``lstm_out`` has shape (batch, seq, hidden*2)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out + residual

        # Compute gated attention weights across the sequence dimension
        attn_scores = self.attention(lstm_out)
        gate = torch.sigmoid(self.attn_gate(lstm_out))
        attn_scores = attn_scores * gate
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
