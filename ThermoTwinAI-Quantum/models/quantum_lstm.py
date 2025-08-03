"""Quantum-enhanced LSTM model.

This file restores the simple architecture that previously yielded the best
results.  The model feeds the final LSTM timestep directly into a shallow
quantum layer.  A small classical head then produces the regression output.
Stability is improved by normalising the quantum inputs while keeping the
quantum circuit depth low.
"""

import torch
import torch.nn as nn

from utils.quantum_layers import QuantumLayer, n_qubits


class QLSTMModel(nn.Module):
    """Unidirectional LSTM followed by a shallow quantum readout."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 16,
        q_depth: int = 2,
        use_conv: bool = True,
    ) -> None:
        super().__init__()

        # Optional 1D convolution to smooth short-term fluctuations.  The
        # convolution operates on the feature/channel dimension and preserves
        # dimensionality so downstream interfaces remain unchanged.
        #
        # Renamed to ``conv1`` to clarify its temporal nature.
        self.conv1 = (
            nn.Conv1d(input_size, input_size, kernel_size=3, padding=1)
            if use_conv
            else None
        )

        # Single-directional LSTM; the last timestep alone is used as features
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
            bidirectional=False,
        )

        # LayerNorm now operates on ``input_size`` (four features) before the
        # quantum circuit as per the new specification.
        self.input_size = input_size  # store for slicing
        self.ln = nn.LayerNorm(input_size)

        # Quantum circuit with entanglement depth fixed to one layer
        # (``q_depth`` kept for API compatibility).
        self.q_layer = QuantumLayer(n_layers=1)

        # Updated output head: Linear(4→16) → ReLU → Linear(16→1)
        self.fc1 = nn.Linear(n_qubits, 16)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optional local smoothing via Conv1d
        if self.conv1 is not None:
            # Conv1d expects (batch, channels, seq_len)
            x = x.transpose(1, 2)
            x = self.conv1(x)
            x = x.transpose(1, 2)

        # LSTM produces representations for each timestep
        lstm_out, _ = self.lstm(x)

        # Use only the final timestep; removed residual/average pooling
        last = lstm_out[:, -1, :]

        # Normalise features and select the qubit-sized subset
        normed = self.ln(last[:, : self.input_size])
        q_out = self.q_layer(normed[:, :n_qubits])

        # Post-quantum MLP head followed by clamping to [-5, 5]
        out = self.act(self.fc1(q_out))
        out = self.fc2(out)
        return torch.clamp(out, -5, 5)


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

    # Evaluate correlation and error on training data for diagnostics
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train).cpu()
        preds = model(X_test).cpu().numpy().flatten()

    # Pearson correlation and MSE on training set
    corr = float(torch.corrcoef(
        torch.stack((train_preds.squeeze(), y_train.squeeze()))
    )[0, 1])
    mse = nn.functional.mse_loss(train_preds, y_train).item()
    print(f"[QLSTM] Train Corr: {corr:.3f} - MSE: {mse:.6f}")

    return preds
