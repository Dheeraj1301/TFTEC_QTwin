"""Quantum-enhanced LSTM model with temporal attention.

The original implementation relied solely on the final LSTM timestep.  Here a
lightweight multi-head attention layer examines the entire sequence and a
residual fusion combines the last state with the sequence mean.  The fused
representation is normalised and passed through a shallow quantum circuit and a
compact MLP head.  The goal is to improve correlation while remaining
CPU-friendly.
"""

import torch
import torch.nn as nn

from utils.quantum_layers import QuantumLayer, n_qubits


class QLSTMModel(nn.Module):
    """Unidirectional LSTM followed by attention, fusion and a quantum readout."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 16,
        q_depth: int = 2,
        use_conv: bool = True,
    ) -> None:
        super().__init__()

        # Optional 1D convolution to smooth local temporal patterns.
        self.conv1 = (
            nn.Conv1d(input_size, input_size, kernel_size=3, padding=1)
            if use_conv
            else None
        )

        # Single-directional LSTM processing the sequence.
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
            bidirectional=False,
        )

        # Temporal attention to capture long-range dependencies.
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=2, batch_first=True
        )

        # LayerNorm on the LSTM hidden dimension prior to the quantum layer.
        self.ln = nn.LayerNorm(hidden_size)

        # Quantum circuit with minimal entanglement (depth=1). ``q_depth`` is
        # retained for API compatibility even though it is fixed.
        self.q_layer = QuantumLayer(n_layers=1)

        # Output head: Linear(4→16) → GELU → Linear(16→1)
        self.fc1 = nn.Linear(n_qubits, 16)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optional local smoothing via Conv1d
        if self.conv1 is not None:
            # Conv1d expects (batch, channels, seq_len)
            x = x.transpose(1, 2)
            x = self.conv1(x)
            x = x.transpose(1, 2)

        # LSTM produces representations for each timestep
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)

        # Attention emphasises informative timesteps
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)

        # Residual fusion: last timestep + mean over sequence
        fused = attn_out[:, -1, :] + attn_out.mean(dim=1)

        # Normalise and keep only the first n_qubits features for the QNode
        normed = self.ln(fused)
        q_out = self.q_layer(normed[:, :n_qubits])

        # Post-quantum MLP head with output clamped to [-3, 3]
        out = self.fc2(self.act(self.fc1(q_out)))
        return torch.clamp(out, -3, 3)


def train_quantum_lstm(
    X_train,
    y_train,
    X_test,
    epochs: int = 50,
    lr: float = 0.001,
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
