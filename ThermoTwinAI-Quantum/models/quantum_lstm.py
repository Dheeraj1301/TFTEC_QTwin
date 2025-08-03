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

        # Single-directional LSTM; final timestep is used as classical features
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
            bidirectional=False,
        )

        # LayerNorm to stabilise the fused temporal features before the
        # quantum layer.  ``hidden_size`` equals the dimensionality of the LSTM
        # outputs and thus the residual fusion vector.
        self.ln = nn.LayerNorm(hidden_size)

        # Quantum circuit depth is fixed at two layers to reduce overfitting and
        # entanglement complexity as per the research specification.  ``q_depth``
        # is kept for API compatibility but ignored internally.
        self.q_layer = QuantumLayer(n_layers=2)

        # Post-quantum head: Linear(4→16) → GELU → Linear(16→1)
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
        lstm_out, _ = self.lstm(x)

        # Residual fusion of final timestep with mean-pooled context
        fused = lstm_out[:, -1, :] + lstm_out.mean(dim=1)

        # Apply LayerNorm before slicing to the quantum-sized subset.  This
        # stabilises the statistics of the temporal features without altering
        # the model's external interface.
        fused = self.ln(fused)
        q_input = fused[:, :n_qubits]
        q_out = self.q_layer(q_input)

        # Post-quantum MLP head
        out = self.act(self.fc1(q_out))
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
