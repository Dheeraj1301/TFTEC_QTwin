"""Quantum-enhanced LSTM model.

The network processes the sequence with a single-directional LSTM and fuses the
final timestep with the mean over all timesteps. The fused representation is
layer-normalised and fed to a shallow quantum layer before a compact MLP head.
Optional convolutional smoothing is available to denoise inputs. The design
keeps simulation costs low while improving correlation and stability.
"""

import random
import numpy as np
import torch
import torch.nn as nn

from utils.quantum_layers import QuantumLayer, n_qubits
from utils.drift_detection import DriftDetector, adjust_learning_rate


class QLSTMModel(nn.Module):
    """Unidirectional LSTM followed by residual fusion and a quantum readout."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 16,
        q_depth: int = 2,
        use_conv: bool = False,
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

        # LayerNorm on the LSTM hidden dimension prior to the quantum layer.
        self.ln = nn.LayerNorm(hidden_size)

        # Quantum circuit with minimal entanglement depth (1–2) and dropout
        # afterwards to reduce overfitting when data augmentation is used. The
        # ``q_depth`` argument is clamped to this safe range.
        depth = max(1, min(q_depth, 2))
        self.q_layer = QuantumLayer(n_layers=depth)
        self.q_dropout = nn.Dropout(0.3)

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

        # Residual fusion: average of last timestep and mean over sequence
        final = lstm_out[:, -1, :]
        mean = torch.mean(lstm_out, dim=1)
        fused = (final + mean) * 0.5

        # Normalise and keep only the first n_qubits features for the QNode
        normed = self.ln(fused)
        q_out = self.q_layer(normed[:, :n_qubits])
        q_out = self.q_dropout(q_out)

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
    drift_detector: DriftDetector | None = None,
):
    """Train ``QLSTMModel`` and return predictions for ``X_test``."""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = X_train.shape[2]
    model = QLSTMModel(num_features, hidden_size=hidden_size, q_depth=q_depth).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train[:, None], dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    def adapt_model(severity: float | None = None) -> None:
        """Retrain the final layers on the most recent window of data."""
        if drift_detector is None:
            return
        window = drift_detector.window_size
        x_recent = X_train[-window:]
        y_recent = y_train[-window:]
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc")
        adapt_opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )
        adjust_learning_rate(adapt_opt, severity, lr)
        model.train()
        adapt_opt.zero_grad()
        out = model(x_recent)
        adapt_loss = criterion(out, y_recent)
        adapt_loss.backward()
        adapt_opt.step()
        for param in model.parameters():
            param.requires_grad = True

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        mae = torch.nn.functional.l1_loss(output, y_train).item()
        if drift_detector is not None:
            drift, prev, curr = drift_detector.update(mae)
            if drift:
                severity = (curr - prev) / prev if prev else None
                adjust_learning_rate(optimizer, severity, lr)
                drift_detector.log("QLSTM", epoch + 1, prev, curr)
                print(f"[QLSTM] Drift detected at epoch {epoch + 1}. Adapting...")
                adapt_model(severity)

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
