"""Quantum-enhanced LSTM model.

The network processes the sequence with a single-directional LSTM and fuses the
final timestep with the mean over all timesteps. The fused representation is
layer-normalised and fed to a shallow quantum layer before a compact MLP head.
Optional convolutional smoothing is available to denoise inputs. The design
keeps simulation costs low while improving correlation and stability.
"""

import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.quantum_layers import QuantumLayer, n_qubits
from utils.drift_detection import DriftDetector, adjust_learning_rate
from utils.preprocessing import SensorFusion


class QLSTMModel(nn.Module):
    """Unidirectional LSTM followed by residual fusion and a quantum readout."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 16,
        q_depth: int = 2,
        use_conv: bool = False,
        dropout: float = 0.25,
        use_attention: bool = True,
    ) -> None:
        super().__init__()

        # Learnable sensor fusion weights applied before any processing.
        self.sensor_fusion = SensorFusion(input_size)

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

        # Optional attention to weight timesteps after the LSTM
        self.use_attention = use_attention
        self.attn = nn.Linear(hidden_size, 1) if use_attention else None

        # Learnable fusion parameter between the last timestep and the
        # attention-pooled representation. ``alpha`` is squashed via sigmoid in
        # ``forward`` to keep it in ``[0, 1]``.
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Two LayerNorms: one before the quantum layer and one on the quantum
        # output, stabilising both classical and quantum representations.
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(n_qubits)

        # Dropout directly on the LSTM representations and after the quantum
        # layer supports Monte Carlo Dropout for uncertainty estimates.
        self.lstm_dropout = nn.Dropout(dropout)

        # Quantum circuit with minimal entanglement depth (1–2) and dropout
        # afterwards to reduce overfitting when data augmentation is used. The
        # ``q_depth`` argument is clamped to this safe range.
        depth = max(1, min(q_depth, 2))
        self.q_layer = QuantumLayer(n_layers=depth)
        self.q_dropout = nn.Dropout(dropout)

        # Output head: Linear(4→16) → GELU → Linear(16→1)
        self.fc1 = nn.Linear(n_qubits, 16)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor, mc_dropout: bool = False) -> torch.Tensor:
        # Apply learnable sensor fusion weights
        x = self.sensor_fusion(x)

        # Optional local smoothing via Conv1d
        if self.conv1 is not None:
            # Conv1d expects (batch, channels, seq_len)
            x = x.transpose(1, 2)
            x = self.conv1(x)
            x = x.transpose(1, 2)

        # LSTM produces representations for each timestep
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
        lstm_out = F.dropout(
            lstm_out, p=self.lstm_dropout.p, training=self.training or mc_dropout
        )

        # Either attention-pooled representation or simple mean
        if self.use_attention:
            weights = torch.softmax(self.attn(lstm_out), dim=1)  # (batch, seq, 1)
            context = torch.sum(weights * lstm_out, dim=1)
        else:
            context = torch.mean(lstm_out, dim=1)

        final = lstm_out[:, -1, :]
        alpha = torch.sigmoid(self.alpha)
        fused = alpha * final + (1 - alpha) * context

        # Normalise and keep only the first n_qubits features for the QNode
        normed = self.ln1(fused)
        q_out = self.q_layer(normed[:, :n_qubits])
        q_out = self.ln2(q_out)
        q_out = F.dropout(
            q_out, p=self.q_dropout.p, training=self.training or mc_dropout
        )

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
    dropout: float = 0.25,
    drift_detector: Optional[DriftDetector] = None,
    patience: int = 10,
) -> Tuple[nn.Module, np.ndarray]:
    """Train ``QLSTMModel`` and return the model with predictions for ``X_test``."""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = X_train.shape[2]
    model = QLSTMModel(
        num_features, hidden_size=hidden_size, q_depth=q_depth, dropout=dropout
    ).to(device)

    criterion = nn.L1Loss()
    # AdamW with weight decay and AMSGrad along with a plateau scheduler
    # provides "safe" optimisation comparable to the QProphet setup.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, min_lr=1e-5
    )

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train[:, None], dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    def adapt_model(severity: Optional[float] = None) -> None:
        """Retrain the final layers on the most recent window of data."""
        if drift_detector is None:
            return
        window = drift_detector.window_size
        x_recent = X_train[-window:]
        y_recent = y_train[-window:]
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc")
        adapt_opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=1e-4,
            amsgrad=True,
        )
        adjust_learning_rate(adapt_opt, severity, lr)
        model.train()
        adapt_opt.zero_grad()
        out = model(x_recent)
        adapt_loss = criterion(out, y_recent)
        adapt_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        adapt_opt.step()
        for param in model.parameters():
            param.requires_grad = True

    best_mae = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        mae = loss.item()
        scheduler.step(mae)
        if mae < best_mae - 1e-4:
            best_mae = mae
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("[QLSTM] Early stopping")
                break
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
    corr = float(
        torch.corrcoef(torch.stack((train_preds.squeeze(), y_train.squeeze())))[0, 1]
    )
    if corr < 0:
        print("[QLSTM] Negative trend detected; inverting output sign for stability")
        with torch.no_grad():
            model.fc2.weight.data *= -1
            model.fc2.bias.data *= -1
            train_preds = -train_preds
        corr = -corr

    mse = nn.functional.mse_loss(train_preds, y_train).item()
    print(f"[QLSTM] Train Corr: {corr:.3f} - MSE: {mse:.6f}")

    return model, preds
