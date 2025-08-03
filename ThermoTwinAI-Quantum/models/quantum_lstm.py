# models/quantum_lstm.py
import torch
import torch.nn as nn
from utils.quantum_layers import QuantumLayer


class QLSTMModel(nn.Module):
    """Quantum enhanced LSTM model.

    A deeper/bidirectional LSTM encodes the temporal dynamics. The final
    hidden state is projected down to four features which are passed through
    one or more quantum layers. The outputs of the last quantum layer are fed
    to a small MLP for the final prediction. An attention mechanism is
    included to allow the network to emphasise informative time steps and the
    quantum inputs are normalised to keep rotations in a stable range.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 16,
        num_layers: int = 2,
        bidirectional: bool = True,
        q_layers: int = 8,
        n_q_layers: int = 1,
        mlp_hidden: int = 8,
    ) -> None:
        super().__init__()

        self.bidirectional = bidirectional
        lstm_hidden = hidden_size * (2 if bidirectional else 1)

        # Stacked/bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Simple self-attention over the time dimension
        self.attn = nn.MultiheadAttention(lstm_hidden, num_heads=1, batch_first=True)

        # Project to the number of qubits expected by the quantum layer
        self.q_proj = nn.Linear(lstm_hidden, 4)
        self.q_norm = nn.LayerNorm(4)

        # Allow stacking of multiple quantum layers
        self.q_layers = nn.ModuleList([QuantumLayer(n_layers=q_layers) for _ in range(n_q_layers)])

        # Lightweight MLP for post-quantum processing
        self.post_mlp = nn.Sequential(
            nn.Linear(4, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)

        # Attention emphasises informative time steps
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        last_out = attn_out[:, -1, :]

        # Quantum feature map with normalisation and optional stacking
        q_input = torch.tanh(self.q_norm(self.q_proj(last_out)))
        for q in self.q_layers:
            q_input = q(q_input)

        return self.post_mlp(q_input)


def train_quantum_lstm(X_train, y_train, X_test, config: dict | None = None):
    """Train the :class:`QLSTMModel` and return predictions for ``X_test``.

    Parameters are provided via ``config`` for easy hyper-parameter tuning.
    """

    if config is None:
        config = {}

    epochs = config.get("epochs", 50)
    lr = config.get("lr", 0.005)
    hidden_size = config.get("hidden_size", 16)
    num_layers = config.get("num_layers", 2)
    bidirectional = config.get("bidirectional", True)
    q_layers = config.get("q_layers", 8)
    n_q_layers = config.get("n_q_layers", 1)
    mlp_hidden = config.get("mlp_hidden", 8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = X_train.shape[2]
    model = QLSTMModel(
        num_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        q_layers=q_layers,
        n_q_layers=n_q_layers,
        mlp_hidden=mlp_hidden,
    ).to(device)

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
