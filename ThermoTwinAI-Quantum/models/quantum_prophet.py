# models/quantum_prophet.py
import torch
import torch.nn as nn
from utils.quantum_layers import QuantumLayer


class QProphetModel(nn.Module):
    """Temporal encoder followed by quantum feature maps and a residual MLP."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 16,
        q_layers: int = 8,
        n_q_layers: int = 1,
        conv_channels: int = 8,
    ):
        super().__init__()

        # Temporal encoding using a lightweight 1D CNN followed by a GRU
        self.conv = nn.Conv1d(n_features, conv_channels, kernel_size=3, padding=1)
        self.gru = nn.GRU(conv_channels, hidden_dim, batch_first=True)

        # Project GRU output to qubit rotations and normalise
        self.q_proj = nn.Linear(hidden_dim, 4)
        self.q_norm = nn.LayerNorm(4)
        self.q_layers = nn.ModuleList([QuantumLayer(n_layers=q_layers) for _ in range(n_q_layers)])

        # Post-quantum processing with skip connection
        self.post_net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.out = nn.Linear(hidden_dim + 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, features)
        x = x.transpose(1, 2)  # (batch, features, window)
        x = torch.relu(self.conv(x))
        x = x.transpose(1, 2)  # (batch, window, conv_channels)
        gru_out, _ = self.gru(x)
        last = gru_out[:, -1, :]

        q_in = torch.tanh(self.q_norm(self.q_proj(last)))
        for q in self.q_layers:
            q_in = q(q_in)

        post = self.post_net(q_in)
        return self.out(torch.cat([post, q_in], dim=1))


def train_quantum_prophet(X_train, y_train, X_test, config: dict | None = None):
    """Train the :class:`QProphetModel` and return predictions for ``X_test``."""

    if config is None:
        config = {}

    epochs = config.get("epochs", 50)
    lr = config.get("lr", 0.005)
    hidden_dim = config.get("hidden_dim", 16)
    q_layers = config.get("q_layers", 8)
    n_q_layers = config.get("n_q_layers", 1)
    conv_channels = config.get("conv_channels", 8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_features = X_train.shape[2]
    model = QProphetModel(
        n_features,
        hidden_dim=hidden_dim,
        q_layers=q_layers,
        n_q_layers=n_q_layers,
        conv_channels=conv_channels,
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
        print(f"[QProphet] Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test).cpu().numpy().flatten()
    return preds
