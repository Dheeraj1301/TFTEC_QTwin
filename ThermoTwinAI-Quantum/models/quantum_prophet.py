# models/quantum_prophet.py
import torch
import torch.nn as nn
from utils.quantum_layers import QuantumLayer


class QProphetModel(nn.Module):
    """Feed-forward network augmented with a quantum feature map."""

    def __init__(self, input_size: int, hidden_dim: int = 16, q_layers: int = 8):
        super().__init__()

        # Reduce the potentially large multivariate window into four features
        self.feature_proj = nn.Linear(input_size, 4)
        self.q_layer = QuantumLayer(n_layers=q_layers)

        # Classical post-processing of quantum outputs
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_proj(x)
        q_out = self.q_layer(x)
        return self.net(q_out)


def train_quantum_prophet(
    X_train,
    y_train,
    X_test,
    epochs: int = 50,
    lr: float = 0.005,
    hidden_dim: int = 16,
    q_layers: int = 8,
):
    """Train the ``QProphetModel`` and return predictions for ``X_test``."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Flatten windowed data to a single feature vector per sample
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    input_size = X_train.shape[1]

    model = QProphetModel(input_size, hidden_dim=hidden_dim, q_layers=q_layers).to(device)
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
