# models/quantum_prophet.py
import torch
import torch.nn as nn
from utils.quantum_layers import QuantumLayer
import numpy as np

class QProphetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_layer = QuantumLayer()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        q_out = self.q_layer(x[:, :4])
        x = self.relu(self.fc1(q_out))
        return self.fc2(x)

def train_quantum_prophet(X_train, y_train, X_test, epochs=50, lr=0.005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QProphetModel().to(device)
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
        print(f"[QProphet] Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test).cpu().numpy().flatten()
    return preds
