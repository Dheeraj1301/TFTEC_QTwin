# models/quantum_lstm.py
import torch
import torch.nn as nn
from utils.quantum_layers import QuantumLayer
import numpy as np

class QLSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 8, batch_first=True)
        self.q_layer = QuantumLayer()
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        q_input = last_out[:, :4]
        q_out = self.q_layer(q_input)
        return self.fc(q_out)

def train_quantum_lstm(X_train, y_train, X_test, epochs=50, lr=0.005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QLSTMModel(1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train = torch.tensor(X_train[:, :, None], dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train[:, None], dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test[:, :, None], dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        print(f"[QLSTM] Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test).cpu().numpy().flatten()
    return preds
