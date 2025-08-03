import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=4, depth=2):
        super().__init__()
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (depth, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        return self.qlayer(x)

class QProphetModel(nn.Module):
    def __init__(self, input_channels=5, sequence_length=30):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4)  # Output: (batch, 8, 4)
        )

        self.linear_in = nn.Linear(8 * 4, 4)  # Reduce to 4 for 4-qubit input
        self.norm = nn.LayerNorm(4)

        self.q_layer = QuantumLayer(n_qubits=4, depth=2)

        self.head = nn.Sequential(
            nn.Linear(4, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # to (batch, features, seq_len)
        x = self.cnn(x)  # (batch, 8, 4)
        x = x.view(x.size(0), -1)  # (batch, 32)
        x = self.linear_in(x)     # (batch, 4)
        x = self.norm(x)
        x = self.q_layer(x)       # (batch, 4)
        return self.head(x)       # (batch, 1)
