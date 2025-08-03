# utils/quantum_layers.py
import pennylane as qml
import torch
from torch.nn import Module

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    """Basic variational circuit used inside :class:`QuantumLayer`.

    Args:
        inputs (tensor): Rotational angles for the input layer.
        weights (tensor): Trainable weights of the entangling layers.
    """

    # Encode features on the qubits
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    # Strong entanglement between qubits with a configurable depth
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class QuantumLayer(Module):
    """Thin wrapper around ``qml.qnn.TorchLayer`` allowing configurable depth.

    Parameters
    ----------
    n_layers: int
        Number of ``StronglyEntanglingLayers`` repetitions. Increasing this
        value increases the expressive power of the quantum circuit at the
        cost of additional simulation time.
    """

    def __init__(self, n_layers: int = 6):
        super().__init__()
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the underlying quantum circuit for a batch of inputs.

        Pennylane's :class:`~pennylane.qnn.TorchLayer` does not automatically
        broadcast over the batch dimension in some environments. This wrapper
        manually iterates over the batch dimension so that ``x`` can be of
        shape ``(batch, n_qubits)``.
        """

        # Ensure batched tensor
        if x.ndim == 1:
            x = x.unsqueeze(0)

        outputs = [self.q_layer(sample) for sample in x]
        return torch.stack(outputs, dim=0)
