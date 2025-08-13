import pytest

try:  # pragma: no cover - used only when torch is available
    import torch
except Exception:  # pragma: no cover - torch missing in minimal envs
    torch = None

if torch is None:  # pragma: no cover - skip module import when torch missing
    pytest.skip("torch not installed", allow_module_level=True)

from utils.causal_graph_attention import AdaptiveCausalGraphAttention
from utils.quantum_layers import n_qubits


def test_acga_shape_and_lambda_adjust():
    acga = AdaptiveCausalGraphAttention(n_sensors=5, out_dim=n_qubits)
    x = torch.randn(2, 10, 5)
    emb = acga(x)
    assert emb.shape == (2, n_qubits)
    lam0 = acga.lambda_value().item()
    acga.adjust_lambda(0.5)
    lam1 = acga.lambda_value().item()
    assert lam1 > lam0
