import torch
import torch.nn as nn

from .quantum_layers import n_qubits


class AdaptiveCausalGraphAttention(nn.Module):
    """Learn sensor-wise causal influence via self-attention."""

    def __init__(self, n_sensors: int, out_dim: int | None = None, n_heads: int = 4) -> None:
        super().__init__()
        out_dim = out_dim or n_qubits
        # Ensure valid head configuration
        n_heads = max(1, min(n_heads, n_sensors))
        if n_sensors % n_heads != 0:
            n_heads = 1
        self.attn = nn.MultiheadAttention(
            embed_dim=n_sensors, num_heads=n_heads, batch_first=True
        )
        self.proj = nn.Linear(n_sensors, out_dim)
        self._lambda = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a causal embedding of shape ``(batch, out_dim)``."""
        if x.dim() == 3:
            x = x.mean(dim=1)
        x = x.unsqueeze(1)
        attn_out, _ = self.attn(x, x, x)
        attn_out = attn_out.squeeze(1)
        return self.proj(attn_out)

    def lambda_value(self) -> torch.Tensor:
        return torch.sigmoid(self._lambda)

    def adjust_lambda(self, severity: float | None) -> None:
        if severity is None:
            return
        with torch.no_grad():
            self._lambda.add_(float(severity))
