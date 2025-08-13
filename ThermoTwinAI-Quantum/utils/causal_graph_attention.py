import torch
import torch.nn as nn

from .quantum_layers import n_qubits


class AdaptiveCausalGraphAttention(nn.Module):
    """Learn sensor-wise causal influence via self-attention."""

    def __init__(self, n_sensors: int, out_dim: int | None = None) -> None:
        super().__init__()
        out_dim = out_dim or n_qubits
        # Treat each sensor as a token with a single feature to obtain a
        # meaningful sensor→sensor attention matrix.
        self.attn = nn.MultiheadAttention(
            embed_dim=1, num_heads=1, batch_first=True
        )
        self.proj = nn.Linear(n_sensors, out_dim)
        self._lambda = nn.Parameter(torch.zeros(1))
        self._last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a causal embedding of shape ``(batch, out_dim)``."""
        if x.dim() == 3:
            x = x.mean(dim=1)
        # ``x`` is now (batch, sensors). Reshape so sensors form the sequence
        # dimension for self-attention.
        x = x.unsqueeze(-1)  # (batch, sensors, 1)
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True)
        # Average heads and batch for a stable sensor→sensor matrix
        self._last_attn = attn_weights.mean(dim=0).mean(dim=0)
        attn_out = attn_out.squeeze(-1)
        return self.proj(attn_out)

    def lambda_value(self) -> torch.Tensor:
        return torch.sigmoid(self._lambda)

    def adjust_lambda(self, severity: float | None) -> None:
        if severity is None:
            return
        with torch.no_grad():
            self._lambda.add_(float(severity))

    def attention_matrix(self) -> torch.Tensor | None:
        """Return the last computed attention matrix if available."""
        return self._last_attn
