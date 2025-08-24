import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.dropout = nn.Dropout(0.15)
        self._lambda = nn.Parameter(torch.zeros(1))
        self._last_attn: torch.Tensor | None = None
        self._ema_attn: torch.Tensor | None = None
        self._alpha = 0.1  # EMA smoothing factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a causal embedding of shape ``(batch, out_dim)``."""
        if x.dim() == 3:
            x = x.mean(dim=1)
        # ``x`` is now (batch, sensors). Reshape so sensors form the sequence
        # dimension for self-attention.
        x = x.unsqueeze(-1)  # (batch, sensors, 1)
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True)
        # Average heads and batch for a stable sensor→sensor matrix
        current = attn_weights.mean(dim=0).mean(dim=0)
        if self._ema_attn is None:
            self._ema_attn = current
        else:
            self._ema_attn = self._alpha * current + (1 - self._alpha) * self._ema_attn
        self._last_attn = self._ema_attn
        attn_out = attn_out.squeeze(-1)
        emb = self.proj(attn_out)
        emb = F.normalize(emb, p=2, dim=-1)
        emb = self.dropout(emb)
        return emb

    def lambda_value(self) -> torch.Tensor:
        return torch.clamp(torch.sigmoid(self._lambda), 0.1, 0.9)

    def adjust_lambda(self, severity: float | None) -> None:
        if severity is None:
            return
        with torch.no_grad():
            self._lambda.add_(float(severity))
            lower = torch.logit(torch.tensor(0.1))
            upper = torch.logit(torch.tensor(0.9))
            self._lambda.clamp_(lower, upper)

    def attention_matrix(self) -> torch.Tensor | None:
        """Return the last computed attention matrix if available."""
        return self._last_attn
