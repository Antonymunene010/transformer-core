"""
Modern architectural layers: RMSNorm and SwiGLU Feed-Forward Network.
Used in frontier models like LLaMA 3, Mistral, and Gemma.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Simplifies LayerNorm by enforcing invariance to scaling without mean centering,
    reducing computational overhead by ~7-10% without loss of training stability.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwiGLU(nn.Module):
    """
    Swish Gated Linear Unit (SwiGLU) Feed-Forward Network.
    Uses three linear projections: gate_proj, up_proj, and down_proj.
    Activation: F.silu(gate_proj(x)) * up_proj(x)
    """
    def __init__(self, hidden_dim: int, ffn_dim: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=bias)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=bias)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
