"""
Multi-Head Causal Attention with Rotary Position Embeddings (RoPE) and KV-Caching.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute complex exponential frequencies for Rotary Position Embeddings (RoPE).
    freqs = 1.0 / (theta ** (2i / dim)) for i in [0, dim/2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embeddings to query and key tensors in complex space.
    xq: [batch_size, seq_len, n_heads, head_dim]
    xk: [batch_size, seq_len, n_kv_heads, head_dim]
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Slice freqs_cis to current sequence length
    freqs_cis = freqs_cis[: xq.shape[1]].unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim // 2]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Attention with RoPE and autoregressive Key-Value Cache.
    """
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        max_seq_len: int = 2048,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0, f"embed_dim {embed_dim} must be divisible by n_heads {n_heads}"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout_p = dropout
        self.max_seq_len = max_seq_len

        # Precompute RoPE frequencies
        freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = x.shape

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [bsz, seq_len, n_heads, head_dim]
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim)

        # Apply Rotary Position Embeddings
        freqs_cis = self.freqs_cis.to(x.device)
        if kv_cache is not None:
            # During incremental decode, slice frequencies starting from cache length
            past_len = kv_cache[0].shape[1]
            q, k = apply_rotary_emb(q, k, freqs_cis[past_len : past_len + seq_len])
        else:
            q, k = apply_rotary_emb(q, k, freqs_cis[:seq_len])

        # Handle KV-Caching for efficient autoregressive inference
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=1)
            v = torch.cat([prev_v, v], dim=1)

        new_kv_cache = (k, v) if use_cache else None

        # Permute to [bsz, n_heads, seq_len, head_dim] for PyTorch scaled_dot_product_attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # PyTorch SDPA with causal masking when computing on full sequence
        is_causal = (seq_len > 1) and (kv_cache is None)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape back to [bsz, seq_len, embed_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        output = self.out_proj(attn_out)

        return output, new_kv_cache
