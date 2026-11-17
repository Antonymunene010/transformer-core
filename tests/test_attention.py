"""
Unit tests for Causal Self Attention, RoPE, and KV caching.
"""

import pytest
import torch
from transformer_core.attention import CausalSelfAttention, precompute_freqs_cis, apply_rotary_emb


def test_rope_frequencies_and_shapes():
    dim = 64
    seq_len = 128
    freqs = precompute_freqs_cis(dim, seq_len)
    assert freqs.shape == (seq_len, dim // 2)
    assert freqs.dtype == torch.complex64

    # Apply RoPE
    xq = torch.randn(2, seq_len, 4, dim)
    xk = torch.randn(2, seq_len, 4, dim)
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs)
    assert xq_out.shape == xq.shape
    assert xk_out.shape == xk.shape


def test_attention_causality():
    """
    Ensure earlier tokens do NOT attend to later tokens (no future leakage).
    """
    embed_dim = 64
    n_heads = 4
    attn = CausalSelfAttention(embed_dim, n_heads)

    # Input sequence
    x = torch.randn(1, 10, embed_dim, requires_grad=True)
    out, _ = attn(x)

    # Loss computed only on the 3rd token
    loss = out[0, 2].sum()
    loss.backward()

    # Gradients for future tokens (indices 3..9) must be zero!
    future_grad_norm = x.grad[0, 3:].abs().sum().item()
    assert future_grad_norm == 0.0, f"Future token gradient leakage detected: {future_grad_norm}"


def test_kv_cache_equivalence():
    """
    Incremental generation with KV cache must match full forward pass exactly.
    """
    torch.manual_seed(42)
    embed_dim = 64
    n_heads = 4
    attn = CausalSelfAttention(embed_dim, n_heads)
    attn.eval()

    prompt = torch.randn(1, 6, embed_dim)
    with torch.no_grad():
        full_out, _ = attn(prompt)

        # Decode token by token with KV cache
        prefix_out, cache = attn(prompt[:, :5], use_cache=True)
        next_out, _ = attn(prompt[:, 5:6], kv_cache=cache, use_cache=True)

    # Compare 6th token output
    diff = (full_out[:, 5:6] - next_out).abs().max().item()
    assert diff < 1e-5, f"KV cache discrepancy exceeds tolerance: {diff}"
