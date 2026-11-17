"""
transformer_core: Modular PyTorch implementation of modern Transformer architectures.
"""

from .model import TransformerConfig, TransformerLM, TransformerBlock
from .attention import CausalSelfAttention, apply_rotary_emb, precompute_freqs_cis
from .layers import RMSNorm, SwiGLU

__all__ = [
    "TransformerConfig",
    "TransformerLM",
    "TransformerBlock",
    "CausalSelfAttention",
    "apply_rotary_emb",
    "precompute_freqs_cis",
    "RMSNorm",
    "SwiGLU",
]
