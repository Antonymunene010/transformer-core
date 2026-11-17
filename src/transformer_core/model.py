"""
Full Transformer Language Model implementation with KV-caching generation.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention
from .layers import RMSNorm, SwiGLU


@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    embed_dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    ffn_dim: int = 1408  # Typically ~8/3 * embed_dim for SwiGLU
    max_seq_len: int = 2048
    dropout: float = 0.0
    bias: bool = False
    norm_eps: float = 1e-6
    tie_weights: bool = True


class TransformerBlock(nn.Module):
    """
    Standard Pre-Norm Transformer Decoder Block:
    x = x + Attention(RMSNorm(x))
    x = x + SwiGLU(RMSNorm(x))
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.embed_dim, eps=config.norm_eps)
        self.attn = CausalSelfAttention(
            embed_dim=config.embed_dim,
            n_heads=config.n_heads,
            max_seq_len=config.max_seq_len,
            bias=config.bias,
            dropout=config.dropout,
        )
        self.ffn_norm = RMSNorm(config.embed_dim, eps=config.norm_eps)
        self.ffn = SwiGLU(
            hidden_dim=config.embed_dim,
            ffn_dim=config.ffn_dim,
            bias=config.bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-Norm Attention
        attn_out, new_cache = self.attn(self.attn_norm(x), kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out

        # Pre-Norm FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class TransformerLM(nn.Module):
    """
    Autoregressive Decoder-only Transformer Language Model.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.embed_dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.tok_embeddings.weight

        # Parameter initialization (standard scaled normal)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        bsz, seq_len = input_ids.shape
        x = self.dropout(self.tok_embeddings(input_ids))

        new_caches = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, cache = layer(x, kv_cache=layer_cache, use_cache=use_cache)
            if use_cache:
                new_caches.append(cache)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Flatten across sequence for cross-entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)

        return logits, loss, new_caches

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation using Key-Value Caching for O(1) step latency.
        """
        self.eval()
        curr_ids = prompt_ids
        kv_caches = None

        for step in range(max_new_tokens):
            if step == 0:
                # First prefill step: process whole prompt
                logits, _, kv_caches = self(curr_ids, use_cache=True)
            else:
                # Incremental decode: process only newest token
                logits, _, kv_caches = self(curr_ids[:, -1:], kv_caches=kv_caches, use_cache=True)

            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)

        return curr_ids
