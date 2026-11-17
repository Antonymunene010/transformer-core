"""
Unit tests for TransformerLM forward pass, loss, and autoregressive generation.
"""

import pytest
import torch
from transformer_core.model import TransformerConfig, TransformerLM


def test_model_forward_and_backward():
    config = TransformerConfig(
        vocab_size=256,
        embed_dim=64,
        n_layers=2,
        n_heads=4,
        ffn_dim=128,
        max_seq_len=64,
    )
    model = TransformerLM(config)

    input_ids = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))

    logits, loss, _ = model(input_ids, targets=targets)
    assert logits.shape == (2, 16, 256)
    assert loss is not None
    assert not torch.isnan(loss)

    loss.backward()
    # Check that embeddings received gradients
    assert model.tok_embeddings.weight.grad is not None


def test_model_generation():
    config = TransformerConfig(
        vocab_size=128,
        embed_dim=32,
        n_layers=2,
        n_heads=2,
        ffn_dim=64,
        max_seq_len=32,
    )
    model = TransformerLM(config)

    prompt = torch.tensor([[10, 20, 30]])
    gen_tokens = model.generate(prompt, max_new_tokens=5, temperature=0.8)
    assert gen_tokens.shape == (1, 8)
