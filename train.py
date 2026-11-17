#!/usr/bin/env python3
"""
Toy Autoregressive Training Script for TransformerLM.
Demonstrates training loop, gradient clipping, loss decay, and autoregressive text generation.
"""

import time
import torch
from transformer_core.model import TransformerConfig, TransformerLM


def main():
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Training on device: {device}")

    # Synthetic text dataset (toy algorithmic corpus)
    text = (
        "def fibonacci(n):\n"
        "    if n <= 1: return n\n"
        "    return fibonacci(n - 1) + fibonacci(n - 2)\n\n"
        "def factorial(n):\n"
        "    return 1 if n <= 1 else n * factorial(n - 1)\n\n"
    ) * 200

    # Character-level vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([char_to_ix[c] for c in text], dtype=torch.long)

    print(f"[*] Corpus length: {len(data)} characters | Vocab size: {vocab_size}")

    # Initialize model
    config = TransformerConfig(
        vocab_size=vocab_size,
        embed_dim=128,
        n_layers=4,
        n_heads=4,
        ffn_dim=340,
        max_seq_len=64,
        dropout=0.1,
    )
    model = TransformerLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    print(f"[*] Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Training Loop
    block_size = 32
    batch_size = 16
    start_time = time.time()

    model.train()
    print("[*] Starting training run (50 steps)...")
    for step in range(1, 51):
        # Sample batch
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)

        optimizer.zero_grad()
        logits, loss, _ = model(x, targets=y)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 10 == 0 or step == 1:
            print(f"    Step {step:02d}/50 | Loss: {loss.item():.4f} | Perplexity: {torch.exp(loss).item():.2f}")

    elapsed = time.time() - start_time
    print(f"[*] Training finished in {elapsed:.2f}s")

    # Autoregressive generation demo
    print("\n[*] Testing Autoregressive Generation with KV Cache:")
    prompt = "def "
    prompt_ids = torch.tensor([[char_to_ix[c] for c in prompt]], dtype=torch.long).to(device)
    generated = model.generate(prompt_ids, max_new_tokens=40, temperature=0.7)
    decoded = "".join([ix_to_char[idx] for idx in generated[0].tolist()])
    print("--------------------------------------------------")
    print(decoded)
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()
