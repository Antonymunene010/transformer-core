# transformer-core

A modular, production-grade implementation of a modern autoregressive **Decoder-only Transformer Language Model** built from scratch in pure PyTorch.

This project implements the foundational architecture used in state-of-the-art Large Language Models (LLaMA 3, Mistral, Gemma), focusing on mathematical correctness, numerical stability, and inference performance.

---

## Key Architectural Features

* **Multi-Head Causal Attention with RoPE:** Implements Rotary Position Embeddings (Su et al.) via complex polar frequencies, allowing relative position modeling with length extrapolation.
* **Pre-Layer Normalization (RMSNorm):** Root Mean Square normalization without mean centering, reducing computational overhead while stabilizing gradient flow.
* **SwiGLU Non-Linearity:** Swish Gated Linear Unit feed-forward network ($\approx \frac{8}{3}d_{\text{model}}$ hidden projection) for superior representation capacity.
* **Key-Value Cache (KV-Cache):** Incremental decoding caching past $K$ and $V$ tensors for $O(1)$ token generation step latency.
* **Numerical Verification & Tests:** Rigorous test suite checking causal mask non-leakage, backprop stability, and KV-cache equivalence.

---

## Architecture Blueprint

```
Input Tokens (IDs)
        │
        ▼
 Token Embedding
        │
┌───────┴─────────────────────────────────────────┐
│ Transformer Block (repeated N times)            │
│                                                 │
│    x ───► RMSNorm ───► Causal Self-Attention    │
│    │                        (with RoPE)         │
│    └───► (+) ◄───────────────────┘              │
│           │                                     │
│           ├──► RMSNorm ───► SwiGLU FFN          │
│           │                   (gate/up/down)    │
│           └──► (+) ◄─────────────┘              │
└───────┬─────────────────────────────────────────┘
        ▼
   Final RMSNorm
        │
        ▼
  LM Head (Linear Projection / Weight-Tied)
        │
        ▼
 Logits [Batch, SeqLen, VocabSize]
```

---

## Directory Layout

```
transformer-core/
├── src/
│   └── transformer_core/
│       ├── __init__.py      # Public exports
│       ├── attention.py     # Causal attention, RoPE, KV-caching
│       ├── layers.py        # RMSNorm and SwiGLU modules
│       └── model.py         # TransformerConfig, TransformerBlock, TransformerLM
├── tests/
│   ├── test_attention.py    # Causality & KV cache equivalence tests
│   └── test_model.py        # Forward/backward pass and generation tests
├── train.py                 # Self-contained toy training & generation demo
├── pyproject.toml           # Project metadata & test configuration
└── README.md
```

---

## Installation & Quickstart

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/Antonymunene010/transformer-core.git
cd transformer-core

# Using uv (fastest)
uv venv
source .venv/bin/activate
uv pip install torch pytest

# Or with standard pip
python -m venv .venv
source .venv/bin/activate
pip install -e . pytest
```

### 2. Run Toy Training & Text Generation
```bash
python train.py
```

### 3. Run Verification Tests
```bash
pytest tests/ -v
```

---

## Mathematical Highlights

### Rotary Position Embedding (RoPE)
Given query vector $\mathbf{q}_m$ at token position $m$, RoPE applies an orthogonal rotation in 2D coordinate pairs:

$$R_{\Theta, m}^d = \text{diag}\left(R_{\theta_1, m}, R_{\theta_2, m}, \dots, R_{\theta_{d/2}, m}\right)$$

where $\theta_i = 10000^{-2(i-1)/d}$.

### RMSNorm Formulation
$$\bar{a}_i = \frac{a_i}{\text{RMS}(\mathbf{a})} g_i, \quad \text{where} \quad \text{RMS}(\mathbf{a}) = \sqrt{\frac{1}{d} \sum_{i=1}^d a_i^2 + \epsilon}$$

---

## Author
**Antony Munene** — [GitHub](https://github.com/Antonymunene010) • [LinkedIn](https://www.linkedin.com/in/antony-munene-ml/)
License: MIT
