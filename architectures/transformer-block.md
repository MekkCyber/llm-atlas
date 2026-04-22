# Transformer Block
*Depth — the repeating structural unit of every Transformer.*

**TL;DR:** The repeating unit of a Transformer: attention (mix information across tokens) → feed-forward (transform within each token), each wrapped in a residual connection and a LayerNorm. Stack N of these and you have a Transformer.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](multi-head-attention.md)
**Related:** [positional-encoding](../fundamentals/_positional-encoding.md)

---

## What it is

The fundamental structural motif of every Transformer-based model. Each block has two sub-layers:

1. **Multi-head self-attention** — mixes information across token positions.
2. **Position-wise feed-forward network (FFN)** — a small MLP applied independently to each token.

Each sub-layer is wrapped with:
- A **residual connection** (`x + sublayer(x)`)
- A **LayerNorm** (the position of which — pre or post — matters, see below)

## How it works

### The classic (post-norm) block from AIAYN

```
x₁ = LayerNorm(x + MultiHeadAttention(x))
x₂ = LayerNorm(x₁ + FFN(x₁))
```

### The modern (pre-norm) block used by almost every current LLM

```
x₁ = x + MultiHeadAttention(LayerNorm(x))
x₂ = x₁ + FFN(LayerNorm(x₁))
```

Pre-norm is more stable at depth because the residual pathway stays clean (no norm on the skip connection). Every large modern LLM uses pre-norm.

### The FFN

```
FFN(x) = σ(xW₁ + b₁) W₂ + b₂
```

- `W₁ ∈ ℝ^(d × 4d)`, `W₂ ∈ ℝ^(4d × d)` — the "expand by 4× then contract" shape.
- σ was ReLU in the original, GeLU in GPT-2/3, **SwiGLU** in modern LLMs (LLaMA onward).
- **Two-thirds of the model's parameters live in FFNs**, not attention. Often overlooked.

### Decoder-only variant

Modern LLMs (GPT, LLaMA, Mistral) drop the encoder and cross-attention entirely. They stack decoder-only blocks with **causal masking** in the attention — position `i` can only attend to positions `≤ i`. That's the whole architectural difference.

## Why it matters

- The block is the **unit of scaling**. Doubling depth = doubling blocks. Every scaling result (Chinchilla, Kaplan, etc.) is about how to stack these.
- The **alternating pattern** — mix across tokens (attention), transform within token (FFN) — is the inductive bias that makes Transformers work. Information flows "horizontally" in attention and "vertically" (through the channel dimension) in the FFN.
- Every architectural innovation after 2017 is a modification to one piece of this block: attention (MQA, GQA, sliding window), FFN (MoE, SwiGLU), norm (RMSNorm, pre-norm), or glue.

## Gotchas & tricks

- **Pre-norm vs. post-norm**: post-norm is in the paper; pre-norm is what you actually want. Post-norm at depth requires careful learning rate warmup to avoid divergence.
- **RMSNorm is usually substituted for LayerNorm** in modern LLMs — fewer parameters, similar quality, slightly faster.
- **The FFN hidden dim is not always 4×** anymore. LLaMA uses `8/3≈ 2.67×` because SwiGLU has three weight matrices instead of two, and they budget for total params.
- **Bias terms are often dropped** in modern implementations (LLaMA removes them) — small quality hit, small speedup, cleaner code.
- The residual stream is where every interpretability paper lives — it's the "shared workspace" that attention and FFNs read from and write to.

## Sources

- Paper: *Attention Is All You Need* — Vaswani et al., 2017.
- Paper: *On Layer Normalization in the Transformer Architecture* (pre-norm analysis) — Xiong et al., 2020.
- Paper: *GLU Variants Improve Transformer (SwiGLU)* — Shazeer, 2020.
- Paper: *LLaMA: Open and Efficient Foundation Language Models* — Touvron et al., 2023 — modern block defaults.
