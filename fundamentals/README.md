# Fundamentals

*The prerequisites everyone assumes you already know — tokenization, embeddings, optimizers, losses, normalization, and the core primitives that every other section builds on.*

---

## What This Is

Before you can reason about training, post-training, inference, or any of the frontier work, there's a set of primitives that the rest of the field treats as background knowledge. This folder is that background — explicit, in one place, so you don't have to reconstruct it from a dozen sources.

If you're doing zero-to-hero, start here.

---

## What Belongs Here

- **Tokenization** — BPE, SentencePiece, tiktoken, vocabulary design, tokenizer effects on quality.
- **Embeddings & positional encoding** — token embeddings, RoPE, ALiBi, learned vs. fixed.
- **Optimizers** — SGD, Adam, AdamW, Muon, Shampoo, Lion — what they do and why AdamW became the default.
- **Losses** — cross-entropy, label smoothing, KL, and variants used in RL post-training.
- **Normalization** — LayerNorm, RMSNorm, pre-norm vs. post-norm, why it matters for stability.
- **Activations** — GeLU, SwiGLU, and why modern LLMs picked what they picked.
- **Initialization** — why init matters at scale, common schemes.
- **Prompting & in-context learning** — few-shot, chain-of-thought basics, prompt sensitivity.

## Reading Order

1. Tokenization
2. Embeddings & positional encoding
3. Normalization & activations
4. Optimizers & losses
5. Initialization
6. Prompting & in-context learning

---

## Overview Pages (taxonomies)

Entry points for each class of primitives. Start here to orient yourself before diving into a specific technique.

- [Tokenization](_tokenization.md) — how raw text becomes integer tokens.
- [Positional Encoding](_positional-encoding.md) — how Transformers are told where each token sits.

## Concept Pages (depth)

One technique per page, grounded in its source paper(s).

- [Scaled Dot-Product Attention](attention.md)
- [Byte-Pair Encoding (BPE)](bpe.md)
- [Sinusoidal Positional Encoding](sinusoidal-encoding.md)
- [Z-loss](z-loss.md)

---

## Related

- [architectures/](../architectures/) — where these primitives are composed into models
- [pre-training/](../pre-training/) — how these primitives behave at scale
