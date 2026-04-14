# Positional Encoding
*Taxonomy — how Transformers are told where each token sits in the sequence.*

**TL;DR:** Attention is permutation-equivariant — shuffle the tokens and you get the shuffled output. Models need position information injected explicitly. There are four main families (absolute, relative, rotary, bias-based), and modern LLMs have converged on rotary (RoPE) with interpolation tricks for long context.

**Related:** [attention](attention.md), [transformer-block](../architectures/transformer-block.md)

---

## The problem

A Transformer block is made of attention (permutation-equivariant) and a position-wise FFN (applied independently per token). Neither sees order. Without extra signal, the model literally cannot distinguish "the cat sat on the mat" from "mat the on sat cat the."

The job of positional encoding is to inject order information somewhere in the computation — at the input, at the attention scores, or on the Q/K vectors — in a way that (a) is learnable or principled, (b) composes cleanly with attention, and ideally (c) extrapolates to sequence lengths longer than those seen in training.

## Variants at a glance

| Technique | Family | Key idea | When it wins |
| --- | --- | --- | --- |
| [Sinusoidal](sinusoidal-encoding.md) | absolute | fixed sin/cos of position added to embeddings | simple, some length extrapolation |
| Learned absolute | absolute | trainable vector per position | simple, needs no math; doesn't extrapolate |
| T5 relative bias | relative | learned scalar bias on attention logits based on distance | strong, used in T5 family |
| RoPE | rotary | rotate Q and K by position-dependent angles | modern default (LLaMA, Qwen, DeepSeek, Mistral) |
| ALiBi | bias | static distance-proportional bias on logits, no params | extrapolates to longer context cheaply |
| YaRN / NTK-aware RoPE | rotary extension | interpolate/scale RoPE frequencies | stretch a trained RoPE model's context |

## How to choose

**If you're building a modern LLM from scratch, use RoPE.** That's where the field has converged. The reasons: it encodes *relative* position implicitly while keeping the absolute API (just apply it to Q and K before the dot product), it plays well with modern attention kernels, and there's a mature ecosystem of interpolation methods (YaRN, NTK-aware, LongRoPE) for extending context length after training.

**If you need aggressive length extrapolation with no retraining**, ALiBi is the strongest baseline. It has no learned parameters and extrapolates cleanly because the bias is a simple monotonic function of distance.

**Sinusoidal and learned absolute** are mostly historical now. Learned absolute is a trap for any use case involving longer-than-training-length sequences.

**T5 relative bias** is a solid alternative to RoPE used in some encoder-decoder lines, but the LLM decoder-only lineage has mostly picked RoPE.

## Why the design choice matters

- Positional encoding is one of the most-swapped components in Transformer research — always check which one a paper is using before assuming behavior.
- Length generalization is downstream of positional encoding choice. A model trained on 2048 tokens with learned absolute positions is *useless* at 4096; the same model with RoPE + YaRN can serve 128k.
- RoPE is applied to Q and K only, not V. Easy to mis-implement.

## Sources

- Paper: *Attention Is All You Need* — Vaswani et al., 2017 — sinusoidal encoding.
- Paper: *RoFormer: Enhanced Transformer with Rotary Position Embedding* — Su et al., 2021 — https://arxiv.org/abs/2104.09864
- Paper: *Train Short, Test Long: Attention with Linear Biases (ALiBi)* — Press et al., 2021 — https://arxiv.org/abs/2108.12409
- Paper: *YaRN: Efficient Context Window Extension* — Peng et al., 2023 — https://arxiv.org/abs/2309.00071
- Paper: *Exploring the Limits of Transfer Learning (T5)* — Raffel et al., 2019 — relative position bias.
