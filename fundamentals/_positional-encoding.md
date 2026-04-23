# Positional Encoding

*Taxonomy — how Transformers are told where each token sits in the sequence.*

**TL;DR:** Attention is permutation-equivariant — shuffle the tokens and you get the shuffled output. Models need position information injected explicitly. There are four main families (absolute, relative, rotary, bias-based), and modern LLMs have converged on rotary (RoPE) with interpolation tricks (YaRN, NTK-aware) for long context.

**Related taxonomies:** *(none yet)*
**Depth files covered here:** [sinusoidal-encoding](sinusoidal-encoding.md) · [rope](rope.md)

---

## The problem

A Transformer block is made of attention (permutation-equivariant) and a position-wise FFN (applied independently per token). Neither sees order. Without extra signal, the model literally cannot distinguish "the cat sat on the mat" from "mat the on sat cat the".

The job of positional encoding is to inject order information somewhere in the computation — at the input, at the attention scores, or on the Q/K vectors — in a way that (a) is learnable or principled, (b) composes cleanly with attention, and ideally (c) extrapolates to sequence lengths longer than those seen in training.

## The shared pattern

Every positional-encoding scheme injects a signal that is a **function of position** into the attention computation. They differ on three orthogonal axes:

1. **What is encoded** — the *absolute* position of each token, or the *relative* distance between pairs of tokens.
2. **Where the signal enters** — added to the input embedding, added to the Q/K vectors, or added as a bias to the attention logits.
3. **Parameterization** — fixed (sinusoidal, ALiBi), learned per-position (learned absolute), or a small learned function of position (RoPE, T5 bias).

"Relative" matters more than "absolute" for most tasks — language cares about token proximity, not index 47 vs 48 — which is why relative and rotary schemes dominate modern models.

## Variants

| Technique | Family | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- | --- |
| [Sinusoidal](sinusoidal-encoding.md) | Absolute | Fixed sin/cos of position added to embeddings | Limited length extrapolation | Historical reference; some length generalization vs. learned absolute |
| Learned absolute (no depth file yet) | Absolute | Trainable vector per position | Cannot extrapolate past trained max length | Simple, historical (GPT-2 era) |
| T5 relative bias (no depth file yet) | Relative | Learned scalar bias on attention logits based on distance | Bias is a learned lookup, bounded by training |  Encoder-decoder line (T5 family) |
| [RoPE](rope.md) | Rotary | Rotate Q and K by position-dependent angles | Rotation implements relative position implicitly | **Modern default** — LLaMA, Qwen, DeepSeek, Mistral, Kimi |
| ALiBi (no depth file yet) | Bias | Static distance-proportional bias on logits, no params | No learned position params at all | Cheap, strong length extrapolation without retraining |
| YaRN / NTK-aware RoPE (covered in [rope](rope.md)) | Rotary extension | Interpolate/scale RoPE frequencies post-hoc | Requires a short fine-tune to fully adapt | Extending a trained RoPE model's context (e.g. 4k → 128k) |
| LongRoPE (no depth file yet) | Rotary extension | Non-uniform per-dimension frequency scaling | More complex than YaRN, slightly stronger | Aggressive context extension |

## How to choose

**If you're building a modern LLM from scratch, use RoPE.** That's where the field has converged. The reasons: it encodes *relative* position implicitly while keeping the absolute API (just apply it to Q and K before the dot product), it plays well with modern attention kernels (FlashAttention etc.), and there's a mature ecosystem of interpolation methods (YaRN, NTK-aware, LongRoPE) for extending context length after training.

**If you need aggressive length extrapolation with no retraining**, ALiBi is the strongest baseline. It has no learned parameters and extrapolates cleanly because the bias is a simple monotonic function of distance.

**Sinusoidal and learned absolute** are mostly historical now. Learned absolute is a trap for any use case involving longer-than-training-length sequences.

**T5 relative bias** is a solid alternative to RoPE used in some encoder-decoder lines, but the LLM decoder-only lineage has mostly picked RoPE.

### Context extension at inference time

Once a RoPE model is trained, YaRN and NTK-aware scaling let you multiply its context length by 4–32× with a short fine-tune. This has become the standard recipe — start with a sensible training context (say 8k), then extend to the target (32k, 128k, 1M) via rotary-frequency interpolation.

## Adjacent but distinct

- **Embedding layer.** The embedding maps token IDs to vectors; positional encoding *modifies* those vectors (or their dot products) with position signal. Different concerns.
- **Attention variants** (GQA, MQA, MLA). Orthogonal to positional encoding — you pick one of each independently.
- **Context-length schedules.** How you gradually expand sequence length during training. Uses positional encoding as a building block but is its own topic.

## Sources

- Paper: *Attention Is All You Need* — Vaswani et al., 2017 — sinusoidal encoding.
- Paper: *RoFormer: Enhanced Transformer with Rotary Position Embedding* — Su et al., 2021 — https://arxiv.org/abs/2104.09864
- Paper: *Train Short, Test Long: Attention with Linear Biases (ALiBi)* — Press et al., 2021 — https://arxiv.org/abs/2108.12409
- Paper: *YaRN: Efficient Context Window Extension* — Peng et al., 2023 — https://arxiv.org/abs/2309.00071
- Paper: *LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens* — Ding et al., 2024.
- Paper: *Exploring the Limits of Transfer Learning (T5)* — Raffel et al., 2019 — relative position bias.
