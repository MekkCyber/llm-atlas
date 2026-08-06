# ALiBi (Attention with Linear Biases)
*Depth — a parameter-free positional-encoding scheme that biases attention logits by token distance.*

**TL;DR:** ALiBi (Press, Smith, Lewis 2021) skips positional embeddings entirely: it adds a **static, distance-proportional negative bias** directly to the attention logits before softmax. The bias for a query at position $i$ attending to a key at position $j$ is $-m_h \cdot (i - j)$, where $m_h$ is a fixed per-head slope from a geometric series. No learned parameters, no rotations, no embeddings. Its selling point is **length extrapolation**: models trained at 1k tokens work reasonably at 2k+ with no fine-tune, because the bias formula is defined for any distance. Modern LLMs have mostly moved to RoPE, but ALiBi persists in BLOOM, MPT, and some long-context variants — and a 2026 result (Schröder et al., paper 15 in the 2026-08-05 digest) shows the linear bias formula has a **precision-level failure mode** that can quietly zero out attention heads.

**Prereqs:** [attention.md](./attention.md), [_positional-encoding.md](./_positional-encoding.md)
**Related:** [rope.md](./rope.md) · [sinusoidal-encoding.md](./sinusoidal-encoding.md) · [dca.md](./dca.md)

---

## What it is

Positional encoding whose three structural choices are:

1. **What is injected** — a scalar bias, one per (query, key) distance.
2. **Where it enters** — added directly to the attention logits before softmax. Not to embeddings, not to Q/K.
3. **Parameterization** — **fixed, no learned parameters.** Per-head slopes $m_h$ come from a geometric sequence: for $H$ heads, $m_h = 2^{-8h/H}$.

For a causal decoder, the attention score between query at position $i$ and key at position $j \le i$ becomes:

$$
\text{score}_{i,j}^{(h)} = \frac{q_i^\top k_j}{\sqrt{d_k}} \;-\; m_h \cdot (i - j)
$$

Softmax over $j$. Distant keys get a strong negative bias; the model attends preferentially to nearby positions, with each head using a different decay rate.

## How it works

- **Slopes** span roughly $2^{-1}$ down to $2^{-8}$ across heads: some heads see far, some see very local context. This is what gives ALiBi its multi-scale character.
- **No RoPE-style rotation.** Q and K are left alone; the bias enters at the score level.
- **Extrapolation** falls out automatically. The bias formula is defined for any $i - j$; nothing in the model has a "trained max length" beyond which behavior is undefined. Empirically, ALiBi degrades far more gracefully than sinusoidal or learned-absolute encodings when tested beyond the training length.

## Why it matters

- **Zero-parameter positional signal.** The only extra state is the per-head slope table, and it's fixed.
- **Length extrapolation without fine-tuning.** ALiBi was the strongest published extrapolation baseline in 2021, and remains competitive vs the RoPE + YaRN family for pure "train short, test long" without re-tuning.
- **Cheap kernel.** The bias is a simple `logits += m * (i - j)` broadcast; fuses cleanly with softmax.
- **Historical importance.** Convinced the field that *relative* schemes could be strong length extrapolators, feeding into the design space that eventually converged on RoPE + scaling tricks.

Losses vs RoPE:

- No mature ecosystem of context-extension tricks (there's no ALiBi equivalent of YaRN / NTK-aware scaling — you just extrapolate).
- Weaker on high-precision retrieval tasks than a properly-scaled RoPE model of the same size.
- The precision-level failure mode below is unique to ALiBi's specific bias magnitude.

## Gotchas & tricks

- **FP16/BF16 underflow ("ALiBi goes blind").** For long distances, $-m_h \cdot (i - j)$ becomes a very negative number; combined with softmax numerics in FP16/BF16, its softmax weight underflows to zero. Some heads then attend to *nothing but the last few tokens*, permanently. Schröder et al. 2026 characterize this in production ALiBi models and show it *substantially* impairs passkey retrieval while only mildly affecting standard decoder benchmarks — which is why the failure went unnoticed.
- **Mitigation: log-scaled distances.** Replacing $(i - j)$ with $\log(1 + i - j)$ in the bias formula was the most consistent fix in Schröder et al.'s pretraining experiments; the paper also evaluates three other training-time mitigations.
- **Only apply during training.** Adding ALiBi bias to a model pretrained without it doesn't retrofit position awareness. The heads' slope specialization is a training-time signal.
- **Head order matters if you probe slopes.** ALiBi's slopes are assigned per head index; two heads with the same slope are redundant. When ablating, don't shuffle head order without accounting for the slope assignment.
- **Doesn't compose with RoPE.** Pick one — stacking both on the same layer double-counts positional signal and empirically hurts.

## Sources

- Paper: *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation* — Press, Smith, Lewis, 2021 — [arXiv 2108.12409](https://arxiv.org/abs/2108.12409). The ALiBi paper.
- Paper: *When Attention Goes Blind: Numerical Failure in ALiBi Positional Encodings* — Schröder, Gienapp, Schlatt, Potthast, Heyer, 2026 — [arXiv 2608.03994](https://arxiv.org/abs/2608.03994). The precision failure mode and four training-time mitigations.
- Models using ALiBi: BLOOM (176B), MPT (7B/30B), Replit-code-v1.
