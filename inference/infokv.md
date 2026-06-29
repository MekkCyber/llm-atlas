# InfoKV
*Depth — entropy-aware KV cache compression that complements attention-based eviction with information-theoretic signals.*

**TL;DR:** Attention-based KV eviction (H2O, SnapKV, etc.) keeps tokens with high attention to the current step. The hidden assumption is that attention captures importance. **InfoKV** shows it captures only *local* importance: attention-selected tokens influence nearby contexts, while tokens with high predictive uncertainty (entropy) influence *distant* future contexts and are exactly what attention-based eviction throws away. Combining attention scores with a token-level entropy signal — formalized as **Forward Influence** — gives a clean Pareto improvement on long-reasoning workloads.

**Prereqs:** [_kv-cache-compression](_kv-cache-compression.md), [../architectures/mla.md](../architectures/mla.md)
**Related:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

A KV cache compression scheme for long-reasoning inference that augments attention-score-based eviction with an **information-theoretic importance signal**. The motivating observation:

> *Tokens that attention selects affect nearby contexts; tokens with high predictive uncertainty affect distant contexts.*

So picking tokens by attention alone systematically discards tokens that matter for the long tail of the reasoning trace. InfoKV adds an entropy score and combines the two.

## How it works

The construction has three pieces:

1. **Forward Influence.** A metric that measures how much a *compressed* token affects *future* contexts. Empirically: attention-selected tokens have high Forward Influence on the next few steps; high-entropy tokens have high Forward Influence over hundreds of steps.

2. **Entropy score.** Two components:
   - **Token-level predictive uncertainty** at the position.
   - **Layer-wise representation evolution** — how much the token's hidden state changes across transformer layers (a coarse proxy for "the model is still processing this token").
   These combine into a per-token entropy/uncertainty score.

3. **Combined eviction.** At each compression step, the keep/evict decision uses a weighted combination of attention scores and the entropy score. The weighting is the only knob; defaults work across the tested models.

## Why it matters

- **Reasoning workloads produce the longest KV caches in production today** — long-CoT outputs, agent traces, multi-turn tool use. Existing eviction strategies leak quality precisely there.
- Orthogonal to attention-based methods, so it composes with them rather than replacing them — the wins are additive.
- Consistent improvements across **Llama-3.1, Llama-3.2, and DeepSeek-R1** in both long-prefill and long-decode regimes.
- Complementary to architectural KV compression like [MLA](../architectures/mla.md): MLA shrinks each token's stored KV; InfoKV decides *which tokens* to keep. Use both.

## Gotchas & tricks

- The entropy signal requires *the model's own probability distribution*, not just attention — adds a small per-token compute cost. Amortizes well at long-context.
- Forward Influence is the right framing but expensive to compute exactly; the entropy + layer-evolution combo is the cheap surrogate that actually ships.
- Pairs especially well with reasoning models (DeepSeek-R1) where the long-CoT distribution has many high-uncertainty tokens encoded mid-trace; less impactful on short, peaky chat workloads.
- Sensitive to the weighting between attention and entropy components — start with the paper's defaults before tuning.

## Sources

- Paper: *Information-Aware KV Cache Compression for Long Reasoning* — Xiao, Birch, Lin, 2026 — [arXiv:2606.26875](https://arxiv.org/abs/2606.26875). SJTU LUMIA Lab / Edinburgh.
