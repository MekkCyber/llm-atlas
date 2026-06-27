# KV Cache Compression

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Long-CoT reasoning has made KV-cache size the dominant inference cost: every reasoning token has to keep its keys/values around. Compression schemes drop or merge some of those entries. Most prior work scores a token's *importance* by attention weight, but attention selects locally-relevant tokens; **InfoKV** shows that tokens with high predictive uncertainty matter most for *distant* future contexts and proposes combining the two signals.

**Prereqs:** [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../architectures/mla.md](../architectures/mla.md), [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md), [_speculative-decoding](_speculative-decoding.md)

---

## What it is

A family of inference-time techniques that **selectively retain** a subset of the KV entries during prefill and/or decode, dropping or merging the rest. The goal is to bound peak GPU memory and (often) reduce the per-step attention compute, while preserving the model's downstream quality.

Important distinction from architectural KV reductions:

- **[MLA](../architectures/mla.md), GQA, MQA** shrink the *per-token* KV size at training time, baked into the architecture.
- **KV compression** is *which tokens to keep at inference*, applied to a frozen model.

The two compose: MLA decides how big each KV entry is; KV compression decides how many entries you carry.

## How it works

The classic schemes (H2O, SnapKV, StreamingLLM) all share one move: assign each past token an *importance score* and keep the top-K. They differ mostly in the score:

- **Recency** — keep the last N tokens; drop the rest.
- **Sink tokens** — always keep the first few; they soak up attention mass.
- **Attention-based** — keep tokens with high accumulated attention weight from recent queries.

InfoKV (2026) argues attention-based scoring is incomplete:

1. Define **Forward Influence** — how much compressing a token shifts future predicted representations.
2. Empirically, **attention-selected tokens dominate the influence on nearby future tokens**, but **tokens with high token-level predictive entropy dominate influence on distant future tokens**.
3. Therefore: combine attention scores with (a) per-token predictive uncertainty (entropy) and (b) per-layer representation drift. The combined entropy-aware score keeps both the local and the long-range carriers.

The mechanism is training-free; the entropy signal comes from the model's own logits during prefill and is reused as a static per-token weight during eviction.

## Why it matters

Long-CoT reasoning models (DeepSeek-R1, Llama-3.x reasoning, etc.) routinely emit 10K–100K tokens of internal chain. Prefill grows linearly; KV memory grows linearly; serving cost grows linearly. Attention-only eviction was the default lever and was hitting accuracy walls on multi-step reasoning where mid-chain "boring" tokens carry necessary state for *much* later steps. Forward Influence gives a principled reason to keep them.

Practical use: drop-in for the same serving stacks that already implement H2O / SnapKV style eviction, on Llama-3.1, Llama-3.2, DeepSeek-R1 in the paper.

## Gotchas & tricks

- **Entropy is computed once** (during prefill, from the model's logits) and reused. Cheap, but it freezes the importance ranking — a token that was uncertain at emission can later become irrelevant; no scheme catches that without expensive recomputation.
- **Layer-wise drift signal** matters: scoring tokens with a single global signal underperforms layer-aware scoring because deep layers re-distribute information.
- **Composes with KV quantization**, not a substitute for it. Compression chooses *which* entries to keep; quantization decides *how many bits* per kept entry.
- **Reasoning evaluations only.** The InfoKV paper validates on long-context reasoning benchmarks. Open-question whether the entropy-importance shift survives on retrieval-heavy long-context workloads where critical tokens may be lexically distinctive instead of uncertain.

## Sources

- Paper: *Information-Aware KV Cache Compression for Long Reasoning* — Xiao, Birch, Lin, 2026 — [arXiv:2606.26875](https://arxiv.org/abs/2606.26875).
- Background: *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models* — Zhang et al., 2023 — earliest attention-importance KV eviction.
- Background: *SnapKV: LLM Knows What You are Looking for Before Generation* — Li et al., 2024.
- Background: *Efficient Streaming Language Models with Attention Sinks* — Xiao et al., 2023 — sink-token + recency baseline.
