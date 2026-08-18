# Mobius (Decoupled Knowledge and Reasoning)
*Depth — Transformer variant that factors the FFN into one shared memory and multiple attention-based reasoning operators.*

**TL;DR:** Mobius-v0 restructures the Transformer so that a single **globally shared FFN** (Memory) holds knowledge vectors and **multiple self-attention operators** (Reasoners) iteratively query it. The hidden state serves as both cache and message bus between reasoning rounds. A 7B trained from scratch matches a 7B Transformer with 62.6% of the tokens; continually pretrained from a 35B Transformer checkpoint (Intern-S2-Mobius), it matches the base's downstream scores with ~4× end-to-end inference speedup.

**Prereqs:** [transformer-block](transformer-block.md), [multi-head-attention](multi-head-attention.md)
**Related:** [_moe](_moe.md), [deepseek-moe](deepseek-moe.md), [mla](mla.md)

---

## What it is

A rearrangement of the Transformer stack around a *shared-storage / iterated-compute* pattern. Where a vanilla Transformer alternates attention and FFN once per layer, Mobius consolidates all FFN parameters into one large Memory bank at model level and re-runs multiple lightweight Reasoner attention operators against it. The design pushes MoE's "separate what-you-know from what-you-do" intuition to a global scope.

## How it works

Architecture at inference:

- **Memory (M).** One globally shared FFN acting as a key-value store: keys are learned, values encode knowledge vectors.
- **Reasoners (R_1 … R_k).** Self-attention operators (thin, no per-layer FFN).
- **Hidden state `h`.** Doubles as the query into M and as the working tape passed between reasoners.

Per token, the update is roughly:

```
h ← R_1(h)                # attention update
k, v = M(h)               # query the shared memory for knowledge
h ← merge(h, v)           # write knowledge back to hidden state
h ← R_2(h)
k, v = M(h)
h ← merge(h, v)
...                       # k rounds
```

Where a vanilla Transformer of similar total-parameter count spreads FFN capacity across layers, Mobius concentrates it once and iterates the attention side. Total FFN parameters end up lower per-forward because M is shared across all reasoners.

## Why it matters

- **Inference-cost lever.** Sharing FFN across reasoners cuts per-token FLOPs materially; the 4× end-to-end speedup on Intern-S2-Mobius (continual-pretrained from Qwen3.5-35B) is a striking demonstration.
- **Data-efficient training.** From-scratch 7B matches Transformer 7B at 62.6% of the tokens — architectural inductive bias substituting for data.
- **Architectural upgrade path.** Continual pretraining onto an existing checkpoint means labs don't need to redo the base run from scratch to adopt Mobius — a rare property for a structural change.

## Gotchas & tricks

- The shared-memory bottleneck can become a serialization point at very high concurrency; kernel design matters for realizing the 4× wall-clock number.
- Continual-pretraining transfer requires reshaping the source Transformer's FFN weights into Mobius's Memory bank; the reshaping recipe is architecture-specific and not fully documented in the abstract.
- Compute savings depend on the ratio between Memory size and per-reasoner attention cost — tuning `k` (rounds) trades quality for speed.

## Sources

- Intern-S2-Mobius: Foundation Model with Decoupled Knowledge and Reasoning — Kai Chen et al., 2026 — [arXiv:2608.14290](https://arxiv.org/abs/2608.14290)
