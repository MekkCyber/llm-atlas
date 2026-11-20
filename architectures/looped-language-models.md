# Looped Language Models
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A family of transformers that **share a single stack of layers and iterate it T times** per token at inference, trading parameter count for depth-at-inference. Ouro, Huginn and relatives deepen reasoning at a compact parameter budget by passing the hidden state through the same shared layer stack repeatedly. All T passes run on every token in standard inference; smarter schedules (early-exit, late-entry) try to skip some.

**Prereqs:** [transformer-block.md](transformer-block.md), [attention](../fundamentals/attention.md)
**Related:** [late-entry.md](../inference/late-entry.md), [multi-head-attention.md](multi-head-attention.md)

---

## What it is

A standard transformer with L distinct layers becomes a **looped** transformer with K distinct layers iterated T times, so effective depth is K × T but parameter count is K layers worth. Weight tying across loop iterations is the whole trick; each iteration sees the previous iteration's hidden state and refines it.

The idea predates LLMs (Universal Transformer, 2018), but recent compact reasoning models — **Ouro-1.4B / 2.6B** and **Huginn-3.5B** — have revived it as a way to get stronger per-parameter reasoning than a dense same-size baseline.

---

## How it works

Given a shared layer stack $f_\theta$, a token's representation evolves as

$$
h^{(0)} = \text{embed}(x), \quad h^{(t)} = f_\theta(h^{(t-1)}) \text{ for } t = 1, \ldots, T
$$

The final hidden state $h^{(T)}$ is read out to logits. Key design choices across variants:

- **T fixed vs adaptive.** Ouro/Huginn fix T at training; adaptive-compute transformers predict T per token.
- **Residual across loops or reset.** Most variants keep a residual stream flowing through loops; some reset norms each loop.
- **Same T for all tokens.** All tokens in a sequence share T, so KV cache is standard — but the **compute per token is T× a non-looped model of the same parameter count**.

At inference, this means a looped model with T=64 does 64× more FLOPs per token than its parameter count suggests — the cost is in compute, not memory.

---

## Why it matters

- **Parameter-efficient reasoning.** A 2.6B looped model with T=64 can approach the quality of much larger dense models on reasoning tasks, at a small memory footprint.
- **Natural fit for inference-time scaling.** T can be varied at inference, trading latency for quality without retraining.
- **New inference-efficiency surface.** Methods like early-exit (quit early for easy tokens) and [late-entry](../inference/late-entry.md) (skip the first K passes for newly appended tokens on a cached context) are only possible because of the looped structure.

---

## Gotchas & tricks

- **Compute per token scales with T.** Looped LMs are *not* free — memory-efficient, compute-expensive. The win is quality-per-parameter, not quality-per-flop.
- **Weight-tied training needs care.** Gradients flow through all T unrolls; depending on T this can be numerically delicate (gradient explosion in deep loops). Norm / init tricks matter.
- **Different from Mixture-of-Depths.** MoD routes tokens to different depths dynamically; looped LMs share weights across a *fixed* depth loop. The two are orthogonal and can be stacked.
- **KV cache behavior.** Within a single token's T passes, all passes see the same KV from cached context, so cache memory doesn't multiply with T — only compute does.

---

## Sources

- Paper: *Late Entry in Looped Language Models* — 2026 — [arXiv:2611.24576](https://arxiv.org/abs/2611.24576). Introduces late-entry inference and references Ouro/Huginn with hyperparameters.
- Paper: *Universal Transformers* — Dehghani et al., 2018 — the ancestor of the looped family.
- Model: Ouro-1.4B / 2.6B and Huginn-3.5B — open-weight looped reasoning LMs referenced in late-entry.
