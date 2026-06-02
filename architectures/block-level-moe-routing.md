# Block-Level MoE Routing

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A routing variant for MoE layers in *parallel-decoded* models (notably diffusion LLMs). Standard token-level top-K routing inside a block activates almost every expert, because each token picks independently. Block-level routing aggregates the per-token routing distributions into a single block-level distribution and routes the entire block coherently — slashing unique-expert activations per block from ~70 → ~15 with <1% quality loss.

**Prereqs:** [_moe](_moe.md), [load-balancing-loss](load-balancing-loss.md)
**Related:** [deepseek-moe](deepseek-moe.md) · [aux-loss-free-balancing](aux-loss-free-balancing.md) · [capacity-factor](capacity-factor.md)

---

## What it is

In autoregressive transformers, MoE layers route one token at a time, and the next token's routing depends on the previous token's output — so the *natural unit* of MoE routing is a token. In parallel-decoded models (diffusion LLMs, MaskGIT-style, block-parallel inference) an entire block of B tokens is denoised or revealed in one forward pass. Token-level routing in this regime means B independent expert selections per layer; with top-K = 8 and B = 16, you can easily activate 60–80 distinct experts per block — defeating the memory-bandwidth advantage that MoE inference relies on.

Block-level MoE routing changes the unit. The block has one routing distribution; one set of top-K experts processes every token in it. Expert weights are read from HBM once per block instead of once per token.

---

## How it works

### Aggregation rule

For each block of B tokens at MoE layer ℓ, compute per-token routing scores as usual:

```
s_{t,i} = softmax( router(h_t) )_i        for token t, expert i
```

Aggregate to a block-level score by averaging (or max-pooling) across tokens in the block:

```
S_i = (1/B) · Σ_t s_{t,i}
```

Select the block's experts as the top-K of `S_i`. Every token in the block is then routed only to that shared set of experts.

The token's gating weights are still per-token: `g_{t,i} = s_{t,i} / Σ_{j ∈ Top-K} s_{t,j}`. So the per-token output is still a token-specific weighted sum — what changes is the *set* of experts read from memory.

### Why this works for diffusion LLMs

Within a block, tokens have **bidirectional dependencies** under the denoising/masking objective. Adjacent tokens being processed by *different* experts forces redundant cross-expert communication that the bidirectional context already implies should be shared. The block-aggregated distribution captures this correlation explicitly.

### What it doesn't change

The expert layer's weight matrices, the router, the load-balancing loss, and the inference graph are all reused as-is. Only the routing decision boundary moves from per-token to per-block.

---

## Why it matters

- **Restores MoE's memory-bandwidth advantage in parallel-decoded models.** The whole reason to use MoE is that activated params ≪ total params. Token-level routing in diffusion LLMs silently activates most experts and burns through HBM bandwidth. dMoE fixes this without touching anything else.
- **76–80% memory reduction and 1.14–1.66× latency speedup** at 99.11% retained quality, per the source paper across multiple benchmarks.
- **Likely to become the default for dLLM × MoE.** As diffusion LLMs scale, MoE will be standard for capacity. Block-level routing is the natural counterpart.

---

## Gotchas & tricks

- **Block size matters.** Tiny blocks (B = 2–4) recover only modest savings; huge blocks (B = 64+) start to lose per-token specialization. The paper's sweet spot is in the 8–32 range.
- **Aggregation choice is a knob.** Mean-pool is the default. Max-pool routes by the most confident token's preference (more specialization, more imbalance). The choice interacts with the load-balance loss.
- **Load balancing still required.** Block routing does not free you from balancing experts — the block-level distribution can collapse just like a token-level one. Reuse [load-balancing-loss](load-balancing-loss.md) or [aux-loss-free-balancing](aux-loss-free-balancing.md), now computed on block-level statistics.
- **Not for standard autoregressive models.** Autoregressive decoding processes one token at a time anyway; the memory amortization is already token-local. Block routing is specifically a parallel-decoding optimization.

---

## Sources

- Paper: *dMoE: dLLMs with Learnable Block Experts* — Feng, Chen, Fang, Ma, Wang, 2026 — National University of Singapore. Introduces block-level MoE routing for diffusion LLMs; reports 76.64–79.84% memory reduction and 1.14–1.66× speedup at 99.11% quality.
- Code: https://github.com/fscdc/dMoE
