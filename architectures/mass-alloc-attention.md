# Mass Allocation Attention (MALA)
*Depth — a fused attention primitive that keeps full causal scoring but skips post-score work on interactions carrying negligible normalized mass.*

**TL;DR:** MALA reads the online-softmax normalizer built during dense attention and uses it in-flight to decide which value projections (and, at the backward pass, which dV contributions) to actually compute. A single scalar tolerance shared across training and inference gives distribution-adaptive sparsity without changing model semantics.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](../architectures/multi-head-attention.md)
**Related:** [_attention-sparsity](_attention-sparsity.md), [mla](mla.md), [dca](../fundamentals/dca.md)

---

## What it is

An attention primitive that preserves the full QK score matrix (so no interaction is silently dropped from the causal path) but replaces the "always compute the whole post-score sequence" pattern of dense attention with a per-token retention gate driven by normalized attention contribution. It targets the situation where FullAttn assigns negligible mass to most of the causal score space but still performs the full post-score work anyway.

## How it works

- **Forward.** During the fused kernel, MALA runs the standard online-softmax normalizer as usual. At each block it compares each interaction's normalized contribution to a scalar tolerance `τ`; contributions below `τ` are treated as retained but with post-score work (V projection, `p*V` accumulation) omitted. The normalizer itself is *not* modified, so downstream numerical invariants hold.
- **Backward.** MALA reuses the finalized normalizer from the forward pass to derive the nested retained support, so dV, dK, dQ can be computed only on the retained set using standard attention state — no auxiliary index or sketch.
- **Single knob.** Training and inference share the same `τ`, which keeps the training/inference gap small and avoids the mode-switching issue seen when sparse attention is only used at serving time.

## Why it matters

Existing attention-sparsity families (top-k, block-sparse, sliding window) trade correctness for speed via a rule *chosen before you see the distribution*. MALA turns that rule into a quantity *read off the softmax itself*, so it adapts per query-block. Reported wins are large on long contexts: at 128K tokens with tensor parallelism, MALA cuts training forward/backward attention latency by 2.2×/3.0× and decoding latency by 1.6× vs FullAttn, while tracking FullAttn perplexity in 0.6B–14B scaling-law runs.

## Gotchas & tricks

- The reported gains are on the **post-score** work, not on the score matrix itself, which still has quadratic memory. MALA does not by itself unlock context lengths dense attention can't fit — pair with paged KV or a streaming attention layout for that.
- `τ` interacts with model behavior: too tight → measurable perplexity gap; too loose → no speedup. The paper's sweet spot leaves omitted-mass ≈ 0.02% at matched work.
- Because backward reuses the forward normalizer, activation checkpointing must save the normalizer state along with the retention mask — otherwise dK/dV drift from FullAttn.

## Sources

- Paper: MassAlloc Attention: Let Attention Allocate Its Own Compute — Shi et al., 2026 — [arXiv:2609.32712](https://arxiv.org/abs/2609.32712)
