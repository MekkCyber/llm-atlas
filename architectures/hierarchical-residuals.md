# Hierarchical Residuals (HiRes)
*Depth — widen the residual stream without widening per-layer compute.*

**TL;DR:** The standard Transformer residual connection ties the width of the residual stream to the compute width inside each block, so scaling one forces you to scale the other. HiRes decouples them by running PreNorm blocks at compute width and wrapping the stack in a wider PostNorm residual with learned projections at block boundaries. Only stock operators (dense projections, RMSNorm) — no new ops — and reports a 1.30× FLOPs-efficiency gain across sizes.

**Prereqs:** [transformer-block](transformer-block.md), [_normalization](_normalization.md)
**Related:** [reordered-norm](reordered-norm.md), [qk-norm](qk-norm.md), [multi-head-attention](multi-head-attention.md), [_training-stability](../pre-training/_training-stability.md)

---

## What it is

In a vanilla Transformer, each block looks like:

```
x = x + Attn(Norm(x))
x = x + MLP(Norm(x))
```

The residual stream `x` has the same width `d_model` as the attention and MLP, so the three quantities are coupled. If you want more residual capacity, you must pay the quadratic attention and MLP cost for it.

HiRes keeps the inside of each block the same (PreNorm, compute width `d_c`) but wraps *groups* of blocks inside an **outer** residual at a larger width `d_r > d_c`, with learned down- and up-projections at the boundaries:

```
h_outer = PostNorm(h_outer + Up( Blocks_PreNorm( Down(h_outer) ) ))
```

Residual width `d_r` becomes a separate scaling knob from compute width `d_c`.

## How it works

Three pieces:

1. **Compute-width inner blocks.** Standard PreNorm attention + MLP at width `d_c`. No changes to any familiar operator.
2. **Residual-width outer stream.** A wider stream `h_outer ∈ R^(T × d_r)` runs across the whole network.
3. **Boundary projections.** Before entering a block (or group of blocks), `Down: d_r → d_c` compresses to compute width; after the block, `Up: d_c → d_r` lifts back. The outer residual uses PostNorm (`Norm(h_outer + ...)`) so the high-width stream is renormalized on every outer step.

Blocks can be grouped — several PreNorm inner blocks inside one Up/Down pair — which amortizes the projection cost across depth. Projections are dense matrix multiplies, so the entire scheme stays inside standard dense + norm primitives.

Memory-wise, activations at `d_r` are only kept at outer-residual boundaries; the hot path inside the block is at `d_c`. Residual-write I/O and activation memory therefore track `d_c` more than `d_r`.

## Why it matters

- **A new, cheap scaling axis.** For years the implicit rule "residual width = compute width" was treated as a constraint. HiRes shows it was a bottleneck. You can now trade residual capacity (which is mostly about *carrying* information across layers) for compute capacity (which is about *transforming* it).
- **FLOPs-efficient.** Reports ~1.30× FLOPs-efficiency improvement across sizes and depths versus vanilla residuals. On OLMoE-1B-7B pretraining over 500B tokens, lower training loss and higher downstream accuracy.
- **Low implementation risk.** The paper is explicit: only dense projections and normalizations. Any training stack that supports a Transformer supports HiRes.
- **Competitive with other residual-widening schemes at lower cost.** Alternative "wide residual" ideas exist (parallel residual branches, DeepNet, residual pre-conditioning) — HiRes matches or beats them at lower activation memory and I/O.

## Gotchas & tricks

- **PreNorm inside, PostNorm outside.** The paper is specific about the ordering: inner blocks stay PreNorm (for stable deep training), the outer residual uses PostNorm (to normalize the accumulated outer stream). Mixing the two orientations is doing real work; don't reverse them.
- **Grouping matters.** Fewer, bigger groups mean fewer `Up/Down` projections (cheaper) but less outer-stream renormalization (less stable at depth). The paper's recipes group a handful of inner blocks per outer-residual step.
- **Init the projections carefully.** `Up` and `Down` are the only new learnable modules. The paper uses standard dense initialization; the outer-residual gain starts around 1 so the model behaves like vanilla early in training.
- **Adds parameters.** `Up` and `Down` are `d_r × d_c` each — sometimes non-negligible at small models. The FLOPs-efficiency number is reported inclusive of these.
- **Not a positional encoding fix.** HiRes is orthogonal to RoPE/GQA/MLA — it operates on the residual stream, not inside attention. You compose it with whatever attention you already use.

## Sources

- Paper: *Hierarchical Residuals: A Minimalist Decoupling of Residual Width from Compute Width* — Wang, Lu, Ma, Zhang, Zhu, Ma, Chen, 2026 — arXiv 2610.41711.
- Paper: *OLMoE: Open Mixture-of-Experts Language Models* — Muennighoff et al., 2024 — the OLMoE-1B-7B pretraining setup used for HiRes's main scaling experiments.
