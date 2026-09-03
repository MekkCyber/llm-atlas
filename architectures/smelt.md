# SMELT (Sparse MoE with Middle Layers Looped Twice)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A layer-looping recipe for MoE transformers that iterates a shared block of layers to gain effective depth *without* growing parameters, FLOPs, or KV cache. SMELT loops the **middle half of layers twice** while matching per-token FLOPs, total non-embedding parameters, and KV cache against an unlooped MoE baseline. Compute-optimal Chinchilla-style fits show 6.8–18.0% training-FLOPs savings up to 54B non-embedding parameters.

**Prereqs:** [_moe.md](_moe.md), [transformer-block.md](transformer-block.md)
**Related:** [deepseek-moe.md](deepseek-moe.md), [aux-loss-free-balancing.md](aux-loss-free-balancing.md)

---

## What it is

**Looped transformers** iterate a shared block of layers so the model has more effective depth than its parameter count implies. Prior evaluations were unfair: they compared looped and unlooped models at fixed model size, so any gain conflated the loop with the extra FLOPs from re-executing layers.

SMELT is the *budget-matched* version for MoE. It loops **only the middle half of layers**, exactly **twice**, and matches the unlooped baseline on three budgets at once:

- per-token FLOPs,
- total non-embedding parameters,
- KV-cache size.

Under those constraints, any observed gain is attributable to looping, not to more compute.

## How it works

Let the model have $L$ layers, all MoE. SMELT partitions them into three groups:

1. **Early layers** ($L/4$): unshared, run once.
2. **Middle layers** ($L/2$): shared block, run **twice** back-to-back.
3. **Late layers** ($L/4$): unshared, run once.

Compared to an unlooped baseline with the same active parameter count and FLOP budget, SMELT reduces distinct-layer count (fewer unique weights) but doubles the effective depth of the middle block. The paper searches loop position (early / middle / late) and count (2× / 3× / …) and settles on **middle + twice** as the compute-optimal setting.

### Mechanistic signature

The paper analyzes what changes on the second visit through the looped block:

- **Attention sink reduction.** On the second visit, less attention mass concentrates on the sink token (usually BOS).
- **Content-token reallocation.** The freed mass moves to content-relevant tokens, giving the model an in-context refinement pass.

This is the causal story for the observed downstream gains — the loop acts as a self-refinement operator rather than a raw depth increase.

## Why it matters

- **Real architectural win under honest accounting.** SMELT is the first compute-matched scaling-law study to show looping is a genuine architectural improvement for MoE, not a FLOPs-injection artifact.
- **Downstream > loss gain.** Benchmark improvements exceed what validation-loss deltas predict, with **largest gains on code** and gains that **grow with sequence length and in-context example count**. Consistent with the "self-refinement" story.
- **Composes with existing MoE recipes.** Orthogonal to routing algorithm, balancer, and granularity; drops into DeepSeekMoE-style or Mixtral-style stacks.
- **Scaling laws separately fit.** SMELT's Chinchilla fit has a steeper loss-vs-compute slope than the baseline's, so the gap grows at frontier scale.

## Gotchas & tricks

- **Loop position matters.** Looping early or late layers is strictly worse in the paper's ablations. Middle-only preserves the input-embedding and output-projection roles of the outer layers.
- **Two visits, not more.** 3× or 4× loops start to overfit the middle representation and hurt loss. Twice is a sweet spot in the search.
- **Budget matching is non-trivial.** To match per-token FLOPs while looping, the unshared layers or MoE granularity must be adjusted; a naive loop increases FLOPs. Reproduce the paper's compute accounting before comparing.
- **KV-cache matching cuts inference concerns.** Because KV cache is matched, SMELT's inference-time memory is the baseline's — no hidden cost for the loop.
- **Second-visit attention sink is the signature.** In interp studies, this metric distinguishes SMELT's loop from other effective-depth tricks.

## Sources

- Paper: *SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers* — Wang et al., 2026 — [arXiv:2609.01343](https://arxiv.org/abs/2609.01343).
